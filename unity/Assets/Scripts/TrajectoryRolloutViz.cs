using System;
using System.Collections.Generic;
using UnityEngine;
// Active Input Handling is "Input System Package (New)" only, so the legacy
// UnityEngine.Input.GetKeyDown throws at runtime - use the new API.
using UnityEngine.InputSystem;
// Alias only the type we need rather than `using RosMessageTypes.Std;` - that
// namespace also defines a `Time` message which collides with UnityEngine.Time.
using StringMsg = RosMessageTypes.Std.StringMsg;

/// <summary>
/// Renders the policy's candidate future trajectories as a fan of lines in
/// the live Unity client. The trainer (see rl_agent/rollout_viz.py) samples
/// K action sequences (each H steps) from the policy's action head and
/// publishes them on the `policy_rollouts` topic as a std_msgs/String JSON
/// payload. We forward-simulate each sequence with a kinematic bicycle model
/// from the car's current pose and draw it with a pooled LineRenderer.
///
/// Phase 1 is OPEN-LOOP (see docs/trajectory-rollout-viz.md): each sequence
/// is i.i.d. samples from the CURRENT distribution, so the fan shows the
/// policy's immediate action *spread*, not a true closed-loop prediction.
///
/// Setup: drop this component on any GameObject in the scene (e.g. the
/// SimController object). It self-subscribes; the car is looked up lazily
/// (it's re-instantiated on every episode reset). Only the Unity client whose
/// ros-server the trainer publishes to (actor 0) receives messages, so no
/// per-client gating is needed here.
/// </summary>
public class TrajectoryRolloutViz : MonoBehaviour
{
    [Header("Topic")]
    [Tooltip("ROS topic the trainer publishes rollout payloads on.")]
    public string topic = "policy_rollouts";

    [Header("Bicycle model (calibrate against CarController)")]
    [Tooltip("Front-to-rear axle distance (m). Used for the kinematic yaw rate. "
             + "Tune so predicted curves match the real car's turning.")]
    public float wheelbase = 2.0f;
    [Tooltip("Accel command -> acceleration scale (m/s^2 per unit command). "
             + "Tune so the predicted speed change matches the real car.")]
    public float kAccel = 6.0f;
    [Tooltip("Speed clamp (m/s) for the forward sim.")]
    public float vMax = 30.0f;
    [Tooltip("If > 0, override the car's maxSteeringAngle; else read it from "
             + "CarController at runtime.")]
    public float steeringAngleDegOverride = 0f;

    [Header("Rendering")]
    public Color lineColor = new Color(0.2f, 0.9f, 1f, 0.9f);
    // Default tuned for a FAR overhead camera (whole track in view): a thin
    // 0.25m line is sub-pixel from there. Bump up if still hard to see.
    public float lineWidth = 1.5f;
    [Tooltip("Vertical offset above the car's Y so lines float just over the road.")]
    public float yOffset = 0.5f;
    [Tooltip("Forward distance (m) from the car pivot to the fan origin, so the "
             + "trajectories emanate from the FRONT of the car (Waymo-style) "
             + "rather than its center.")]
    public float frontOffset = 2.2f;
    [Tooltip("Re-simulate + redraw the fan from the car's CURRENT pose at this "
             + "rate (Hz). Decoupled from the (sparse) publish rate so the fan "
             + "tracks the moving car smoothly instead of freezing at the spot "
             + "where the last payload arrived.")]
    public float redrawHz = 20f;
    [Tooltip("Hide the fan if no payload arrives within this many seconds. "
             + "Set generously: EVAL publishes are sparse (one per episode). The "
             + "fan keeps redrawing from the latest actions until this elapses, "
             + "so a frozen ghost only appears if training fully stops.")]
    public float staleTimeoutSeconds = 30.0f;
    [Tooltip("Hard cap on rendered trajectories (safety vs. a huge K).")]
    public int maxLines = 64;

    [Header("Probability weighting")]
    [Tooltip("Color + width each trajectory by its relative probability "
             + "(weights[] from the policy), so the most-/least-likely paths "
             + "are easy to tell apart.")]
    public bool weightByProbability = true;
    [Tooltip("Single trajectory color; probability is conveyed by ALPHA "
             + "(more transparent = less likely) and width.")]
    public Color probColor = new Color(0.10f, 0.45f, 1f);
    [Tooltip("Opacity of the LEAST- / MOST-likely trajectory (exponential "
             + "falloff: lower probability fades to more transparent). minAlpha "
             + "is the floor so alternative paths stay visible.")]
    public float minAlpha = 0.30f;
    public float maxAlpha = 0.95f;
    [Tooltip("Exponential falloff for both alpha-fading and width-shrinking as "
             + "probability drops. Lower = SMOOTHER/gentler decay so more "
             + "alternative paths are visible; higher = sharper emphasis on the "
             + "single most-likely path.")]
    public float probFalloff = 1.0f;
    [Tooltip("Width scale for the least- / most-likely trajectory.")]
    public float minWidthScale = 0.40f;
    public float maxWidthScale = 1.6f;
    private bool _vizEnabled = true;

    [Serializable]
    private class RolloutPayload
    {
        public double stamp;
        public int step;
        public float dt;
        public int horizon;   // H
        public int k;         // K
        public float[] accel; // length k*H, k-major: accel[k*H + h]
        public float[] steer; // length k*H
        public float[] weights; // length k: relative probability per trajectory
        // Optional render-style knobs (published by Python so Unity can be
        // tuned without a rebuild). maxAlpha>0 means "style present"; otherwise
        // JsonUtility zero-fills these and we use the inspector defaults.
        public float lineWidth;
        public float probFalloff;
        public float minAlpha;
        public float maxAlpha;
        public float minWidth;
        public float maxWidth;
        public string mode;   // "train" / "eval" (for the HUD)
        public int actor;     // actor index (for the HUD cross-check)
    }

    // Latest train/eval mode reported by the trainer (consumed by HudOverlay).
    // Empty when no recent payload has arrived.
    private string _lastMode = "";
    private float _lastModeTime = -999f;
    public string CurrentMode =>
        (Time.time - _lastModeTime) < 5f ? _lastMode : "";

    private readonly List<LineRenderer> _pool = new List<LineRenderer>();
    private Material _lineMaterial;
    private CarController _car;
    private RolloutPayload _pending;   // set on ROS thread, handed off on main thread
    private RolloutPayload _active;    // latest payload, kept for continuous redraw
    private readonly object _lock = new object();
    private float _lastDataTime = -999f;  // when the last payload arrived
    private float _redrawAccum = 0f;      // time since last redraw
    private bool _subscribed = false;

    void Start()
    {
        TrySubscribe();
    }

    void TrySubscribe()
    {
        if (_subscribed) return;
        var ros = ROSConnection.instance;
        if (ros == null) return;  // retried in Update until the singleton exists
        ros.Subscribe<StringMsg>(topic, OnRollouts);
        _subscribed = true;
        Debug.Log($"[TrajectoryRolloutViz] subscribed to '{topic}'");
    }

    // ROS callback may run off the main thread; just stash the parsed payload
    // and do all Unity API work (transforms, LineRenderers) in Update().
    void OnRollouts(StringMsg msg)
    {
        if (msg == null || string.IsNullOrEmpty(msg.data)) return;
        try
        {
            var payload = JsonUtility.FromJson<RolloutPayload>(msg.data);
            if (payload != null && payload.accel != null && payload.steer != null)
            {
                lock (_lock) { _pending = payload; }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[TrajectoryRolloutViz] payload parse failed: {e.Message}");
        }
    }

    void Update()
    {
        if (!_subscribed) { TrySubscribe(); }

        var kb = Keyboard.current;
        if (kb != null && kb.tKey.wasPressedThisFrame)
        {
            _vizEnabled = !_vizEnabled;
            if (!_vizEnabled)
            {
                for (int i = 0; i < _pool.Count; i++)
                    if (_pool[i].enabled) _pool[i].enabled = false;
            }
            Debug.Log($"[TrajectoryRolloutViz] fan {( _vizEnabled ? "ON" : "OFF")}");
        }
        if (!_vizEnabled)
        {
            // Drain any pending payload so we don't draw a stale burst on re-enable.
            lock (_lock) { _pending = null; }
            return;
        }

        // Pull any newly-arrived payload into the active cache. We keep the
        // latest actions and re-simulate them from the car's CURRENT pose on a
        // fixed cadence below, rather than drawing once when it arrives.
        RolloutPayload fresh = null;
        lock (_lock) { if (_pending != null) { fresh = _pending; _pending = null; } }
        if (fresh != null)
        {
            _active = fresh;
            _lastDataTime = Time.time;
            _lastMode = fresh.mode ?? "";
            _lastModeTime = Time.time;
            Debug.Log($"[TrajectoryRolloutViz] new payload step={fresh.step} "
                      + $"K={fresh.k} H={fresh.horizon} "
                      + $"accelLen={(fresh.accel == null ? -1 : fresh.accel.Length)}");
        }

        // No data for a while -> hide the fan (training stopped).
        if (_active == null || Time.time - _lastDataTime > staleTimeoutSeconds)
        {
            HideAll();
            return;
        }

        // Redraw at redrawHz from the car's current pose so the fan tracks the
        // moving car smoothly (Waymo-style), decoupled from the publish rate.
        _redrawAccum += Time.deltaTime;
        float period = 1f / Mathf.Max(1f, redrawHz);
        if (_redrawAccum >= period)
        {
            _redrawAccum = 0f;
            DrawPayload(_active);
        }
    }

    void HideAll()
    {
        for (int i = 0; i < _pool.Count; i++)
            if (_pool[i].enabled) _pool[i].enabled = false;
    }

    // Map a normalized probability p in [0,1] through an exponential curve:
    // e(0)=0, e(1)=1, convex for falloff>0 so the most-likely path stays bold
    // (deep blue / wide) while lower-probability paths fall off fast to lighter
    // + thinner. falloff=0 degrades to linear.
    static float ExpNorm(float p, float falloff)
    {
        p = Mathf.Clamp01(p);
        if (Mathf.Abs(falloff) < 1e-4f) return p;
        return (Mathf.Exp(falloff * p) - 1f) / (Mathf.Exp(falloff) - 1f);
    }

    CarController ResolveCar()
    {
        if (_car != null) return _car;
        // The car is destroyed + re-instantiated on every reset, so re-find it.
        _car = FindObjectOfType<CarController>();
        return _car;
    }

    void DrawPayload(RolloutPayload p)
    {
        var car = ResolveCar();
        if (car == null)
            return;  // car briefly absent during episode reset; silently skip

        int H = Mathf.Max(1, p.horizon);
        int K = Mathf.Max(0, p.k);
        // Defensive: the flat arrays must hold K*H entries.
        if (p.accel.Length < K * H || p.steer.Length < K * H)
        {
            K = Mathf.Min(K, Mathf.Min(p.accel.Length, p.steer.Length) / H);
        }
        K = Mathf.Min(K, maxLines);

        Transform t = car.transform;
        // Originate from the FRONT of the car (Waymo-style) rather than the
        // pivot: push forward along the car's heading by frontOffset.
        Vector3 origin = t.position + t.forward * frontOffset + Vector3.up * yOffset;
        float yaw0 = t.eulerAngles.y * Mathf.Deg2Rad;  // Unity yaw: CW from +z
        float v0 = 0f;
        try { v0 = car.GetSpeed(); } catch { v0 = 0f; }

        float steerMaxDeg = steeringAngleDegOverride > 0f
            ? steeringAngleDegOverride : Mathf.Max(1f, car.maxSteeringAngle);
        float dt = p.dt > 0f ? p.dt : 0.1f;

        // Effective render style: published payload knobs override the inspector
        // defaults when present (maxAlpha>0 = "style present"), so opacity/width
        // can be tuned from Python (env vars) without a Unity rebuild.
        bool hasStyle = p.maxAlpha > 0f;
        float effFalloff = hasStyle ? p.probFalloff : probFalloff;
        float effMinAlpha = hasStyle ? p.minAlpha : minAlpha;
        float effMaxAlpha = hasStyle ? p.maxAlpha : maxAlpha;
        float effMinWidth = hasStyle ? p.minWidth : minWidthScale;
        float effMaxWidth = hasStyle ? p.maxWidth : maxWidthScale;
        float effLineWidth = (hasStyle && p.lineWidth > 0f) ? p.lineWidth : lineWidth;

        // Probability weighting: normalize the K weights to [0,1] (min-max) so
        // the most-likely trajectory is brightest/thickest and the least-likely
        // is faint/thin. Falls back to all-equal if no weights were published.
        bool haveW = weightByProbability && p.weights != null
                     && p.weights.Length >= K && K > 0;
        float wmin = 0f, wmax = 1f;
        if (haveW)
        {
            wmin = float.MaxValue; wmax = float.MinValue;
            for (int k = 0; k < K; k++)
            {
                float w = p.weights[k];
                if (w < wmin) wmin = w;
                if (w > wmax) wmax = w;
            }
        }

        for (int k = 0; k < K; k++)
        {
            LineRenderer lr = GetLine(k);

            float w01 = 1f;
            if (haveW)
                w01 = (wmax > wmin)
                    ? Mathf.Clamp01((p.weights[k] - wmin) / (wmax - wmin)) : 1f;
            // Opacity decays EXPONENTIALLY (not linearly) as the probability
            // rank drops: alpha = maxAlpha * exp(-falloff*(1-w01)), floored at
            // minAlpha. So each notch less likely multiplies the opacity down.
            // With uniform weights (eval) w01=1 -> full alpha + width.
            float a = Mathf.Max(effMinAlpha,
                                effMaxAlpha * Mathf.Exp(-effFalloff * (1f - w01)));
            Color c = new Color(probColor.r, probColor.g, probColor.b, a);
            lr.startColor = c;
            lr.endColor = new Color(c.r, c.g, c.b, a * 0.6f);
            float ww = effLineWidth * Mathf.Lerp(effMinWidth, effMaxWidth,
                                                 ExpNorm(w01, effFalloff));
            lr.startWidth = ww;
            lr.endWidth = ww * 0.5f;

            lr.positionCount = H + 1;
            lr.SetPosition(0, origin);

            float x = origin.x, z = origin.z, yaw = yaw0, v = v0;
            for (int h = 0; h < H; h++)
            {
                int idx = k * H + h;
                float accel = Mathf.Clamp(p.accel[idx], -1f, 1f);
                float steer = Mathf.Clamp(p.steer[idx], -1f, 1f);
                float steerRad = steer * steerMaxDeg * Mathf.Deg2Rad;

                v = Mathf.Clamp(v + kAccel * accel * dt, 0f, vMax);
                yaw += (v / Mathf.Max(0.01f, wheelbase)) * Mathf.Tan(steerRad) * dt;
                x += v * Mathf.Sin(yaw) * dt;
                z += v * Mathf.Cos(yaw) * dt;
                lr.SetPosition(h + 1, new Vector3(x, origin.y, z));
            }
        }

        // Hide any pooled lines beyond the current K.
        for (int i = K; i < _pool.Count; i++)
        {
            if (_pool[i].enabled) _pool[i].enabled = false;
        }
    }

    LineRenderer GetLine(int index)
    {
        while (_pool.Count <= index)
        {
            var go = new GameObject($"RolloutLine_{_pool.Count}");
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = true;
            lr.material = LineMaterial();
            lr.startColor = lineColor;
            lr.endColor = new Color(lineColor.r, lineColor.g, lineColor.b, lineColor.a * 0.25f);
            lr.startWidth = lineWidth;
            lr.endWidth = lineWidth * 0.5f;
            lr.numCapVertices = 2;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;
            _pool.Add(lr);
        }
        var line = _pool[index];
        if (!line.enabled) line.enabled = true;
        return line;
    }

    Material LineMaterial()
    {
        if (_lineMaterial == null)
        {
            // Sprites/Default is an always-available unlit, vertex-color-aware
            // shader - good for thin translucent lines without lighting setup.
            var shader = Shader.Find("Sprites/Default");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null)
            {
                // Both stripped from the build -> lines would render as the
                // pink error material (i.e. invisible/wrong). Log loudly so
                // we know to add the shader to "Always Included Shaders".
                Debug.LogError("[TrajectoryRolloutViz] no usable line shader found "
                               + "(Sprites/Default + Unlit/Color both missing - likely "
                               + "stripped from the build). Lines may not render.");
                shader = Shader.Find("Legacy Shaders/Diffuse");
            }
            else
            {
                Debug.Log($"[TrajectoryRolloutViz] line shader = '{shader.name}'");
            }
            _lineMaterial = new Material(shader);
        }
        return _lineMaterial;
    }
}
