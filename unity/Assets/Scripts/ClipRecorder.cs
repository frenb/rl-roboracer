using System;
using System.Collections;
using UnityEngine;
using StringMsg = RosMessageTypes.Std.StringMsg;

/// <summary>
/// Follow-camera frames for training clips (rl_agent/media.py). The trainer
/// sends <c>{"record": true, "fps": 10, "width": 1280, "max_s": 30}</c> on
/// <c>clip_control</c> and gets 16:9 JPEG frames back on <c>camera/overhead</c>,
/// header.seq counting from 0, until <c>{"record": false}</c> or max_s.
///
/// Frames come from a camera of its own rather than the window's: top-down and
/// orthographic, on SimController's current car (a reset spawns a new one),
/// turned with it so the road ahead points up, and zoomed to a fixed number of
/// car lengths so every course frames alike - the walls are no guide, since
/// some courses only fence their outside. Auto-attached by SimController.
/// </summary>
public class ClipRecorder : MonoBehaviour
{
    public const string ControlTopic = "clip_control";
    public const string FrameTopic = "camera/overhead";
    const string FrameId = "clip_follow";
    const int JpegQuality = 90;

    // Framing, in sim units unless noted.
    const float CarLengthsTall = 7f;       // frame height, in car lengths
    const float DefaultCarLength = 1.5f;   // until the car's meshes have been measured
    const float LookAhead = 0.4f;          // fraction of the half-height shifted ahead of the car
    const float CameraHeight = 60f;
    const float PositionTime = 0.25f;      // smoothing time constants, seconds
    const float ZoomTime = 0.6f;
    const float HeadingTime = 0.35f;
    const float SnapDistance = 1f;         // in half-heights; further than this is a reset, not driving
                                           // (a spin moves the look-ahead point at most 2 x LookAhead)

    [Serializable]
    class Control
    {
        public bool record;
        public float fps = 10f;
        public int width = 1280;
        public float max_s = 30f;
    }

    readonly object _lock = new object();
    Control _pending;
    bool _subscribed;
    ROSConnection _ros;
    Coroutine _recording;
    RenderTexture _rt;
    Texture2D _tex;

    UnityEngine.Camera _cam;
    SimController _sim;
    GameObject _car;
    float _carLength;
    bool _following;
    bool _haveView;
    Vector3 _center;
    float _halfHeight;
    float _heading;

    void Update()
    {
        if (!_subscribed) Subscribe();
        Control control;
        lock (_lock)
        {
            control = _pending;
            _pending = null;
        }
        if (control == null) return;
        if (_recording != null)
        {
            StopCoroutine(_recording);
            _recording = null;
            _following = false;
        }
        if (control.record) _recording = StartCoroutine(Record(control));
    }

    void LateUpdate()
    {
        if (_following) Follow(Time.unscaledDeltaTime);
    }

    void Subscribe()
    {
        _ros = ROSConnection.instance;
        if (_ros == null) return;  // retried next frame until the singleton exists
        _ros.Subscribe<StringMsg>(ControlTopic, OnControl);
        _subscribed = true;
    }

    // ROS callbacks may run off the main thread; hand over and act in Update.
    void OnControl(StringMsg msg)
    {
        if (msg == null || string.IsNullOrEmpty(msg.data)) return;
        try
        {
            var control = JsonUtility.FromJson<Control>(msg.data);
            lock (_lock) { _pending = control; }
        }
        catch (Exception e)
        {
            Debug.LogWarning("[ClipRecorder] bad " + ControlTopic + ": " + e.Message);
        }
    }

    IEnumerator Record(Control control)
    {
        float interval = 1f / Mathf.Clamp(control.fps, 1f, 30f);
        float stopAt = Time.unscaledTime + Mathf.Clamp(control.max_s, 1f, 120f);
        int width = Mathf.Clamp(control.width, 64, 1920) & ~1;
        int height = Mathf.Max(2, Mathf.RoundToInt(width * 9f / 16f) & ~1);
        EnsureBuffers(width, height);
        EnsureCamera();
        _haveView = false;
        _following = true;
        Follow(0f);
        var endOfFrame = new WaitForEndOfFrame();
        uint seq = 0;
        float next = Time.unscaledTime;
        Debug.Log("[ClipRecorder] recording " + width + "x" + height + " at " + (1f / interval) + " fps");
        while (Time.unscaledTime < stopAt)
        {
            if (Time.unscaledTime >= next)
            {
                // A slow frame skips ahead rather than bursting to catch up.
                next = Mathf.Max(next + interval, Time.unscaledTime);
                yield return endOfFrame;
                if (_haveView) Publish(width, height, seq++);
            }
            yield return null;
        }
        Debug.Log("[ClipRecorder] stopped after " + seq + " frames");
        _following = false;
        _recording = null;
    }

    void EnsureCamera()
    {
        if (_cam != null) return;
        var go = new GameObject("ClipFollowCamera");
        go.hideFlags = HideFlags.HideAndDontSave;
        _cam = go.AddComponent<UnityEngine.Camera>();
        _cam.enabled = false;  // rendered by hand, only when a frame is due
        _cam.orthographic = true;
        _cam.clearFlags = CameraClearFlags.SolidColor;
        _cam.backgroundColor = Color.black;
        _cam.nearClipPlane = 0.3f;
        _cam.farClipPlane = CameraHeight * 2f;
        _cam.aspect = 16f / 9f;
        _cam.cullingMask = ~(1 << FlyBrainViz.OverlayLayer);
    }

    /// <summary>Move the follow camera toward the framing the car's position
    /// asks for; <paramref name="dt"/> 0 or a reset-sized jump snaps to it.</summary>
    void Follow(float dt)
    {
        if (_sim == null) _sim = GetComponent<SimController>();
        GameObject car = _sim != null ? _sim.car : null;
        if (car == null || _cam == null) return;
        if (car != _car)
        {
            // A new car is a reset, wherever it spawned: jump to it rather than glide.
            _haveView = false;
            _car = car;
            _carLength = MeasureLength(car);
            Debug.Log("[ClipRecorder] following " + car.name + ", length " + _carLength.ToString("F2"));
        }

        Transform t = car.transform;
        Vector3 fwd = Vector3.ProjectOnPlane(t.forward, Vector3.up);
        if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
        fwd.Normalize();

        float wantHalfHeight = _carLength * CarLengthsTall * 0.5f;
        float wantHeading = Mathf.Atan2(fwd.x, fwd.z) * Mathf.Rad2Deg;

        bool snap = !_haveView || dt <= 0f;
        if (!snap)
        {
            Vector3 aheadNow = t.position + fwd * (LookAhead * _halfHeight);
            snap = Vector3.Distance(Flat(aheadNow), Flat(_center)) > SnapDistance * _halfHeight;
        }
        if (snap)
        {
            _halfHeight = wantHalfHeight;
            _heading = wantHeading;
            _center = t.position + fwd * (LookAhead * _halfHeight);
            _haveView = true;
        }
        else
        {
            _halfHeight = Mathf.Lerp(_halfHeight, wantHalfHeight, Ease(dt, ZoomTime));
            _heading = Mathf.LerpAngle(_heading, wantHeading, Ease(dt, HeadingTime));
            Vector3 wantCenter = t.position + fwd * (LookAhead * _halfHeight);
            _center = Vector3.Lerp(_center, wantCenter, Ease(dt, PositionTime));
        }

        Vector3 up = Quaternion.Euler(0f, _heading, 0f) * Vector3.forward;
        _cam.orthographicSize = _halfHeight;
        _cam.transform.SetPositionAndRotation(_center + Vector3.up * CameraHeight,
                                              Quaternion.LookRotation(Vector3.down, up));
    }

    static float Ease(float dt, float timeConstant) => 1f - Mathf.Exp(-dt / timeConstant);

    static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

    /// <summary>The car's length along its own forward axis, from its meshes
    /// (not its renderers' world boxes, which grow as the car turns, and not
    /// the sensor-ray LineRenderers).</summary>
    static float MeasureLength(GameObject car)
    {
        Matrix4x4 toCar = car.transform.worldToLocalMatrix;
        float min = float.MaxValue, max = float.MinValue;
        foreach (var mf in car.GetComponentsInChildren<MeshFilter>())
        {
            if (mf.sharedMesh == null) continue;
            Bounds b = mf.sharedMesh.bounds;
            Matrix4x4 m = toCar * mf.transform.localToWorldMatrix;
            for (int i = 0; i < 8; i++)
            {
                var corner = new Vector3((i & 1) == 0 ? b.min.x : b.max.x,
                                         (i & 2) == 0 ? b.min.y : b.max.y,
                                         (i & 4) == 0 ? b.min.z : b.max.z);
                float z = m.MultiplyPoint3x4(corner).z * car.transform.lossyScale.z;
                min = Mathf.Min(min, z);
                max = Mathf.Max(max, z);
            }
        }
        float length = max - min;
        return length > 0.05f && length < 50f ? length : DefaultCarLength;
    }

    void Publish(int width, int height, uint seq)
    {
        if (_cam == null || _ros == null) return;

        var prev = RenderTexture.active;
        _cam.targetTexture = _rt;
        _cam.Render();
        _cam.targetTexture = null;

        RenderTexture.active = _rt;
        _tex.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
        _tex.Apply(false, false);
        RenderTexture.active = prev;

        byte[] jpg = _tex.EncodeToJPG(JpegQuality);
        var img = new RosMessageTypes.Sensor.Image
        {
            header = new RosMessageTypes.Std.Header
            {
                seq = seq,
                stamp = new RosMessageTypes.Std.Time(),
                frame_id = FrameId
            },
            height = (uint)height,
            width = (uint)width,
            encoding = "jpeg",
            is_bigendian = 0,
            step = 0,
            data = jpg
        };
        _ros.Send(FrameTopic, new RosMessageTypes.Roboracer.Camera(img));
    }

    void EnsureBuffers(int width, int height)
    {
        if (_rt != null && (_rt.width != width || _rt.height != height))
        {
            _rt.Release();
            _rt = null;
            Destroy(_tex);
            _tex = null;
        }
        if (_rt == null)
        {
            _rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            _rt.antiAliasing = 4;
            _rt.Create();
        }
        if (_tex == null) _tex = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    void OnDestroy()
    {
        if (_rt != null) _rt.Release();
        if (_tex != null) Destroy(_tex);
        if (_cam != null) Destroy(_cam.gameObject);
    }
}
