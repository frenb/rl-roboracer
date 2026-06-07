using UnityEngine;
// Active Input Handling is "Input System Package (New)" only, so the legacy
// UnityEngine.Input would throw - use the new API for the toggle key.
using UnityEngine.InputSystem;

/// <summary>
/// Real-time HUD drawn over the live Unity client: a steering-wheel icon that
/// rotates with the car's steer command, a speed readout, and the throttle
/// ("force") command. All values are read locally from the CarController each
/// frame (no ROS), so every client shows its own car. Toggle with the H key.
///
/// Self-contained IMGUI (OnGUI): the steering-wheel texture is generated
/// procedurally at startup, so there's no asset or scene setup. Auto-attached
/// by SimController (like TrajectoryRolloutViz).
/// </summary>
public class HudOverlay : MonoBehaviour
{
    [Header("Placement / size (top-middle)")]
    public float topMargin = 10f;
    public float wheelSize = 96f;
    public float panelWidth = 190f;

    [Header("Steering wheel")]
    [Tooltip("Full steer command (|angle|=1) maps to this many degrees of wheel "
             + "rotation - amplified like a real wheel (~1.25 turns each way).")]
    public float maxWheelDeg = 450f;
    [Tooltip("Flip if the wheel rotates the wrong way vs. the turn direction.")]
    public float steerSign = 1f;

    [Header("Colors")]
    public Color wheelColor = new Color(0.92f, 0.95f, 1f, 1f);

    private bool _hudOn = true;
    private CarController _car;
    private Texture2D _wheelTex;
    private Texture2D _panelTex;
    private GUIStyle _speedStyle;
    private GUIStyle _steerStyle;
    private GUIStyle _forceStyle;
    private GUIStyle _labelStyle;
    private GUIStyle _headerStyle;
    private bool _assetsReady = false;
    private TrajectoryRolloutViz _viz;   // source of the train/eval mode
    private int _clientIndex = -1;        // derived from the ROS unityPort

    string ClientName()
    {
        if (_clientIndex < 0)
        {
            // Launch convention (RosBootstrap): unityPort = 5005 + actor_index,
            // so each window maps 1:1 to its actor/client index.
            var ros = ROSConnection.instance;
            if (ros != null)
            {
                int idx = ros.unityPort - 5005;
                _clientIndex = idx >= 0 ? idx : 0;
            }
        }
        return _clientIndex >= 0 ? "client-" + _clientIndex : "client-?";
    }

    string ModeText()
    {
        if (_viz == null) _viz = FindObjectOfType<TrajectoryRolloutViz>();
        string m = _viz != null ? _viz.CurrentMode : "";
        if (m == "train") return "TRAINING";
        if (m == "eval") return "EVAL";
        return "(mode: waiting)";
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null && kb.hKey.wasPressedThisFrame)
            _hudOn = !_hudOn;
    }

    CarController ResolveCar()
    {
        if (_car != null) return _car;  // Unity-null after episode reset -> refind
        _car = FindObjectOfType<CarController>();
        return _car;
    }

    void OnGUI()
    {
        if (!_hudOn) return;
        EnsureAssets();
        var car = ResolveCar();
        if (car == null) return;

        float steerNorm = Mathf.Clamp(car.angle, -1f, 1f);
        float wheelDeg = steerSign * steerNorm * maxWheelDeg;
        float speed = 0f;
        try { speed = car.GetSpeed(); } catch { speed = 0f; }
        float force = car.acceleration;

        float cx = Screen.width * 0.5f;
        float top = topMargin;
        float headerH = 22f;
        float contentTop = top + headerH;

        // Background panel for readability.
        Rect panel = new Rect(cx - panelWidth * 0.5f, top - 6f,
                              panelWidth, wheelSize + 104f + headerH);
        Color prevC = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.38f);
        GUI.DrawTexture(panel, _panelTex);
        GUI.color = prevC;

        // Header: client name + explicit TRAINING / EVAL mode (word + color).
        string mode = ModeText();
        string header = ClientName() + "    " + mode;
        _headerStyle.normal.textColor =
            mode == "EVAL" ? new Color(1f, 0.80f, 0.40f)          // amber
            : (mode == "TRAINING" ? new Color(0.55f, 1f, 0.65f)   // green
            : new Color(0.72f, 0.72f, 0.78f));                    // gray waiting
        GUI.Label(new Rect(cx - panelWidth * 0.5f, top - 2f, panelWidth, 20f),
                  header, _headerStyle);

        // Steering wheel (rotated about its center).
        Rect wheelRect = new Rect(cx - wheelSize * 0.5f, contentTop,
                                  wheelSize, wheelSize);
        Matrix4x4 m = GUI.matrix;
        GUIUtility.RotateAroundPivot(wheelDeg, wheelRect.center);
        GUI.DrawTexture(wheelRect, _wheelTex);
        GUI.matrix = m;

        // Steering angle (road-wheel angle, degrees) right below the wheel.
        float steerDeg = car.steering;
        GUI.Label(new Rect(cx - panelWidth * 0.5f, contentTop + wheelSize + 2f,
                           panelWidth, 24f),
                  "steer " + steerDeg.ToString("0.0") + "\u00B0", _steerStyle);

        // Speed readout.
        GUI.Label(new Rect(cx - panelWidth * 0.5f, contentTop + wheelSize + 28f,
                           panelWidth, 30f),
                  speed.ToString("0.0") + " m/s", _speedStyle);

        // Force (throttle command) readout, color-coded by sign.
        _forceStyle.normal.textColor = force > 0.02f
            ? new Color(0.55f, 1f, 0.65f)
            : (force < -0.02f ? new Color(1f, 0.55f, 0.5f) : Color.white);
        GUI.Label(new Rect(cx - panelWidth * 0.5f, contentTop + wheelSize + 60f,
                           panelWidth, 26f),
                  "force " + (force >= 0f ? "+" : "") + force.ToString("0.00"),
                  _forceStyle);
    }

    void EnsureAssets()
    {
        if (_assetsReady) return;
        _wheelTex = GenerateWheelTexture(128);
        _panelTex = SolidTexture(new Color(1f, 1f, 1f, 1f));

        _speedStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 26,
            fontStyle = FontStyle.Bold,
        };
        _speedStyle.normal.textColor = Color.white;

        _steerStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 18,
        };
        _steerStyle.normal.textColor = new Color(0.85f, 0.90f, 1f);

        _headerStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 15,
            fontStyle = FontStyle.Bold,
        };
        _headerStyle.normal.textColor = new Color(0.6f, 0.85f, 1f);

        _forceStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 18,
        };
        _forceStyle.normal.textColor = Color.white;

        _labelStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 12,
        };
        _labelStyle.normal.textColor = new Color(0.8f, 0.85f, 0.95f);

        _assetsReady = true;
    }

    static Texture2D SolidTexture(Color c)
    {
        var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        t.SetPixel(0, 0, c);
        t.Apply();
        t.wrapMode = TextureWrapMode.Clamp;
        return t;
    }

    Texture2D GenerateWheelTexture(int size)
    {
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        var px = new Color32[size * size];
        Color32 clear = new Color32(0, 0, 0, 0);
        Color32 col = wheelColor;
        for (int i = 0; i < px.Length; i++) px[i] = clear;

        float cx = (size - 1) * 0.5f, cy = (size - 1) * 0.5f;
        float R = size * 0.46f;            // outer ring radius
        float ringThick = size * 0.075f;
        float hubR = size * 0.13f;
        float spokeThick = size * 0.06f;

        // Outer ring + center hub.
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float dx = x - cx, dy = y - cy;
                float d = Mathf.Sqrt(dx * dx + dy * dy);
                if ((d <= R && d >= R - ringThick) || d <= hubR)
                    px[y * size + x] = col;
            }
        }

        // Three spokes (down, upper-left, upper-right) so rotation is obvious.
        float[] spokeAng = { 90f, 210f, 330f };
        foreach (float a in spokeAng)
        {
            float rad = a * Mathf.Deg2Rad;
            float ex = cx + Mathf.Cos(rad) * R;
            float ey = cy + Mathf.Sin(rad) * R;
            DrawThickLine(px, size, cx, cy, ex, ey, spokeThick, col);
        }

        tex.SetPixels32(px);
        tex.Apply();
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    static void DrawThickLine(Color32[] px, int size, float x0, float y0,
                              float x1, float y1, float thick, Color32 col)
    {
        float half = thick * 0.5f;
        int minx = Mathf.Max(0, (int)(Mathf.Min(x0, x1) - thick));
        int maxx = Mathf.Min(size - 1, (int)(Mathf.Max(x0, x1) + thick));
        int miny = Mathf.Max(0, (int)(Mathf.Min(y0, y1) - thick));
        int maxy = Mathf.Min(size - 1, (int)(Mathf.Max(y0, y1) + thick));
        float dx = x1 - x0, dy = y1 - y0;
        float len2 = dx * dx + dy * dy;
        for (int y = miny; y <= maxy; y++)
        {
            for (int x = minx; x <= maxx; x++)
            {
                float t = len2 > 0f
                    ? Mathf.Clamp01(((x - x0) * dx + (y - y0) * dy) / len2) : 0f;
                float pxs = x0 + t * dx, pys = y0 + t * dy;
                float dd = Mathf.Sqrt((x - pxs) * (x - pxs) + (y - pys) * (y - pys));
                if (dd <= half) px[y * size + x] = col;
            }
        }
    }
}
