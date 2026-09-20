using System;
using UnityEngine;
// Active Input Handling is "Input System Package (New)" only, so the legacy
// UnityEngine.Input API throws at runtime - use the new one.
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
// Alias only the type we need rather than `using RosMessageTypes.Std;` - that
// namespace also defines a `Time` message which collides with UnityEngine.Time.
using StringMsg = RosMessageTypes.Std.StringMsg;

/// <summary>
/// Draws the fly connectome that is driving the car. The trainer (see
/// rl_agent/fly_brain/viz.py) publishes two topics: `fly_brain_geometry`,
/// static soma positions and edge pairs for the display subset, and
/// `fly_brain_activity`, one intensity byte per neuron at 20 Hz.
///
/// Geometry uploads a mesh once; activity only rewrites vertex colors, which
/// is what keeps a few thousand neurons free at frame rate. Both payloads
/// carry their numeric arrays as base64 of a little-endian buffer rather than
/// JSON number arrays, so parsing is a Convert.FromBase64String plus a
/// Buffer.BlockCopy instead of allocating 24,000 boxed floats every resend.
///
/// Auto-attached by SimController like HudOverlay/TrajectoryRolloutViz, so no
/// scene setup is required. Toggle with the B key. Only the Unity client whose
/// ros-server the trainer publishes to (actor 0) receives messages.
/// </summary>
public class FlyBrainViz : MonoBehaviour
{
    [Header("Topics")]
    public string geometryTopic = "fly_brain_geometry";
    public string activityTopic = "fly_brain_activity";

    [Header("Placement")]
    [Tooltip("Half-extent (m) of the drawn brain. Positions arrive normalized "
             + "to a unit box, so this is the only scale factor.")]
    public float displaySize = 12f;
    [Tooltip("Offset from this GameObject's origin, in world metres. The "
             + "default lifts the overlay above the track so it doesn't sit "
             + "inside the geometry.")]
    public Vector3 worldOffset = new Vector3(0f, 40f, 0f);
    [Tooltip("Spin the brain slowly so its depth reads on a 2D screen. "
             + "Degrees per second; 0 to hold still.")]
    public float spinDegreesPerSecond = 8f;

    [Header("Neurons")]
    [Tooltip("Half-size (m) of each neuron's camera-facing quad.")]
    public float pointSize = 0.10f;
    public Color sensoryColor = new Color(0.25f, 0.85f, 1.00f);    // LC4/LPLC2/...
    public Color interneuronColor = new Color(0.65f, 0.65f, 0.72f);
    public Color commandColor = new Color(1.00f, 0.85f, 0.20f);    // DNp01, MDN, ...
    public Color descendingColor = new Color(1.00f, 0.35f, 0.25f);
    [Tooltip("Role 4, the silhouette population: most of the nervous system, "
             + "drawn only so the shape is recognizable. Unlike the roles "
             + "above it does not dim its own colour but crosses from a cold "
             + "slate at rest to amber when firing, which is what makes "
             + "active neuropils stand out of the cloud.")]
    public Color contextRestColor = new Color(0.20f, 0.23f, 0.30f);
    public Color contextActiveColor = new Color(1.00f, 0.60f, 0.12f);
    [Tooltip("Brightness of a fully silent neuron, as a fraction of its role "
             + "colour. Keeps the structure readable when the brain is quiet.")]
    public float restBrightness = 0.18f;
    // 0.45, not the 0.20 this was against the grey background: over the dark
    // backdrop a resting neuron has to carry the silhouette, and at 0.20 a
    // slate point on near-black is nothing at all.
    public float minAlpha = 0.45f;
    public float maxAlpha = 1.00f;
    [Tooltip("Exponent on intensity. >1 darkens the midrange so only genuinely "
             + "active cells stand out.")]
    public float intensityGamma = 1.6f;

    [Header("Edges")]
    [Tooltip("Edge opacity as a fraction of its endpoint neuron's alpha.")]
    public float edgeAlpha = 0.22f;

    [Header("Backdrop")]
    [Tooltip("Panel drawn behind the cloud. The sim's camera background is a "
             + "flat mid-grey; a connectome whose resting state is near-black "
             + "is simply invisible against it. Darkening the camera instead "
             + "would drag the track's look along with it, so the overlay "
             + "brings its own background.")]
    public Color backdropColor = new Color(0.04f, 0.05f, 0.09f, 0.94f);
    [Tooltip("Margin around the cloud's own extent, as a fraction of it.")]
    public float backdropMargin = 0.10f;

    [Header("Staleness")]
    [Tooltip("Hide the overlay if no activity frame arrives within this many "
             + "seconds (i.e. no fly policy is driving).")]
    public float staleTimeoutSeconds = 5f;

    [Serializable]
    private class GeometryPayload
    {
        public string kind;
        public double stamp;
        public int n;
        public int nEdges;
        public string pos;         // float32[n*3], xyz interleaved, unit box
        public string edgeSrc;     // int32[nEdges]
        public string edgeDst;     // int32[nEdges]
        public string edgeWeight;  // float32[nEdges], signed
        public string role;        // uint8[n]: 0 inter, 1 sensory, 2 command, 3 descending
        public string side;        // uint8[n]: 0 other, 1 L, 2 R
        // Placement knobs published by Python so the overlay can be moved and
        // resized without an Editor rebuild (a build has no inspector). All
        // optional: displaySize>0 is the "present" sentinel, since JsonUtility
        // zero-fills absent fields.
        public float displaySize;
        public float pointSize;
        public float offsetX, offsetY, offsetZ;
        public float spin;
        public float edgeAlpha;
    }

    [Serializable]
    private class ActivityPayload
    {
        public string kind;
        public double stamp;
        public int step;
        public int n;
        public int spikes;
        public string act;         // uint8[n], 0..255
    }

    private GeometryPayload _pendingGeometry;
    private ActivityPayload _pendingActivity;
    private readonly object _lock = new object();

    private GameObject _edgeObject, _pointObject, _backdropObject;
    private Mesh _edgeMesh, _pointMesh, _backdropMesh;
    private Shader _shader;

    private Vector3[] _positions;      // local, already scaled by displaySize
    private byte[] _role;
    private Color[] _edgeColors;       // one per neuron (line endpoints interpolate)
    private Color[] _pointColors;      // four per neuron
    private Vector3[] _pointVertices;  // four per neuron, rebuilt to face the camera
    private int _n, _nEdges;
    private double _geometryStamp = -1;

    private float _lastActivityTime = -999f;
    private bool _subscribed;
    private bool _vizEnabled = true;
    private int _lastSpikes;
    private float _spin;

    /// <summary>Total spikes in the last frame, or -1 when stale. For the HUD.</summary>
    public int CurrentSpikes =>
        (Time.time - _lastActivityTime) < staleTimeoutSeconds ? _lastSpikes : -1;

    void Start()
    {
        TrySubscribe();
    }

    void TrySubscribe()
    {
        if (_subscribed) return;
        var ros = ROSConnection.instance;
        if (ros == null) return;   // retried in Update until the singleton exists
        ros.Subscribe<StringMsg>(geometryTopic, OnGeometry);
        ros.Subscribe<StringMsg>(activityTopic, OnActivity);
        _subscribed = true;
        Debug.Log($"[FlyBrainViz] subscribed to '{geometryTopic}' + '{activityTopic}'");
    }

    // ---- ROS callbacks: may run off the main thread, so only stash ----------
    void OnGeometry(StringMsg msg)
    {
        if (msg == null || string.IsNullOrEmpty(msg.data)) return;
        try
        {
            var p = JsonUtility.FromJson<GeometryPayload>(msg.data);
            if (p != null && p.n > 0 && !string.IsNullOrEmpty(p.pos))
                lock (_lock) { _pendingGeometry = p; }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[FlyBrainViz] geometry parse failed: {e.Message}");
        }
    }

    void OnActivity(StringMsg msg)
    {
        if (msg == null || string.IsNullOrEmpty(msg.data)) return;
        try
        {
            var p = JsonUtility.FromJson<ActivityPayload>(msg.data);
            if (p != null && !string.IsNullOrEmpty(p.act))
                lock (_lock) { _pendingActivity = p; }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[FlyBrainViz] activity parse failed: {e.Message}");
        }
    }

    // ---- base64 helpers ----------------------------------------------------
    static float[] Floats(string b64)
    {
        var bytes = Convert.FromBase64String(b64);
        var outp = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, outp, 0, outp.Length * 4);
        return outp;
    }

    static int[] Ints(string b64)
    {
        var bytes = Convert.FromBase64String(b64);
        var outp = new int[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, outp, 0, outp.Length * 4);
        return outp;
    }

    void Update()
    {
        if (!_subscribed) TrySubscribe();

        var kb = Keyboard.current;
        if (kb != null && kb.bKey.wasPressedThisFrame)
        {
            _vizEnabled = !_vizEnabled;
            SetVisible(false);
            Debug.Log($"[FlyBrainViz] overlay {(_vizEnabled ? "ON" : "OFF")}");
        }
        if (!_vizEnabled)
        {
            lock (_lock) { _pendingActivity = null; }
            return;
        }

        GeometryPayload geom = null;
        ActivityPayload act = null;
        lock (_lock)
        {
            if (_pendingGeometry != null) { geom = _pendingGeometry; _pendingGeometry = null; }
            if (_pendingActivity != null) { act = _pendingActivity; _pendingActivity = null; }
        }

        // Geometry is resent on a slow heartbeat so a late-connecting client
        // still gets it; rebuild only when it is actually new.
        if (geom != null && geom.stamp != _geometryStamp)
            BuildMeshes(geom);

        if (act != null && _positions != null && act.n == _n)
        {
            ApplyActivity(Convert.FromBase64String(act.act));
            _lastActivityTime = Time.time;
            _lastSpikes = act.spikes;
        }

        // No fly policy driving -> hide rather than leave a frozen brain up.
        SetVisible(_positions != null
                   && (Time.time - _lastActivityTime) < staleTimeoutSeconds);
    }

    void LateUpdate()
    {
        if (_positions == null || _pointMesh == null || !IsVisible()) return;

        if (spinDegreesPerSecond != 0f) _spin += spinDegreesPerSecond * Time.deltaTime;

        var container = _pointObject.transform.parent;
        container.localPosition = worldOffset;
        container.localRotation = Quaternion.Euler(0f, _spin, 0f);

        // Billboard every neuron quad toward the camera. Done in the
        // container's local space so the spin above doesn't fight it.
        var cam = Camera.main;
        if (cam == null) return;
        Vector3 right = container.InverseTransformDirection(cam.transform.right) * pointSize;
        Vector3 up = container.InverseTransformDirection(cam.transform.up) * pointSize;
        for (int i = 0; i < _n; i++)
        {
            Vector3 p = _positions[i];
            int b = i * 4;
            _pointVertices[b + 0] = p - right - up;
            _pointVertices[b + 1] = p + right - up;
            _pointVertices[b + 2] = p - right + up;
            _pointVertices[b + 3] = p + right + up;
        }
        _pointMesh.vertices = _pointVertices;
    }

    void BuildMeshes(GeometryPayload g)
    {
        var flat = Floats(g.pos);
        _n = Mathf.Min(g.n, flat.Length / 3);
        _nEdges = g.nEdges;
        _geometryStamp = g.stamp;

        // Python's placement wins when it sent any, so the overlay can be
        // repositioned from the trainer's environment instead of a rebuild.
        if (g.displaySize > 0f)
        {
            displaySize = g.displaySize;
            pointSize = g.pointSize;
            worldOffset = new Vector3(g.offsetX, g.offsetY, g.offsetZ);
            spinDegreesPerSecond = g.spin;
            edgeAlpha = g.edgeAlpha;
        }

        _positions = new Vector3[_n];
        for (int i = 0; i < _n; i++)
            _positions[i] = new Vector3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])
                            * displaySize;

        var roleBytes = Convert.FromBase64String(g.role);
        _role = new byte[_n];
        Array.Copy(roleBytes, _role, Mathf.Min(roleBytes.Length, _n));

        EnsureObjects();

        // Edges: one vertex per neuron, two indices per edge. Colours are per
        // endpoint and the rasterizer interpolates along the line, so an edge
        // out of a firing cell lights from that end.
        int[] src = Ints(g.edgeSrc), dst = Ints(g.edgeDst);
        int m = Mathf.Min(src.Length, dst.Length);
        var indices = new int[m * 2];
        int kept = 0;
        for (int e = 0; e < m; e++)
        {
            if (src[e] < 0 || src[e] >= _n || dst[e] < 0 || dst[e] >= _n) continue;
            indices[kept * 2] = src[e];
            indices[kept * 2 + 1] = dst[e];
            kept++;
        }
        if (kept * 2 != indices.Length) Array.Resize(ref indices, kept * 2);

        _edgeColors = new Color[_n];
        _edgeMesh.Clear();
        _edgeMesh.indexFormat = IndexFormat.UInt32;
        _edgeMesh.vertices = _positions;
        _edgeMesh.SetIndices(indices, MeshTopology.Lines, 0);
        _edgeMesh.RecalculateBounds();

        // Neurons: a camera-facing quad each, positioned in LateUpdate.
        _pointVertices = new Vector3[_n * 4];
        _pointColors = new Color[_n * 4];
        var tris = new int[_n * 6];
        for (int i = 0; i < _n; i++)
        {
            int b = i * 4, t = i * 6;
            // Sprites/Default is Cull Off, so winding does not matter here.
            tris[t] = b; tris[t + 1] = b + 1; tris[t + 2] = b + 2;
            tris[t + 3] = b + 2; tris[t + 4] = b + 1; tris[t + 5] = b + 3;
        }
        _pointMesh.Clear();
        _pointMesh.indexFormat = IndexFormat.UInt32;
        _pointMesh.vertices = _pointVertices;
        _pointMesh.SetIndices(tris, MeshTopology.Triangles, 0);
        // The quads are rebuilt every frame around the real positions, so a
        // recalculated bound would be wrong on frame 0; set it from the extent.
        _pointMesh.bounds = new Bounds(Vector3.zero,
                                       Vector3.one * (displaySize * 2.5f));

        BuildBackdrop();

        ApplyActivity(new byte[_n]);   // draw the resting structure immediately
        Debug.Log($"[FlyBrainViz] geometry: {_n} neurons, {kept} edges "
                  + $"(of {_nEdges} sent)");
    }

    /// <summary>
    /// A quad behind the cloud, sized to what actually arrived. Fitted to the
    /// real extent rather than a square of displaySize: the nervous system is
    /// about 1.4x taller than wide, and a square panel would put a wide dark
    /// band either side of it.
    /// </summary>
    void BuildBackdrop()
    {
        // Local x and z are the screen plane and local y faces the camera;
        // see the axis note in viz.py's get_config.
        float xMin = float.MaxValue, xMax = float.MinValue;
        float zMin = float.MaxValue, zMax = float.MinValue;
        float yMin = float.MaxValue;
        for (int i = 0; i < _n; i++)
        {
            Vector3 p = _positions[i];
            if (p.x < xMin) xMin = p.x;
            if (p.x > xMax) xMax = p.x;
            if (p.z < zMin) zMin = p.z;
            if (p.z > zMax) zMax = p.z;
            if (p.y < yMin) yMin = p.y;
        }
        if (xMin > xMax) return;

        float padX = (xMax - xMin) * backdropMargin;
        float padZ = (zMax - zMin) * backdropMargin;
        xMin -= padX; xMax += padX; zMin -= padZ; zMax += padZ;
        // Just past the deepest neuron, and no further. The camera is a
        // perspective one, so every metre this sits further away shrinks it on
        // screen relative to the cloud it is supposed to be backing; parked at
        // a fixed fraction of displaySize it ends up smaller than the brain.
        float y = yMin - displaySize * 0.02f;

        _backdropMesh.Clear();
        _backdropMesh.vertices = new[]
        {
            new Vector3(xMin, y, zMin), new Vector3(xMax, y, zMin),
            new Vector3(xMin, y, zMax), new Vector3(xMax, y, zMax),
        };
        _backdropMesh.SetIndices(new[] { 0, 1, 2, 2, 1, 3 },
                                 MeshTopology.Triangles, 0);
        _backdropMesh.colors = new[]
        {
            backdropColor, backdropColor, backdropColor, backdropColor,
        };
        _backdropMesh.RecalculateBounds();
    }

    /// <summary>Rest and full-activity colours for a role code.</summary>
    void RoleRamp(byte role, out Color rest, out Color full)
    {
        if (role == 4)
        {
            rest = contextRestColor;
            full = contextActiveColor;
            return;
        }
        switch (role)
        {
            case 1: full = sensoryColor; break;
            case 2: full = commandColor; break;
            case 3: full = descendingColor; break;
            default: full = interneuronColor; break;
        }
        rest = full * restBrightness;
    }

    void ApplyActivity(byte[] act)
    {
        if (_positions == null) return;
        for (int i = 0; i < _n; i++)
        {
            float t = (i < act.Length ? act[i] : (byte)0) / 255f;
            if (intensityGamma != 1f) t = Mathf.Pow(t, intensityGamma);

            Color rest, full;
            RoleRamp(_role[i], out rest, out full);
            Color lit = Color.Lerp(rest, full, t);
            lit.a = Mathf.Lerp(minAlpha, maxAlpha, t);

            _edgeColors[i] = new Color(lit.r, lit.g, lit.b, lit.a * edgeAlpha);
            int b = i * 4;
            _pointColors[b] = lit;
            _pointColors[b + 1] = lit;
            _pointColors[b + 2] = lit;
            _pointColors[b + 3] = lit;
        }
        _edgeMesh.colors = _edgeColors;
        _pointMesh.colors = _pointColors;
    }

    void EnsureObjects()
    {
        if (_edgeObject != null) return;

        var container = new GameObject("FlyBrain").transform;
        container.SetParent(transform, false);
        container.localPosition = worldOffset;

        _backdropMesh = new Mesh { name = "FlyBrainBackdrop" };
        _edgeMesh = new Mesh { name = "FlyBrainEdges" };
        _pointMesh = new Mesh { name = "FlyBrainNeurons" };
        // Explicit queues rather than trusting the transparent sort: these
        // meshes share one material and the backdrop must lose to everything
        // drawn on top of it.
        _backdropObject = NewMeshObject("FlyBrainBackdrop", container,
                                        _backdropMesh, 3000);
        _edgeObject = NewMeshObject("FlyBrainEdges", container, _edgeMesh, 3002);
        _pointObject = NewMeshObject("FlyBrainNeurons", container, _pointMesh, 3003);
    }

    GameObject NewMeshObject(string name, Transform parent, Mesh mesh, int queue)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var mr = go.AddComponent<MeshRenderer>();
        // One material per object, not a shared one: renderQueue lives on the
        // material, and these three need different queues.
        mr.sharedMaterial = new Material(OverlayShader()) { renderQueue = queue };
        mr.shadowCastingMode = ShadowCastingMode.Off;
        mr.receiveShadows = false;
        return go;
    }

    Shader OverlayShader()
    {
        if (_shader != null) return _shader;
        // Same fallback chain as TrajectoryRolloutViz: Sprites/Default is an
        // always-available unlit, vertex-colour-aware, Cull Off shader.
        _shader = Shader.Find("Sprites/Default");
        if (_shader == null) _shader = Shader.Find("Unlit/Color");
        if (_shader == null)
        {
            Debug.LogError("[FlyBrainViz] no usable shader found (Sprites/Default "
                           + "+ Unlit/Color both missing - likely stripped from the "
                           + "build). Add one to 'Always Included Shaders'.");
            _shader = Shader.Find("Legacy Shaders/Diffuse");
        }
        return _shader;
    }

    bool IsVisible()
    {
        return _edgeObject != null && _edgeObject.activeSelf;
    }

    void SetVisible(bool on)
    {
        bool want = on && _vizEnabled;
        if (_edgeObject != null && _edgeObject.activeSelf != want)
            _edgeObject.SetActive(want);
        if (_pointObject != null && _pointObject.activeSelf != want)
            _pointObject.SetActive(want);
        if (_backdropObject != null && _backdropObject.activeSelf != want)
            _backdropObject.SetActive(want);
    }
}
