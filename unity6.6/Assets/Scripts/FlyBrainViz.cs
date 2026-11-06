using System;
using System.Collections.Generic;
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
///
/// Hold the left mouse button over the brain and drag to turn it; release and
/// it returns to the neutral, front-facing pose. See UpdateDragRotation.
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

    [Header("Orientation")]
    [Tooltip("Persistent correction to the NEUTRAL pose, degrees about the "
             + "camera's up axis. Positive turns the brain right. This is what "
             + "squares the overlay up with the top-down camera when the "
             + "parent transform leaves it a few degrees off flush. Left/Right "
             + "arrows adjust it live and the value is logged; bake a value "
             + "you like into this field before the next build. '/' returns "
             + "to this value, not to zero.")]
    public float baseYawDegrees = -144.3f;
    [Tooltip("Same, about the camera's right axis. Positive tips the near face "
             + "up. Up/Down arrows.")]
    public float basePitchDegrees = -9.1f;
    [Tooltip("Degrees per second while an arrow key is held.")]
    public float orientationDegreesPerSecond = 20f;

    [Header("Controls legend")]
    [Tooltip("Draw the key legend while the overlay is up. ';' toggles it. "
             + "Off by default - both panels overlap the brain, which is the "
             + "thing being looked at. The one-line hint in the corner keeps "
             + "them discoverable in a build, which has no inspector.")]
    public bool showControls = false;
    [Tooltip("Draw the colour key while the overlay is up. \"'\" toggles it. "
             + "Independent of the controls list: someone presenting this "
             + "wants the colour key up without the key bindings. Off by "
             + "default for the same reason as showControls.")]
    public bool showColorLegend = false;
    [Tooltip("Show a label for the neuron under the cursor: its cell type, "
             + "side, role, what it does in the fly, and how many of that type "
             + "are drawn and firing. Needs a trainer that publishes the type "
             + "table; against an older one the label falls back to the role.")]
    public bool showHoverLabel = true;
    [Tooltip("How close (screen pixels) the cursor must be to a neuron to "
             + "label it. Roughly the drawn radius of a neuron at the default "
             + "displaySize; larger makes a hover easier to land in a sparse "
             + "region but starts picking neighbours in a dense one.")]
    public float hoverRadiusPixels = 18f;
    [Tooltip("Fraction of full scale a neuron must reach to count as firing "
             + "in the hover label's tally. Read off the raw activity byte, "
             + "before intensityGamma, so the display curve cannot move it.")]
    public float hoverFiringThreshold = 0.25f;
    [Tooltip("Point size of the legend text. The whole panel is laid out in "
             + "multiples of this, so raising it scales the box with the type "
             + "rather than overflowing it. IMGUI does not scale with display "
             + "DPI, so this is the only way to make the panel legible on a "
             + "high-resolution screen.")]
    public int controlsFontSize = 24;

    [Header("Drag to rotate")]
    [Tooltip("Degrees turned per pixel of mouse movement while the left "
             + "button is held over the overlay.")]
    public float dragDegreesPerPixel = 0.35f;
    [Tooltip("Seconds to return to the neutral pose after release. 0 snaps "
             + "instantly; a little easing reads better than a hard cut.")]
    public float snapBackSeconds = 0.15f;
    [Tooltip("Grab radius as a multiple of displaySize. The overlay has no "
             + "collider, so the hit test is a screen-space circle about its "
             + "centre and this sets how big that circle is.")]
    public float grabRadiusScale = 1f;

    [Header("Depth")]
    [Tooltip("Runtime multiplier on the depth axis, on top of whatever "
             + "FLY_VIZ_DEPTH_SCALE the trainer already applied. viz.py clips "
             + "to +/-1 BEFORE scaling, so this is exactly equivalent to "
             + "having set a different FLY_VIZ_DEPTH_SCALE - nothing is lost. "
             + "1 would be the published depth; [ and ] change it, \\ returns "
             + "to this value. Setting FLY_VIZ_DEPTH_SCALE to the absolute "
             + "figure the legend shows and leaving this at 1 is equivalent, "
             + "and does not need a rebuild to change.")]
    public float depthMultiplier = 3.81f;
    [Tooltip("Factor applied per keypress, so the steps stay proportional "
             + "across the range.")]
    public float depthStepFactor = 1.25f;
    public float depthMultiplierMin = 0.1f;
    public float depthMultiplierMax = 20f;

    [Header("Overlay camera")]
    [Tooltip("Draw the brain through a dedicated orthographic camera on its "
             + "own layer, pinned to a fixed corner of the screen. Off, the "
             + "overlay is a world-space object in the main camera: it drifts "
             + "with the window's aspect, smears under perspective at high "
             + "depth, and - because the CSI camera renders every layer - can "
             + "end up inside a policy's camera observation.")]
    public bool useOverlayCamera = true;
    [Tooltip("Where the overlay sits, as a fraction of the window: x, y from "
             + "the bottom-left, then width and height. x still anchors the "
             + "column's left edge. The height and y are only a fallback, "
             + "used until OverheadCameraFit has measured the track - after "
             + "that the overlay takes its height and centre from the track "
             + "and its width from the model's own proportions, and the width "
             + "here becomes a floor. The legends draw over it.")]
    public Rect overlayViewport = new Rect(0.004f, 0.02f, 0.35f, 0.96f);
    [Tooltip("Margin around the model inside its viewport. 1 exactly touches "
             + "the edges. Applied to the viewport and to the camera alike, "
             + "so it is breathing room in the column and does NOT make the "
             + "model shorter than the track.")]
    public float overlayZoom = 1.08f;
    [Tooltip("Height of the overlay as a multiple of the track's. - and = "
             + "adjust it live, 0 resets. At 1 the model is exactly as tall "
             + "as the track and shares its centre line; above that it grows "
             + "past the track and the column widens to keep its shape.")]
    public float overlaySizeScale = 1f;
    [Tooltip("Multiplier per press of - / =.")]
    public float sizeStepFactor = 1.08f;
    public float overlaySizeMin = 0.35f;
    public float overlaySizeMax = 3.0f;

    [Header("Neurons")]
    [Tooltip("Half-size (m) of each neuron's camera-facing quad.")]
    public float pointSize = 0.10f;
    public Color sensoryColor = new Color(0.25f, 0.85f, 1.00f);    // LC4/LPLC2/LPLC1
    [Tooltip("The chase half of role 1 - LC10a, the cells the chase cues are "
             + "injected into. Split out of sensoryColor so the two cue "
             + "families read apart while they fire. Violet rather than the "
             + "pinker purple you might reach for first: scored in CIEDE2000 "
             + "against the rest of this palette under simulated dichromat "
             + "vision, a magenta lands on top of descendingColor for a "
             + "tritanope - rgb(1.00,0.30,0.95) comes out 0.6 from it, where "
             + "under 2 is indistinguishable - while this stays 27 clear of "
             + "every colour here and 28 clear of the loom cyan in its worst "
             + "view. Push it toward red and that margin is what goes.")]
    public Color chaseColor = new Color(0.60f, 0.15f, 1.00f);      // LC10a
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
    // 0.45, not the 0.20 this was against the old grey background: on black a
    // resting neuron has to carry the silhouette on its own, and at 0.20 a
    // slate point is nothing at all.
    public float minAlpha = 0.45f;
    public float maxAlpha = 1.00f;
    [Tooltip("Exponent on intensity. >1 darkens the midrange so only genuinely "
             + "active cells stand out.")]
    public float intensityGamma = 1.6f;

    [Header("Edges")]
    [Tooltip("Edge opacity as a fraction of its endpoint neuron's alpha.")]
    public float edgeAlpha = 0.22f;

    [Header("Staleness")]
    [Tooltip("Hide the overlay if no activity frame arrives within this many "
             + "seconds (i.e. no fly policy is driving).")]
    public float staleTimeoutSeconds = 5f;

    [Header("View mode")]
    [Tooltip("Connectome draws the CNS on its own. FlyAnatomy places it, at "
             + "true proportions, inside a see-through animated male fruit fly "
             + "(FlyAnatomyView). BrainCloseUp zooms in until the head, bent "
             + "down, fills the space left of the track. N cycles the three. "
             + "The fly's transparency, animation "
             + "schedule and CNS placement are tuned on the asset "
             + "Resources/FruitFly/FlyAnatomySettings, not here - this "
             + "component is added at runtime and forgets inspector edits.")]
    public FlyViewMode viewMode = FlyViewMode.Connectome;

    public enum FlyViewMode { Connectome, FlyAnatomy, BrainCloseUp }

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
        // FLY_VIZ_DEPTH_SCALE as the trainer applied it. Provenance only: the
        // flattening is already baked into `pos`, so this must never be
        // applied again - it exists so the depth keys can report the absolute
        // value their multiplier works out to. 0 means an older trainer that
        // does not send it (JsonUtility zero-fills absent fields).
        public float depthScale;
        // Cell type per neuron, for the hover label: a table of distinct names
        // and a uint16 index each. Both absent on a trainer that predates
        // them, which the hover code treats as "types unknown" rather than
        // failing - JsonUtility leaves the array null.
        public string[] typeNames;
        public string typeIdx;

        /// <summary>Same neurons and placement, whatever the stamp. Each
        /// trainer process stamps its own copy with its own clock, so with
        /// several publishing, the stamp changes on nearly every resend.</summary>
        public bool SameContent(GeometryPayload o)
        {
            return o != null && n == o.n && nEdges == o.nEdges && pos == o.pos
                   && role == o.role && side == o.side && typeIdx == o.typeIdx
                   && edgeSrc == o.edgeSrc && edgeDst == o.edgeDst
                   && displaySize == o.displaySize && pointSize == o.pointSize
                   && offsetX == o.offsetX && offsetY == o.offsetY && offsetZ == o.offsetZ
                   && spin == o.spin && edgeAlpha == o.edgeAlpha && depthScale == o.depthScale
                   && (typeNames == null) == (o.typeNames == null)
                   && (typeNames == null || typeNames.Length == o.typeNames.Length);
        }
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

    [Tooltip("Most activity frames applied per second (each recolours every "
             + "neuron); newer frames in between replace the waiting one.")]
    public float maxActivityHz = 10f;

    private GeometryPayload _pendingGeometry;
    private GeometryPayload _builtGeometry;
    private float _nextActivityApply;
    private ActivityPayload _pendingActivity;
    private readonly object _lock = new object();

    private GameObject _edgeObject, _pointObject;
    private Mesh _edgeMesh, _pointMesh;
    private Material _material;

    private Vector3[] _positions;      // local, already scaled by displaySize
    private float[] _baseDepth;        // _positions[i].y as published, pre-multiplier
    private float _publishedDepthScale;   // 0 when the trainer didn't send it
    private byte[] _side;                 // SIDE_CODES: 0 none, 1 L, 2 R
    private string[] _typeNames;          // null on a trainer that omits them
    private ushort[] _typeIdx;            // index into _typeNames, or null
    private byte[] _act;                  // last activity frame, for live counts
    // Per distinct type: how many are drawn, and how many are firing now.
    // Rebuilt on geometry; the firing half is refreshed on a timer, since it
    // is a full pass over every neuron and the label only needs to feel live.
    private int[] _typeShown, _typeFiring;
    private int _hoverIndex = -1;         // neuron under the cursor, or -1
    private Vector2 _hoverPos;
    private float _nextCountRefresh;
    // Five is what DrawHoverLabel can currently emit (name, tuning, role, cue
    // population, counts); the spare is headroom, since overflowing this in
    // OnGUI would throw once per frame rather than fail quietly.
    private readonly string[] _hoverLines = new string[6];
    private Camera _overlayCam;
    private Camera _overheadCam;
    private float _nextOverheadLookup;
    private float _modelRadius = 1f;      // furthest neuron from the centre

    // What the brain covers as the overlay camera sees it, and the pose that
    // was measured for. See UpdateProjectedExtents.
    private float _projHalfW, _projHalfH;
    private Vector3 _projCentre;
    private Quaternion _framingRot = Quaternion.identity;
    private bool _haveFraming;
    private bool _framingDirty = true;
    private float _overlaySizeDefault = 1f;
    private float _lastAchievedScale = 1f;
    // Captured in Start, before any key can touch them, so the reset keys go
    // back to however this component was configured rather than to hardcoded
    // identity values.
    private float _depthMultiplierDefault, _baseYawDefault, _basePitchDefault;
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

    // User-applied rotation, composed on top of the idle spin. Identity is the
    // neutral pose. _snapT is the return animation's 0..1 progress and sits at
    // 1 whenever there is nothing to return from.
    private Quaternion _dragRotation = Quaternion.identity;
    private Quaternion _snapFrom = Quaternion.identity;
    private bool _dragging;
    private float _snapT = 1f;

    // ---- fly anatomy mode (see BuildAnatomy) ----
    private Vector3[] _unit;              // published positions, unit box, before displaySize
    private string _pendingGeometryJson;  // raw payload, cached to disk for the fly view
    private bool _triedGeometryCache;
    private FlyAnatomySettings _anatSettings;
    private FlyAnatomyView _fly;
    private bool _flyFailed;
    private Vector3[] _anatHead, _anatThorax;   // per neuron, in its bone's rest frame
    private float[] _anatW;               // 0 rides the head, 1 the thorax
    private Vector3 _anatBrainCentre;     // where the brain is centred, head-bone rest frame
    private Vector3[] _anatPos;           // this frame, fly-Root local
    private double _anatStamp = -2;
    private int _anatHash;
    private FlyAnatomyView _anatFly;
    private float _anatPointSize;
    private float _anatSpin;
    private Vector3 _framedLift;
    private float _viewYawDefault, _viewElevationDefault;
    private float _closeUpYawDefault, _closeUpElevationDefault;

    bool Anatomy { get { return viewMode != FlyViewMode.Connectome && _fly != null; } }
    bool CloseUp { get { return viewMode == FlyViewMode.BrainCloseUp && _fly != null; } }

    /// <summary>What is drawn this frame, in the container's space.</summary>
    Vector3[] DrawPositions
    {
        get { return Anatomy && _anatPos != null && _anatPos.Length == _n ? _anatPos : _positions; }
    }

    /// <summary>Total spikes in the last frame, or -1 when stale. For the HUD.</summary>
    public int CurrentSpikes =>
        (Time.time - _lastActivityTime) < staleTimeoutSeconds ? _lastSpikes : -1;

    void Start()
    {
        _overlaySizeDefault = overlaySizeScale;
        _depthMultiplierDefault = depthMultiplier;
        _baseYawDefault = baseYawDegrees;
        _basePitchDefault = basePitchDegrees;
        LoadAnatSettings();
        Camera.onPostRender += BlitOverlayCache;
        TrySubscribe();
        if (viewMode != FlyViewMode.Connectome) ApplyViewMode();
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
                lock (_lock) { _pendingGeometry = p; _pendingGeometryJson = msg.data; }
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

    static ushort[] UShorts(string b64)
    {
        if (string.IsNullOrEmpty(b64)) return null;
        var bytes = Convert.FromBase64String(b64);
        var outp = new ushort[bytes.Length / 2];
        Buffer.BlockCopy(bytes, 0, outp, 0, outp.Length * 2);
        return outp;
    }

    // ---- frame-time log ----------------------------------------------------
    // The car's commands and observations only move between frames, at
    // Time.timeScale 3, so frame length is driving latency. Logged per view
    // so the views can be compared on the same run.
    private readonly float[] _frameMs = new float[8192];
    private int _frameCount;
    private float _frameLogStart = -1f;
    private string _frameLogView;
    private int _redrawCount;

    void LogFrameTimes()
    {
        string view = !_vizEnabled ? "off" : !IsVisible() ? "hidden" : viewMode.ToString();
        float now = Time.unscaledTime;
        if (_frameLogStart < 0f || view != _frameLogView || now - _frameLogStart >= 10f)
        {
            if (_frameCount > 30 && _frameLogView != null)
            {
                Array.Sort(_frameMs, 0, _frameCount);
                float sum = 0f;
                for (int i = 0; i < _frameCount; i++) sum += _frameMs[i];
                float secs = now - _frameLogStart;
                Debug.Log($"[FlyBrainViz] frames view={_frameLogView} {_frameCount / secs:0} fps, "
                          + $"mean {sum / _frameCount:0.0} ms, p95 {_frameMs[(int)(_frameCount * 0.95f)]:0.0} ms, "
                          + $"p99 {_frameMs[(int)(_frameCount * 0.99f)]:0.0} ms, max {_frameMs[_frameCount - 1]:0.0} ms, "
                          + $"redraws {_redrawCount} over {secs:0} s");
            }
            _frameCount = 0;
            _redrawCount = 0;
            _frameLogStart = now;
            _frameLogView = view;
        }
        if (_frameCount < _frameMs.Length) _frameMs[_frameCount++] = Time.unscaledDeltaTime * 1000f;
    }

    void Update()
    {
        LogFrameTimes();
        if (!_subscribed) TrySubscribe();

        var kb = Keyboard.current;
        if (kb != null && kb.bKey.wasPressedThisFrame)
        {
            _vizEnabled = !_vizEnabled;
            SetVisible(false);
            Debug.Log($"[FlyBrainViz] overlay {(_vizEnabled ? "ON" : "OFF")}");
        }
        if (kb != null && kb.nKey.wasPressedThisFrame)
        {
            viewMode = viewMode == FlyViewMode.Connectome ? FlyViewMode.FlyAnatomy
                     : viewMode == FlyViewMode.FlyAnatomy ? FlyViewMode.BrainCloseUp
                     : FlyViewMode.Connectome;
            ApplyViewMode();
            Debug.Log($"[FlyBrainViz] view mode {viewMode}");
        }
        // The P car-camera view is the CSI frame the policy would see, so the
        // brain stands aside for it without touching the B setting - P again
        // brings the overlay back exactly as it was.
        if (!_vizEnabled || CameraViewSwitcher.CarCameraOn)
        {
            SetVisible(false);
            lock (_lock) { _pendingActivity = null; }
            // Drop any grab in progress rather than resuming mid-drag when the
            // overlay comes back: this early-return skips UpdateDragRotation,
            // so _dragging would otherwise stay set with no way to clear it.
            CancelDrag();
            return;
        }

        HandleDepthKeys(kb);
        HandleOrientationKeys(kb);
        HandleSizeKeys(kb);
        HandleFlyOpacityKeys(kb);

        GeometryPayload geom = null;
        ActivityPayload act = null;
        string geomJson = null;
        lock (_lock)
        {
            if (_pendingGeometry != null)
            {
                geom = _pendingGeometry; _pendingGeometry = null;
                geomJson = _pendingGeometryJson; _pendingGeometryJson = null;
            }
            // Every trainer process publishes its own activity, so frames can
            // arrive faster than they are worth recolouring 19k neurons for;
            // the newest waits in the slot until the next apply.
            if (_pendingActivity != null && Time.unscaledTime >= _nextActivityApply)
            {
                act = _pendingActivity; _pendingActivity = null;
                _nextActivityApply = Time.unscaledTime + 1f / Mathf.Max(1f, maxActivityHz);
            }
        }

        // Geometry is resent on a slow heartbeat so a late-connecting client
        // still gets it; rebuild only when it is actually new.
        if (geom != null && geom.stamp != _geometryStamp && !geom.SameContent(_builtGeometry))
        {
            BuildMeshes(geom);
            _builtGeometry = geom;
            SaveGeometryCache(geomJson);
        }
        if (_positions == null && viewMode != FlyViewMode.Connectome) TryLoadGeometryCache();

        if (act != null && _positions != null && act.n == _n)
        {
            ApplyActivity(Convert.FromBase64String(act.act));
            _lastActivityTime = Time.time;
            _lastSpikes = act.spikes;
        }

        // No fly policy driving -> hide rather than leave a frozen brain up.
        // The fly view may stay up regardless, as a presentation piece.
        bool fresh = (Time.time - _lastActivityTime) < staleTimeoutSeconds;
        bool preview = Anatomy && _anatSettings.showWithoutActivity;
        SetVisible(_positions != null && (fresh || preview));
        SyncCloseUpLook();
        if (!fresh && preview) DriveTestActivity();

        // After SetVisible, so a grab can only start on an overlay that is
        // actually on screen this frame.
        UpdateDragRotation();
    }

    /// <summary>
    /// '[' and ']' scale the depth axis live, '\' returns it to what the
    /// trainer published.
    ///
    /// This exists because FLY_VIZ_DEPTH_SCALE cannot be changed on a running
    /// job: viz.py reads it in FlyBrainViz.__init__, once per job, and caches
    /// the packed geometry for the instance's lifetime. Rescaling here is not
    /// an approximation of that knob - viz.py clips to +/-1 BEFORE multiplying
    /// by depth_scale, so the scale is a pure final multiplier and doing it on
    /// this side gives bit-comparable positions to having set
    /// FLY_VIZ_DEPTH_SCALE * depthMultiplier in the first place.
    /// </summary>
    void HandleDepthKeys(Keyboard kb)
    {
        // The fly view always draws true proportions (see BuildAnatomy).
        if (kb == null || _positions == null || Anatomy) return;
        bool down = kb.leftBracketKey.wasPressedThisFrame;
        bool up = kb.rightBracketKey.wasPressedThisFrame;
        bool reset = kb.backslashKey.wasPressedThisFrame;
        if (!down && !up && !reset) return;

        // Reset goes back to the configured default rather than to 1. The
        // default is a tuned value, not the identity, so 1 would be a pose
        // nobody asked for and would mean re-climbing the ladder to recover.
        depthMultiplier = reset
            ? _depthMultiplierDefault
            : Mathf.Clamp(depthMultiplier * (up ? depthStepFactor : 1f / depthStepFactor),
                          depthMultiplierMin, depthMultiplierMax);
        ApplyDepth();
        Debug.Log($"[FlyBrainViz] depth x{depthMultiplier:0.###} - to keep it, "
                  + $"{DepthKeepHint()}");
    }

    /// <summary>
    /// How to make the current depth permanent. Two routes, because there are
    /// two places it can live: the trainer's env var (applies to every client
    /// and every future job) or this component's default (Unity-side, needs a
    /// rebuild). The env var is preferred, and quotable directly because the
    /// trainer tells us what it used - but only since the depthScale field was
    /// added to the geometry payload, so fall back to the multiplier alone.
    /// </summary>
    string DepthKeepHint()
    {
        if (_publishedDepthScale > 0f)
            return $"set FLY_VIZ_DEPTH_SCALE={_publishedDepthScale * depthMultiplier:0.####} "
                   + $"(it is {_publishedDepthScale:0.####} now), or depthMultiplier="
                   + $"{depthMultiplier:0.###} on FlyBrainViz";
        return $"set depthMultiplier={depthMultiplier:0.###} on FlyBrainViz, or "
               + $"multiply the trainer's FLY_VIZ_DEPTH_SCALE by {depthMultiplier:0.###}";
    }

    /// <summary>The absolute depth for the legend, or null if unknowable.</summary>
    string DepthAbsoluteText()
    {
        if (_publishedDepthScale <= 0f) return null;
        return (_publishedDepthScale * depthMultiplier).ToString("0.####");
    }

    /// <summary>
    /// Arrow keys square the overlay up with the camera, '/' resets that,
    /// ';' toggles the legend.
    ///
    /// Deliberately separate from the mouse drag. The drag springs back, so it
    /// is a look-around and cannot hold a correction; this is the persistent
    /// pose, and it is what the drag springs back TO. The case it exists for
    /// is the overlay sitting a few degrees off flush to the top-down camera,
    /// which comes from the rotation of the parent transform and is not
    /// expressible through FLY_VIZ_ROTATE (that one turns the picture within
    /// the screen plane, which cannot fix a tilt).
    ///
    /// Held rather than tapped, because finding "flush" by eye wants a
    /// continuous sweep. Unscaled, like the snap-back, so Time.timeScale 3-5
    /// does not make it uncontrollable.
    /// </summary>
    void HandleOrientationKeys(Keyboard kb)
    {
        if (kb == null) return;
        if (kb.semicolonKey.wasPressedThisFrame) { showControls = !showControls; return; }
        if (kb.quoteKey.wasPressedThisFrame) { showColorLegend = !showColorLegend; return; }

        if (Anatomy) { HandleFlyViewKeys(kb); return; }

        if (kb.slashKey.wasPressedThisFrame)
        {
            // Back to the configured default, not to zero: the default is the
            // measured correction that squares the overlay up, so zero is the
            // known-wrong pose rather than a neutral one.
            baseYawDegrees = _baseYawDefault;
            basePitchDegrees = _basePitchDefault;
            LogOrientation();
            return;
        }

        float dYaw = (kb.rightArrowKey.isPressed ? 1f : 0f)
                     - (kb.leftArrowKey.isPressed ? 1f : 0f);
        float dPitch = (kb.upArrowKey.isPressed ? 1f : 0f)
                       - (kb.downArrowKey.isPressed ? 1f : 0f);
        if (dYaw != 0f || dPitch != 0f)
        {
            float step = orientationDegreesPerSecond * Time.unscaledDeltaTime;
            baseYawDegrees = Wrap180(baseYawDegrees + dYaw * step);
            basePitchDegrees = Wrap180(basePitchDegrees + dPitch * step);
        }
        // Logged on release rather than per frame, which would be ~60 lines a
        // second while a key is held.
        if (kb.rightArrowKey.wasReleasedThisFrame || kb.leftArrowKey.wasReleasedThisFrame
            || kb.upArrowKey.wasReleasedThisFrame || kb.downArrowKey.wasReleasedThisFrame)
            LogOrientation();
    }

    /// <summary>
    /// The arrows' fly-view meaning: turn the fly (yaw) and raise or lower the
    /// camera over it (elevation). Written to FlyAnatomySettings, so in the
    /// editor the pose you settle on is the one saved with the project.
    /// </summary>
    void HandleFlyViewKeys(Keyboard kb)
    {
        var s = _anatSettings;
        if (kb.slashKey.wasPressedThisFrame)
        {
            if (CloseUp) { s.closeUpViewYaw = _closeUpYawDefault; s.closeUpViewElevation = _closeUpElevationDefault; }
            else { s.viewYaw = _viewYawDefault; s.viewElevation = _viewElevationDefault; }
            s.MarkChanged();
            return;
        }
        float dYaw = (kb.rightArrowKey.isPressed ? 1f : 0f) - (kb.leftArrowKey.isPressed ? 1f : 0f);
        float dElev = (kb.upArrowKey.isPressed ? 1f : 0f) - (kb.downArrowKey.isPressed ? 1f : 0f);
        if (dYaw != 0f || dElev != 0f)
        {
            float step = orientationDegreesPerSecond * Time.unscaledDeltaTime;
            FlyYaw = Wrap180(FlyYaw + dYaw * step);
            FlyElevation = Mathf.Clamp(FlyElevation + dElev * step, -89f, 89f);
        }
        if (kb.rightArrowKey.wasReleasedThisFrame || kb.leftArrowKey.wasReleasedThisFrame
            || kb.upArrowKey.wasReleasedThisFrame || kb.downArrowKey.wasReleasedThisFrame)
        {
            s.MarkChanged();
            Debug.Log($"[FlyBrainViz] {(CloseUp ? "close-up" : "fly view")} yaw {FlyYaw:0.#} "
                      + $"elevation {FlyElevation:0.#} (FlyAnatomySettings)");
        }
    }

    // The orientation dials of whichever fly view is up; each view has its own.
    float FlyYaw
    {
        get { return CloseUp ? _anatSettings.closeUpViewYaw : _anatSettings.viewYaw; }
        set { if (CloseUp) _anatSettings.closeUpViewYaw = value; else _anatSettings.viewYaw = value; }
    }
    float FlyElevation
    {
        get { return CloseUp ? _anatSettings.closeUpViewElevation : _anatSettings.viewElevation; }
        set { if (CloseUp) _anatSettings.closeUpViewElevation = value; else _anatSettings.viewElevation = value; }
    }
    float FlyRoll { get { return CloseUp ? _anatSettings.closeUpViewRoll : _anatSettings.viewRoll; } }
    float FlyBodyPitch { get { return CloseUp ? _anatSettings.closeUpBodyPitch : _anatSettings.viewBodyPitch; } }

    /// <summary>, and . make the fly's body more or less see-through; Y swaps
    /// it for the fully textured model and back.</summary>
    void HandleFlyOpacityKeys(Keyboard kb)
    {
        if (kb == null || !Anatomy) return;
        if ((kb.digit1Key.wasPressedThisFrame || kb.numpad1Key.wasPressedThisFrame)
            && _fly != null && IsVisible() && _fly.TriggerFlight())
            Debug.Log("[FlyBrainViz] fly: takeoff (1)");
        if (kb.yKey.wasPressedThisFrame)
        {
            _flyTextured = !_flyTextured;
            Debug.Log($"[FlyBrainViz] fly {(_flyTextured ? "textured" : "see-through")}");
        }
        bool down = kb.commaKey.wasPressedThisFrame;
        bool up = kb.periodKey.wasPressedThisFrame;
        if (!down && !up) return;
        var s = _anatSettings;
        s.bodyOpacity = Mathf.Clamp01(s.bodyOpacity + (up ? 0.02f : -0.02f));
        s.MarkChanged();
        Debug.Log($"[FlyBrainViz] fly body opacity {s.bodyOpacity:0.00} (FlyAnatomySettings)");
    }

    static float Wrap180(float deg)
    {
        return Mathf.Repeat(deg + 180f, 360f) - 180f;
    }

    void LogOrientation()
    {
        Debug.Log($"[FlyBrainViz] orientation yaw {baseYawDegrees:0.##} pitch "
                  + $"{basePitchDegrees:0.##} - to keep it, set baseYawDegrees "
                  + $"/ basePitchDegrees to these before the next build");
    }

    /// <summary>
    /// The persistent correction, about the CAMERA's axes so "flush with the
    /// camera" means what it says. Same sign convention as the drag: positive
    /// yaw turns the brain right, positive pitch tips its near face up.
    /// </summary>
    Quaternion BaseOrientation(Camera cam)
    {
        if (cam == null || (baseYawDegrees == 0f && basePitchDegrees == 0f))
            return Quaternion.identity;
        Vector3 up = transform.InverseTransformDirection(cam.transform.up);
        Vector3 right = transform.InverseTransformDirection(cam.transform.right);
        return Quaternion.AngleAxis(-baseYawDegrees, up)
               * Quaternion.AngleAxis(basePitchDegrees, right);
    }

    /// <summary>Rewrite the depth axis from the published values.</summary>
    void ApplyDepth()
    {
        if (_positions == null || _baseDepth == null) return;
        for (int i = 0; i < _n; i++)
            _positions[i].y = _baseDepth[i] * depthMultiplier;
        UpdateModelRadius();
        // Edge vertices ARE the positions, so they have to be re-uploaded. The
        // neuron quads are rebuilt from _positions every LateUpdate and need
        // nothing. The vertexCount guard covers the call from BuildMeshes,
        // which runs before the mesh has been filled.
        if (_edgeMesh != null && _edgeMesh.vertexCount == _n)
        {
            _edgeMesh.vertices = _positions;
            _edgeMesh.RecalculateBounds();
        }
        UpdatePointBounds();
    }

    /// <summary>
    /// The neuron quads are rebuilt every frame around the real positions, so
    /// a recalculated bound would be wrong on frame 0 - it is set from the
    /// extent instead. It has to grow with depthMultiplier, or pulling the
    /// depth out far enough frustum-culls the whole cloud.
    /// </summary>
    void UpdatePointBounds()
    {
        if (_pointMesh == null) return;
        if (Anatomy)
        {
            // Fly-Root local, which is model units; the overlay camera sees
            // nothing else, so an oversized bound costs nothing.
            _pointMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 1e4f);
            return;
        }
        float reach = displaySize * 2.5f * Mathf.Max(1f, depthMultiplier);
        _pointMesh.bounds = new Bounds(Vector3.zero, Vector3.one * reach);
    }

    /// <summary>
    /// Left-drag over the brain to turn it; release and it returns to neutral.
    ///
    /// Rotation is about the CAMERA's right and up axes rather than world
    /// ones. The sim camera looks straight down, so rotating about Unity's y
    /// would spin the brain in the screen plane and "left" would stop meaning
    /// left - the same reason viz.py ships spin=0 by default. Against camera
    /// axes the mapping holds whatever the camera is doing.
    ///
    /// The near face follows the cursor, like turning a globe with a fingertip:
    /// drag right and the surface facing you travels right, drag up and it
    /// travels up.
    /// </summary>
    void UpdateDragRotation()
    {
        var mouse = Mouse.current;
        // The overlay's own camera when it has one: the drag axes have to be
        // the axes the brain is actually being viewed through.
        var cam = _overlayCam != null ? _overlayCam : Camera.main;
        bool canDrag = IsVisible() && mouse != null && cam != null;

        if (_dragging && (!canDrag || !mouse.leftButton.isPressed))
            ReleaseDrag();

        if (!canDrag)
        {
            // Nothing to see, so settle immediately rather than animating a
            // return the user cannot watch.
            CancelDrag();
            return;
        }

        if (!_dragging && mouse.leftButton.wasPressedThisFrame
            && IsOverOverlay(cam, mouse.position.ReadValue()))
        {
            _dragging = true;
            _snapT = 1f;          // abandon any return still in flight
        }

        if (_dragging)
        {
            Vector2 d = mouse.delta.ReadValue();
            if (d.sqrMagnitude > 0f)
            {
                // Axes in the container's PARENT space, because _dragRotation
                // is composed into localRotation and that parent is rotated.
                Vector3 up = transform.InverseTransformDirection(cam.transform.up);
                Vector3 right = transform.InverseTransformDirection(cam.transform.right);
                // A positive Unity rotation about up carries +forward toward
                // +right, which swings the near face LEFT. Negate so dragging
                // right sends the face you can see right. Pitch needs no such
                // flip: positive about camera-right lifts the near face.
                Quaternion yaw = Quaternion.AngleAxis(-d.x * dragDegreesPerPixel, up);
                Quaternion pitch = Quaternion.AngleAxis(d.y * dragDegreesPerPixel, right);
                // Pre-multiplied, so each frame's delta is applied in the
                // camera's frame rather than in the brain's own - which is what
                // keeps "right" meaning right after the pose has been turned.
                // Normalized because this compounds once per frame for as long
                // as the button is held, and Unity does not renormalize.
                _dragRotation = Quaternion.Normalize(yaw * pitch * _dragRotation);
            }
            return;
        }

        if (_snapT < 1f)
        {
            // Unscaled: the sim runs at Time.timeScale 3-5, which would make a
            // 0.15 s return finish in 30-50 ms of wall clock.
            _snapT = snapBackSeconds <= 0f
                ? 1f
                : Mathf.Min(1f, _snapT + Time.unscaledDeltaTime / snapBackSeconds);
            float eased = _snapT * _snapT * (3f - 2f * _snapT);   // smoothstep
            _dragRotation = Quaternion.Slerp(_snapFrom, Quaternion.identity, eased);
        }
    }

    /// <summary>Begin the animated return to the neutral pose.</summary>
    void ReleaseDrag()
    {
        _dragging = false;
        _snapFrom = _dragRotation;
        _snapT = 0f;
    }

    /// <summary>Drop the grab and the return, straight back to neutral.</summary>
    void CancelDrag()
    {
        _dragging = false;
        _dragRotation = Quaternion.identity;
        _snapFrom = Quaternion.identity;
        _snapT = 1f;
    }

    /// <summary>
    /// Is the cursor over the drawn brain? There is no collider to raycast -
    /// the overlay is billboarded quads and lines - so this is a screen-space
    /// circle about the container's centre, sized from displaySize because
    /// that is what the unit-box positions were scaled by.
    /// </summary>
    bool IsOverOverlay(Camera cam, Vector2 screenPos)
    {
        if (_pointObject == null) return false;
        // With a dedicated camera the overlay's extent on screen is exactly
        // that camera's viewport, so there is nothing to approximate.
        if (_overlayCam != null) return _overlayCam.pixelRect.Contains(screenPos);
        Vector3 centre = _pointObject.transform.parent.position;
        Vector3 c = cam.WorldToScreenPoint(centre);
        if (c.z <= 0f) return false;        // behind the camera
        Vector3 edge = cam.WorldToScreenPoint(
            centre + cam.transform.right * (displaySize * grabRadiusScale));
        var centre2 = new Vector2(c.x, c.y);
        float radiusPx = Vector2.Distance(centre2, new Vector2(edge.x, edge.y));
        return Vector2.Distance(screenPos, centre2) <= radiusPx;
    }

    void LateUpdate()
    {
        PoseOverlay();
        SyncOverlayCache();
    }

    void PoseOverlay()
    {
        // Ahead of the visibility bail-out: the overlay camera has to be told
        // to switch off, and the track's reserved column has to be handed
        // back, precisely when the overlay stops being visible.
        if (_overlayCam != null) UpdateOverlayCamera();

        if (_positions == null || _pointMesh == null || !IsVisible()) return;

        // Held still while the user has hold of it and while it returns, so
        // the pose under the cursor is the one they put there. Resumes from
        // where it stopped, not from zero. Moot at the shipped spin of 0.
        bool interacting = _dragging || _snapT < 1f;
        if (spinDegreesPerSecond != 0f && !interacting)
            _spin += spinDegreesPerSecond * Time.deltaTime;

        // Fetched before the rotation is composed, because the persistent
        // orientation correction is expressed in this camera's axes. In
        // overlay-camera mode that is the overlay's own camera, which is what
        // decouples the brain's framing from wherever the scene camera is
        // pointing.
        var cam = _overlayCam != null ? _overlayCam : Camera.main;

        var container = _pointObject.transform.parent;
        float size = pointSize;
        // Fly views only: nothing to redo on a frame where neither the fly's
        // pose nor the view moved. The fly is posed at FlyAnatomySettings
        // .animationHz, not every frame, so most frames skip the per-neuron
        // passes below.
        bool unchanged = false;
        if (Anatomy)
        {
            if (cam == null) { _hoverIndex = -1; return; }
            if (!interacting && _anatSettings.spinDegreesPerSecond != 0f)
                _anatSpin += _anatSettings.spinDegreesPerSecond * Time.unscaledDeltaTime;
            // The drag is kept in this component's parent frame (see
            // UpdateDragRotation); conjugate it into world space for Root.
            Quaternion t = transform.rotation;
            Quaternion rootRot = t * _dragRotation * Quaternion.Inverse(t) * SettledFlyRotation(cam);
            _fly.Root.rotation = rootRot;
            size = _anatPointSize * (CloseUp ? Mathf.Max(0.1f, _anatSettings.closeUpPointScale)
                                    : GlowOn ? Mathf.Max(0.1f, _anatSettings.glowPointScale) : 1f);
            int placement = _anatSettings.PlacementHash();
            unchanged = _drawn.valid && !_fly.PosedThisFrame && rootRot == _drawn.rootRot
                        && cam.transform.rotation == _drawn.camRot && size == _drawn.size
                        && CloseUp == _drawn.closeUp && placement == _drawn.placement
                        && _geometryStamp == _drawn.stamp;
            if (!unchanged) UpdateAnatomyPositions();
            _drawn = new DrawnPose {
                valid = true, rootRot = rootRot, camRot = cam.transform.rotation, size = size,
                closeUp = CloseUp, placement = placement, stamp = _geometryStamp,
            };
        }
        else
        {
            _drawn.valid = false;
            // In overlay-camera mode the rig is parked at a fixed world point
            // and framed by its own camera, so the trainer's placement offset
            // does not apply - writing it here every frame would drag the
            // brain out of its own viewport.
            if (_overlayCam == null) container.localPosition = worldOffset;
            // Read right to left: the brain's own turntable, then the
            // persistent correction that squares it up with the camera, then
            // the temporary drag. Both rotations act in the parent's frame,
            // where their axes were computed. The drag returning to identity
            // therefore lands on the corrected pose, not on the uncorrected one.
            container.localRotation =
                _dragRotation * BaseOrientation(cam) * Quaternion.Euler(0f, _spin, 0f);
        }

        if (!Anatomy && GlowOn) size *= Mathf.Max(0.1f, _anatSettings.glowPointScale);

        // Billboard every neuron quad toward the camera. Done in the
        // container's local space so the rotations above don't fight it.
        if (cam == null) { _hoverIndex = -1; return; }
        Vector3[] pos = DrawPositions;
        if (!unchanged)
        {
            Vector3 right = container.InverseTransformDirection(cam.transform.right) * size;
            Vector3 up = container.InverseTransformDirection(cam.transform.up) * size;
            for (int i = 0; i < _n; i++)
            {
                Vector3 p = pos[i];
                int b = i * 4;
                _pointVertices[b + 0] = p - right - up;
                _pointVertices[b + 1] = p + right - up;
                _pointVertices[b + 2] = p - right + up;
                _pointVertices[b + 3] = p + right + up;
            }
            _pointMesh.vertices = _pointVertices;
            UpdateBackdrop(container, cam, pos);
        }

        // After the container transform is final for this frame, so the pick
        // projects through the pose the user is actually looking at. A full
        // pass over the neurons, so skipped while neither they nor the mouse
        // have moved.
        Vector2 mousePos = Mouse.current != null ? Mouse.current.position.ReadValue() : Vector2.zero;
        if (!unchanged || mousePos != _hoverMouse || _dragging)
        {
            _hoverMouse = mousePos;
            UpdateHover(cam);
        }

        // Outside the hover path: the colour key's cue rows show firing counts
        // whether or not anything is under the cursor. A full pass over every
        // neuron, so it runs on a timer - the numbers only need to feel live.
        if ((showColorLegend || _hoverIndex >= 0) && Time.unscaledTime >= _nextCountRefresh)
        {
            _nextCountRefresh = Time.unscaledTime + 0.25f;
            RefreshFiringCounts();
        }
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
        // Outside the displaySize gate: it is reporting metadata, not
        // placement, and a trainer that sends it should be believed whether or
        // not it also sent the placement block.
        _publishedDepthScale = g.depthScale;

        _positions = new Vector3[_n];
        _unit = new Vector3[_n];
        for (int i = 0; i < _n; i++)
        {
            _unit[i] = new Vector3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
            _positions[i] = _unit[i] * displaySize;
        }

        // Local y is the axis viz.py flattens (it writes pos[:, 1]). Keep the
        // published values so the depth keys can rescale without a resend, and
        // re-apply the current multiplier so a geometry heartbeat or a new job
        // does not silently undo what the user set.
        _baseDepth = new float[_n];
        for (int i = 0; i < _n; i++) _baseDepth[i] = _positions[i].y;
        ApplyDepth();

        var roleBytes = Convert.FromBase64String(g.role);
        _role = new byte[_n];
        Array.Copy(roleBytes, _role, Mathf.Min(roleBytes.Length, _n));

        // Side was already on the wire but unused; the hover label needs it,
        // because the cues are side-specific and "LC4" alone does not say
        // which eye. 0 is the unsided code (see SIDE_CODES).
        var sideBytes = Convert.FromBase64String(g.side);
        _side = new byte[_n];
        Array.Copy(sideBytes, _side, Mathf.Min(sideBytes.Length, _n));

        _typeNames = g.typeNames;
        _typeIdx = UShorts(g.typeIdx);
        if (_typeIdx != null && _typeIdx.Length < _n) _typeIdx = null;   // unusable
        _act = new byte[_n];
        BuildTypeCounts();
        BuildCueCounts();

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
        var uvs = new Vector2[_n * 4];
        for (int i = 0; i < _n; i++)
        {
            int b = i * 4, t = i * 6;
            // Sprites/Default is Cull Off, so winding does not matter here.
            tris[t] = b; tris[t + 1] = b + 1; tris[t + 2] = b + 2;
            tris[t + 3] = b + 2; tris[t + 4] = b + 1; tris[t + 5] = b + 3;
            // Corner order matches LateUpdate's billboard; the glow shader
            // draws a round spot from these.
            uvs[b] = new Vector2(0f, 0f); uvs[b + 1] = new Vector2(1f, 0f);
            uvs[b + 2] = new Vector2(0f, 1f); uvs[b + 3] = new Vector2(1f, 1f);
        }
        _pointMesh.Clear();
        _pointMesh.indexFormat = IndexFormat.UInt32;
        _pointMesh.vertices = _pointVertices;
        _pointMesh.uv = uvs;
        _pointMesh.SetIndices(tris, MeshTopology.Triangles, 0);
        UpdatePointBounds();

        ApplyActivity(new byte[_n]);   // draw the resting structure immediately
        Debug.Log($"[FlyBrainViz] geometry: {_n} neurons, {kept} edges "
                  + $"(of {_nEdges} sent)");
    }

    /// <summary>
    /// Rest and full-activity colours for a role code. `chase` splits role 1
    /// into its two cue families and is ignored by every other role.
    /// </summary>
    void RoleRamp(byte role, bool chase, out Color rest, out Color full)
    {
        if (role == 4)
        {
            rest = contextRestColor;
            full = contextActiveColor;
            return;
        }
        switch (role)
        {
            case 1: full = chase ? chaseColor : sensoryColor; break;
            case 2: full = commandColor; break;
            case 3: full = descendingColor; break;
            default: full = interneuronColor; break;
        }
        rest = full * restBrightness;
    }

    void ApplyActivity(byte[] act)
    {
        if (_positions == null) return;
        // Under the glow, silent neurons are lifted so the brain's shape reads
        // on its own, and their weight goes to the shader as alpha.
        bool glow = GlowOn;
        float restGain = glow ? Mathf.Max(1f, _anatSettings.glowRestGain) : 1f;
        float restAlpha = glow ? Mathf.Clamp01(_anatSettings.glowRestAlpha) : minAlpha;
        for (int i = 0; i < _n; i++)
        {
            // Kept so the hover label can count what is firing right now. The
            // raw byte, before the gamma below, so the count means "above this
            // fraction of full scale" rather than something bent by a display
            // curve.
            if (_act != null) _act[i] = i < act.Length ? act[i] : (byte)0;
            float t = (i < act.Length ? act[i] : (byte)0) / 255f;
            if (intensityGamma != 1f) t = Mathf.Pow(t, intensityGamma);

            Color rest, full;
            RoleRamp(_role[i], _isChase != null && i < _isChase.Length && _isChase[i],
                     out rest, out full);
            if (restGain != 1f)
                rest = new Color(Mathf.Min(1f, rest.r * restGain), Mathf.Min(1f, rest.g * restGain),
                                 Mathf.Min(1f, rest.b * restGain), rest.a);
            Color lit = Color.Lerp(rest, full, t);
            lit.a = Mathf.Lerp(restAlpha, maxAlpha, t);

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

    void OnDestroy()
    {
        Camera.onPostRender -= BlitOverlayCache;
        if (_overlayRt != null) { _overlayRt.Release(); Destroy(_overlayRt); }
        if (_overlayBlitMat != null) Destroy(_overlayBlitMat);
        DrawsBehindTrack = false;
        HiddenCamera.Destroy(_behindCam);
        HiddenCamera.Destroy(_overlayCam);
        // The rig is unparented to park it away from the course, so it does
        // not go down with this component on its own.
        var rig = _pointObject != null ? _pointObject.transform.parent : null;
        if (rig != null && rig.parent == null) Destroy(rig.gameObject);
        TrackGutterFraction = 0f;
    }

    void EnsureObjects()
    {
        if (_edgeObject != null) return;

        var container = new GameObject("FlyBrain").transform;
        container.SetParent(transform, false);
        container.localPosition = worldOffset;

        _edgeMesh = new Mesh { name = "FlyBrainEdges" };
        _pointMesh = new Mesh { name = "FlyBrainNeurons" };
        _edgeObject = NewMeshObject("FlyBrainEdges", container, _edgeMesh);
        _pointObject = NewMeshObject("FlyBrainNeurons", container, _pointMesh);

        if (useOverlayCamera) EnsureOverlayCamera(container);
        if (Anatomy) ParentContainer();
    }

    /// <summary>
    /// Put the overlay on its own layer, in its own orthographic camera, in a
    /// fixed corner of the screen. This is three fixes in one rig.
    ///
    /// It cannot be rendered by anything else. JetRacerCsiIntrinsics sets the
    /// CSI camera's cullingMask to ~0 - every layer - and that camera is what
    /// CsiFramePublisher reads for `camera/front`. Without a layer of its own
    /// the brain can appear in a policy's camera observation, which is a
    /// silent data-corruption bug rather than a visual one.
    ///
    /// It cannot drift off-screen. A viewport rect is a fraction of the
    /// window, so the overlay sits in the same place and takes the same share
    /// of the screen at any resolution, instead of being a world-space object
    /// whose framing depends on the main camera's aspect.
    ///
    /// It cannot smear. viz.py warns that depth outliers "project far off to
    /// the side and smear the whole structure into a radial fan" under a
    /// perspective camera, and the shipped depth of x3.81 was doing exactly
    /// that - the structure split into a bright cloud and a dim streak below
    /// it. An orthographic camera has no such divergence: depth changes what
    /// occludes what, and nothing else.
    /// </summary>
    void EnsureOverlayCamera(Transform container)
    {
        SetLayerRecursive(container.gameObject, OverlayLayer);

        // Parked far from the course so the overlay camera can only ever see
        // the brain: no clip planes to tune against scene geometry, and no
        // chance of the track wandering into frame behind it.
        container.SetParent(null, false);
        container.position = OverlayWorldOrigin;

        _overlayCam = HiddenCamera.Create("FlyBrainOverlayCamera");
        _overlayCam.orthographic = true;
        _overlayCam.cullingMask = 1 << OverlayLayer;
        // Cleared to black itself: its rect lies in the column the track
        // gives up (OverheadCameraFit), never over the track, and relying on
        // anything else to clear it leaves every past frame of the animated
        // fly smeared on screen.
        _overlayCam.clearFlags = CameraClearFlags.SolidColor;
        _overlayCam.backgroundColor = Color.black;
        _overlayCam.depth = 100f;             // after every scene camera
        _overlayCam.nearClipPlane = 0.01f;
        _overlayCam.farClipPlane = OverlayCamDistance * 4f;
        _overlayCam.allowHDR = false;
        _overlayCam.allowMSAA = false;
        _overlayCam.useOcclusionCulling = false;
        AimOverlayCamera();

        // Belt and braces: the scene cameras should not draw this layer even
        // though the rig is parked out of their way.
        StripOverlayLayer(Camera.main);
    }

    public const int OverlayLayer = 7;        // "FlyBrainOverlay" in TagManager
    public static int OverlayLayerMask { get { return 1 << OverlayLayer; } }
    static readonly Vector3 OverlayWorldOrigin = new Vector3(0f, 5000f, 0f);
    const float OverlayCamDistance = 500f;

    /// <summary>
    /// Frame the overlay on the track: the model ends up exactly as tall as
    /// the road, top and bottom level with it, sharing its centre line.
    ///
    /// The model here is the whole published cloud - brain, neck connective
    /// and nerve cord - matched as one object, so its proportions are the
    /// anatomy's and not something this code chose. The column's width falls
    /// out of that rather than being set independently, which is what keeps
    /// the shape undistorted while the height is pinned to the track's.
    /// </summary>
    void UpdateOverlayCamera()
    {
        // Orthographic unless a flight's pull-back turns it perspective
        // below; reset every frame so no view is ever left in perspective.
        _overlayCam.orthographic = true;
        _overlayCam.nearClipPlane = 0.01f;
        _overlayCam.farClipPlane = OverlayCamDistance * 4f;
        // Aim first: the projected extents are measured along this camera's
        // right and up, so its rotation has to be settled before they mean
        // anything.
        AimOverlayCamera();
        UpdateProjectedExtents();

        Rect vp = OverlayViewportScaled();
        bool on = IsVisible();
        float flightFrame = 0f;
        if (_fly != null)
        {
            _fly.HeadPitchTarget = CloseUp ? _anatSettings.closeUpHeadPitch : 0f;
            _fly.RearOpacityScale = CloseUp ? _anatSettings.closeUpRearOpacity : 1f;
            _fly.FlightRight = _fly.Root.InverseTransformDirection(_overlayCam.transform.right);
            _fly.FlightUp = _fly.Root.InverseTransformDirection(_overlayCam.transform.up);
            // Out to the whole flight at takeoff, back in once it has landed.
            float target = Anatomy && _fly.Flying ? 1f : 0f;
            _flightZoom = Mathf.MoveTowards(_flightZoom, target,
                Time.unscaledDeltaTime / Mathf.Max(0.01f, _anatSettings.flightCameraSeconds));
            flightFrame = Mathf.SmoothStep(0f, 1f, _flightZoom);
        }

        // The close-up keeps the fly view's column for the track's sake, and
        // draws from the left edge of the window up to it: the fly runs off
        // the left of the screen and is cut off where the track starts.
        Rect framed = vp;
        if (CloseUp)
        {
            float right = framed.xMax;
            float road = OverheadCameraFit.TrackScreenLeft;
            if (road > 0.05f) right = Mathf.Min(right, road);
            vp = new Rect(0f, 0f, Mathf.Clamp(right, 0.05f, 1f), 1f);
        }

        _overlayCam.rect = vp;
        _overlayCam.enabled = on;

        // Claim the column so OverheadCameraFit can squeeze the track out of
        // it, and give it straight back when the overlay is toggled off - the
        // track should reclaim the full window rather than keep a gap.
        TrackGutterFraction = on ? Mathf.Clamp01(framed.xMax) : 0f;
        if (CloseUp)
        {
            FrameCloseUp(vp);
            BlendToFlightFrame(flightFrame);
            return;
        }

        // Framed on what the brain actually COVERS on screen, not on how far
        // its furthest neuron is from the centre.
        //
        // The old sizing fitted a sphere of _modelRadius, which is wrong in
        // both directions at once. The radius is set by the single most
        // distant neuron along the model's long axis, so the camera zoomed
        // out far enough to fit that distance on *every* axis - including the
        // short one, and including depth, which points straight at an
        // orthographic camera and covers no screen at all. The brain then sat
        // in the middle of its column at roughly two thirds the size it could
        // have been, and stretching the depth with ] made it smaller still.
        //
        // Measuring the projected box instead means the height below is the
        // height you see, which is what lets it be matched to the track's.
        float zoom = Mathf.Max(0.01f, overlayZoom);
        float halfH = _haveFraming ? _projHalfH * zoom : _modelRadius * zoom;
        float halfW = _haveFraming ? _projHalfW * zoom : _modelRadius * zoom;
        halfH = Mathf.Max(1e-3f, halfH);
        halfW = Mathf.Max(1e-3f, halfW);

        // orthographicSize is the half-HEIGHT; the half-width it implies is
        // that times the aspect. Taking the max of the two requirements lets
        // whichever axis is tighter decide, so a column too narrow to show the
        // brain at track height shrinks it rather than cropping it.
        Rect px = _overlayCam.pixelRect;
        float aspect = px.height > 1f ? px.width / px.height : 1f;
        _overlayCam.orthographicSize = Mathf.Max(halfH, halfW / Mathf.Max(0.01f, aspect));
        if (Anatomy) BlendToFlightFrame(flightFrame);
    }

    private float _flightZoom;             // 0 framed as usual, 1 pulled back to the whole flight

    /// <summary>
    /// Move the overlay camera part way (w) from the framing just set toward
    /// one that holds the fly at rest and hovering at the top of the takeoff.
    /// Not the orbit: it is sized from this shot to carry the fly out of it
    /// and back, so the shot's size is passed on every frame, before the
    /// camera starts to move. The hovering fly turns, so it is allowed its
    /// largest rest extent in every direction.
    /// </summary>
    void BlendToFlightFrame(float w)
    {
        if (_fly == null || _fly.FramePoints == null) return;
        var cam = _overlayCam;
        Transform root = _fly.Root;
        Vector3 r = cam.transform.right, u = cam.transform.up;
        float sc = root.lossyScale.x;
        Quaternion rot = root.rotation;
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        foreach (var p in _fly.FramePoints)
        {
            Vector3 q = rot * (p * sc);
            float a = Vector3.Dot(q, r), b = Vector3.Dot(q, u);
            if (a < minX) minX = a;
            if (a > maxX) maxX = a;
            if (b < minY) minY = b;
            if (b > maxY) maxY = b;
        }
        float cx = (minX + maxX) * 0.5f, cy = (minY + maxY) * 0.5f;
        float reach = Mathf.Max(maxX - minX, maxY - minY) * 0.5f;
        Vector3 lift = rot * (_fly.Lift * sc);
        float hx = cx + Vector3.Dot(lift, r), hy = cy + Vector3.Dot(lift, u);
        float ax = reach, ay = reach;
        minX = Mathf.Min(minX, hx - ax); maxX = Mathf.Max(maxX, hx + ax);
        minY = Mathf.Min(minY, hy - ay); maxY = Mathf.Max(maxY, hy + ay);

        float pad = Mathf.Max(1f, _anatSettings.flightFramePadding);
        Rect px = cam.pixelRect;
        float aspect = px.height > 1f ? px.width / px.height : 1f;
        float size = Mathf.Max((maxY - minY) * 0.5f, (maxX - minX) * 0.5f / Mathf.Max(0.01f, aspect)) * pad;
        Vector3 pos = root.position + r * ((minX + maxX) * 0.5f) + u * ((minY + maxY) * 0.5f)
                      - cam.transform.forward * OverlayCamDistance;
        _fly.ShotHalfSize = new Vector2(size * aspect, size) / Mathf.Max(1e-6f, sc);
        _fly.CamTanHalfFov = 0f;
        if (w <= 0f) return;
        cam.orthographicSize = Mathf.Lerp(cam.orthographicSize, size, w);
        cam.transform.position = Vector3.Lerp(cam.transform.position, pos, w);

        // Perspective, so a fly flying away shrinks. Framed like the
        // orthographic shot on the plane through the fly, and the field of
        // view opens from nearly nothing as the camera pulls back: at w = 0
        // the two are indistinguishable, so there is no jump either way.
        float fov = Mathf.Lerp(1f, Mathf.Clamp(_anatSettings.flightFov, 5f, 90f), w);
        float tan = Mathf.Tan(0.5f * fov * Mathf.Deg2Rad);
        Vector3 fwd = cam.transform.forward;
        Vector3 centre = cam.transform.position + fwd * OverlayCamDistance;
        float dist = cam.orthographicSize / tan;
        cam.orthographic = false;
        cam.fieldOfView = fov;
        cam.transform.position = centre - fwd * dist;
        cam.nearClipPlane = Mathf.Max(0.01f, dist * 0.05f);
        cam.farClipPlane = dist + Mathf.Max(dist, _fly.OrbitReach * sc) * 3f;
        _fly.CamPos = root.InverseTransformPoint(cam.transform.position);
        _fly.CamForward = root.InverseTransformDirection(fwd);
        _fly.CamTanHalfFov = tan;
    }

    /// <summary>
    /// Frame the close-up on the head alone: the head's box, bent as the head
    /// is and carried up with the fly in flight, is fitted to closeUpHeadBox
    /// (screen fractions) and centred in it. The rest of the fly lands
    /// wherever that puts it, including off screen.
    /// </summary>
    void FrameCloseUp(Rect vp)
    {
        var s = _anatSettings;
        Rect box = s.closeUpHeadBox;
        Transform root = _fly.Root;
        Vector3 r = _overlayCam.transform.right, u = _overlayCam.transform.up;
        Quaternion bend = Quaternion.AngleAxis(_fly.HeadPitch, _fly.Right);
        // Framed on the head at rest during a 1-key flight: the pull-back and
        // the return then only zoom, and the fly moves against a still camera.
        Vector3 lift = _fly.Flying ? Vector3.zero : _fly.ShownLift;
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        for (int k = 0; k < 8; k++)
        {
            Vector3 p = _fly.HeadCentre
                        + _fly.Right * (((k & 1) == 0 ? -0.5f : 0.5f) * _fly.HeadWidth)
                        + _fly.Up * (((k & 2) == 0 ? -0.5f : 0.5f) * _fly.HeadHeight)
                        + _fly.Forward * (((k & 4) == 0 ? -0.5f : 0.5f) * _fly.HeadDepth);
            if (_fly.HasNeck) p = _fly.NeckRest + bend * (p - _fly.NeckRest);
            Vector3 w = root.TransformPoint(p + lift) - root.position;
            float a = Vector3.Dot(w, r), b = Vector3.Dot(w, u);
            if (a < minX) minX = a;
            if (a > maxX) maxX = a;
            if (b < minY) minY = b;
            if (b > maxY) maxY = b;
        }
        float halfW = Mathf.Max(1e-4f, (maxX - minX) * 0.5f);
        float halfH = Mathf.Max(1e-4f, (maxY - minY) * 0.5f);

        // orthographicSize is half the viewport's height in world units; pick
        // the one that makes the head exactly as tall, or as wide, as the box,
        // whichever is tighter.
        float sw = Mathf.Max(1f, Screen.width), sh = Mathf.Max(1f, Screen.height);
        float fill = Mathf.Max(0.05f, s.closeUpHeadFill);
        float size = Mathf.Max(halfH * vp.height / Mathf.Max(0.01f, box.height),
                               halfW * vp.height * sh / (Mathf.Max(0.01f, box.width) * sw)) / fill;
        _overlayCam.orthographicSize = size;

        // Shift the camera so the head's centre lands on the box's centre
        // rather than the viewport's.
        float perPixel = 2f * size / (vp.height * sh);
        float dx = (box.center.x - vp.center.x) * sw * perPixel;
        float dy = (box.center.y - vp.center.y) * sh * perPixel;
        Vector3 head = r * ((minX + maxX) * 0.5f) + u * ((minY + maxY) * 0.5f);
        _overlayCam.transform.position = root.position + head - r * dx - u * dy
                                         - _overlayCam.transform.forward * OverlayCamDistance;
    }

    /// <summary>
    /// Half-width and half-height of the brain as the overlay camera sees it,
    /// plus the offset from the container's origin to the centre of that box.
    ///
    /// Measured over the real neuron positions rather than a bounding sphere
    /// or an axis-aligned box: the cloud is neither, and both approximations
    /// round in the direction of drawing it too small.
    ///
    /// The drag rotation is deliberately excluded. The framing is recomputed
    /// when the pose changes, and folding the drag in would rescale the brain
    /// while the user is turning it - it would swell and shrink under the
    /// cursor instead of just rotating. The settled pose is what gets framed;
    /// the extra reach of a turned one is covered by overlayZoom's padding.
    /// </summary>
    void UpdateProjectedExtents()
    {
        if (_positions == null || _n <= 0 || _overlayCam == null) return;
        if (Anatomy) { UpdateFlyExtents(); return; }

        Quaternion rot = BaseOrientation(_overlayCam) * Quaternion.Euler(0f, _spin, 0f);
        // A full pass over every neuron, so it runs only when the answer can
        // have changed: a new pose, or new positions (geometry, or a depth
        // keypress). At the shipped spin of 0 that means it is idle.
        if (_haveFraming && !_framingDirty && Quaternion.Angle(rot, _framingRot) < 0.25f)
            return;

        // Into the model's own frame, so the per-neuron work is two dot
        // products against a fixed pair of vectors.
        Quaternion inv = Quaternion.Inverse(rot);
        Vector3 right = inv * _overlayCam.transform.right;
        Vector3 up = inv * _overlayCam.transform.up;

        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        for (int i = 0; i < _n; i++)
        {
            Vector3 p = _positions[i];
            float a = p.x * right.x + p.y * right.y + p.z * right.z;
            float b = p.x * up.x + p.y * up.y + p.z * up.z;
            if (a < minX) minX = a;
            if (a > maxX) maxX = a;
            if (b < minY) minY = b;
            if (b > maxY) maxY = b;
        }
        if (maxX < minX || maxY < minY) return;

        _projHalfW = (maxX - minX) * 0.5f;
        _projHalfH = (maxY - minY) * 0.5f;

        // Min/max rather than max-absolute, so a cloud whose centre of mass is
        // off its origin is centred on what it covers instead of being framed
        // around empty space on the opposite side. World-space, because the
        // coefficients are already distances along the camera's own axes.
        _projCentre = _overlayCam.transform.right * ((maxX + minX) * 0.5f)
                      + _overlayCam.transform.up * ((maxY + minY) * 0.5f);

        _framingRot = rot;
        _haveFraming = true;
        _framingDirty = false;
    }

    /// <summary>
    /// Share of the window's width, measured from the left edge, that the
    /// brain overlay is occupying - 0 when it is hidden. OverheadCameraFit
    /// narrows the track's viewport by this much so the two do not overlap.
    /// </summary>
    public static float TrackGutterFraction { get; private set; }

    /// <summary>
    /// Most of the window the overlay may ever claim. Shared with
    /// OverheadCameraFit, which clamps to the same number - if the overlay
    /// could grow past it the two would disagree and overlap.
    /// </summary>
    public const float MaxGutter = 0.6f;

    /// <summary>
    /// - and = shrink and grow the overlay, 0 back to the configured default.
    ///
    /// 1 is "as tall as the track", so this is a deliberate departure from
    /// the thing it is meant to line up with rather than a size in the
    /// abstract.
    ///
    /// Reports the size actually achieved alongside the one requested. They
    /// diverge once the column hits MaxGutter: past that the model has to
    /// shrink to fit the width it is allowed, so a request with nowhere left
    /// to go says so instead of looking like a dead key.
    /// </summary>
    void HandleSizeKeys(Keyboard kb)
    {
        if (kb == null) return;
        bool down = kb.minusKey.wasPressedThisFrame;
        bool up = kb.equalsKey.wasPressedThisFrame;
        bool reset = kb.digit0Key.wasPressedThisFrame;
        if (!down && !up && !reset) return;

        overlaySizeScale = reset
            ? _overlaySizeDefault
            : Mathf.Clamp(overlaySizeScale * (up ? sizeStepFactor : 1f / sizeStepFactor),
                          overlaySizeMin, overlaySizeMax);

        float achieved = _lastAchievedScale;
        string note = achieved < overlaySizeScale - 0.02f
            ? $" (showing x{achieved:0.00} - the window has no more room)"
            : "";
        Debug.Log($"[FlyBrainViz] overlay size x{overlaySizeScale:0.00}{note}"
                  + $" - to keep it, set overlaySizeScale = {overlaySizeScale:0.00}f");
    }

    /// <summary>
    /// Point the overlay camera at the brain from the same angle the scene's
    /// overhead camera views the track.
    ///
    /// The brain's neurons are laid out in its local X-Z plane with depth on
    /// local Y, so it only reads as a brain when viewed down that Y axis. The
    /// scene's overhead camera does exactly that, which is why the overlay
    /// looked right as a world-space object; a camera looking along +Z
    /// instead catches the whole thing edge-on and it collapses to a sliver.
    ///
    /// Copying the overhead rotation rather than hard-coding "straight down"
    /// also keeps the two views in the same frame, so the brain's forward is
    /// the track's forward and the -144.3 / -9.1 square-up stays meaningful.
    /// </summary>
    void AimOverlayCamera()
    {
        if (_overlayCam == null) return;
        Quaternion rot = OverheadRotation();
        _overlayCam.transform.rotation = rot;
        // Offset onto the centre of what the brain covers, so it sits in the
        // middle of its column rather than hanging off one side of it. One
        // frame stale after a pose change, which is invisible on a value that
        // only moves when an arrow key is pressed.
        _overlayCam.transform.position = OverlayWorldOrigin + _projCentre
                                         - (rot * Vector3.forward) * OverlayCamDistance;
    }

    /// <summary>
    /// The scene overhead camera's orientation, or a straight-down fallback
    /// looking along -Y with +Z up - which is what Euler(90,0,0) gives and so
    /// matches an unyawed overhead camera.
    /// </summary>
    Quaternion OverheadRotation()
    {
        // Throttled: GameObject.Find walks the scene, and this runs every
        // frame. It resolves on the first call in practice; the retry only
        // matters if the camera is rebuilt under us.
        if (_overheadCam == null && Time.unscaledTime >= _nextOverheadLookup)
        {
            _nextOverheadLookup = Time.unscaledTime + 1f;
            var go = GameObject.Find("Main Camera");
            if (go != null) _overheadCam = go.GetComponent<Camera>();
        }
        if (_overheadCam != null) return _overheadCam.transform.rotation;
        return Quaternion.LookRotation(Vector3.down, Vector3.forward);
    }

    /// <summary>
    /// The overlay's slice of the window: height and centre from the track,
    /// width from the model's own proportions at that height.
    ///
    /// Deliberately takes no account of the legends. An earlier version kept
    /// the overlay clear of the controls panel, which sounds tidier but made
    /// that panel's height the thing that decided how big the brain could be,
    /// which capped the size outright and left the - and = keys with nothing
    /// to do. The legends are translucent and draw on top;
    /// letting them overlap costs a corner of the overlay and buys back the
    /// whole column.
    /// </summary>
    Rect OverlayViewportScaled()
    {
        float s = Mathf.Clamp(overlaySizeScale, overlaySizeMin, overlaySizeMax);
        Rect vp = overlayViewport;

        // The fly view is a portrait rather than the track's counterpart: it
        // fills its own configured area and the camera fits the fly to it.
        if (Anatomy)
        {
            Rect a = _anatSettings.viewport;
            float w = Mathf.Clamp(a.width * s, 0.05f, MaxGutter);
            float h = Mathf.Clamp(a.height * s, 0.05f, 1f);
            _lastAchievedScale = s;
            return new Rect(Mathf.Clamp(a.center.x - w * 0.5f, 0f, MaxGutter - w),
                            Mathf.Clamp(a.center.y - h * 0.5f, 0f, 1f - h), w, h);
        }

        // Height and centre come from the track, so the brain reads as the
        // track's counterpart rather than a widget floating next to it. The
        // fallback is the configured viewport, for the frames before the fit
        // has measured anything and for a scene with no road in it.
        float trackH = OverheadCameraFit.TrackScreenHeight;
        bool haveTrack = trackH > 0.01f;
        float baseH = haveTrack ? trackH : overlayViewport.height;
        float centreY = haveTrack
            ? OverheadCameraFit.TrackScreenCentreY
            : overlayViewport.y + overlayViewport.height * 0.5f;

        // The viewport is overlayZoom TALLER than the band being matched, not
        // equal to it. orthographicSize carries the same factor as breathing
        // room around the brain, so a viewport sized to the track exactly
        // would render the brain a margin short of it. Putting the padding in
        // both places cancels it: the brain, not its viewport, is what ends up
        // the same height as the track.
        float zoom = Mathf.Max(0.01f, overlayZoom);
        float wantH = baseH * zoom;
        vp.height = Mathf.Clamp(wantH * s, 0.05f, 1f);
        vp.y = Mathf.Clamp(centreY - vp.height * 0.5f, 0f, 1f - vp.height);

        // Width follows from the brain's own proportions at that height,
        // instead of being a number that the brain then has to fit inside.
        // The column is only as wide as the brain needs, so the track is
        // charged for exactly the space being used and no more.
        //
        // This does feed back - a wider column leaves the track narrower, the
        // fit pulls back, and the band this is measured against gets shorter.
        // It settles rather than oscillates because each pass shrinks the
        // correction (the track is wider than it is tall, so a column sized
        // from its height is a fraction of the width it cost), and the floor
        // below keeps a bad measurement from collapsing the column.
        float want = overlayViewport.width;
        if (_haveFraming && _projHalfH > 1e-4f && Screen.width > 1)
        {
            float heightPx = vp.height * Screen.height;
            want = heightPx * (_projHalfW / _projHalfH) / Screen.width;
        }
        // Capped on the RIGHT edge, not the width: OverheadCameraFit refuses
        // to give the track away past MaxGutter, so a wider overlay would
        // claim a strip the track is still drawing into.
        float maxW = Mathf.Max(0.05f, MaxGutter - vp.x);
        vp.width = Mathf.Clamp(Mathf.Max(want, overlayViewport.width * s), 0.05f, maxW);

        // Reported in terms of the brain's rendered height, which is what the
        // key is asking to change. Height alone would claim success on a
        // column too narrow to show it, where the brain shrinks to fit the
        // width instead - so a request that ran out of room says so.
        float heightScale = wantH > 1e-4f ? vp.height / wantH : s;
        float widthLimit = want > 1e-6f ? Mathf.Min(1f, vp.width / want) : 1f;
        _lastAchievedScale = heightScale * widthLimit;
        return vp;
    }

    /// <summary>
    /// Distance from the container's origin to the furthest neuron, for
    /// framing. Recomputed whenever the positions change, which is geometry
    /// and every depth keypress.
    /// </summary>
    void UpdateModelRadius()
    {
        float r2 = 0f;
        if (_positions != null)
            for (int i = 0; i < _n; i++) r2 = Mathf.Max(r2, _positions[i].sqrMagnitude);
        _modelRadius = r2 > 0f ? Mathf.Sqrt(r2) : Mathf.Max(0.01f, displaySize);
        // Every route that moves a neuron comes through here - geometry
        // arriving and every [ or ] - so this is the one place the overlay's
        // framing needs to be told the positions are stale.
        _framingDirty = true;
    }

    // ---- fly anatomy mode --------------------------------------------------

    /// <summary>Bring the scene in line with viewMode, falling back to the
    /// connectome view if the fly cannot be built.</summary>
    void ApplyViewMode()
    {
        if (viewMode != FlyViewMode.Connectome)
        {
            if (!useOverlayCamera)
            {
                Debug.LogWarning("[FlyBrainViz] the fly view needs useOverlayCamera");
                viewMode = FlyViewMode.Connectome;
            }
            else
            {
                EnsureFly();
                if (_fly == null) viewMode = FlyViewMode.Connectome;
            }
        }
        ParentContainer();
        CancelDrag();
        _framingDirty = true;
        UpdatePointBounds();
        if (_fly != null && !Anatomy) _fly.SetVisible(false);
    }

    /// <summary>Loaded up front, not with the fly: the neuron glow settings on
    /// it apply to the brain-only view as well.</summary>
    void LoadAnatSettings()
    {
        if (_anatSettings != null) return;
        _anatSettings = FlyAnatomySettings.Load();
        _viewYawDefault = _anatSettings.viewYaw;
        _viewElevationDefault = _anatSettings.viewElevation;
        _closeUpYawDefault = _anatSettings.closeUpViewYaw;
        _closeUpElevationDefault = _anatSettings.closeUpViewElevation;
    }

    void EnsureFly()
    {
        if (_fly != null || _flyFailed) return;
        LoadAnatSettings();
        _fly = FlyAnatomyView.Create(OverlayWorldOrigin, OverlayLayer, _anatSettings);
        if (_fly == null)
        {
            _flyFailed = true;
            Debug.LogWarning("[FlyBrainViz] fly model unavailable; staying in the connectome view");
            return;
        }
        _fly.SetVisible(false);
    }

    /// <summary>
    /// The neuron container rides inside the fly's Root in the fly view, so
    /// the per-neuron positions can be written in the fly's own frame; in the
    /// connectome view it goes back to where EnsureObjects put it.
    /// </summary>
    void ParentContainer()
    {
        if (_pointObject == null) return;
        var c = _pointObject.transform.parent;
        if (Anatomy)
        {
            c.SetParent(_fly.Root, false);
            c.localPosition = Vector3.zero;
            c.localRotation = Quaternion.identity;
        }
        else if (_overlayCam != null)
        {
            c.SetParent(null, false);
            c.position = OverlayWorldOrigin;
        }
        else
        {
            c.SetParent(transform, false);
            c.localPosition = worldOffset;
        }
        c.localScale = Vector3.one;
    }

    /// <summary>
    /// The fly's pose without the drag: head to the camera's right, turned by
    /// the view's yaw about its own vertical, then tipped toward the camera by
    /// its elevation so the view is from above, and rolled in the screen
    /// plane. The fly view and the close-up each have their own dials.
    /// </summary>
    Quaternion SettledFlyRotation(Camera cam)
    {
        Quaternion align = Quaternion.LookRotation(Vector3.right, Vector3.up)
                           * Quaternion.Inverse(Quaternion.LookRotation(_fly.Forward, _fly.Up));
        Quaternion view = Quaternion.AngleAxis(FlyRoll, Vector3.forward)
                          * Quaternion.AngleAxis(-FlyElevation, Vector3.right)
                          * Quaternion.AngleAxis(FlyYaw + _anatSpin, Vector3.up);
        // Applied in the fly's own frame, first: a nose-down pitch about its
        // left-right axis. Positive tips the head down, as for the neck bend.
        Quaternion tilt = Quaternion.AngleAxis(FlyBodyPitch, _fly.Right);
        return cam.transform.rotation * view * align * tilt;
    }

    /// <summary>UpdateProjectedExtents for the fly view: frame the fly at
    /// rest, plus the share of the takeoff climb FlyAnatomyView lets show, so
    /// the fly fills its column and flight does not rescale the picture.</summary>
    void UpdateFlyExtents()
    {
        Quaternion rot = SettledFlyRotation(_overlayCam);
        Vector3 lift = _fly.Lift * Mathf.Clamp01(_anatSettings.flightLiftShown);
        if (_haveFraming && !_framingDirty && Quaternion.Angle(rot, _framingRot) < 0.25f
            && lift == _framedLift)
            return;
        float sc = _fly.Root.lossyScale.x;
        Vector3 r = _overlayCam.transform.right, u = _overlayCam.transform.up;
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        foreach (var p0 in _fly.FramePoints)
        {
            for (int k = 0; k < 2; k++)
            {
                Vector3 w = rot * ((k == 0 ? p0 : p0 + lift) * sc);
                float a = Vector3.Dot(w, r), b = Vector3.Dot(w, u);
                if (a < minX) minX = a;
                if (a > maxX) maxX = a;
                if (b < minY) minY = b;
                if (b > maxY) maxY = b;
            }
        }
        _framedLift = lift;
        _projHalfW = (maxX - minX) * 0.5f;
        _projHalfH = (maxY - minY) * 0.5f;
        _projCentre = r * ((maxX + minX) * 0.5f) + u * ((maxY + minY) * 0.5f);
        _framingRot = rot;
        _haveFraming = true;
        _framingDirty = false;
    }

    void UpdateAnatomyPositions()
    {
        EnsureAnatomyLayout();
        if (_anatHead == null) return;
        Matrix4x4 toRoot = _fly.Root.worldToLocalMatrix;
        Matrix4x4 mh = toRoot * _fly.HeadBone.localToWorldMatrix;
        Matrix4x4 mt = toRoot * _fly.ThoraxBone.localToWorldMatrix;
        // The close-up shrinks the brain about its own centre, riding the
        // head; the cord keeps its size and the connective blends between.
        float k = CloseUp ? Mathf.Max(0.05f, _anatSettings.closeUpBrainScale) : 1f;
        Vector3 c = mh.MultiplyPoint3x4(_anatBrainCentre);
        for (int i = 0; i < _n; i++)
        {
            float w = _anatW[i];
            if (w >= 1f) { _anatPos[i] = mt.MultiplyPoint3x4(_anatThorax[i]); continue; }
            Vector3 ph = mh.MultiplyPoint3x4(_anatHead[i]);
            if (k != 1f) ph = c + (ph - c) * k;
            _anatPos[i] = w <= 0f ? ph : Vector3.Lerp(ph, mt.MultiplyPoint3x4(_anatThorax[i]), w);
        }
    }

    void EnsureAnatomyLayout()
    {
        if (_unit == null || _fly == null) return;
        int hash = _anatSettings.PlacementHash();
        if (_anatStamp == _geometryStamp && hash == _anatHash && _anatFly == _fly
            && _anatPos != null && _anatPos.Length == _n)
            return;
        BuildAnatomy();
        _anatStamp = _geometryStamp;
        _anatHash = hash;
        _anatFly = _fly;
        _framingDirty = true;
    }

    /// <summary>
    /// Place every neuron inside the fly, at true proportions.
    ///
    /// The published cloud is the whole CNS - brain, neck connective, nerve
    /// cord - reoriented and flattened for the top-down connectome view
    /// (viz.py _build_geometry, default axes "-z,y,x"): local x is the body
    /// axis, y the axis it squashes by depthScale, z the bilateral one. The
    /// squash is a pure final multiply after a +/-1 clip, so dividing it back
    /// out restores the anatomy exactly for everything inside the clip.
    ///
    /// Orientation is read from the data rather than assumed:
    ///  - The neck connective holds no somata, so it is the widest empty run
    ///    of the histogram along the body axis. The side of it with the wider
    ///    lateral spread is the brain.
    ///  - Left is where the side==L neurons sit.
    ///  - The fly brain is taller than it is deep, so if its extent along the
    ///    body axis beats its extent along y, the CNS is in the dissected,
    ///    straightened pose (brain dorsal away from the cord), and the cord is
    ///    bent back neckBendDegrees at the neck to lie along the thorax.
    ///    Otherwise it is already in body pose and is placed rigidly.
    ///  - The cord attaches behind the brain (straightened) or below it (body
    ///    pose), which signs y.
    /// The brain is centred on the head and scaled so its width is
    /// brainWidthFraction of the head's; that one scale applies to the whole
    /// CNS, so the cord's size and position relative to the brain are real.
    ///
    /// Each neuron is stored in the rest frame of the bone it rides - the head
    /// for the brain, the thorax for the cord, a blend across the connective -
    /// so idle_look turns the brain with the head and flight carries the lot.
    /// </summary>
    void BuildAnatomy()
    {
        var s = _anatSettings;
        var f = _fly;
        int n = _n;
        float ds = _publishedDepthScale > 0f ? _publishedDepthScale : 0.12f;
        var u = new float[n];
        var y = new float[n];
        var l = new float[n];
        float umin = float.MaxValue, umax = float.MinValue;
        for (int i = 0; i < n; i++)
        {
            u[i] = _unit[i].x;
            y[i] = _unit[i].y / ds;
            l[i] = _unit[i].z;
            umin = Mathf.Min(umin, u[i]);
            umax = Mathf.Max(umax, u[i]);
        }

        const int Bins = 96;
        float bw = Mathf.Max(1e-6f, (umax - umin) / Bins);
        var hist = new int[Bins];
        int peak = 0;
        for (int i = 0; i < n; i++)
            hist[Mathf.Clamp((int)((u[i] - umin) / bw), 0, Bins - 1)]++;
        for (int b = 0; b < Bins; b++) peak = Mathf.Max(peak, hist[b]);
        int thr = Mathf.Max(1, peak / 100);
        int bestLo = -1, bestLen = 0, edge = Bins / 10;
        for (int b = edge; b < Bins - edge;)
        {
            if (hist[b] > thr) { b++; continue; }
            int e = b;
            while (e < Bins - edge && hist[e] <= thr) e++;
            if (e - b > bestLen) { bestLen = e - b; bestLo = b; }
            b = e;
        }
        bool haveGap = bestLo >= 0;
        float gapLo = umin + bestLo * bw, gapHi = umin + (bestLo + bestLen) * bw;

        if (haveGap)
        {
            var above = new List<float>();
            var below = new List<float>();
            for (int i = 0; i < n; i++)
            {
                if (u[i] >= gapHi) above.Add(l[i]);
                else if (u[i] <= gapLo) below.Add(l[i]);
            }
            if (Range(below, 0.02f, 0.98f) > Range(above, 0.02f, 0.98f))
            {
                for (int i = 0; i < n; i++) u[i] = -u[i];
                float t = gapLo;
                gapLo = -gapHi;
                gapHi = -t;
            }
        }

        // Brain is now at high u, the cord at low u.
        var w = new float[n];
        var bu = new List<float>();
        var by = new List<float>();
        var bl = new List<float>();
        float vySum = 0f;
        int nBrain = 0, nCord = 0;
        for (int i = 0; i < n; i++)
        {
            w[i] = haveGap ? Mathf.InverseLerp(gapHi, gapLo, u[i]) : 0f;
            if (w[i] <= 0f) { bu.Add(u[i]); by.Add(y[i]); bl.Add(l[i]); nBrain++; }
            else if (w[i] >= 1f) { vySum += y[i]; nCord++; }
        }
        if (nBrain < 10)
        {
            bu.Clear(); by.Clear(); bl.Clear();
            for (int i = 0; i < n; i++) { bu.Add(u[i]); by.Add(y[i]); bl.Add(l[i]); }
        }
        float medU = Percentile(bu, 0.5f), medY = Percentile(by, 0.5f), medL = Percentile(bl, 0.5f);
        float spanU = Range(bu, 0.05f, 0.95f), spanY = Range(by, 0.05f, 0.95f);
        float width = Mathf.Max(1e-6f, Range(bl, 0.02f, 0.98f));

        float lSum = 0f, rSum = 0f;
        int nL = 0, nR = 0;
        if (_side != null)
            for (int i = 0; i < n && i < _side.Length; i++)
            {
                if (_side[i] == 1) { lSum += l[i]; nL++; }
                else if (_side[i] == 2) { rSum += l[i]; nR++; }
            }
        float sL = (nL > 0 && nR > 0 && lSum / nL < rSum / nR) ? -1f : 1f;
        if (s.flipLeftRight) sL = -sL;
        float sY = nCord > 0 && vySum / nCord < medY ? -1f : 1f;
        if (s.flipFrontBack) sY = -sY;

        bool straightened = s.layout == FlyAnatomySettings.CnsLayout.Straightened
                            || (s.layout == FlyAnatomySettings.CnsLayout.Auto && spanU > spanY);

        // Anatomical components per neuron, in published units: right, up, forward.
        var ar = new float[n];
        var au = new float[n];
        var af = new float[n];
        float pivotFwd = 0f;
        for (int i = 0; i < n; i++)
        {
            float du = u[i] - medU, dy = y[i] - medY, dl = l[i] - medL;
            ar[i] = -sL * dl;
            if (straightened) { au[i] = du; af[i] = -sY * dy; }
            else { af[i] = du; au[i] = -sY * dy; }
            if (w[i] >= 1f) pivotFwd += af[i];
        }
        if (straightened && haveGap)
        {
            pivotFwd = nCord > 0 ? pivotFwd / nCord : 0f;
            float pivotUp = (gapLo + gapHi) * 0.5f - medU;
            for (int i = 0; i < n; i++)
            {
                if (w[i] <= 0f) continue;
                float th = s.neckBendDegrees * w[i] * Mathf.Deg2Rad;
                float c = Mathf.Cos(th), sn = Mathf.Sin(th);
                float du2 = au[i] - pivotUp, df2 = af[i] - pivotFwd;
                au[i] = pivotUp + du2 * c - df2 * sn;
                af[i] = pivotFwd + du2 * sn + df2 * c;
            }
        }

        float scale = s.brainWidthFraction * f.HeadWidth / width;
        float hw = f.HeadWidth;
        Vector3 shift = (s.brainOffset.x * f.Right + s.brainOffset.y * f.Up + s.brainOffset.z * f.Forward) * hw;
        Vector3 cordShift = (s.vncOffset.x * f.Right + s.vncOffset.y * f.Up + s.vncOffset.z * f.Forward) * hw;
        Matrix4x4 hInv = f.HeadRestInRoot.inverse, tInv = f.ThoraxRestInRoot.inverse;

        _anatHead = new Vector3[n];
        _anatThorax = new Vector3[n];
        _anatPos = new Vector3[n];
        _anatW = w;
        _anatBrainCentre = hInv.MultiplyPoint3x4(f.HeadCentre + shift);
        for (int i = 0; i < n; i++)
        {
            Vector3 p = f.HeadCentre + shift + cordShift * w[i]
                        + (ar[i] * f.Right + au[i] * f.Up + af[i] * f.Forward) * scale;
            _anatHead[i] = hInv.MultiplyPoint3x4(p);
            _anatThorax[i] = tInv.MultiplyPoint3x4(p);
            _anatPos[i] = p;
        }
        _anatPointSize = pointSize / Mathf.Max(1e-6f, displaySize) * scale * s.pointScale;

        Debug.Log($"[FlyBrainViz] fly view: {(straightened ? "straightened" : "body-pose")} CNS "
                  + $"(brain {spanU:0.###} along body axis vs {spanY:0.###} deep), "
                  + (haveGap ? $"neck gap {bestLen} bins, " : "no neck gap found, ")
                  + $"brain {nBrain} / cord {nCord} / connective {n - nBrain - nCord} neurons, "
                  + $"depth restored x{1f / ds:0.##}, left sign {sL:+0;-0}, y sign {sY:+0;-0}");
    }

    static float Percentile(List<float> v, float q)
    {
        if (v.Count == 0) return 0f;
        var a = v.ToArray();
        Array.Sort(a);
        return a[Mathf.Clamp(Mathf.RoundToInt(q * (a.Length - 1)), 0, a.Length - 1)];
    }

    static float Range(List<float> v, float lo, float hi)
    {
        if (v.Count == 0) return 0f;
        var a = v.ToArray();
        Array.Sort(a);
        int n = a.Length - 1;
        return a[Mathf.RoundToInt(hi * n)] - a[Mathf.RoundToInt(lo * n)];
    }

    // The fly view is a presentation piece and should come up without a
    // trainer running, so the last geometry received is kept on disk.
    static string GeometryCachePath
    {
        get { return System.IO.Path.Combine(Application.persistentDataPath, "fly_brain_geometry.json"); }
    }

    void SaveGeometryCache(string json)
    {
        if (string.IsNullOrEmpty(json)) return;
        try { System.IO.File.WriteAllText(GeometryCachePath, json); }
        catch (Exception e) { Debug.LogWarning($"[FlyBrainViz] geometry cache not written: {e.Message}"); }
    }

    void TryLoadGeometryCache()
    {
        if (_triedGeometryCache) return;
        _triedGeometryCache = true;
        try
        {
            if (!System.IO.File.Exists(GeometryCachePath)) return;
            var p = JsonUtility.FromJson<GeometryPayload>(System.IO.File.ReadAllText(GeometryCachePath));
            if (p == null || p.n <= 0 || string.IsNullOrEmpty(p.pos)) return;
            BuildMeshes(p);
            _builtGeometry = p;
            Debug.Log($"[FlyBrainViz] fly view using cached geometry from {GeometryCachePath}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[FlyBrainViz] geometry cache unreadable: {e.Message}");
        }
    }

    /// <summary>Remove the overlay layer from a camera that should not see it.</summary>
    public static void StripOverlayLayer(Camera cam)
    {
        if (cam != null) cam.cullingMask &= ~OverlayLayerMask;
    }

    static void SetLayerRecursive(GameObject go, int layer)
    {
        go.layer = layer;
        for (int i = 0; i < go.transform.childCount; i++)
            SetLayerRecursive(go.transform.GetChild(i).gameObject, layer);
    }

    GameObject NewMeshObject(string name, Transform parent, Mesh mesh)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = OverlayMaterial();
        mr.shadowCastingMode = ShadowCastingMode.Off;
        mr.receiveShadows = false;
        return go;
    }

    // ---- brain close-up look ------------------------------------------------

    private Material _glowMaterial, _backdropMaterial;
    private GameObject _backdrop;
    private Mesh _backdropMesh;
    private bool _lookGlow;
    private bool _flyTextured;             // Y: textured fly instead of the see-through shell

    // What the neuron billboards were last built for in the fly views.
    struct DrawnPose
    {
        public bool valid, closeUp;
        public Quaternion rootRot, camRot;
        public float size;
        public int placement;
        public double stamp;
    }
    private DrawnPose _drawn;
    private Vector2 _hoverMouse = new Vector2(-1f, -1f);
    private readonly Vector3[] _backdropVerts = new Vector3[4];

    private float[] _testLevel;
    private byte[] _testAct;
    private float _nextTestTick;
    private System.Random _testRng;

    /// <summary>
    /// Stand-in firing for tuning the look in the editor with no fly job
    /// running: random neurons fire and fade, bursts light a random run of
    /// neighbours in the published order. Never runs in a build, and gives way
    /// as soon as real activity arrives.
    /// </summary>
    void DriveTestActivity()
    {
        var s = _anatSettings;
        if (!Application.isEditor || s == null || !s.editorTestActivity || _positions == null) return;
        if (Time.unscaledTime < _nextTestTick) return;
        float dt = 0.1f;
        _nextTestTick = Time.unscaledTime + dt;
        if (_testLevel == null || _testLevel.Length != _n)
        {
            _testLevel = new float[_n];
            _testAct = new byte[_n];
            _testRng = new System.Random(1);
        }
        float decay = Mathf.Exp(-dt / 0.35f);
        for (int i = 0; i < _n; i++) _testLevel[i] *= decay;

        int sparse = Mathf.RoundToInt(_n * Mathf.Clamp01(s.editorTestFiringFraction));
        for (int k = 0; k < sparse; k++) _testLevel[_testRng.Next(_n)] = 1f;
        if (_testRng.NextDouble() < 0.3)
        {
            int start = _testRng.Next(_n), len = Mathf.Min(_n, 150 + _testRng.Next(450));
            for (int k = 0; k < len; k++)
                if (_testRng.NextDouble() < 0.6) _testLevel[(start + k) % _n] = 1f;
        }
        for (int i = 0; i < _n; i++) _testAct[i] = (byte)(Mathf.Clamp01(_testLevel[i]) * 255f);
        ApplyActivity(_testAct);
    }

    /// <summary>Neurons drawn as glowing spots, in whichever view.</summary>
    bool GlowOn
    {
        get { return _anatSettings != null && _anatSettings.glow && GlowMaterial() != null; }
    }

    // ---- cached overlay (fly views) ----------------------------------------
    // The fly views draw a skinned see-through fly and thousands of large
    // glow spots. Drawn every frame that stretches the sim's frames, and at
    // Time.timeScale 3 each frame is sim time the car drives on stale
    // commands. So they are drawn into _overlayRt at overlayRenderHz, and on
    // every frame the overlay camera draws nothing itself and just copies the
    // last drawing into its column. The column is opaque (the camera clears
    // it to black and it never lies over the track), so a plain copy is exact.
    private RenderTexture _overlayRt;
    private Material _overlayBlitMat;
    private bool _overlayCached;
    private float _nextOverlayRender;
    private int _cachedHoverIndex = -1;

    void SyncOverlayCache()
    {
        if (_overlayCam == null) return;
        float hz = _anatSettings != null ? _anatSettings.overlayRenderHz : 0f;
        // Smooth while the fly flies and while the camera moves.
        if (hz > 0f && _fly != null && Anatomy && (_fly.Flying || _flightZoom > 0f))
            hz = _anatSettings.flightRenderHz > 0f ? Mathf.Max(hz, _anatSettings.flightRenderHz) : 0f;
        if (!_overlayCam.enabled || !Anatomy || hz <= 0f || OverlayBlitMaterial() == null)
        {
            if (_overlayCached)
            {
                _overlayCam.cullingMask = OverlayLayerMask;
                _overlayCam.clearFlags = CameraClearFlags.SolidColor;
                _overlayCached = false;
            }
            SetBehindTrack(false);
            return;
        }

        // Behind the track: drawn over the whole window, before the track
        // camera, which then draws the road over it and leaves the fly
        // showing on the black around it. The camera is still framed on its
        // column, so the fly rests where it always has.
        bool behind = _anatSettings.flyBehindTrack && OverheadCameraFit.TrackCamera != null;
        Rect px = behind ? new Rect(0f, 0f, Screen.width, Screen.height) : _overlayCam.pixelRect;
        int w = Mathf.Max(1, Mathf.RoundToInt(px.width));
        int h = Mathf.Max(1, Mathf.RoundToInt(px.height));
        bool resized = _overlayRt == null || _overlayRt.width != w || _overlayRt.height != h;
        if (resized)
        {
            if (_overlayRt != null) { _overlayRt.Release(); Destroy(_overlayRt); }
            _overlayRt = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32)
                { name = "FlyBrainOverlayCache", antiAliasing = 1 };
            _overlayRt.Create();
        }

        float now = Time.unscaledTime;
        // In flight, draw exactly the frames the fly was posed on: two
        // independent 30 Hz clocks beat against each other and the motion
        // stutters, showing some poses twice and skipping others.
        bool flying = _fly != null && _fly.Flying;
        bool timed = flying ? _fly.PosedThisFrame : now >= _nextOverlayRender;
        bool due = resized || !_overlayCached || _dragging || _snapT < 1f
                   || _hoverIndex != _cachedHoverIndex || timed || behind != DrawsBehindTrack;
        if (due)
        {
            _nextOverlayRender = now + 1f / hz;
            _cachedHoverIndex = _hoverIndex;
            Rect rect = _overlayCam.rect;
            Matrix4x4 columnProj = _overlayCam.projectionMatrix;
            _overlayCam.cullingMask = OverlayLayerMask;
            _overlayCam.clearFlags = CameraClearFlags.SolidColor;
            _overlayCam.targetTexture = _overlayRt;
            _overlayCam.rect = new Rect(0f, 0f, 1f, 1f);
            if (behind) _overlayCam.projectionMatrix = ColumnToWindow(rect) * columnProj;
            _overlayCam.Render();
            if (behind) _overlayCam.ResetProjectionMatrix();
            _redrawCount++;
            _overlayCam.targetTexture = null;
            _overlayCam.rect = rect;
            CaptureFlightFrame(now);
        }
        _overlayCam.cullingMask = 0;
        _overlayCam.clearFlags = CameraClearFlags.Nothing;
        _overlayCached = true;
        SetBehindTrack(behind);
    }

    /// <summary>Clip-space map from a camera's own viewport to the whole
    /// window, so it draws exactly what it would in its rect, plus whatever
    /// its view extends to beyond it.</summary>
    static Matrix4x4 ColumnToWindow(Rect r)
    {
        var m = Matrix4x4.identity;
        m.m00 = r.width;
        m.m03 = 2f * r.x + r.width - 1f;
        m.m11 = r.height;
        m.m13 = 2f * r.y + r.height - 1f;
        return m;
    }

    /// <summary>True while the fly view is drawn over the whole window
    /// behind the track; OverheadCameraFit then stops clearing over it.</summary>
    public static bool DrawsBehindTrack { get; private set; }
    private Camera _behindCam;

    void SetBehindTrack(bool on)
    {
        DrawsBehindTrack = on;
        if (!on)
        {
            if (_behindCam != null) _behindCam.enabled = false;
            return;
        }
        if (_behindCam == null)
        {
            _behindCam = HiddenCamera.Create("FlyBehindTrackCamera");
            _behindCam.clearFlags = CameraClearFlags.SolidColor;
            _behindCam.backgroundColor = Color.black;
            _behindCam.cullingMask = 0;          // clears, then takes the cached fly
            _behindCam.orthographic = true;
            _behindCam.allowHDR = false;
            _behindCam.allowMSAA = false;
            _behindCam.useOcclusionCulling = false;
        }
        _behindCam.rect = new Rect(0f, 0f, 1f, 1f);
        _behindCam.depth = OverheadCameraFit.TrackCamera != null
            ? OverheadCameraFit.TrackCamera.depth - 0.5f : -1.5f;
        _behindCam.enabled = true;
    }

    // ---- flight capture (editor review) ------------------------------------
    private bool _captureWasFlying;
    private int _captureCount;
    private float _nextCapture;
    private Texture2D _captureTex;

    /// <summary>Editor only: save the just-drawn column to
    /// Temp/FlyFlightFrames while a flight, or the camera's return after it,
    /// is under way.</summary>
    void CaptureFlightFrame(float now)
    {
        if (!Application.isEditor || _anatSettings == null || !_anatSettings.editorCaptureFlight
            || _fly == null || _overlayRt == null) return;
        bool active = _fly.Flying || _flightZoom > 0f;
        string dir = System.IO.Path.GetFullPath(
            System.IO.Path.Combine(Application.dataPath, "..", "Temp", "FlyFlightFrames"));
        if (!active)
        {
            if (_captureWasFlying)
                Debug.Log($"[FlyBrainViz] flight capture: {_captureCount} frames in {dir}");
            _captureWasFlying = false;
            return;
        }
        if (!_captureWasFlying)
        {
            if (System.IO.Directory.Exists(dir)) System.IO.Directory.Delete(dir, true);
            System.IO.Directory.CreateDirectory(dir);
            _captureCount = 0;
            _nextCapture = now;
            _captureWasFlying = true;
        }
        if (now < _nextCapture) return;
        _nextCapture = now + 1f / Mathf.Max(1f, _anatSettings.editorCaptureHz);
        int w = _overlayRt.width, h = _overlayRt.height;
        if (_captureTex == null || _captureTex.width != w || _captureTex.height != h)
        {
            if (_captureTex != null) Destroy(_captureTex);
            _captureTex = new Texture2D(w, h, TextureFormat.RGB24, false);
        }
        var prev = RenderTexture.active;
        RenderTexture.active = _overlayRt;
        _captureTex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        RenderTexture.active = prev;
        System.IO.File.WriteAllBytes(System.IO.Path.Combine(dir, $"flight_{_captureCount:000}.png"),
                                     _captureTex.EncodeToPNG());
        _captureCount++;
    }

    Material OverlayBlitMaterial()
    {
        if (_overlayBlitMat == null)
        {
            var sh = Shader.Find("Hidden/FlyOverlayBlit");
            if (sh == null) return null;
            _overlayBlitMat = new Material(sh) { hideFlags = HideFlags.HideAndDontSave };
        }
        return _overlayBlitMat;
    }

    /// <summary>Camera.onPostRender: on the overlay camera's own (empty) pass
    /// to the screen, fill its column with the cached drawing.</summary>
    void BlitOverlayCache(Camera c)
    {
        Camera into = DrawsBehindTrack ? _behindCam : _overlayCam;
        if (c != into || !_overlayCached || c.targetTexture != null
            || _overlayRt == null || _overlayBlitMat == null) return;
        _overlayBlitMat.mainTexture = _overlayRt;
        GL.PushMatrix();
        GL.LoadOrtho();
        _overlayBlitMat.SetPass(0);
        GL.Begin(GL.QUADS);
        GL.TexCoord2(0f, 0f); GL.Vertex3(0f, 0f, 0f);
        GL.TexCoord2(0f, 1f); GL.Vertex3(0f, 1f, 0f);
        GL.TexCoord2(1f, 1f); GL.Vertex3(1f, 1f, 0f);
        GL.TexCoord2(1f, 0f); GL.Vertex3(1f, 0f, 0f);
        GL.End();
        GL.PopMatrix();
    }

    /// <summary>
    /// Keep the neurons' material and the fly's shell in line with the view
    /// and the glow setting. Recolours at once when the glow switches rather
    /// than waiting for the next activity message, which never comes while
    /// the sim is idle.
    /// </summary>
    void SyncCloseUpLook()
    {
        if (_fly != null)
        {
            _fly.OutlineOnly = CloseUp;
            _fly.Textured = _flyTextured;
        }
        bool glow = GlowOn;
        if (_pointObject != null)
        {
            Material m = glow ? GlowMaterial() : OverlayMaterial();
            var mr = _pointObject.GetComponent<MeshRenderer>();
            if (mr.sharedMaterial != m) mr.sharedMaterial = m;
        }
        if (_glowMaterial != null && _anatSettings != null)
            _glowMaterial.SetFloat("_Gain", _anatSettings.glowGain
                * (Anatomy && !CloseUp ? Mathf.Max(0f, _anatSettings.flyViewBrainBrightness) : 1f));
        if (glow != _lookGlow)
        {
            _lookGlow = glow;
            if (_act != null && _positions != null && _act.Length == _n)
                ApplyActivity((byte[])_act.Clone());
        }
    }

    Material GlowMaterial()
    {
        if (_glowMaterial != null) return _glowMaterial;
        var shader = Shader.Find("Hidden/FlyNeuronGlow");
        if (shader == null) return null;
        _glowMaterial = new Material(shader);
        return _glowMaterial;
    }

    /// <summary>
    /// Fit the dark oval to the brain as the camera sees it, set just behind
    /// the brain's middle. Brain here is every neuron that rides the head.
    /// Positions and axes are in the neuron container's space, which in the
    /// fly view is the fly's Root.
    /// </summary>
    void UpdateBackdrop(Transform container, Camera cam, Vector3[] pos)
    {
        var s = _anatSettings;
        // Drawn over everything, so it would black out a solid head.
        bool on = CloseUp && s.closeUpBackdrop && !_flyTextured && _anatW != null && _anatW.Length == _n
                  && pos != null && pos.Length == _n;
        if (!on)
        {
            if (_backdrop != null && _backdrop.activeSelf) _backdrop.SetActive(false);
            return;
        }
        Vector3 r = container.InverseTransformDirection(cam.transform.right);
        Vector3 u = container.InverseTransformDirection(cam.transform.up);
        Vector3 f = container.InverseTransformDirection(cam.transform.forward);
        float minA = float.MaxValue, maxA = float.MinValue;
        float minB = float.MaxValue, maxB = float.MinValue;
        float sumC = 0f;
        int count = 0;
        for (int i = 0; i < _n; i++)
        {
            if (_anatW[i] >= 0.5f) continue;
            Vector3 p = pos[i];
            float a = Vector3.Dot(p, r), b = Vector3.Dot(p, u);
            if (a < minA) minA = a;
            if (a > maxA) maxA = a;
            if (b < minB) minB = b;
            if (b > maxB) maxB = b;
            sumC += Vector3.Dot(p, f);
            count++;
        }
        if (count == 0)
        {
            if (_backdrop != null) _backdrop.SetActive(false);
            return;
        }
        if (!EnsureBackdrop(container)) return;

        float pad = Mathf.Max(0.1f, s.closeUpBackdropPadding);
        float hw = (maxA - minA) * 0.5f * pad, hh = (maxB - minB) * 0.5f * pad;
        Vector3 c = r * ((minA + maxA) * 0.5f) + u * ((minB + maxB) * 0.5f) + f * (sumC / count);
        _backdropVerts[0] = c - r * hw - u * hh;
        _backdropVerts[1] = c + r * hw - u * hh;
        _backdropVerts[2] = c - r * hw + u * hh;
        _backdropVerts[3] = c + r * hw + u * hh;
        _backdropMesh.vertices = _backdropVerts;

        _backdropMaterial.SetColor("_Color", new Color(0f, 0f, 0f, Mathf.Clamp01(s.closeUpBackdropOpacity)));
        _backdropMaterial.SetFloat("_Inner", 1f - Mathf.Clamp(s.closeUpBackdropSoftness, 0.05f, 1f));
        if (!_backdrop.activeSelf) _backdrop.SetActive(true);
    }

    bool EnsureBackdrop(Transform container)
    {
        if (_backdrop == null)
        {
            var shader = Shader.Find("Hidden/FlyBrainBackdrop");
            if (shader == null) return false;
            // After the shell (2990), before the neurons (3000).
            _backdropMaterial = new Material(shader) { renderQueue = 2995 };
            _backdropMesh = new Mesh { name = "FlyBrainBackdrop" };
            _backdropMesh.vertices = _backdropVerts;
            _backdropMesh.uv = new[] { new Vector2(0f, 0f), new Vector2(1f, 0f),
                                       new Vector2(0f, 1f), new Vector2(1f, 1f) };
            _backdropMesh.triangles = new[] { 0, 1, 2, 2, 1, 3 };
            _backdropMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 1e4f);
            _backdrop = NewMeshObject("FlyBrainBackdrop", container, _backdropMesh);
            _backdrop.GetComponent<MeshRenderer>().sharedMaterial = _backdropMaterial;
            _backdrop.layer = OverlayLayer;
        }
        if (_backdrop.transform.parent != container)
        {
            _backdrop.transform.SetParent(container, false);
            _backdrop.transform.localPosition = Vector3.zero;
            _backdrop.transform.localRotation = Quaternion.identity;
            _backdrop.transform.localScale = Vector3.one;
        }
        return true;
    }

    Material OverlayMaterial()
    {
        if (_material != null) return _material;
        // Same fallback chain as TrajectoryRolloutViz: Sprites/Default is an
        // always-available unlit, vertex-colour-aware, Cull Off shader.
        var shader = Shader.Find("Sprites/Default");
        if (shader == null) shader = Shader.Find("Unlit/Color");
        if (shader == null)
        {
            Debug.LogError("[FlyBrainViz] no usable shader found (Sprites/Default "
                           + "+ Unlit/Color both missing - likely stripped from the "
                           + "build). Add one to 'Always Included Shaders'.");
            shader = Shader.Find("Legacy Shaders/Diffuse");
        }
        _material = new Material(shader);
        return _material;
    }

    bool IsVisible()
    {
        return _pointObject != null && _pointObject.activeSelf;
    }

    // ---- controls legend ---------------------------------------------------
    //
    // IMGUI like HudOverlay, and for the same reason: a build has no inspector
    // and this script is auto-attached with no scene setup, so a procedural
    // OnGUI panel is the only way anyone finds out these keys exist. Drawn
    // only while the overlay itself is up, so non-fly jobs never see it.

    private Texture2D _guiPanelTex;
    private GUIStyle _guiTitleStyle, _guiKeyStyle, _guiTextStyle, _guiHeadStyle;
    private bool _guiReady;
    private int _guiFontSize = -1;     // what the cached styles were built at

    void OnGUI()
    {
        if (!_vizEnabled || !IsVisible()) return;
        EnsureGuiAssets();

        // Everything below lays out in logical pixels; OverlayUi maps them to
        // the real window. Sizes here are therefore "at 1080p" and hold their
        // apparent size on any display.
        Matrix4x4 guiPrev = OverlayUi.Begin();

        // Independently toggled: the controls are for whoever is driving the
        // overlay, the colour key is for anyone watching over their shoulder.
        // Someone presenting this wants the second without the first.
        float ctrlW = 0f, ctrlH = 0f;
        // One or the other, never nothing: something always says how to get
        // the panels back.
        if (showControls) DrawControlsLegend(out ctrlW, out ctrlH);
        else DrawControlsHint(out ctrlW, out ctrlH);
        // Given its actual drawn size rather than a recomputed guess, so the
        // colour key can step out of its way without the two having to agree
        // on a layout formula.
        if (showColorLegend) DrawColorLegend(ctrlW, ctrlH);
        // Last, so it draws over both panels rather than under them.
        DrawHoverLabel();

        OverlayUi.End(guiPrev);
    }

    /// <summary>Widest of `items` as this style would draw it.</summary>
    static float MaxWidth(GUIStyle style, string[] items)
    {
        float w = 0f;
        for (int i = 0; i < items.Length; i++)
            w = Mathf.Max(w, style.CalcSize(new GUIContent(items[i])).x);
        return w;
    }

    /// <summary>
    /// Line box for one row. Measured off a string with both an ascender and a
    /// descender, because a row shorter than the glyphs crops them.
    /// </summary>
    float RowHeight()
    {
        var probe = new GUIContent("Ayg");
        return Mathf.Ceil(Mathf.Max(_guiKeyStyle.CalcSize(probe).y,
                                    _guiTextStyle.CalcSize(probe).y) * 1.18f);
    }

    void DrawControlsLegend(out float panelW, out float panelH)
    {
        // Parallel arrays rather than tuples, to stay compatible with the
        // oldest C# version this project might be compiled under.
        string[] keys = {
            "hover", "drag LMB", "-   =", "0", "[   ]", "\\", "arrows", "/",
            "N", ",   .", "Y", "1", "B", ";", "'",
        };
        bool anat = Anatomy;
        // The absolute value is what gets pasted into FLY_VIZ_DEPTH_SCALE, so
        // show it next to the multiplier rather than making the user read the
        // log to find it. Absent on a trainer that predates the depthScale
        // field, in which case the multiplier is all there is.
        string depthAbs = DepthAbsoluteText();
        // Flags when the requested size is not the one on screen, so a key
        // that has run out of room does not read as broken. Kept terse: this
        // string sets the panel's width, and a long one pushed the panel out
        // over the track.
        string sizeText = "size  \u00d7" + overlaySizeScale.ToString("0.00");
        if (_lastAchievedScale < overlaySizeScale - 0.02f)
            sizeText += "  (fits \u00d7" + _lastAchievedScale.ToString("0.00") + ")";

        string[] text = {
            "name the neuron under it",
            "turn it (springs back)",
            sizeText,
            "reset size",
            anat ? "depth  (true in the fly view)"
                 : "depth  \u00d7" + depthMultiplier.ToString("0.##")
                   + (depthAbs != null ? "   = " + depthAbs : ""),
            "reset depth",
            anat ? "turn fly  " + FlyYaw.ToString("0.#") + "\u00b0 yaw / "
                   + FlyElevation.ToString("0.#") + "\u00b0 elev"
                 : "square up  " + baseYawDegrees.ToString("0.#") + "\u00b0 yaw / "
                   + basePitchDegrees.ToString("0.#") + "\u00b0 pitch",
            "reset orientation",
            anat ? "brain-only view" : "fly anatomy view",
            anat ? "fly opacity  " + _anatSettings.bodyOpacity.ToString("0.00")
                 : "fly opacity (fly view)",
            anat ? (_flyTextured ? "see-through fly" : "textured fly")
                 : "textured fly (fly view)",
            anat ? (_fly != null && _fly.Flying ? "flying\u2026" : "fly a loop and land")
                 : "fly a loop (fly view)",
            "hide overlay",
            "hide this list",
            // Reads the state rather than asserting it: with the key hidden,
            // "hide colour key" is telling the user to do what they already
            // did, and the panel stops being a reliable answer to "what does
            // this key do right now".
            showColorLegend ? "hide colour key" : "show colour key",
        };

        const string foot = "else:  H hud   T fan   C stages   P cam   F csi";

        // Columns are measured, not assumed. Guessing an em width per
        // character got this wrong in both directions: too narrow for the bold
        // key column, which wrapped and then cropped the wrapped line, and too
        // generous elsewhere. CalcSize asks the font, so it also survives a
        // change of font size or of the strings themselves.
        float fs = Mathf.Max(8, controlsFontSize);
        float pad = fs * 0.83f, gap = fs * 0.6f;
        float rowH = RowHeight();
        float keyW = MaxWidth(_guiKeyStyle, keys) + gap;
        float titleH = Mathf.Ceil(_guiTitleStyle.CalcSize(new GUIContent("FLY BRAIN")).y * 1.35f);
        float footH = rowH;
        panelW = Mathf.Ceil(Mathf.Max(keyW + MaxWidth(_guiTextStyle, text),
                                      _guiTextStyle.CalcSize(new GUIContent(foot)).x)
                            + pad * 2f);
        panelH = Mathf.Ceil(pad * 2f + titleH + keys.Length * rowH + footH);
        float x = 14f, y = OverlayUi.LogicalHeight - panelH - 14f;

        Color prev = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.55f);
        GUI.DrawTexture(new Rect(x, y, panelW, panelH), _guiPanelTex);
        GUI.color = prev;

        GUI.Label(new Rect(x + pad, y + pad - fs * 0.17f, panelW - pad * 2f, titleH),
                  "FLY BRAIN", _guiTitleStyle);

        float row = y + pad + titleH;
        for (int i = 0; i < keys.Length; i++)
        {
            GUI.Label(new Rect(x + pad, row, keyW, rowH), keys[i], _guiKeyStyle);
            GUI.Label(new Rect(x + pad + keyW, row, panelW - pad * 2f - keyW, rowH),
                      text[i], _guiTextStyle);
            row += rowH;
        }

        // The other overlays' toggles, listed here purely so one panel answers
        // "what are the keys". They belong to HudOverlay, TrajectoryRolloutViz,
        // CurriculumStageButtons, CameraViewSwitcher and CsiFramePublisher - if
        // one of those changes its key, this line is what goes stale.
        GUI.Label(new Rect(x + pad, row, panelW - pad * 2f, footH), foot, _guiTextStyle);
    }

    /// <summary>
    /// The one line that survives hiding the panels: which key brings them
    /// back. Without it the only route back from ; or ' is knowing the answer
    /// already, and both panels sit over the brain now, so hiding them is a
    /// normal thing to want rather than an edge case.
    /// </summary>
    void DrawControlsHint(out float panelW, out float panelH)
    {
        string hint = (showColorLegend ? "; fly-brain keys"
                                       : "; fly-brain keys    ' colours")
                      + (CloseUp ? "    N brain only" : Anatomy ? "    N brain close-up" : "    N fly view");
        float fs = Mathf.Max(8, controlsFontSize);
        float pad = fs * 0.45f;
        panelW = Mathf.Ceil(_guiTextStyle.CalcSize(new GUIContent(hint)).x + pad * 2f);
        panelH = Mathf.Ceil(RowHeight() + pad * 2f);
        float x = 14f, y = OverlayUi.LogicalHeight - panelH - 14f;

        Color prev = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.45f);
        GUI.DrawTexture(new Rect(x, y, panelW, panelH), _guiPanelTex);
        GUI.color = prev;
        GUI.Label(new Rect(x + pad, y + pad, panelW - pad * 2f, RowHeight()),
                  hint, _guiTextStyle);
    }

    /// <summary>
    /// What the colours mean, so someone watching can read the overlay without
    /// being told. Each row is the swatch actually used to draw that role -
    /// taken from the same fields RoleRamp switches on, not a copy - so
    /// retinting a role in the inspector retints its key entry too.
    ///
    /// Roles come from display_subset.build (viz.py ROLE_CODES); the cue text
    /// is the encoder's POPULATION_CELLS mapping. Both are on the Python side,
    /// so if either changes this text is what goes stale.
    /// </summary>
    void DrawColorLegend(float ctrlW, float ctrlH)
    {
        const string title = "WHAT THE COLOURS MEAN";
        const string foot = "brightness = firing rate  \u00b7  L/R = side";

        // Role 4 is drawn as a rest->active crossfade rather than one colour,
        // so its row needs two swatches and is handled separately below. The
        // context entry is last in names/cues but has no entry in `swatch`.
        Color[] swatch = {
            sensoryColor, chaseColor, interneuronColor, commandColor, descendingColor,
        };
        // "loom" and "chase" rather than one "sensory" row: both are role 1,
        // but the two cue families are the comparison someone watching is
        // actually making, so the key names them the way the cues do.
        string[] names = { "loom", "chase", "relay", "command", "descending", "context" };
        // Cell types by name, since that is what the overlay is actually
        // drawing and what the hover label will echo back. Which of them
        // carries which cue is the block below, so it is not repeated here.
        string[] cues = {
            "LC4, LPLC2, LPLC1",
            "LC10a",
            "sensory \u2192 descending path",
            "DNp01, DNa02, DNg100, MDN\u2026",
            "1314 descending, the policy's input",
            "silhouette \u00b7 amber = firing",
        };

        // Measured, for the reason given in DrawControlsLegend - this is the
        // panel where guessing showed: "descending" is the widest name and
        // overran the name column, and since GUI.skin.label word-wraps by
        // default it took a second line the row had no height for and lost its
        // descender. Styles now have wordWrap off and the column is measured.
        float fs = Mathf.Max(8, controlsFontSize);
        float pad = fs * 0.83f, gap = fs * 0.6f;
        float rowH = RowHeight();
        float swatchW = fs * 1.4f;
        float nameW = MaxWidth(_guiKeyStyle, names) + gap;
        float titleH = Mathf.Ceil(_guiTitleStyle.CalcSize(new GUIContent(title)).y * 1.35f);
        float footH = rowH;
        float textX = pad + swatchW + gap;

        // The cue block: each injected cue, the neuron types and side it
        // reaches, and how many of them are drawn and firing.
        string[] cueRows = BuildCueRows();
        float cueNameW = MaxWidth(_guiKeyStyle, CueNames) + gap;

        float panelW = Mathf.Ceil(Mathf.Max(Mathf.Max(
            Mathf.Max(textX + nameW + MaxWidth(_guiTextStyle, cues),
                      pad + _guiTitleStyle.CalcSize(new GUIContent(title)).x),
            pad + _guiTextStyle.CalcSize(new GUIContent(foot)).x),
            textX + cueNameW + MaxWidth(_guiTextStyle, cueRows)) + pad);
        float panelH = Mathf.Ceil(pad * 2f + titleH + names.Length * rowH + footH
                                  + titleH + cueRows.Length * rowH);
        float x = OverlayUi.LogicalWidth - panelW - 14f;
        float y = OverlayUi.LogicalHeight - panelH - 14f;

        // Both bottom corners are ours, and at 1280x720 the two panels are
        // wider than the screen between them. Rather than shrink either, stack
        // this one above the controls: still bottom-right, just lifted clear.
        if (ctrlW > 0f && x < 14f + ctrlW + 8f)
            y -= ctrlH + 8f;

        Color prev = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.55f);
        GUI.DrawTexture(new Rect(x, y, panelW, panelH), _guiPanelTex);
        GUI.color = prev;

        GUI.Label(new Rect(x + pad, y + pad, panelW - pad * 2f, titleH),
                  title, _guiTitleStyle);

        float row = y + pad + titleH;
        float sh = Mathf.Min(fs * 0.85f, rowH - 4f);
        float cueX = textX + nameW;
        float cueW = panelW - pad - cueX;
        for (int i = 0; i < names.Length; i++)
        {
            float sy = row + (rowH - sh) * 0.5f;
            if (i < swatch.Length)
            {
                DrawSwatch(x + pad, sy, swatchW, sh, swatch[i]);
            }
            else
            {
                // Context: two swatches, because the thing worth conveying is
                // the transition, not either endpoint on its own.
                float half = (swatchW - fs * 0.22f) * 0.5f;
                DrawSwatch(x + pad, sy, half, sh, contextRestColor);
                DrawSwatch(x + pad + half + fs * 0.22f, sy, half, sh, contextActiveColor);
            }
            GUI.Label(new Rect(x + textX, row, nameW, rowH), names[i], _guiKeyStyle);
            GUI.Label(new Rect(x + cueX, row, cueW, rowH), cues[i], _guiTextStyle);
            row += rowH;
        }

        // Brightness is the overlay's other channel and is easy to miss, and
        // the left/right split carries the steering signal while being spatial
        // rather than coloured - so neither is self-evident from swatches.
        GUI.Label(new Rect(x + pad, row, panelW - pad * 2f, footH), foot, _guiTextStyle);
        row += footH;

        GUI.Label(new Rect(x + pad, row, panelW - pad * 2f, titleH),
                  "CUES INJECTED", _guiTitleStyle);
        row += titleH;
        for (int c = 0; c < cueRows.Length; c++)
        {
            // The colour that cue's cells are actually drawn in, so a row here
            // and the dots it counts are matched by eye.
            DrawSwatch(x + pad, row + (rowH - sh) * 0.5f, swatchW, sh, CueColor(c));
            GUI.Label(new Rect(x + textX, row, cueNameW, rowH), CueNames[c], _guiKeyStyle);
            GUI.Label(new Rect(x + textX + cueNameW, row,
                               panelW - pad - (textX + cueNameW), rowH),
                      cueRows[c], _guiTextStyle);
            row += rowH;
        }
    }

    /// <summary>
    /// "LC4+LPLC2, left  ·  165 cells  ·  23 firing" per cue. Counts come from
    /// the connectome via the published type table, so they are resolved at
    /// runtime rather than transcribed - the LC4+LPLC2 left figure should
    /// agree with step 1's LC4 (L71/R55) + LPLC2 (L94/R91).
    /// </summary>
    string[] BuildCueRows()
    {
        if (_cueRows == null) _cueRows = new string[CueNames.Length];
        for (int c = 0; c < CueNames.Length; c++)
        {
            string types = _cueTypeText != null ? _cueTypeText[c] : CueTypeText(c);
            if (_cueShown == null || _cueShown[c] == 0)
            {
                // No type table from this trainer, so the cells cannot be
                // identified. Say what the cue reaches and leave the counts
                // out rather than showing a confident zero.
                _cueRows[c] = types;
                continue;
            }
            int firing = _cueFiring != null ? _cueFiring[c] : 0;
            _cueRows[c] = types + "  \u00b7  " + _cueShown[c] + " cells  \u00b7  "
                          + firing + " firing";
        }
        return _cueRows;
    }

    // ---- hover label -------------------------------------------------------

    /// <summary>
    /// What a cell type does in the fly, for the hover label. Only types the
    /// repo has actually measured or named get an entry; everything else -
    /// most relay interneurons, most of the 1,314 descending neurons, and the
    /// context sample - falls through to its role description. Inventing
    /// plausible-sounding functions for the rest would make the label
    /// untrustworthy exactly where it is least checkable.
    ///
    /// Sources, all from docs/flybrain-driver-plan.md: LC4/LPLC2 -> DNp01 and
    /// LC10a -> DNa02 are step 2's measured pathways, DNg100's unresponsiveness
    /// is step 2's strength sweep, and LPLC1 is drawn only because it is in
    /// display_subset.SENSORY_TYPES.
    /// </summary>
    static string TypeFunction(string type)
    {
        switch (type)
        {
            case "LC4":
            case "LPLC2":  return "drives DNp01 +25 spikes/s here";
            case "LPLC1":  return "drawn, but no cue is injected into it";
            case "LC10a":  return "drives DNa02 +3.9 here";
            case "DNp01":  return "+25.2 under loom, +0.1 under chase";
            case "DNa02":  return "+3.9 under chase, -0.1 under loom";
            case "DNg100": return "flat under every cue and strength tested";
            default:       return null;
        }
    }

    /// <summary>
    /// What the cell type IS, anatomically. The abbreviations are opaque
    /// unless you already know them, which defeats the point of a label whose
    /// job is to explain the picture to someone who does not.
    ///
    /// LC = lobula columnar, LPLC = lobula plate / lobula columnar (dendrites
    /// in both neuropils), DN = descending neuron with the Namiki et al. 2018
    /// group letter and number.
    /// </summary>
    static string TypeLongName(string type)
    {
        switch (type)
        {
            case "LC4":    return "lobula columnar neuron, type 4";
            case "LPLC2":  return "lobula plate / lobula columnar, type 2";
            case "LPLC1":  return "lobula plate / lobula columnar, type 1";
            case "LC10a":  return "lobula columnar neuron, type 10a";
            case "DNp01":  return "descending neuron, posterior group 1";
            case "DNa02":  return "descending neuron, anterior group 2";
            case "DNg100": return "descending neuron, gnathal group 100";
            default:       return null;
        }
    }

    /// <summary>
    /// What the type REPRESENTS - the published visual tuning, led by the word
    /// that matters here: looming or chasing.
    ///
    /// Each line is a finding from the Drosophila literature, not this repo's
    /// measurements (TypeFunction carries those) and not a paraphrase of the
    /// encoder's intent:
    ///
    ///   LC4    encodes the ANGULAR VELOCITY of a looming edge; with LPLC2 it
    ///          is one of the two functionally distinct inputs to the giant
    ///          fibre (von Reyn et al. 2017; Ache et al. 2019).
    ///   LPLC2  is ultra-selective for OUTWARD radial motion by opponency,
    ///          which is the optic-flow signature of an object on a direct
    ///          collision course, and encodes angular size (Klapoetke et al.
    ///          2017).
    ///   LC10a  tracks small moving objects and is required for visually
    ///          guided courtship pursuit, steering via DNa02 (Ribeiro et al.
    ///          2018; Sten et al. 2021).
    ///   DNp01  is the giant fibre, one cell per side, driving short-mode
    ///          escape take-off.
    ///   DNa02  drives ipsilateral turning during walking.
    ///
    /// LPLC1 and DNg100 get no tuning claim. LPLC1 is looming-family but the
    /// encoder does not inject into it, and DNg100's function is not something
    /// this repo can source - the plan called it "forward walking", which the
    /// literature does not establish, so the label now says only what step 2
    /// actually measured about it.
    /// </summary>
    static string TypeScience(string type)
    {
        switch (type)
        {
            case "LC4":    return "looming \u00b7 speed of an expanding edge";
            case "LPLC2":  return "looming \u00b7 outward motion = collision course";
            case "LPLC1":  return "looming family \u00b7 no cue injected here";
            case "LC10a":  return "chasing \u00b7 tracks a small moving object";
            case "DNp01":  return "looming target \u00b7 giant fibre, escape take-off";
            case "DNa02":  return "chasing target \u00b7 turns toward the target";
            case "DNg100": return null;
            default:       return null;
        }
    }

    static string RoleName(byte role)
    {
        switch (role)
        {
            case 1:  return "sensory";
            case 2:  return "command";
            case 3:  return "descending";
            case 4:  return "context";
            default: return "relay";
        }
    }

    static string RoleFunction(byte role)
    {
        switch (role)
        {
            case 1:  return "feature detector the encoder injects into";
            case 2:  return "named command neuron";
            case 3:  return "one of the 1314 the policy reads";
            case 4:  return "silhouette only, not part of the circuit";
            default: return "relay on the sensory \u2192 descending path";
        }
    }

    // The four cues the encoder injects, as rl_agent/fly_brain/encoder.py
    // defines them in POPULATION_CELLS. Named here so the colour key can show
    // the same "LC4+LPLC2, left" breakdown the cue table in
    // docs/fly-brain-driver-how-it-works.md carries, with counts resolved live
    // from the connectome rather than transcribed.
    static readonly string[] CueNames = { "loom_L", "loom_R", "chase_L", "chase_R" };
    static readonly string[][] CueTypes = {
        new[] { "LC4", "LPLC2" }, new[] { "LC4", "LPLC2" },
        new[] { "LC10a" },        new[] { "LC10a" },
    };
    static readonly byte[] CueSides = { 1, 2, 1, 2 };       // SIDE_CODES
    static readonly string[] CueSideWords = { "left", "right", "left", "right" };
    private int[] _cueShown, _cueFiring;
    private string[] _cueTypeText, _cueRows;
    // Which neurons draw in chaseColor. Resolved once with the geometry
    // because ApplyActivity recolours all ~10k neurons on every activity
    // message and CueOf is a string search over the type table.
    private bool[] _isChase;

    /// <summary>
    /// Is this cue one of the chase pair? Read off the cue name rather than
    /// the index so reordering CueNames cannot silently recolour the overlay.
    /// </summary>
    static bool IsChaseCue(int cue)
    {
        return cue >= 0 && cue < CueNames.Length && CueNames[cue].StartsWith("chase");
    }

    /// <summary>The colour cue `c`'s population is drawn in.</summary>
    Color CueColor(int cue)
    {
        return IsChaseCue(cue) ? chaseColor : sensoryColor;
    }

    /// <summary>
    /// "LC4 + LPLC2, left" - the cue table's neuron-types column, spaced to
    /// read the same way it does in docs/fly-brain-driver-how-it-works.md.
    /// Columns are measured with CalcSize, so the spacing costs nothing.
    /// </summary>
    static string CueTypeText(int cue)
    {
        return string.Join(" + ", CueTypes[cue]) + ", " + CueSideWords[cue];
    }

    void BuildCueCounts()
    {
        _cueShown = new int[CueNames.Length];
        _cueFiring = new int[CueNames.Length];
        // All false without a type table, which is the pre-split look: every
        // sensory cell cyan. Degrading to one colour beats guessing at two.
        _isChase = new bool[_n];
        if (_cueTypeText == null)
        {
            _cueTypeText = new string[CueNames.Length];
            for (int c = 0; c < CueNames.Length; c++) _cueTypeText[c] = CueTypeText(c);
        }
        if (_typeNames == null || _typeIdx == null || _side == null) return;
        for (int i = 0; i < _n; i++)
        {
            int c = CueOf(i);
            if (c < 0) continue;
            _cueShown[c]++;
            _isChase[i] = IsChaseCue(c);
        }
    }

    /// <summary>Which cue drives neuron i, or -1. Matches on type *and* side.</summary>
    int CueOf(int i)
    {
        if (_typeNames == null || _typeIdx == null || _side == null) return -1;
        int slot = _typeIdx[i];
        if (slot < 0 || slot >= _typeNames.Length) return -1;
        string t = _typeNames[slot];
        byte sd = _side[i];
        for (int c = 0; c < CueTypes.Length; c++)
        {
            if (CueSides[c] != sd) continue;
            for (int k = 0; k < CueTypes[c].Length; k++)
                if (CueTypes[c][k] == t) return c;
        }
        return -1;
    }

    void BuildTypeCounts()
    {
        if (_typeNames == null || _typeIdx == null) { _typeShown = null; _typeFiring = null; return; }
        _typeShown = new int[_typeNames.Length];
        _typeFiring = new int[_typeNames.Length];
        for (int i = 0; i < _n; i++)
        {
            int t = _typeIdx[i];
            if (t >= 0 && t < _typeShown.Length) _typeShown[t]++;
        }
    }

    void RefreshFiringCounts()
    {
        if (_act == null || _typeIdx == null) return;
        if (_typeFiring != null) Array.Clear(_typeFiring, 0, _typeFiring.Length);
        if (_cueFiring != null) Array.Clear(_cueFiring, 0, _cueFiring.Length);
        byte cut = (byte)Mathf.Clamp(Mathf.RoundToInt(hoverFiringThreshold * 255f), 1, 255);
        for (int i = 0; i < _n; i++)
        {
            if (_act[i] < cut) continue;
            int t = _typeIdx[i];
            if (_typeFiring != null && t >= 0 && t < _typeFiring.Length) _typeFiring[t]++;
            if (_cueFiring != null)
            {
                int c = CueOf(i);
                if (c >= 0) _cueFiring[c]++;
            }
        }
    }

    /// <summary>
    /// Nearest drawn neuron to the cursor, or -1. Projects every neuron rather
    /// than raycasting, because the overlay is billboarded quads with no
    /// colliders - the same reason IsOverOverlay works in screen space.
    ///
    /// Circuit neurons win ties against context within the same radius even if
    /// slightly further away: context is 16k points of background that would
    /// otherwise swallow every hover, and it is the circuit the label is
    /// worth reading about.
    /// </summary>
    void UpdateHover(Camera cam)
    {
        _hoverIndex = -1;
        if (!showHoverLabel || cam == null || _positions == null || !IsVisible()) return;
        if (_dragging) return;              // turning it, not inspecting it
        var mouse = Mouse.current;
        if (mouse == null) return;

        Vector2 sp = mouse.position.ReadValue();
        _hoverPos = sp;
        if (!IsOverOverlay(cam, sp)) return;   // cheap reject before the full pass

        // Two matrices and a manual transform per neuron: 19k
        // Camera.WorldToScreenPoint calls a frame is the expensive way to do
        // exactly this. Split at view space rather than going straight to clip
        // so the behind-the-camera test works under an orthographic camera
        // too, where w is a constant 1 and a clip-space test would happily
        // report points behind the lens as visible.
        Matrix4x4 toView = cam.worldToCameraMatrix
                           * _pointObject.transform.parent.localToWorldMatrix;
        Matrix4x4 proj = cam.projectionMatrix;
        // Viewport, not Screen: Camera.main may not fill the window, and the
        // mouse position this is compared against is in screen pixels.
        Rect vp = cam.pixelRect;
        float r2 = hoverRadiusPixels * hoverRadiusPixels;
        float bestCircuit = r2, bestAny = r2;
        int hitCircuit = -1, hitAny = -1;
        Vector3[] pos = DrawPositions;

        for (int i = 0; i < _n; i++)
        {
            Vector3 v = toView.MultiplyPoint3x4(pos[i]);
            if (v.z >= 0f) continue;                       // view space looks down -Z
            Vector4 clip = proj * new Vector4(v.x, v.y, v.z, 1f);
            if (clip.w <= 0f) continue;
            float inv = 1f / clip.w;
            float dx = (vp.x + (clip.x * inv * 0.5f + 0.5f) * vp.width) - sp.x;
            float dy = (vp.y + (clip.y * inv * 0.5f + 0.5f) * vp.height) - sp.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < bestAny) { bestAny = d2; hitAny = i; }
            if (_role[i] != 4 && d2 < bestCircuit) { bestCircuit = d2; hitCircuit = i; }
        }
        _hoverIndex = hitCircuit >= 0 ? hitCircuit : hitAny;
    }

    void DrawHoverLabel()
    {
        if (_hoverIndex < 0 || _hoverIndex >= _n) return;

        byte role = _role[_hoverIndex];
        byte side = _side != null && _hoverIndex < _side.Length ? _side[_hoverIndex] : (byte)0;
        string type = null;
        int slot = -1;
        if (_typeNames != null && _typeIdx != null)
        {
            slot = _typeIdx[_hoverIndex];
            if (slot >= 0 && slot < _typeNames.Length) type = _typeNames[slot];
        }

        string sideText = side == 1 ? " \u00b7 left" : side == 2 ? " \u00b7 right" : "";
        // Falls back to the role when the trainer sends no type table, so the
        // label still says something useful against an older trainer.
        string head = (string.IsNullOrEmpty(type) ? RoleName(role) : type) + sideText;
        string fn = (type != null ? TypeFunction(type) : null) ?? RoleFunction(role);

        // Fixed buffer rather than a List: OnGUI runs more than once per frame
        // (layout, then repaint), so anything allocated here is allocated at
        // several times the frame rate.
        int nLines = 0;

        // Read as: what it is, what it detects, what it does here, which cue
        // population it belongs to. Every line is optional, so an unnamed
        // relay or a context cell still gets a short, honest label instead of
        // blank rows.
        string longName = type != null ? TypeLongName(type) : null;
        if (longName != null) _hoverLines[nLines++] = longName;

        string science = type != null ? TypeScience(type) : null;
        if (science != null) _hoverLines[nLines++] = science;

        _hoverLines[nLines++] = RoleName(role) + "  \u00b7  " + fn;

        // Say why the label is short, rather than letting a stale trainer look
        // like a bug in the label. Everything below this point - the type name,
        // the tuning line, the cue population, the counts - needs the type
        // table, so without it the hover silently degrades to two lines of role
        // text and there is nothing on screen to say why.
        //
        // The table is published once with the geometry, so this can only clear
        // on a trainer restart: naming that action is the whole value of the
        // line. It cost an hour to work out the first time, from a trainer
        // whose process predated the viz.py that sends it.
        if (_typeNames == null || _typeIdx == null)
            _hoverLines[nLines++] = "no cell types on the wire \u00b7 restart the trainer";

        // The population, not just the cue name - "LC4+LPLC2, left" is what
        // the encoder actually injects into, so a hovered LC4 says what it is
        // grouped with. Resolved through CueOf so it matches on type AND side
        // exactly the way the injection does, rather than a second table that
        // could disagree with it.
        int cueIdx = CueOf(_hoverIndex);
        if (cueIdx >= 0)
            _hoverLines[nLines++] = "cue " + CueNames[cueIdx] + "  \u00b7  "
                                    + CueTypeText(cueIdx);

        if (slot >= 0 && _typeShown != null && slot < _typeShown.Length)
        {
            int shown = _typeShown[slot];
            int firing = _typeFiring != null ? _typeFiring[slot] : 0;
            _hoverLines[nLines++] = shown + (shown == 1 ? " cell drawn" : " cells drawn")
                                    + "  \u00b7  " + firing + " firing now";
        }

        float fs = Mathf.Max(8, controlsFontSize);
        float pad = fs * 0.55f;
        float rowH = RowHeight();
        float headH = Mathf.Ceil(_guiKeyStyle.CalcSize(new GUIContent(head)).y * 1.15f);
        float w = _guiKeyStyle.CalcSize(new GUIContent(head)).x;
        for (int i = 0; i < nLines; i++)
            w = Mathf.Max(w, _guiTextStyle.CalcSize(new GUIContent(_hoverLines[i])).x);
        w = Mathf.Ceil(w + pad * 2f);
        float h = Mathf.Ceil(pad * 2f + headH + nLines * rowH);

        // The cursor is a screen-space position and GUI.matrix does not apply
        // to it, so convert before mixing it with logical-pixel layout.
        Vector2 cursor = OverlayUi.ToLogical(_hoverPos);
        float screenW = OverlayUi.LogicalWidth, screenH = OverlayUi.LogicalHeight;

        // Offset from the cursor so the pointer does not sit on the text, then
        // pulled back inside the screen - at the right or bottom edge the box
        // flips to the other side of the cursor rather than hanging off.
        float x = cursor.x + fs * 0.9f;
        float y = (screenH - cursor.y) + fs * 0.6f;
        if (x + w > screenW - 6f) x = cursor.x - w - fs * 0.9f;
        if (y + h > screenH - 6f) y = (screenH - cursor.y) - h - fs * 0.6f;
        x = Mathf.Clamp(x, 6f, Mathf.Max(6f, screenW - w - 6f));
        y = Mathf.Clamp(y, 6f, Mathf.Max(6f, screenH - h - 6f));

        Color prev = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.82f);
        GUI.DrawTexture(new Rect(x, y, w, h), _guiPanelTex);
        GUI.color = prev;

        // Headline in the colour the neuron is drawn in, so the label and the
        // thing it describes are visibly the same object.
        Color restC, fullC;
        RoleRamp(role, IsChaseCue(cueIdx), out restC, out fullC);
        _guiHeadStyle.normal.textColor = new Color(fullC.r, fullC.g, fullC.b, 1f);
        GUI.Label(new Rect(x + pad, y + pad, w - pad * 2f, headH), head, _guiHeadStyle);

        float row = y + pad + headH;
        for (int i = 0; i < nLines; i++)
        {
            GUI.Label(new Rect(x + pad, row, w - pad * 2f, rowH), _hoverLines[i],
                      _guiTextStyle);
            row += rowH;
        }
    }

    /// <summary>Flat colour block, tinting the shared 1x1 white texture.</summary>
    void DrawSwatch(float x, float y, float w, float h, Color c)
    {
        Color prev = GUI.color;
        // Full alpha regardless of the role colour's own: the swatch is
        // identifying a hue, not reproducing a resting neuron's faintness.
        GUI.color = new Color(c.r, c.g, c.b, 1f);
        GUI.DrawTexture(new Rect(x, y, w, h), _guiPanelTex);
        GUI.color = prev;
    }

    void EnsureGuiAssets()
    {
        // Font size is part of what is cached, so a change to it has to
        // invalidate the styles - otherwise editing the field does nothing
        // visible and looks broken.
        if (_guiReady && _guiFontSize == controlsFontSize) return;
        _guiFontSize = controlsFontSize;
        int fs = Mathf.Max(8, controlsFontSize);

        if (_guiPanelTex == null)
        {
            _guiPanelTex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _guiPanelTex.SetPixel(0, 0, Color.white);
            _guiPanelTex.Apply();
            _guiPanelTex.wrapMode = TextureWrapMode.Clamp;
        }

        // wordWrap off on all three. GUI.skin.label turns it ON by default,
        // which is what squashed the rows: a label wider than its column took
        // a second line, the row had height for one, and the glyphs were cut
        // through the middle. Off, a column that is somehow still too narrow
        // truncates at the right edge - visibly wrong rather than deceptively
        // wrong - and CalcSize below measures a single line, as intended.
        // Also required for CalcSize to be meaningful at all: with wrapping on
        // it reports an unhelpfully tall box for long strings.
        _guiTitleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = fs,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.MiddleLeft,
            wordWrap = false,
        };
        _guiTitleStyle.normal.textColor = new Color(1.00f, 0.60f, 0.12f);

        _guiKeyStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = fs,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.MiddleLeft,
            wordWrap = false,
        };
        _guiKeyStyle.normal.textColor = new Color(1f, 1f, 1f, 0.95f);

        _guiTextStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = fs,
            alignment = TextAnchor.MiddleLeft,
            wordWrap = false,
        };
        _guiTextStyle.normal.textColor = new Color(0.78f, 0.83f, 0.92f);

        // The hover headline is recoloured per neuron, so it needs its own
        // style object - tinting _guiKeyStyle would drag the legends' key
        // column along with it.
        _guiHeadStyle = new GUIStyle(_guiKeyStyle);

        _guiReady = true;
    }

    void SetVisible(bool on)
    {
        bool want = on && _vizEnabled && !CameraViewSwitcher.CarCameraOn;
        // Edges are laid out in the connectome view's space only, so the fly
        // view draws neurons alone (edges ship off by default anyway).
        bool edges = want && !Anatomy;
        if (_edgeObject != null && _edgeObject.activeSelf != edges)
            _edgeObject.SetActive(edges);
        if (_pointObject != null && _pointObject.activeSelf != want)
            _pointObject.SetActive(want);
        if (_fly != null) _fly.SetVisible(want && Anatomy);
    }
}
