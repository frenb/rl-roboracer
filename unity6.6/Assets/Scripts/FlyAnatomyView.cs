using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

/// <summary>
/// The see-through male fruit fly that FlyBrainViz's anatomy mode (N key)
/// draws the connectome inside. Owns the model, its X-ray materials, and the
/// animation schedule; FlyBrainViz owns the pose of Root and the neurons.
///
/// The schedule is a fixed loop: stand still on the idle clip's first frame
/// for a random stretch, idle_look (cleaning hands), stand still again,
/// look_around (head left, then right), and around again. TriggerFlight (the
/// 1 key) interrupts it for takeoff, a figure-eight and a landing, then
/// starts it afresh. Everything runs on unscaled time, because the sim runs
/// at Time.timeScale 3-5 and the fly should not.
///
/// Driven through a manually evaluated PlayableGraph rather than an Animator
/// Controller: controllers can only be authored in the editor, and manual
/// evaluation leaves the bones settled before FlyBrainViz.LateUpdate reads
/// them.
/// </summary>
public class FlyAnatomyView : MonoBehaviour
{
    public const string ModelPath = "FruitFly/FruitFlyMale_animated_v2";
    const string ShaderName = "Hidden/FlyXRay";
    /// <summary>Head width in world units after normalizing the model's scale.</summary>
    public const float HeadWidthUnits = 4f;
    // Before Sprites/Default (3000), which draws the neurons: the shell is
    // blended first and the neurons land on top of it at full strength.
    const int ShellQueue = 2990;

    enum Clip { Idle, IdleLook, Takeoff, Hover, Land, FlyForward, FlyTurnLeft, FlyTurnRight, Look }
    static readonly string[] ClipKeys = {
        "idle", "idle_look", "takeoff", "hover", "land",
        "fly_forward", "fly_turn_left", "fly_turn_right", "look_around",
    };
    /// <summary>The look-around is keyed on the same skeleton in Blender and
    /// exported on its own, so the model file stays as it was.</summary>
    public const string LookClipPath = "FruitFly/FruitFlyMale_look";

    enum Kind { Play, Still, Path }

    struct Step
    {
        public Kind kind;
        public Clip clip;
        public int repeats;
        public float fade;      // blend into this step, seconds
        public float seconds;   // Still and Path: how long
        public bool faceOut;    // flight: turn to face the path's first heading
        public string label;
        public static Step Play(Clip c, int r, float f, string l)
        { return new Step { kind = Kind.Play, clip = c, repeats = r, fade = f, label = l }; }
        public static Step Still(float s, float f, string l)
        { return new Step { kind = Kind.Still, clip = Clip.Idle, repeats = 1, fade = f, seconds = s, label = l }; }
        public static Step Path(float s, float f, string l)
        { return new Step { kind = Kind.Path, clip = Clip.FlyForward, repeats = 1, fade = f, seconds = s, label = l }; }
    }

    public FlyAnatomySettings Settings { get; private set; }
    /// <summary>Rotated and placed by FlyBrainViz. Its local space is the
    /// frame every landmark below is expressed in.</summary>
    public Transform Root { get { return transform; } }
    public bool Ready { get; private set; }

    // Rest-pose landmarks, Root-local. Up/Forward/Right are the fly's own
    // dorsal, anterior and rightward directions.
    public Vector3 Up { get; private set; }
    public Vector3 Forward { get; private set; }
    public Vector3 Right { get; private set; }
    public Vector3 HeadCentre { get; private set; }
    public float HeadWidth { get; private set; }
    public float HeadDepth { get; private set; }
    public float HeadHeight { get; private set; }
    /// <summary>Root-local points that outline the fly at rest, for framing.</summary>
    public Vector3[] FramePoints { get; private set; }
    /// <summary>Root-local pelvis displacement at the top of the takeoff.</summary>
    public Vector3 Lift { get; private set; }
    public Transform HeadBone { get; private set; }
    public Transform ThoraxBone { get; private set; }
    /// <summary>The bones' rest-pose localToWorld, relative to Root.</summary>
    public Matrix4x4 HeadRestInRoot { get; private set; }
    public Matrix4x4 ThoraxRestInRoot { get; private set; }
    /// <summary>Head bend at the neck the fly eases toward, degrees; positive
    /// tips the head down.</summary>
    public float HeadPitchTarget { get; set; }
    /// <summary>Current head bend, degrees.</summary>
    public float HeadPitch { get; private set; }
    /// <summary>Root-local neck pivot at rest; null-safe via HasNeck.</summary>
    public Vector3 NeckRest { get; private set; }
    public bool HasNeck { get { return _neck != null; } }
    /// <summary>Root-local shift of the body this frame from the part of the
    /// flight climb left on screen; zero on the ground.</summary>
    public Vector3 ShownLift { get; private set; }
    /// <summary>Opacity multiplier for the body behind the head; the head,
    /// front legs, wings and eyes keep theirs.</summary>
    public float RearOpacityScale { get; set; } = 1f;
    /// <summary>Draw the body as little more than its rim, in the close-up's
    /// darker tint.</summary>
    public bool OutlineOnly { get; set; }
    /// <summary>Draw the fly with the model's own textured materials, lit,
    /// instead of the see-through shell.</summary>
    public bool Textured { get; set; }

    const float HeadPitchDegreesPerSecond = 40f;

    private GameObject _model;
    private Transform _neck;
    private Quaternion _neckBase, _neckWritten;
    private bool _neckBent;
    private Transform _pelvis;
    private Vector3 _pelvisRest, _modelRestLocalPos;
    private PlayableGraph _graph;
    private AnimationMixerPlayable _mixer;
    private readonly AnimationClipPlayable[] _clipPlayables = new AnimationClipPlayable[ClipKeys.Length];
    private readonly float[] _clipLength = new float[ClipKeys.Length];
    private bool _hasLook;
    private Quaternion _modelRestLocalRot = Quaternion.identity;
    // The clip carrying most weight in the last pose, and its time, so a
    // flight can blend out of whatever the fly was doing.
    private Clip _curClip;
    private float _curTime;

    // ---- flight (1 key) ----
    private bool _flight;
    private float _pathSign = 1f;
    /// <summary>Half-width and half-height of the pulled-back flight shot on
    /// the plane through the fly, Root-local, set by FlyBrainViz; the loop is
    /// sized to leave it.</summary>
    public Vector2 ShotHalfSize { get; set; }
    /// <summary>The overlay camera in Root-local terms while it is in
    /// perspective for a flight (CamTanHalfFov 0 when orthographic), so the
    /// loop can be planned against what the camera actually sees.</summary>
    public Vector3 CamPos { get; set; }
    public Vector3 CamForward { get; set; } = Vector3.forward;
    public float CamTanHalfFov { get; set; }
    /// <summary>Farthest the loop goes from the hover point, Root-local, so
    /// the camera's far clip plane can take it in.</summary>
    public float OrbitReach { get; private set; }

    const int OrbitSamples = 720;
    private readonly Vector3[] _orbit = new Vector3[OrbitSamples + 1];   // level offset from the hover point
    private readonly float[] _orbitTime = new float[OrbitSamples + 1];   // share of the loop's time
    private bool _orbitBuilt;
    private float _orbitFar, _orbitHoverY;
    private float _landProgress;
    // The last pose's blend, and each wingbeat clip's body averaged over its
    // loop (Root-local): the centre between pelvis and head, the pelvis, and
    // the pelvis-to-head direction. Flight holds the body near these.
    private Clip _poseFrom, _poseTo;
    private float _poseW = 1f;
    private readonly Vector3[] _airCentre = new Vector3[ClipKeys.Length];
    private readonly Vector3[] _airPelvis = new Vector3[ClipKeys.Length];
    private readonly Vector3[] _airDir = new Vector3[ClipKeys.Length];
    private Vector3 _modelRestLocalScale = Vector3.one;
    private float _flightYaw, _yawRate, _flightRoll;
    private Vector3 _flightOffset;
    /// <summary>True from takeoff until the landing has finished.</summary>
    public bool Flying { get { return _flight; } }
    /// <summary>Root-local directions of the camera's right and up, set by
    /// FlyBrainViz: the figure-eight is laid out in that plane so it stays
    /// in view whatever the orientation dials say.</summary>
    public Vector3 FlightRight { get; set; } = Vector3.right;
    public Vector3 FlightUp { get; set; } = Vector3.up;
    /// <summary>Pelvis to head bone at rest, Root-local.</summary>
    public float BodyLength { get; private set; } = 1f;
    /// <summary>
    /// The figure-eight's axes, Root-local: across the screen and toward the
    /// camera, both level, and the fly's up. Level from the camera's right
    /// and forward, so the loops run across and into the view whatever the
    /// orientation dials say.
    /// </summary>
    void PathAxes(out Vector3 across, out Vector3 depth)
    {
        across = Vector3.ProjectOnPlane(FlightRight, Up);
        if (across.sqrMagnitude < 1e-6f) across = Right;
        across.Normalize();
        depth = Vector3.Cross(Up, across).normalized;
    }

    private readonly Material[] _materials = new Material[4];   // body, eyes, wings, hair
    private readonly List<Renderer> _hairRenderers = new List<Renderer>();
    private readonly List<Mesh> _meshes = new List<Mesh>();

    struct MaterialSwap { public Renderer renderer; public Material[] textured, xray; }
    // The model's authored materials (albedo, normal and metallic maps), by
    // slot: body, eyes, wings, hair.
    static readonly string[] TexturedMaterialPaths = {
        "FruitFly/Materials/FruitFlyMale_Body", "FruitFly/Materials/FruitFlyMale_Eyes",
        "FruitFly/Materials/FruitFlyMale_Wing", "FruitFly/Materials/FruitFlyMale_Hair",
    };
    private readonly List<MaterialSwap> _swaps = new List<MaterialSwap>();
    private bool _texturedApplied;
    private Light _light;
    private readonly List<Step> _queue = new List<Step>();
    private int _stepIndex = -1;          // -1: the cycle has not started
    private float _stepStart;
    private Clip _fadeFrom;
    private float _fadeFromTime;
    private bool _fading;

    public static FlyAnatomyView Create(Vector3 position, int layer, FlyAnatomySettings settings)
    {
        var go = new GameObject("FlyAnatomy");
        go.transform.position = position;
        var view = go.AddComponent<FlyAnatomyView>();
        view.Settings = settings;
        if (!view.Build(layer))
        {
            Destroy(go);
            return null;
        }
        return view;
    }

    bool Build(int layer)
    {
        var prefab = Resources.Load<GameObject>(ModelPath);
        if (prefab == null)
        {
            Debug.LogError($"[FlyAnatomy] model not found at Resources/{ModelPath}");
            return false;
        }
        var shader = Shader.Find(ShaderName);
        if (shader == null)
        {
            Debug.LogError($"[FlyAnatomy] shader {ShaderName} missing; expected at "
                           + "Resources/FruitFly/FlyXRay.shader");
            return false;
        }

        _model = Instantiate(prefab, transform, false);
        _model.name = "FruitFlyMale";
        SetLayerRecursive(gameObject, layer);

        if (!BuildAnimation()) return false;
        BuildMaterials(shader);
        WriteKeepMask();

        HeadBone = FindBone(Settings.headBone);
        ThoraxBone = FindBone(Settings.thoraxBone);
        if (HeadBone == null || ThoraxBone == null)
        {
            Debug.LogError($"[FlyAnatomy] bones '{Settings.headBone}' / "
                           + $"'{Settings.thoraxBone}' not found in the model");
            return false;
        }

        MeasureLandmarks();
        transform.localScale = Vector3.one * (HeadWidthUnits / Mathf.Max(1e-6f, HeadWidth));

        Ready = true;
        Debug.Log($"[FlyAnatomy] built: clips {string.Join(", ", ClipSummary())}");
        return true;
    }

    // ---- animation ---------------------------------------------------------
    bool BuildAnimation()
    {
        var animator = _model.GetComponentInChildren<Animator>();
        if (animator == null) animator = _model.AddComponent<Animator>();
        animator.runtimeAnimatorController = null;
        animator.applyRootMotion = false;
        animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;

        var clips = Resources.LoadAll<AnimationClip>(ModelPath);
        _graph = PlayableGraph.Create("FlyAnatomy");
        _graph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
        var output = AnimationPlayableOutput.Create(_graph, "fly", animator);
        _mixer = AnimationMixerPlayable.Create(_graph, ClipKeys.Length);
        output.SetSourcePlayable(_mixer);

        for (int k = 0; k < ClipKeys.Length; k++)
        {
            AnimationClip clip = null;
            if (k == (int)Clip.Look)
            {
                // Its own file, whose one take Blender names after the scene.
                foreach (var c in Resources.LoadAll<AnimationClip>(LookClipPath))
                    if (!c.name.StartsWith("__preview__")) clip = c;
                _clipLength[k] = 1f;
                if (clip == null)
                {
                    Debug.LogWarning($"[FlyAnatomy] no look-around clip at Resources/{LookClipPath}; "
                                     + "the idle cycle grooms only");
                    continue;
                }
                _hasLook = true;
            }
            else
            {
                foreach (var c in clips)
                    if (!c.name.StartsWith("__preview__") && c.name.EndsWith("FruitFly_" + ClipKeys[k]))
                        clip = c;
            }
            if (clip == null)
            {
                Debug.LogError($"[FlyAnatomy] clip '{ClipKeys[k]}' not found in {ModelPath}");
                return false;
            }
            var p = AnimationClipPlayable.Create(_graph, clip);
            p.SetApplyFootIK(false);
            _graph.Connect(p, 0, _mixer, k);
            _clipPlayables[k] = p;
            _clipLength[k] = Mathf.Max(1e-3f, clip.length);
        }
        Pose(Clip.Idle, 0f);
        CheckLookBinds();
        return true;
    }

    /// <summary>
    /// A clip from a separate file only moves bones whose paths match the
    /// model's. If the export's hierarchy differs it plays as a frozen
    /// standing pose, silently; catch that here instead.
    /// </summary>
    void CheckLookBinds()
    {
        if (!_hasLook) return;
        // The turn is keyed on the neck, which the head's mesh is skinned to;
        // the head bone (and the brain riding it) just follows.
        var head = FindBone("HeadLock");
        if (head == null) return;
        Quaternion rest = head.rotation;
        Pose(Clip.Look, Mathf.Min(1.3f, _clipLength[(int)Clip.Look] * 0.35f));
        float turned = Quaternion.Angle(rest, head.rotation);
        Pose(Clip.Idle, 0f);
        if (turned < 5f)
        {
            _hasLook = false;
            Debug.LogWarning($"[FlyAnatomy] look-around clip does not move the head "
                             + $"({turned:0.0} deg); its bone paths do not match the model. "
                             + "The idle cycle grooms only");
        }
    }

    IEnumerable<string> ClipSummary()
    {
        for (int k = 0; k < ClipKeys.Length; k++)
            yield return $"{ClipKeys[k]} {_clipLength[k]:0.00}s";
    }

    /// <summary>True on the frames the pose was re-evaluated, so FlyBrainViz
    /// only moves the neurons when the bones they ride have moved.</summary>
    public bool PosedThisFrame { get; private set; }
    private float _lastPose = -1f;

    void Update()
    {
        if (!Ready) return;
        ApplyMaterialSettings();
        PosedThisFrame = false;
        if (!_model.activeInHierarchy) return;
        // Posed at a fixed rate rather than every frame: the sim renders as
        // fast as it can, and every frame spent here is sim time the policy
        // waits for its next camera frame.
        float now = Time.unscaledTime;
        float hz = Settings.animationHz;
        if (_flight && hz > 0f)
            hz = Settings.flightAnimationHz > 0f ? Mathf.Max(hz, Settings.flightAnimationHz) : 0f;
        if (hz > 0f && _lastPose >= 0f && now - _lastPose < 1f / hz) return;
        float dt = _lastPose >= 0f ? now - _lastPose : 0f;
        _lastPose = now;
        bool moved = AdvanceSchedule(now, dt);
        if (moved) ApplyModelTransform(_queue[_stepIndex].clip);
        float pitch = HeadPitch;
        ApplyHeadPitch(dt);
        PosedThisFrame = moved || HeadPitch != pitch;
    }

    /// <summary>
    /// Bend the head at the neck on top of the animated pose. The clips do not
    /// all key the neck, so Evaluate does not always rewrite it: if the value
    /// is still the one written last frame, the bend goes on the remembered
    /// base instead of on itself, or it would compound into a spin.
    /// </summary>
    void ApplyHeadPitch(float dt)
    {
        HeadPitch = Mathf.MoveTowards(HeadPitch, HeadPitchTarget, HeadPitchDegreesPerSecond * dt);
        if (_neck == null) return;
        bool untouched = _neckBent && _neck.localRotation == _neckWritten;
        Quaternion baseLocal = untouched ? _neckBase : _neck.localRotation;
        _neckBase = baseLocal;
        if (Mathf.Abs(HeadPitch) < 1e-3f)
        {
            if (_neckBent) _neck.localRotation = baseLocal;
            _neckBent = false;
            return;
        }
        Quaternion parent = _neck.parent != null ? _neck.parent.rotation : Quaternion.identity;
        Vector3 axis = transform.TransformDirection(Right).normalized;
        Quaternion bent = Quaternion.AngleAxis(HeadPitch, axis) * parent * baseLocal;
        _neck.localRotation = Quaternion.Inverse(parent) * bent;
        _neckWritten = _neck.localRotation;
        _neckBent = true;
    }

    /// <summary>Advance the cycle and pose the fly; false when the pose is
    /// the one already on the bones, so nothing was evaluated.</summary>
    bool AdvanceSchedule(float now, float dt)
    {
        bool started = false;
        if (_stepIndex < 0)
        {
            BuildCycle();
            StartStep(0, now, Clip.Idle, 0f);
            started = true;
        }

        Step step = _queue[_stepIndex];
        float t = now - _stepStart;
        if (t >= StepSeconds(step))
        {
            // A looping clip ends on its last frame, which is also its first.
            Clip endClip = step.clip;
            float endTime = step.kind == Kind.Still ? 0f
                          : step.kind == Kind.Path ? WingTime(Clip.FlyForward, StepSeconds(step))
                          : _clipLength[(int)step.clip];
            int next = _stepIndex + 1;
            if (next >= _queue.Count)
            {
                if (_flight) EndFlight();
                BuildCycle();     // picks up any settings changed mid-cycle
                next = 0;
            }
            StartStep(next, now, endClip, endTime);
            step = _queue[_stepIndex];
            t = 0f;
            started = true;
        }

        float w = (!_fading || step.fade <= 0f) ? 1f : Mathf.Clamp01(t / step.fade);
        bool wasFading = _fading;
        if (w >= 1f) _fading = false;
        UpdateFlightMotion(step, t, dt);
        Clip from = _fading ? _fadeFrom : step.clip;
        switch (step.kind)
        {
            case Kind.Still:
                if (!started && !wasFading && !_flight) return false;
                Pose(step.clip, 0f, from, _fadeFromTime, w);
                return true;
            case Kind.Path:
            {
                float fwd = WingTime(Clip.FlyForward, t);
                if (_fading)
                {
                    Pose(Clip.FlyForward, fwd, from, _fadeFromTime, w);
                    return true;
                }
                // Positive yaw turns the fly to its right.
                Clip turn = _yawRate > 0f ? Clip.FlyTurnRight : Clip.FlyTurnLeft;
                float tw = Mathf.Clamp01(Mathf.Abs(_yawRate) / Mathf.Max(1f, Settings.flightFullBankTurnRate));
                Pose(turn, WingTime(turn, t), Clip.FlyForward, fwd, tw);
                return true;
            }
            default:
            {
                float len = _clipLength[(int)step.clip];
                float played = t * Speed(step.clip);
                float clipTime = step.repeats > 1 ? Mathf.Repeat(played, len) : Mathf.Min(played, len);
                _landProgress = step.clip == Clip.Land ? clipTime / len : 0f;
                Pose(step.clip, clipTime, from, _fadeFromTime, w);
                return true;
            }
        }
    }

    float WingTime(Clip c, float t)
    {
        return Mathf.Repeat(t * Speed(c), _clipLength[(int)c]);
    }

    float StepSeconds(Step s)
    {
        return s.kind != Kind.Play ? s.seconds : _clipLength[(int)s.clip] * s.repeats / Speed(s.clip);
    }

    /// <summary>Playback rate of a clip; 1 is its native speed.</summary>
    float Speed(Clip c)
    {
        var s = Settings;
        switch (c)
        {
            case Clip.Takeoff: return Mathf.Max(0.05f, s.flightTakeoffSpeed);
            case Clip.Land: return Mathf.Max(0.05f, s.flightLandSpeed);
            case Clip.Hover:
            case Clip.FlyForward:
            case Clip.FlyTurnLeft:
            case Clip.FlyTurnRight: return Mathf.Max(0.05f, s.flightWingSpeed);
            default: return 1f;
        }
    }

    /// <summary>Whole loops of a looping clip closest to a wanted duration, so
    /// it always stops on its first frame.</summary>
    int Loops(Clip c, float seconds)
    {
        return Mathf.Max(1, Mathf.RoundToInt(seconds * Speed(c) / _clipLength[(int)c]));
    }

    /// <summary>
    /// The fixed cycle: stand still, groom, stand still, look left and right,
    /// and around again. Grooming holds the front legs up on every frame,
    /// first and last included, so it blends in and out of the standing pose
    /// at a pace the legs can plausibly move. The look-around starts and ends
    /// on the standing pose itself.
    /// </summary>
    void BuildCycle()
    {
        var s = Settings;
        float legs = Mathf.Max(0f, s.groomBlendSeconds);
        _queue.Clear();
        _queue.Add(Step.Still(Mathf.Max(0.1f, StillSeconds()), legs, "standing still"));
        _queue.Add(Step.Play(Clip.IdleLook, Loops(Clip.IdleLook, s.groomSeconds), legs, "idle_look (cleaning hands)"));
        if (!_hasLook) return;
        _queue.Add(Step.Still(Mathf.Max(0.1f, StillSeconds()), legs, "standing still"));
        _queue.Add(Step.Play(Clip.Look, 1, 0.2f, "look_around (left, then right)"));
    }

    // ---- flight (1 key) ----------------------------------------------------
    /// <summary>
    /// Take off from wherever the idle cycle is, fly a figure-eight in the
    /// camera's plane, and land back on the spot. False when already flying
    /// or not on screen.
    /// </summary>
    public bool TriggerFlight()
    {
        if (!Ready || _flight || !_model.activeInHierarchy) return false;
        _pathSign = Settings.flightLoopLeft ? -1f : 1f;
        _flight = true;
        _orbitBuilt = false;
        _flightYaw = _yawRate = _flightRoll = 0f;
        _flightOffset = Vector3.zero;
        Clip from = _curClip;
        float fromTime = _curTime;
        BuildFlight();
        StartStep(0, Time.unscaledTime, from, fromTime);
        return true;
    }

    void BuildFlight()
    {
        var s = Settings;
        float cf = Mathf.Max(0f, s.flightCrossfadeSeconds);
        _queue.Clear();
        _queue.Add(Step.Play(Clip.Takeoff, 1, cf, "takeoff"));
        var hover = Step.Play(Clip.Hover, Loops(Clip.Hover, s.flightHoverSeconds), cf, "hover");
        hover.faceOut = true;
        _queue.Add(hover);
        _queue.Add(Step.Path(s.flightPathSeconds, 0.3f, "orbiting"));
        _queue.Add(Step.Play(Clip.Hover, Loops(Clip.Hover, s.flightReturnHoverSeconds), 0.3f,
                             "hover (back over the start)"));
        _queue.Add(Step.Play(Clip.Land, 1, cf, "land"));
    }

    void EndFlight()
    {
        _flight = false;
        _flightYaw = _yawRate = _flightRoll = 0f;
        _flightOffset = Vector3.zero;
    }

    /// <summary>
    /// Plan the loop against the camera as it is when the loop starts: a
    /// level circle (stretched across the shot by flightLoopWidth), its
    /// nearest point the hover point, leftward first unless flightLoopLeft
    /// is off, as a plane circling
    /// in front of a camera at its own height - across the shot large, round
    /// and away at one side, back across small on the far side, and round
    /// toward the camera at the other. Sized so the far side is
    /// flightFarShrink times smaller than the start, which is all perspective
    /// needs to set the radius: the far side is flightFarShrink times the
    /// camera's distance away. Timed by arc length, the speed rising
    /// flightFarSpeedup-fold toward the far side (1 is a steady speed).
    /// </summary>
    void BuildOrbit()
    {
        var s = Settings;
        PathAxes(out Vector3 across, out Vector3 depth);
        Vector3 away = -depth;
        Vector3 hover = _pelvisRest + Lift;
        float tan = CamTanHalfFov > 1e-4f ? CamTanHalfFov : Mathf.Tan(0.5f * s.flightFov * Mathf.Deg2Rad);
        float halfH = ShotHalfSize.y > 0f ? ShotHalfSize.y : 3f * BodyLength;
        float d = CamTanHalfFov > 1e-4f ? Vector3.Dot(hover - CamPos, CamForward) : halfH / tan;
        d = Mathf.Max(BodyLength, d);
        float r = 0.5f * (Mathf.Max(1.1f, s.flightFarShrink) - 1f) * d;
        float far = 2f * r;
        float width = Mathf.Max(0.5f, s.flightLoopWidth);
        float speedUp = Mathf.Max(1f, s.flightFarSpeedup);
        float cost = 0f;
        Vector2 prev = Vector2.zero;
        for (int i = 0; i <= OrbitSamples; i++)
        {
            float a = 2f * Mathf.PI * i / OrbitSamples;
            var p = new Vector2(_pathSign * width * r * Mathf.Sin(a), r * (1f - Mathf.Cos(a)));
            if (i > 0)
                cost += (p - prev).magnitude / (1f + (speedUp - 1f) * Mathf.Clamp01(0.5f * (p.y + prev.y) / far));
            _orbitTime[i] = cost;
            _orbit[i] = across * p.x + away * p.y;
            prev = p;
        }
        for (int i = 0; i <= OrbitSamples; i++) _orbitTime[i] /= Mathf.Max(1e-6f, cost);
        _orbitFar = far;
        OrbitReach = Mathf.Max(far, width * r);
        Vector3 q = hover - CamPos;
        _orbitHoverY = CamTanHalfFov > 1e-4f
            ? Vector3.Dot(q, FlightUp.normalized) / (Vector3.Dot(q, CamForward) * tan) : 0f;
        _orbitBuilt = true;
    }

    /// <summary>Level offset and direction on the loop at time share x.</summary>
    void SampleOrbit(float x, out Vector3 level, out Vector3 dir)
    {
        int lo = 0, hi = OrbitSamples;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) >> 1;
            if (_orbitTime[mid] <= x) lo = mid; else hi = mid;
        }
        float span = _orbitTime[hi] - _orbitTime[lo];
        float f = span > 1e-7f ? Mathf.Clamp01((x - _orbitTime[lo]) / span) : 0f;
        level = Vector3.Lerp(_orbit[lo], _orbit[hi], f);
        dir = _orbit[hi] - _orbit[lo];
        if (dir.sqrMagnitude < 1e-12f) dir = _orbit[Mathf.Min(OrbitSamples, hi + 1)] - _orbit[lo];
    }

    /// <summary>Share of the orbit done at time share x: steady, with a
    /// constant-acceleration ramp of share r at each end.</summary>
    static float Ramp(float x, float r)
    {
        float p;
        if (x < r) p = x * x / (2f * r);
        else if (x > 1f - r) p = 1f - r - (1f - x) * (1f - x) / (2f * r);
        else p = x - r * 0.5f;
        return p / (1f - r);
    }

    /// <summary>
    /// Where the fly is on its loop (BuildOrbit), and which way it faces.
    /// The camera is in perspective for the flight, so the fly shrinks with
    /// distance on its own. Its height is solved for where it should sit on
    /// screen: from the hover point's height to flightFarRise (of the
    /// half-height) above it on the far side, as if the camera were level
    /// with the loop, rather than wherever the camera's downward look would
    /// put a level path, which lifts the far side toward the horizon.
    ///
    /// The fly faces along the loop and banks with the rate of turn. It
    /// turns to the first heading while it hovers after takeoff, and back to
    /// the way it started in the hover before landing.
    /// </summary>
    void UpdateFlightMotion(Step step, float t, float dt)
    {
        if (!_flight) return;
        var s = Settings;
        float targetYaw = 0f;
        Vector3 offset = Vector3.zero;
        PathAxes(out Vector3 across, out Vector3 depth);
        if (step.kind == Kind.Path)
        {
            if (!_orbitBuilt) BuildOrbit();
            float T = Mathf.Max(0.1f, step.seconds);
            float x = Ramp(Mathf.Clamp01(t / T), Mathf.Clamp(s.flightRampSeconds / T, 0.01f, 0.5f));
            SampleOrbit(x, out Vector3 level, out Vector3 dir);
            float h = 0f;
            if (CamTanHalfFov > 1e-4f)
            {
                Vector3 u = FlightUp.normalized, f = CamForward;
                float awayNow = -Vector3.Dot(level, depth);
                float y = _orbitHoverY + s.flightFarRise
                          * Mathf.Clamp01(awayNow / Mathf.Max(1e-4f, _orbitFar));
                Vector3 q = _pelvisRest + Lift + level - CamPos;
                float yt = y * CamTanHalfFov;
                float den = Vector3.Dot(Up, u) - yt * Vector3.Dot(Up, f);
                if (Mathf.Abs(den) > 1e-3f) h = (yt * Vector3.Dot(q, f) - Vector3.Dot(q, u)) / den;
            }
            offset = level + Up * h;
            if (s.flightFaceHeading && dir.sqrMagnitude > 1e-12f)
                targetYaw = Vector3.SignedAngle(Forward, dir, Up);
        }
        else if (step.faceOut && s.flightFaceHeading)
        {
            // The loop leaves the hover point straight across the shot.
            targetYaw = Vector3.SignedAngle(Forward, across * _pathSign, Up);
        }
        float prev = _flightYaw;
        _flightYaw = Mathf.MoveTowardsAngle(_flightYaw, targetYaw, Settings.flightTurnRate * dt);
        _yawRate = dt > 1e-4f ? Mathf.DeltaAngle(prev, _flightYaw) / dt : 0f;
        // A positive roll about the fly's forward dips its left side, so a
        // turn to the right (positive yaw) rolls negative.
        float bank = -Settings.flightBankDegrees
                     * Mathf.Clamp(_yawRate / Mathf.Max(1f, Settings.flightFullBankTurnRate), -1f, 1f);
        _flightRoll = Mathf.MoveTowards(_flightRoll, bank, 90f * dt);
        _flightOffset = offset;
    }

    Quaternion FlightRotation()
    {
        Quaternion yaw = Quaternion.AngleAxis(_flightYaw, Up);
        return Quaternion.AngleAxis(_flightRoll, yaw * Forward) * yaw;
    }

    float StillSeconds()
    {
        var s = Settings;
        return Random.Range(s.idleMinSeconds, Mathf.Max(s.idleMinSeconds, s.idleMaxSeconds));
    }

    void StartStep(int index, float now, Clip from, float fromTime)
    {
        _stepIndex = index;
        _stepStart = now;
        _fadeFrom = from;
        _fadeFromTime = fromTime;
        // One playable per clip can only sit at one time, so a clip cannot
        // fade into itself; idle and idle_look loop cleanly, so none is needed.
        _fading = from != _queue[index].clip;
        var s = _queue[index];
        Debug.Log($"[FlyAnatomy] {s.label} {StepSeconds(s):0.#} s");
    }

    void Pose(Clip clip, float time) { Pose(clip, time, clip, time, 1f); }

    void Pose(Clip to, float toTime, Clip from, float fromTime, float w)
    {
        for (int k = 0; k < ClipKeys.Length; k++) _mixer.SetInputWeight(k, 0f);
        if (from != to && w < 1f)
        {
            _mixer.SetInputWeight((int)from, 1f - w);
            _clipPlayables[(int)from].SetTime(fromTime);
        }
        _mixer.SetInputWeight((int)to, from != to ? w : 1f);
        _clipPlayables[(int)to].SetTime(toTime);
        _graph.Evaluate(0f);
        _poseFrom = from;
        _poseTo = to;
        _poseW = from != to ? w : 1f;
        bool toWins = from == to || w >= 0.5f;
        _curClip = toWins ? to : from;
        _curTime = toWins ? toTime : fromTime;
    }

    // ---- materials ---------------------------------------------------------
    void BuildMaterials(Shader shader)
    {
        for (int m = 0; m < _materials.Length; m++)
        {
            _materials[m] = new Material(shader) { renderQueue = ShellQueue - (m == 2 ? 1 : 0) };
        }
        var authored = new Material[TexturedMaterialPaths.Length];
        for (int m = 0; m < authored.Length; m++)
        {
            authored[m] = Resources.Load<Material>(TexturedMaterialPaths[m]);
            if (authored[m] == null)
                Debug.LogWarning($"[FlyAnatomy] no material at Resources/{TexturedMaterialPaths[m]}; "
                                 + "the textured fly uses the FBX's own there");
        }
        foreach (var r in _model.GetComponentsInChildren<Renderer>(true))
        {
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
            var src = r.sharedMaterials;
            var dst = new Material[src.Length];
            var tex = new Material[src.Length];
            bool hairRenderer = r.name.ToLowerInvariant().Contains("hair");
            for (int i = 0; i < src.Length; i++)
            {
                int cls = hairRenderer ? 3 : Classify(src[i]);
                dst[i] = _materials[cls];
                tex[i] = authored[cls] != null ? authored[cls] : src[i];
                if (src[i] != null && src[i].HasProperty("_MainTex") && src[i].mainTexture != null
                    && _materials[cls].mainTexture == null)
                    _materials[cls].mainTexture = src[i].mainTexture;
            }
            r.sharedMaterials = dst;
            _swaps.Add(new MaterialSwap { renderer = r, textured = tex, xray = dst });
            if (hairRenderer) _hairRenderers.Add(r);
        }
        ApplyMaterialSettings();
    }

    /// <summary>0 body, 1 eyes, 2 wings, 3 hair. Names are the FBX's own:
    /// phong1 is the eyes, phong3 the wings, anisotropic1 the hair.</summary>
    static int Classify(Material m)
    {
        string n = m != null ? m.name.ToLowerInvariant() : "";
        if (n.Contains("eye") || n.Contains("phong1")) return 1;
        if (n.Contains("wing") || n.Contains("phong3")) return 2;
        if (n.Contains("hair") || n.Contains("anisotropic")) return 3;
        return 0;
    }

    // Bones whose subtrees RearOpacityScale leaves alone, besides the neck:
    // the front legs and the wing roots.
    static readonly string[] KeepBranches = { "FrontLegHip", "WingClav" };
    // Limb roots present as an L/R pair ("L" + name, "R" + name).
    static readonly string[] SymmetricPairs = { "FrontLegHip", "MidLegHip", "RearLegHip", "WingClav" };

    /// <summary>
    /// Mark each body vertex by how much it rides the head, front legs or
    /// wing roots (uv2.x = 1) versus the rest of the body (0), from its skin
    /// weights, so the shader can fade the body behind the head on its own.
    /// uv2.y marks the mouthparts and uv2.z the head alone.
    /// Needs Read/Write on the FBX import.
    /// </summary>
    void WriteKeepMask()
    {
        int kept = 0, total = 0, mouthVerts = 0, headVerts = 0;
        foreach (var smr in _model.GetComponentsInChildren<SkinnedMeshRenderer>(true))
        {
            Mesh src = smr.sharedMesh;
            if (src == null) continue;
            if (!src.isReadable)
            {
                Debug.LogWarning($"[FlyAnatomy] mesh '{src.name}' is not readable; enable "
                                 + "Read/Write on the FBX to fade the body behind the head");
                return;
            }
            var bones = smr.bones;
            var keepBone = new bool[bones.Length];
            var mouthBone = new bool[bones.Length];
            var headBone = new bool[bones.Length];
            for (int b = 0; b < bones.Length; b++)
            {
                keepBone[b] = InKeptBranch(bones[b]);
                mouthBone[b] = InBranch(bones[b], Settings.mouthBone);
                headBone[b] = InBranch(bones[b], Settings.neckBone);
            }

            var weights = src.boneWeights;
            var mask = new List<Vector3>(src.vertexCount);
            for (int v = 0; v < src.vertexCount; v++)
            {
                float k = 0f, mouth = 0f, head = 0f;
                if (v < weights.Length)
                {
                    var w = weights[v];
                    k = Weight(keepBone, w);
                    mouth = Weight(mouthBone, w);
                    head = Weight(headBone, w);
                }
                k = Mathf.Clamp01(k);
                mouth = Mathf.Clamp01(mouth);
                head = Mathf.Clamp01(head);
                mask.Add(new Vector3(k, mouth, head));
                if (k > 0.5f) kept++;
                if (mouth > 0.5f) mouthVerts++;
                if (head > 0.5f) headVerts++;
                total++;
            }
            var mesh = Instantiate(src);
            mesh.name = src.name;
            mesh.SetUVs(2, mask);
            smr.sharedMesh = mesh;
            _meshes.Add(mesh);
        }
        Debug.Log($"[FlyAnatomy] rear-fade mask: {kept} of {total} vertices on head / front legs / wing roots, "
                  + $"{mouthVerts} on the mouthparts, {headVerts} on the head");
    }

    static float Weight(bool[] bones, BoneWeight w)
    {
        return Kept(bones, w.boneIndex0, w.weight0) + Kept(bones, w.boneIndex1, w.weight1)
             + Kept(bones, w.boneIndex2, w.weight2) + Kept(bones, w.boneIndex3, w.weight3);
    }

    bool InBranch(Transform t, string root)
    {
        if (string.IsNullOrEmpty(root)) return false;
        for (; t != null && t != _model.transform; t = t.parent)
            if (t.name == root) return true;
        return false;
    }

    static float Kept(bool[] keep, int bone, float weight)
    {
        return bone >= 0 && bone < keep.Length && keep[bone] ? weight : 0f;
    }

    bool InKeptBranch(Transform t)
    {
        for (; t != null && t != _model.transform; t = t.parent)
        {
            if (t.name == Settings.neckBone) return true;
            foreach (var k in KeepBranches)
                if (t.name.EndsWith(k)) return true;
        }
        return false;
    }

    void ApplyMaterialSettings()
    {
        var s = Settings;
        for (int m = 0; m < _materials.Length; m++)
            if (_materials[m] != null)
            {
                _materials[m].SetFloat("_RearFade", m == 0 ? Mathf.Clamp01(RearOpacityScale) : 1f);
                _materials[m].SetFloat("_MouthFade", m == 0 && OutlineOnly
                                                     ? Mathf.Clamp01(s.closeUpMouthOpacity) : 1f);
                _materials[m].SetFloat("_HeadFade", m == 0
                    ? Mathf.Clamp01(OutlineOnly ? s.closeUpHeadOpacity : s.headOpacity) : 1f);
            }
        Color rimColor = OutlineOnly ? s.closeUpRimColor : s.rimColor;
        Color bodyTint = OutlineOnly ? s.closeUpShellTint : s.bodyTint;
        float bodyFill = s.bodyOpacity * (OutlineOnly ? Mathf.Clamp01(s.closeUpShellFill) : 1f);
        SetShell(_materials[0], bodyTint, bodyFill, s.bodyRimOpacity, s.bodyTextureStrength, rimColor);
        SetShell(_materials[1], s.eyeTint, s.eyeOpacity, s.eyeRimOpacity, s.bodyTextureStrength, rimColor);
        SetShell(_materials[2], s.wingTint, s.wingOpacity, s.wingRimOpacity, s.bodyTextureStrength, rimColor);
        SetShell(_materials[3], bodyTint, s.hairOpacity, 0f, s.bodyTextureStrength, rimColor);
        bool hair = Textured ? s.texturedShowHair : s.showHair;
        for (int i = 0; i < _hairRenderers.Count; i++)
            if (_hairRenderers[i] != null && _hairRenderers[i].enabled != hair)
                _hairRenderers[i].enabled = hair;
        ApplyTextured();
    }

    void ApplyTextured()
    {
        if (Textured != _texturedApplied)
        {
            _texturedApplied = Textured;
            foreach (var sw in _swaps)
                if (sw.renderer != null)
                    sw.renderer.sharedMaterials = Textured ? sw.textured : sw.xray;
        }

        // The overlay layer has no light of its own, and the model's
        // materials are lit; this one lights only that layer.
        if (Textured && _light == null)
        {
            var go = new GameObject("FlyKeyLight");
            go.transform.SetParent(transform, false);
            _light = go.AddComponent<Light>();
            _light.type = LightType.Directional;
            _light.shadows = LightShadows.None;
            _light.cullingMask = FlyBrainViz.OverlayLayerMask;
        }
        if (_light == null) return;
        _light.enabled = Textured && _model.activeInHierarchy;
        _light.intensity = Settings.texturedLightIntensity;
        // From above, in front and a little to one side, in the fly's own
        // frame, so the lighting turns with the fly.
        Vector3 toLight = (Up + 0.7f * Forward + 0.4f * Right).normalized;
        _light.transform.localRotation = Quaternion.LookRotation(-toLight, Up);
    }

    void SetShell(Material m, Color tint, float opacity, float rim, float texStrength, Color rimColor)
    {
        if (m == null) return;
        m.SetColor("_Color", tint);
        m.SetFloat("_Opacity", opacity);
        m.SetFloat("_RimOpacity", rim);
        m.SetColor("_RimColor", rimColor);
        m.SetFloat("_RimPower", Settings.rimPower);
        m.SetFloat("_TexStrength", texStrength);
    }

    // ---- landmarks ---------------------------------------------------------
    // Head proportions of FruitFlyMale_animated_v2, measured in Blender on the
    // rest pose (eyes + head hair, skinned), in units of the pelvis-to-head-bone
    // distance. Taken from the skeleton at runtime rather than from the mesh:
    // Renderer.bounds is stale on the frame the model is created, and skinning
    // the imported vertices by hand came out ~35x too wide in the player.
    const float HeadWidthRatio = 0.77f;
    const float HeadDepthRatio = 0.33f;
    const float HeadHeightRatio = 0.62f;
    const float HeadForwardOffset = 0.02f;
    const float HeadUpOffset = -0.015f;

    /// <summary>
    /// Measure the head, the fly's axes and its framing at rest, with Root
    /// unrotated and unscaled so world and Root-local coincide up to a
    /// translation.
    /// </summary>
    void MeasureLandmarks()
    {
        transform.rotation = Quaternion.identity;
        transform.localScale = Vector3.one;
        Pose(Clip.Idle, 0f);
        HeadRestInRoot = transform.worldToLocalMatrix * HeadBone.localToWorldMatrix;
        ThoraxRestInRoot = transform.worldToLocalMatrix * ThoraxBone.localToWorldMatrix;

        var bones = _model.GetComponentsInChildren<Transform>(true);
        _pelvis = FindBone("Pelvis");
        _neck = FindBone(Settings.neckBone);
        if (_neck != null) NeckRest = transform.InverseTransformPoint(_neck.position);
        Vector3 back = _pelvis != null ? transform.InverseTransformPoint(_pelvis.position) : Vector3.zero;
        Vector3 headBonePos = transform.InverseTransformPoint(HeadBone.position);
        float bodyScale = Mathf.Max(1e-6f, Vector3.Distance(back, headBonePos));
        _pelvisRest = back;
        BodyLength = bodyScale;
        _modelRestLocalPos = _model.transform.localPosition;
        _modelRestLocalRot = _model.transform.localRotation;
        _modelRestLocalScale = _model.transform.localScale;

        // Axes from the anatomy rather than from the import's conventions:
        // forward runs pelvis -> head bone, up is away from the feet.
        Vector3 fwd = headBonePos - back;
        fwd = fwd.sqrMagnitude > 1e-10f ? fwd.normalized : Vector3.forward;
        Vector3 feet = Vector3.zero;
        int nFeet = 0;
        foreach (var b in bones)
            if (b.name.Contains("Toe")) { feet += transform.InverseTransformPoint(b.position); nFeet++; }
        Vector3 up = Vector3.up;
        if (nFeet > 0)
        {
            Vector3 u = Vector3.ProjectOnPlane((back + headBonePos) * 0.5f - feet / nFeet, fwd);
            if (u.sqrMagnitude > 1e-10f) up = u.normalized;
        }
        // The feet only give up roughly: an uneven stance rolls it. Level it
        // on the left-right pairs at the limb roots instead, which are
        // mirror images across the body. The feet still settle which way is
        // dorsal, since the import may mirror the rig's L/R.
        Vector3 across = Vector3.zero;
        foreach (var name in SymmetricPairs)
        {
            Transform l = FindBone("L" + name), r = FindBone("R" + name);
            if (l != null && r != null)
                across += transform.InverseTransformPoint(r.position) - transform.InverseTransformPoint(l.position);
        }
        across = Vector3.ProjectOnPlane(across, fwd);
        float rollFix = 0f;
        if (across.sqrMagnitude > 1e-10f)
        {
            Vector3 u = Vector3.Cross(fwd, across.normalized);
            if (Vector3.Dot(u, up) < 0f) u = -u;
            rollFix = Vector3.Angle(u, up);
            up = u.normalized;
        }
        Up = up;
        Vector3 f = Vector3.ProjectOnPlane(fwd, Up);
        Forward = f.sqrMagnitude > 1e-10f ? f.normalized : Vector3.forward;
        Right = Vector3.Cross(Up, Forward);

        HeadCentre = headBonePos + (Forward * HeadForwardOffset + Up * HeadUpOffset) * bodyScale;
        HeadWidth = HeadWidthRatio * bodyScale;
        HeadDepth = HeadDepthRatio * bodyScale;
        HeadHeight = HeadHeightRatio * bodyScale;

        // What the camera frames: every bone the meshes are skinned to, at
        // rest (toes, wing tips, tail end), plus the head's box, which reaches
        // past the head bone. Skin bones only - the mesh objects' own
        // transforms keep their Blender origins, which are nowhere near the fly.
        var frame = new List<Vector3>();
        var skin = new HashSet<Transform>();
        foreach (var smr in _model.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            foreach (var b in smr.bones)
                if (b != null && skin.Add(b)) frame.Add(transform.InverseTransformPoint(b.position));
        if (frame.Count == 0)
            foreach (var b in bones) frame.Add(transform.InverseTransformPoint(b.position));
        float hh = HeadHeightRatio * bodyScale;
        for (int k = 0; k < 8; k++)
            frame.Add(HeadCentre
                      + Right * (((k & 1) == 0 ? -0.5f : 0.5f) * HeadWidth)
                      + Up * (((k & 2) == 0 ? -0.5f : 0.5f) * hh)
                      + Forward * (((k & 4) == 0 ? -0.5f : 0.5f) * HeadDepth));
        FramePoints = frame.ToArray();

        // How far takeoff carries the body, so the frame can make room for
        // the part of it that flightLiftShown lets through.
        Pose(Clip.Takeoff, _clipLength[(int)Clip.Takeoff]);
        Lift = _pelvis != null ? transform.InverseTransformPoint(_pelvis.position) - back : Vector3.zero;
        foreach (Clip c in new[] { Clip.Hover, Clip.FlyForward, Clip.FlyTurnLeft, Clip.FlyTurnRight })
        {
            const int n = 24;
            Vector3 centre = Vector3.zero, pelvis = Vector3.zero, dir = Vector3.zero;
            for (int k = 0; k < n; k++)
            {
                Pose(c, _clipLength[(int)c] * k / n);
                BodyNow(out Vector3 p, out Vector3 h);
                centre += 0.5f * (p + h);
                pelvis += p;
                dir += (h - p).normalized;
            }
            _airCentre[(int)c] = centre / n;
            _airPelvis[(int)c] = pelvis / n;
            _airDir[(int)c] = dir.sqrMagnitude > 1e-9f ? dir.normalized : Forward;
        }

        Pose(Clip.Idle, 0f);
        Debug.Log($"[FlyAnatomy] head width {HeadWidth:0.####} ({HeadWidthRatio} x pelvis-to-head "
                  + $"{bodyScale:0.####}), forward {Forward}, up {Up} (levelled {rollFix:0.0} deg "
                  + "from the feet), takeoff lift "
                  + $"{Lift.magnitude / bodyScale:0.0} x pelvis-to-head");
    }

    static bool Airborne(Clip c)
    {
        return c == Clip.Takeoff || c == Clip.Hover || c == Clip.Land || c == Clip.FlyForward
               || c == Clip.FlyTurnLeft || c == Clip.FlyTurnRight;
    }

    /// <summary>
    /// Place the model under Root for the pose just evaluated. The flight
    /// clips carry the pelvis ~2.5 body-heights up. A flight on the 1 key
    /// shows all of that climb, moves the fly along its path and turns and
    /// banks it about its pelvis; the camera pulls back to make room. Any
    /// other flight clip is shifted back by all but flightLiftShown of the
    /// climb, so a camera framed on the fly at rest keeps it in view.
    /// </summary>
    void ApplyModelTransform(Clip clip)
    {
        if (_pelvis == null) return;
        var mt = _model.transform;
        mt.localPosition = _modelRestLocalPos;
        mt.localRotation = _modelRestLocalRot;
        mt.localScale = _modelRestLocalScale;
        ShownLift = Vector3.zero;
        bool airborne = Airborne(clip);
        if (!airborne && !_flight) return;
        if (_flight)
        {
            ApplyFlightTransform();
            return;
        }
        Vector3 d = transform.InverseTransformPoint(_pelvis.position) - _pelvisRest;
        Vector3 comp = -d * (1f - Mathf.Clamp01(Settings.flightLiftShown));
        mt.localPosition = _modelRestLocalPos + comp;
        ShownLift = d + comp;
    }

    void BodyNow(out Vector3 pelvis, out Vector3 head)
    {
        pelvis = transform.InverseTransformPoint(_pelvis.position);
        head = HeadBone != null ? transform.InverseTransformPoint(HeadBone.position) : pelvis + Forward * BodyLength;
    }

    /// <summary>How much of a clip's pose flight steadies: all of the
    /// wingbeat clips, the landing less and less as it touches down, not the
    /// takeoff.</summary>
    float SteadyWeight(Clip c)
    {
        if (c == Clip.Hover || c == Clip.FlyForward || c == Clip.FlyTurnLeft || c == Clip.FlyTurnRight) return 1f;
        if (c == Clip.Land) return 1f - Mathf.SmoothStep(0f, 1f, _landProgress);
        return 0f;
    }

    /// <summary>
    /// The model in flight. The wingbeat clips bob and pitch the body with
    /// every beat, and each carries it at its own height (fly_forward and the
    /// turns well below hover), so blending between them dropped the fly on
    /// the path and popped it back up. Here the body - the centre between
    /// pelvis and head, and the pelvis-to-head direction - is pulled toward
    /// the clip's average over its loop, keeping flightBodyBounce of the
    /// bob, and each clip's average is raised to where takeoff leaves the
    /// pelvis. The landing is held the same way at first, from the hover's
    /// average, and let go as it touches down, so it hands over to the
    /// clip's own descent without a jump. Then the fly is turned and banked
    /// about that centre and moved along its path.
    /// </summary>
    void ApplyFlightTransform()
    {
        var mt = _model.transform;
        BodyNow(out Vector3 pNow, out Vector3 hNow);
        Vector3 cNow = 0.5f * (pNow + hNow);
        Vector3 dirNow = (hNow - pNow).normalized;

        float a = SteadyWeight(_poseFrom) * (1f - _poseW), b = SteadyWeight(_poseTo) * _poseW;
        float strength = Mathf.Clamp01(a + b);
        Quaternion steady = Quaternion.identity;
        Vector3 target = cNow;
        if (strength > 1e-4f)
        {
            int ia = (int)(_poseFrom == Clip.Land ? Clip.Hover : _poseFrom);
            int ib = (int)(_poseTo == Clip.Land ? Clip.Hover : _poseTo);
            float sum = a + b;
            Vector3 meanC = (_airCentre[ia] * a + _airCentre[ib] * b) / sum;
            Vector3 meanP = (_airPelvis[ia] * a + _airPelvis[ib] * b) / sum;
            Vector3 meanD = (_airDir[ia] * a + _airDir[ib] * b).normalized;
            float k = strength * (1f - Mathf.Clamp01(Settings.flightBodyBounce));
            steady = Quaternion.Slerp(Quaternion.identity, Quaternion.FromToRotation(dirNow, meanD), k);
            target = cNow + (meanC - cNow) * k + (_pelvisRest + Lift - meanP) * strength;
        }

        Quaternion turn = FlightRotation() * steady;
        mt.localRotation = turn * _modelRestLocalRot;
        mt.localPosition = target + turn * (_modelRestLocalPos - cNow) + _flightOffset;
        ShownLift = target + turn * (pNow - cNow) + _flightOffset - _pelvisRest;
    }

    Transform FindBone(string name)
    {
        if (string.IsNullOrEmpty(name) || _model == null) return null;
        foreach (var t in _model.GetComponentsInChildren<Transform>(true))
            if (t.name == name) return t;
        return null;
    }

    // ---- visibility / teardown ---------------------------------------------
    public void SetVisible(bool on)
    {
        if (_model != null && _model.activeSelf != on) _model.SetActive(on);
    }

    static void SetLayerRecursive(GameObject go, int layer)
    {
        go.layer = layer;
        for (int i = 0; i < go.transform.childCount; i++)
            SetLayerRecursive(go.transform.GetChild(i).gameObject, layer);
    }

    void OnDestroy()
    {
        if (_graph.IsValid()) _graph.Destroy();
        if (_light != null) Destroy(_light.gameObject);
        foreach (var m in _materials) if (m != null) Destroy(m);
        foreach (var m in _meshes) if (m != null) Destroy(m);
    }
}
