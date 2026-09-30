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
/// for a random stretch, then idle_look (cleaning hands), and around again.
/// Everything runs on unscaled time, because the sim runs at Time.timeScale
/// 3-5 and the fly should not.
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

    enum Clip { Idle, IdleLook, Takeoff, Hover, Land }
    static readonly string[] ClipKeys = { "idle", "idle_look", "takeoff", "hover", "land" };

    struct Step
    {
        public Clip clip;
        public int repeats;
        public float fade;      // blend into this step, seconds
        public float hold;      // > 0: stand still on the clip's first frame this long instead
        public string label;
        public Step(Clip c, int r, float f, string l, float h = 0f)
        { clip = c; repeats = r; fade = f; label = l; hold = h; }
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
    private readonly AnimationClipPlayable[] _clipPlayables = new AnimationClipPlayable[5];
    private readonly float[] _clipLength = new float[5];
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
            foreach (var c in clips)
                if (!c.name.StartsWith("__preview__") && c.name.EndsWith("FruitFly_" + ClipKeys[k]))
                    clip = c;
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
        return true;
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
        if (hz > 0f && _lastPose >= 0f && now - _lastPose < 1f / hz) return;
        float dt = _lastPose >= 0f ? now - _lastPose : 0f;
        _lastPose = now;
        bool moved = AdvanceSchedule(now);
        if (moved) ApplyLiftCompensation(_queue[_stepIndex].clip);
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
    bool AdvanceSchedule(float now)
    {
        bool started = false;
        if (_stepIndex < 0)
        {
            BuildCycle();
            StartStep(0, now, Clip.Idle, 0f);
            started = true;
        }

        Step step = _queue[_stepIndex];
        float len = _clipLength[(int)step.clip];
        float t = now - _stepStart;
        if (t >= StepSeconds(step))
        {
            // A looping clip ends on its last frame, which is also its first.
            float endTime = step.hold > 0f ? 0f : len;
            int next = _stepIndex + 1;
            if (next >= _queue.Count)
            {
                BuildCycle();     // picks up any settings changed mid-cycle
                next = 0;
            }
            StartStep(next, now, step.clip, endTime);
            step = _queue[_stepIndex];
            len = _clipLength[(int)step.clip];
            t = 0f;
            started = true;
        }

        float w = (!_fading || step.fade <= 0f) ? 1f : Mathf.Clamp01(t / step.fade);
        bool wasFading = _fading;
        if (w >= 1f) _fading = false;
        if (step.hold > 0f)
        {
            if (!started && !wasFading) return false;
            Pose(step.clip, 0f, _fading ? _fadeFrom : step.clip, _fadeFromTime, w);
            return true;
        }
        float clipTime = step.repeats > 1 ? Mathf.Repeat(t, len) : Mathf.Min(t, len);
        Pose(step.clip, clipTime, _fading ? _fadeFrom : step.clip, _fadeFromTime, w);
        return true;
    }

    float StepSeconds(Step s)
    {
        return s.hold > 0f ? s.hold : _clipLength[(int)s.clip] * s.repeats;
    }

    /// <summary>Whole loops of a looping clip closest to a wanted duration, so
    /// it always stops on its first frame.</summary>
    int Loops(Clip c, float seconds)
    {
        return Mathf.Max(1, Mathf.RoundToInt(seconds / _clipLength[(int)c]));
    }

    /// <summary>
    /// The fixed cycle: stand still, groom, and around again. Grooming holds
    /// the front legs up on every frame, first and last included, so it
    /// blends in and out of the standing pose at a pace the legs can
    /// plausibly move.
    /// </summary>
    void BuildCycle()
    {
        var s = Settings;
        float legs = Mathf.Max(0f, s.groomBlendSeconds);
        _queue.Clear();
        _queue.Add(new Step(Clip.Idle, 1, legs, "standing still", Mathf.Max(0.1f, StillSeconds())));
        _queue.Add(new Step(Clip.IdleLook, Loops(Clip.IdleLook, s.groomSeconds), legs, "idle_look (cleaning hands)"));
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
        _modelRestLocalPos = _model.transform.localPosition;

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

        Pose(Clip.Idle, 0f);
        Debug.Log($"[FlyAnatomy] head width {HeadWidth:0.####} ({HeadWidthRatio} x pelvis-to-head "
                  + $"{bodyScale:0.####}), forward {Forward}, up {Up} (levelled {rollFix:0.0} deg "
                  + "from the feet), takeoff lift "
                  + $"{Lift.magnitude / bodyScale:0.0} x pelvis-to-head");
    }

    /// <summary>
    /// Keep the body in frame while it flies. The flight clips carry the
    /// pelvis ~2.5 body-heights up; shifting the model back by all but
    /// flightLiftShown of that lets the camera frame the fly at rest and
    /// still show a hint of the climb. Idle clips are left alone.
    /// </summary>
    void ApplyLiftCompensation(Clip clip)
    {
        if (_pelvis == null) return;
        _model.transform.localPosition = _modelRestLocalPos;
        ShownLift = Vector3.zero;
        if (clip == Clip.Idle || clip == Clip.IdleLook) return;
        Vector3 d = transform.InverseTransformPoint(_pelvis.position) - _pelvisRest;
        float keep = 1f - Mathf.Clamp01(Settings.flightLiftShown);
        _model.transform.localPosition = _modelRestLocalPos - d * keep;
        ShownLift = d * (1f - keep);
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
