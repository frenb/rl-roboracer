using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Everything tunable about the fly anatomy view (FlyBrainViz, N key): how
/// see-through the fly is, the animation schedule, and where the connectome
/// sits inside the body.
///
/// A ScriptableObject in Resources rather than inspector fields, because
/// FlyBrainViz is added by SimController at runtime - edits to a runtime
/// component vanish when Play stops, while edits to this asset are saved with
/// the project. Every value is read every frame, so dragging a slider on
/// Assets/Resources/FruitFly/FlyAnatomySettings during Play shows up at once.
/// </summary>
[CreateAssetMenu(fileName = "FlyAnatomySettings", menuName = "RL Roboracer/Fly Anatomy Settings")]
public class FlyAnatomySettings : ScriptableObject
{
    public const string ResourcePath = "FruitFly/FlyAnatomySettings";

    [Header("Body transparency")]
    [Tooltip("Opacity across the whole body surface. The rim below adds to it "
             + "at glancing angles, so this can go very low and the outline "
             + "still reads.")]
    [Range(0f, 1f)] public float bodyOpacity = 0.10f;
    public Color bodyTint = new Color(0.75f, 0.85f, 1.00f, 1f);
    [Tooltip("How much of the model's own albedo texture shows through the tint.")]
    [Range(0f, 1f)] public float bodyTextureStrength = 0.35f;
    [Tooltip("Extra opacity where the surface turns edge-on to the camera.")]
    [Range(0f, 1f)] public float bodyRimOpacity = 0.45f;
    public Color rimColor = new Color(0.60f, 0.85f, 1.00f, 1f);
    [Tooltip("Higher narrows the rim to the very edge of the silhouette.")]
    [Range(0.5f, 8f)] public float rimPower = 2.5f;
    [Header("Eyes")]
    [Tooltip("Kept faint: the eyes sit either side of the optic lobes, and a "
             + "strong red there drowns the brain's own colours.")]
    [Range(0f, 1f)] public float eyeOpacity = 0.05f;
    public Color eyeTint = new Color(0.70f, 0.30f, 0.26f, 1f);
    [Range(0f, 1f)] public float eyeRimOpacity = 0.12f;

    [Header("Wings")]
    [Range(0f, 1f)] public float wingOpacity = 0.04f;
    public Color wingTint = new Color(0.80f, 0.90f, 1.00f, 1f);
    [Range(0f, 1f)] public float wingRimOpacity = 0.20f;

    [Header("Hair")]
    [Tooltip("Thousands of overlapping transparent strands; off by default "
             + "because they muddy the view of the brain.")]
    public bool showHair = false;
    [Range(0f, 1f)] public float hairOpacity = 0.05f;

    [Header("Performance")]
    [Tooltip("How often the fly's animation, and the neurons riding it, are "
             + "updated, per second of real time. The sim renders as fast as it "
             + "can and the policy waits on its frames, so every frame spent "
             + "posing the fly costs the car reaction time. 0 updates every "
             + "frame.")]
    [Range(0f, 120f)] public float animationHz = 30f;
    [Tooltip("How often the fly views (second and third N press) are redrawn, "
             + "per second; 0 redraws every frame. In between, the last drawing "
             + "is copied to the screen, so the sim's frames stay as short as "
             + "with the plain connectome. Frame length matters: the sim runs at "
             + "Time.timeScale 3 and the car's commands and observations only "
             + "move between frames. Dragging the fly redraws every frame.")]
    [Range(0f, 60f)] public float overlayRenderHz = 10f;

    [Header("Textured fly (Y key)")]
    [Tooltip("Y swaps the see-through shell for the model's own textured "
             + "materials in both fly views. Brightness of the light added "
             + "for it, which lights the fly and nothing else.")]
    [Range(0f, 3f)] public float texturedLightIntensity = 1.1f;
    [Tooltip("Show the hair strands on the textured fly. Off by default: "
             + "thousands of strands are expensive to draw.")]
    public bool texturedShowHair = false;

    [Header("Animation schedule")]
    [Tooltip("The loop: the fly stands still, then grooms (idle_look, front "
             + "legs rubbing), and repeats. Each still stretch lasts its own "
             + "random time between these two, in seconds.")]
    [Min(0f)] public float idleMinSeconds = 30f;
    [Min(0f)] public float idleMaxSeconds = 60f;
    [Tooltip("Seconds of grooming, rounded to whole cycles of the clip.")]
    [Min(0f)] public float groomSeconds = 5f;
    [Tooltip("Blend into and out of grooming, front legs raised, from and to "
             + "the standing pose, seconds.")]
    [Range(0f, 2f)] public float groomBlendSeconds = 0.5f;

    [Header("View")]
    [Tooltip("Screen area the fly fills, as fractions of the window: x, y from "
             + "the bottom-left, then width and height. The default is the space "
             + "left of the track. - and = scale it about its centre.")]
    public Rect viewport = new Rect(0f, 0.029f, 0.477f, 0.922f);
    [Tooltip("Share of the takeoff climb that shows on screen. The camera frames "
             + "the fly at rest, so at 1 the fly climbs out of the top of its "
             + "column; at 0 it flaps in place.")]
    [Range(0f, 1f)] public float flightLiftShown = 0.25f;
    [Tooltip("Slow turntable, degrees per second (real time). 0 holds still.")]
    public float spinDegreesPerSecond = 0f;
    [Tooltip("Keep the fly view up when no fly policy is publishing, showing "
             + "the last geometry received (cached on disk) at rest.")]
    public bool showWithoutActivity = true;
    [Tooltip("Editor only: with no fly job publishing, make random neurons "
             + "fire so the glow can be tuned in Play mode. Never in a build; "
             + "real activity takes over as soon as it arrives.")]
    public bool editorTestActivity = true;
    [Tooltip("Share of neurons set firing every 0.1 s by the editor test, "
             + "besides occasional bursts.")]
    [Range(0f, 0.05f)] public float editorTestFiringFraction = 0.004f;

    [Header("Fly view only (second N press)")]
    [Tooltip("Turn of the fly about its own vertical, degrees. 90 faces the "
             + "viewer head-on; 0 is side-on, head to the right. Left/Right "
             + "arrows adjust it while this view is up.")]
    public float viewYaw = 90f;
    [Tooltip("How far above the fly the camera looks down from, degrees. "
             + "Up/Down arrows.")]
    [Range(-89f, 89f)] public float viewElevation = 15f;
    [Tooltip("Tilt of the fly in the screen plane, degrees, positive "
             + "counter-clockwise. The fly is levelled from its own left-right "
             + "symmetry; this trims whatever is left.")]
    [Range(-45f, 45f)] public float viewRoll = 0f;
    [Tooltip("Nose-down tilt of the whole fly about its own left-right axis, "
             + "degrees; negative lifts the head.")]
    [Range(-45f, 45f)] public float viewBodyPitch = 0f;
    [Tooltip("Opacity of the head's shell (everything from the neck forward, "
             + "eyes excepted) relative to the rest of the body. 1 leaves it as "
             + "the body; lower makes the head, and the brain inside it, "
             + "clearer. The close-up has its own value.")]
    [Range(0f, 1f)] public float headOpacity = 1f;
    [Tooltip("Brightness of the connectome in this view, as a multiple of "
             + "Glow Gain. 1 matches the other views.")]
    [Range(0.25f, 4f)] public float flyViewBrainBrightness = 1f;

    [Header("Brain close-up view (third N press)")]
    [Tooltip("Screen area the head fills and is centred in, as fractions of the "
             + "window: x, y from the bottom-left, then width and height. The "
             + "track keeps the fly view's layout; the rest of the fly runs off "
             + "the left of the screen and behind the track.")]
    public Rect closeUpHeadBox = new Rect(0.04f, 0.167f, 0.366f, 0.585f);
    [Tooltip("Head size relative to that box: 1 touches its tighter pair of "
             + "sides, above 1 overflows it.")]
    [Range(0.3f, 3f)] public float closeUpHeadFill = 0.67f;
    [Tooltip("Head bent down at the neck, degrees.")]
    [Range(-45f, 45f)] public float closeUpHeadPitch = 15f;
    [Tooltip("This view's own turn of the fly about its vertical, degrees, "
             + "independent of the fly view's. 90 faces the viewer head-on; 270 "
             + "shows it from behind. Left/Right arrows while this view is up.")]
    public float closeUpViewYaw = 90f;
    [Tooltip("How far above the fly the camera looks down from, degrees. "
             + "Up/Down arrows.")]
    [Range(-89f, 89f)] public float closeUpViewElevation = 15f;
    [Tooltip("Tilt of the fly in the screen plane, degrees, positive "
             + "counter-clockwise.")]
    [Range(-45f, 45f)] public float closeUpViewRoll = 0f;
    [Tooltip("Nose-down tilt of the whole fly, degrees, about its left-right "
             + "axis. The camera holds the head in closeUpHeadBox, so the "
             + "abdomen is what rises.")]
    [Range(-45f, 45f)] public float closeUpBodyPitch = 10f;
    [Tooltip("Opacity of the body behind the head, relative to the fly view "
             + "(0.75 = 25% more see-through). Head, eyes, front legs and wings "
             + "are unchanged.")]
    [Range(0f, 1f)] public float closeUpRearOpacity = 0.75f;
    [Tooltip("Head Opacity for the close-up: the head's shell, eyes excepted, "
             + "relative to the rest of the body.")]
    [Range(0f, 1f)] public float closeUpHeadOpacity = 1f;

    [Header("Neuron glow (all three views)")]
    [Tooltip("Draw the neurons as soft spots added to what is behind them, so "
             + "busy regions bloom. Off falls back to the plain blended squares. "
             + "Applies to the brain-only view too.")]
    public bool glow = true;
    [Tooltip("Overall glow strength. Lower it if the brain washes out to white.")]
    [Range(0f, 4f)] public float glowGain = 0.9f;
    [Tooltip("Resting (silent) neuron brightness, as a multiple of the "
             + "unlit look's.")]
    [Range(1f, 6f)] public float glowRestGain = 2.5f;
    [Tooltip("Weight of a silent neuron in the glow; a fully firing one is 1.")]
    [Range(0f, 1f)] public float glowRestAlpha = 0.25f;
    [Tooltip("Neuron size under the glow in the brain-only and fly views. The "
             + "spots fade to their edge, so they need to be bigger than the "
             + "squares they replace to cover the same ground.")]
    [Range(0.5f, 4f)] public float glowPointScale = 1.6f;

    [Header("Brain close-up: making the brain stand out")]
    [Tooltip("Neuron size relative to the fly view.")]
    [Range(0.5f, 4f)] public float closeUpPointScale = 2.5f;
    [Tooltip("Brain size relative to the fly view, about its own centre, so "
             + "the eyes either side cover less of the optic lobes. The nerve "
             + "cord is unchanged.")]
    [Range(0.4f, 1.2f)] public float closeUpBrainScale = 0.85f;

    [Tooltip("Dark oval inside the head, behind the brain, hiding the body "
             + "behind it.")]
    public bool closeUpBackdrop = true;
    [Range(0f, 1f)] public float closeUpBackdropOpacity = 0.85f;
    [Tooltip("Oval size relative to the brain's outline on screen.")]
    [Range(0.8f, 2f)] public float closeUpBackdropPadding = 1.25f;
    [Tooltip("Share of the oval's radius that fades out at its edge.")]
    [Range(0.05f, 1f)] public float closeUpBackdropSoftness = 0.4f;

    [Tooltip("Body fill opacity relative to the fly view; near 0 leaves only "
             + "the outline (rim).")]
    [Range(0f, 1f)] public float closeUpShellFill = 0.1f;
    [Tooltip("Opacity of the mouthparts (proboscis) relative to the rest of "
             + "the head, which sit right in front of the brain in this view.")]
    [Range(0f, 1f)] public float closeUpMouthOpacity = 0.1f;
    [Tooltip("Root of the bone branch the mouthparts are skinned to.")]
    public string mouthBone = "Tongue01";
    public Color closeUpShellTint = new Color(0.30f, 0.36f, 0.46f, 1f);
    public Color closeUpRimColor = new Color(0.35f, 0.50f, 0.65f, 1f);
    [Tooltip("Bone the close-up bends the head at. Eyes and head hair are "
             + "skinned to it; the head bone is its child.")]
    public string neckBone = "Neck";

    [Header("Connectome placement")]
    [Tooltip("Width of the brain (optic lobes included) as a fraction of the "
             + "head's width across the eyes. Sets the scale of the whole CNS.")]
    [Range(0.3f, 1.2f)] public float brainWidthFraction = 0.82f;
    [Tooltip("Nudge the whole CNS, in head widths: x right, y up, z forward.")]
    public Vector3 brainOffset = Vector3.zero;
    [Tooltip("Nudge the ventral nerve cord only, in head widths.")]
    public Vector3 vncOffset = Vector3.zero;
    [Tooltip("How the published CNS is laid out. Auto decides from the brain's "
             + "proportions: Straightened is the dissected pose with the cord "
             + "hanging below the brain, which is bent back 90 degrees at the "
             + "neck to sit in the thorax.")]
    public CnsLayout layout = CnsLayout.Auto;
    [Tooltip("Bend at the neck for a straightened CNS, degrees.")]
    [Range(0f, 120f)] public float neckBendDegrees = 90f;
    public bool flipFrontBack = false;
    public bool flipLeftRight = false;
    [Tooltip("Neuron size relative to the connectome-only view.")]
    [Range(0.1f, 4f)] public float pointScale = 1f;
    public string headBone = "HeadLock";
    public string thoraxBone = "Spine02";

    public enum CnsLayout { Auto, Straightened, Natural }

    /// <summary>Changes whenever a placement value does, so the per-neuron
    /// layout is only rebuilt when it has to be.</summary>
    public int PlacementHash()
    {
        unchecked
        {
            int h = 17;
            h = h * 31 + brainWidthFraction.GetHashCode();
            h = h * 31 + brainOffset.GetHashCode();
            h = h * 31 + vncOffset.GetHashCode();
            h = h * 31 + (int)layout;
            h = h * 31 + neckBendDegrees.GetHashCode();
            h = h * 31 + (flipFrontBack ? 1 : 0);
            h = h * 31 + (flipLeftRight ? 2 : 0);
            h = h * 31 + (headBone ?? "").GetHashCode();
            h = h * 31 + (thoraxBone ?? "").GetHashCode();
            return h;
        }
    }

    public static FlyAnatomySettings Load()
    {
        var s = Resources.Load<FlyAnatomySettings>(ResourcePath);
        if (s != null) return s;
        Debug.LogWarning($"[FlyAnatomy] no settings asset at Resources/{ResourcePath}; "
                         + "using defaults (edits will not persist)");
        return CreateInstance<FlyAnatomySettings>();
    }

    /// <summary>Record a runtime (keyboard) change so the editor saves it.</summary>
    public void MarkChanged()
    {
#if UNITY_EDITOR
        EditorUtility.SetDirty(this);
#endif
    }
}
