using UnityEngine;
// Active Input Handling is "Input System Package (New)" only, matching
// HudOverlay's toggle-key pattern.
using UnityEngine.InputSystem;
using RosMessageTypes.NiryoMoveit;

/// <summary>
/// Manual curriculum-stage picker: draws one button per curriculum stage so
/// a developer can jump the track straight to any stage's geometry inside
/// the live Unity client, without needing a Python trainer connected or
/// waiting for the CurriculumScheduler to advance naturally. Useful for
/// visually spot-checking each stage's corner-radius/chicane geometry (see
/// TrackGenerator.cornerRadius / chicanesNorth/East/South/West).
///
/// The <see cref="stages"/> list is a Unity-side MIRROR of whichever
/// experiment_designs document's curriculum_stages array the current
/// curriculum job actually uses (see CurriculumScheduler in robotaxi.py) -
/// Unity has no live channel to read the Mongo document, so keep these in
/// sync by hand if the design's stages change (defaults below match
/// "AWAC + No-BC + curriculum (5-stage) + reward_scale10" as of 2026-07-18).
///
/// A button click builds an ApplyForce carrying that stage's track-geometry
/// fields and calls SimController.Restart() directly - the SAME regenerate-
/// track + respawn-car path a real Python-driven RESET uses. This bypasses
/// the ROS reset handshake/cmd_id bookkeeping entirely, so it's meant for
/// use while the trainer is idle/disconnected; clicking a button while a
/// trainer IS actively driving this client will end its in-flight episode
/// abruptly (same as any other unexpected reset).
///
/// Toggle visibility with the C key. Auto-attached by SimController like
/// HudOverlay/TrajectoryRolloutViz - no scene setup required.
/// </summary>
public class CurriculumStageButtons : MonoBehaviour
{
    [System.Serializable]
    public class StagePreset
    {
        public string label = "1";
        public float cornerRadius = 10f;
        public int chicanesNorth = 0;
        public int chicanesEast = 0;
        public int chicanesSouth = 0;
        public int chicanesWest = 0;
    }

    [Tooltip("Mirror of the active experiment design's curriculum_stages (see class doc above). " +
             "Edit in the Inspector to match whichever design/job you're testing against.")]
    public StagePreset[] stages = new StagePreset[]
    {
        new StagePreset { label = "1", cornerRadius = 10f,  chicanesNorth = 0, chicanesEast = 0, chicanesSouth = 0, chicanesWest = 0 },
        new StagePreset { label = "2", cornerRadius = 9f,   chicanesNorth = 0, chicanesEast = 0, chicanesSouth = 1, chicanesWest = 0 },
        new StagePreset { label = "3", cornerRadius = 8f,   chicanesNorth = 0, chicanesEast = 1, chicanesSouth = 1, chicanesWest = 0 },
        new StagePreset { label = "4", cornerRadius = 7.5f, chicanesNorth = 2, chicanesEast = 1, chicanesSouth = 2, chicanesWest = 1 },
        new StagePreset { label = "5", cornerRadius = 7.5f, chicanesNorth = 3, chicanesEast = 1, chicanesSouth = 3, chicanesWest = 1 },
    };

    [Header("Placement (top-right, to the right of the HUD)")]
    public float rightMargin = 12f;
    public float topMargin = 12f;
    public float buttonWidth = 120f;
    public float buttonHeight = 32f;
    public float spacing = 4f;
    [Tooltip("Title label ('curriculum stage (C to hide)') is wider than a single "
             + "button, so the panel's anchored width is the max of this and "
             + "buttonWidth - keeps the whole panel's right edge flush with "
             + "rightMargin regardless of which line is longest.")]
    public float titleWidth = 220f;

    private bool _panelOn = true;
    // -1 = no manual switch clicked yet this session (a real running
    // CurriculumScheduler may be at any stage - we don't try to read it
    // back, so this only tracks button-driven switches).
    private int _activeStage = -1;
    private GUIStyle _buttonStyle;
    private GUIStyle _activeButtonStyle;
    private GUIStyle _titleStyle;
    private bool _assetsReady = false;

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null && kb.cKey.wasPressedThisFrame)
            _panelOn = !_panelOn;
    }

    void EnsureAssets()
    {
        if (_assetsReady) return;
        _buttonStyle = new GUIStyle(GUI.skin.button) { fontSize = 14 };
        _activeButtonStyle = new GUIStyle(GUI.skin.button) { fontSize = 14, fontStyle = FontStyle.Bold };
        _activeButtonStyle.normal.textColor = new Color(0.55f, 1f, 0.65f);
        _activeButtonStyle.hover.textColor = new Color(0.55f, 1f, 0.65f);
        _titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 12,
            fontStyle = FontStyle.Bold,
        };
        _titleStyle.normal.textColor = new Color(0.8f, 0.85f, 0.95f);
        _assetsReady = true;
    }

    void OnGUI()
    {
        if (stages == null || stages.Length == 0) return;
        EnsureAssets();

        // Anchor to the screen's right edge (not a fixed pixel x) so the
        // panel stays pinned there across resolutions, sitting to the right
        // of the HUD's screen-center panel (see HudOverlay.panelWidth).
        float panelW = Mathf.Max(titleWidth, buttonWidth);
        float x = Screen.width - rightMargin - panelW;

        float y = topMargin;
        GUI.Label(new Rect(x, y, titleWidth, 18f),
                  _panelOn ? "curriculum stage (C to hide)" : "(C to show stages)",
                  _titleStyle);
        if (!_panelOn) return;
        y += 20f;

        for (int i = 0; i < stages.Length; i++)
        {
            var s = stages[i];
            var style = (i == _activeStage) ? _activeButtonStyle : _buttonStyle;
            string label = "Stage " + s.label;
            if (GUI.Button(new Rect(x, y, buttonWidth, buttonHeight), label, style))
            {
                ApplyStage(i);
            }
            y += buttonHeight + spacing;
        }
    }

    void ApplyStage(int index)
    {
        if (stages == null || index < 0 || index >= stages.Length) return;
        var s = stages[index];
        var sim = SimController.instance;
        if (sim == null)
        {
            Debug.LogWarning("[CurriculumStageButtons] SimController.instance is null; cannot switch stage.");
            return;
        }
        var af = new ApplyForce();
        af.num_obstacles = 0;
        // corner_radius must be > 0 - SimController.ApplyTrackConfig treats
        // <= 0 as "message carried no track params" and skips regeneration.
        // All real stage presets use positive radii, so this is only a
        // concern if someone edits the Inspector array to 0.
        af.corner_radius = s.cornerRadius;
        af.curvature_difficulty = 0.0; // deprecated field - chicane counts below drive geometry
        af.chicanes_north = s.chicanesNorth;
        af.chicanes_east = s.chicanesEast;
        af.chicanes_south = s.chicanesSouth;
        af.chicanes_west = s.chicanesWest;
        Debug.Log($"[CurriculumStageButtons] Manually switching to stage {s.label}: " +
                  $"cornerRadius={s.cornerRadius}, chicanes(N/E/S/W)=" +
                  $"{s.chicanesNorth}/{s.chicanesEast}/{s.chicanesSouth}/{s.chicanesWest}");
        sim.Restart(af);
        _activeStage = index;
    }
}
