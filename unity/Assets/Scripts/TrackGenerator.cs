using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Phase-1 spike: procedurally generate a drivable closed-loop racetrack
/// from the Race Track Construction Kit's modular 20m tiles, with a single
/// "curvature" difficulty knob, and emit ordered Goal-N triggers that the
/// existing CarController/SimController pipeline consumes UNCHANGED.
///
/// Why this works with zero Python-side changes
/// --------------------------------------------
///   * CarController.SetUpGoalsArray() finds goals purely by the name
///     convention Goal-1..Goal-N via GameObject.Find. We just have to
///     create objects with those exact names + a trigger collider + a
///     Goal component (SimController sets goal.GetComponent<Goal>()...).
///   * The agent never receives track geometry over ROS - only raycasts +
///     distance/angle to the next goal. So as long as the track has
///     collider walls on the raycast layer (named "Curb" so crash
///     detection fires) and correctly-named goals, observations + rewards
///     are layout-agnostic.
///   * Goals/track are STATIC across resets (SimController only recreates
///     the car). Generating once in Awake() - which runs before any
///     Start() - guarantees goals exist before SimController.Start() ->
///     InstantiateObjects -> SetUpGoalsArray.
///
/// Robustness to prefab-orientation unknowns
/// -----------------------------------------
/// The kit tiles' pivot/default-facing aren't assumed: straight + curve
/// yaw offsets are exposed in the Inspector so you can calibrate visually
/// without recompiling. Left/right turns use the kit's Curve + Curve_Flipped
/// pair. Walls are generated independently (simple boxes) so raycasts +
/// crashes work regardless of whether the kit road meshes carry their own
/// colliders.
///
/// Spike scope: difficulty is an Inspector field (no ROS/curriculum wiring
/// yet). Once this proves drivable, Phase 2 adds the curriculum schedule +
/// SCHEMA knobs and Phase 3 points the MadScientist researcher at it.
/// </summary>
public class TrackGenerator : MonoBehaviour
{
    [Header("Visual road surface")]
    [Tooltip("FALSE (default, spike): lay clean flat road quads that perfectly fill the walled " +
             "corridor - guaranteed drivable + visually clean, no orientation calibration. " +
             "TRUE: use the kit's straight/curve prefabs (prettier, but curve orientation needs " +
             "calibrating via curveYawOffset / the flipped prefab).")]
    public bool useKitTiles = false;
    [Tooltip("Colour of the generated flat road quads (when useKitTiles = false).")]
    public Color roadColor = new Color(0.18f, 0.18f, 0.2f);
    [Tooltip("Colour of the merged straight guard-rail cubes.")]
    public Color railColor = new Color(0.55f, 0.55f, 0.58f);

    [Header("Tile prefabs (assign from Race_Track_Construction_Kit/Prefabs)")]
    [Tooltip("A straight road tile, e.g. Track_Straight_Mobile.")]
    public GameObject straightPrefab;
    [Tooltip("A 90-degree curve tile, e.g. Track_Curve_A. Used for one turn handedness.")]
    public GameObject curvePrefab;
    [Tooltip("The mirrored curve, e.g. Track_Curve_A_Flipped. Used for the other handedness. " +
             "If left null, curvePrefab is reused (you can fix handedness via curveYawOffset).")]
    public GameObject curveFlippedPrefab;
    [Tooltip("Optional goal prefab (e.g. the cylindrical Goal.prefab from Assets/Scenes/Goal.prefab). " +
             "When assigned, PlaceGoal instantiates this prefab instead of creating a flat cube gate. " +
             "The prefab must have a Collider (set to isTrigger) and will have a Goal component " +
             "added automatically if missing. Leave empty to keep the default cube gate.")]
    public GameObject goalPrefab;

    [Header("Tile geometry / calibration")]
    // Raised from 20 -> 24 (TrackGen-v7) specifically to widen the cornerRadius
    // clamp ceiling (tileSize/2, see cornerRadius tooltip below) from 10 to 12,
    // so the 5-stage curriculum's radius progression isn't squeezed into a
    // ~2.5m band. This also increases the loop's overall footprint by 20%
    // (same loopWidthTiles/loopHeightTiles tile COUNT, bigger tiles) - forward-
    // clearance-based DEMO-driving thresholds tuned in collect_expert_demos()
    // against the old 20m tiles (e.g. FORWARD_CLEAR_REF=26.0) may read
    // proportionally "less urgent" now and should be re-checked/rescaled.
    [Tooltip("Edge length of one square tile in metres. Kit pieces are 20m; " +
             "raised to 24m (TrackGen-v7) to widen the cornerRadius clamp ceiling.")]
    public float tileSize = 24f;
    [Tooltip("Yaw added to straight tiles after aligning to heading. Calibrate so the road points along travel.")]
    public float straightYawOffset = 0f;
    [Tooltip("Yaw added to curve tiles after the entry-edge alignment. Calibrate so curves connect cleanly.")]
    public float curveYawOffset = 0f;
    [Tooltip("Vertical Y the track + goals are placed at.")]
    public float trackY = 0f;

    [Header("Rounded corners (procedural arcs)")]
    [Tooltip("Replace sharp 90-degree corners with a quarter-circle arc (road + inner/outer Curb " +
             "walls following the curve). Only applies in flat-geometry mode (Use Kit Tiles = false).")]
    public bool roundedCorners = true;
    [Tooltip("Centreline turn radius (m). Clamp range is [roadHalfWidth+0.5, tileSize/2]. " +
             "Smaller = tighter/harder corner; this is the curvature-difficulty lever.")]
    public float cornerRadius = 10f;
    // Chicane corners (the 4-turn bump inserted per-edge, see
    // BuildEdgeWithChicanes) are spaced only 1 tile apart - stacking 4
    // turns at whatever the curriculum's cornerRadius happens to be (down to
    // 7.5 at the tightest stages) left too little margin and was confirmed
    // (2026-07-18) unnavigable. Decoupled from cornerRadius so chicane
    // difficulty comes from the per-edge chicane counts (turn frequency)
    // rather than compounding with the main corner-radius curriculum axis.
    // Same clamp range as cornerRadius applies (see PlaceRoundedCorner/PlaceGoal).
    [Tooltip("Centreline turn radius (m) used ONLY for chicane corners, independent of cornerRadius. " +
             "Chicane corners are spaced close together, so keep this generous (near tileSize/2) " +
             "regardless of what cornerRadius the curriculum is currently using.")]
    public float chicaneCornerRadius = 10f;
    [Tooltip("Arc facets per corner (2 = two flat chord walls, 5 = smooth enough for raycasts). " +
             "Lower values reduce object count significantly: 5 facets × 2 walls × 4 corners = 40 objects " +
             "vs 80 at 10 facets. Invisible by default (showArcWallGeometry = false) so count is cosmetic, " +
             "but fewer facets also means fewer physics broadphase entries.")]
    public int cornerFacets = 5;
    [Tooltip("Extra length added to each road/wall segment so adjacent facets overlap and rays " +
             "never slip through a seam.")]
    public float segmentOverlap = 0.4f;

    [Header("Loop shape + difficulty")]
    [Tooltip("Base rectangle width in tiles (>=3).")]
    public int loopWidthTiles = 7;
    // Raised 5 -> 8 (TrackGen-v7) so the west/east ends (the SHORT vertical
    // edges, length = loopHeightTiles) have enough straight cells to survive
    // the corner-adjacency goal skip (see the goal-placement block in
    // Generate()) with 4 goals left over per side, not just 1. Each vertical
    // edge has loopHeightTiles-2 straight cells; the 2 flanking the corners
    // at each end are skipped, so goals-per-side = loopHeightTiles-4.
    [Tooltip("Base rectangle height in tiles (>=3).")]
    public int loopHeightTiles = 8;
    // Legacy/logging only as of 2026-07-18 - no longer drives chicane count
    // (see chicanesNorth/East/South/West below). Kept so old TensorBoard
    // curriculum/curvature_difficulty logging and the ApplyForce wire field
    // stay meaningful for anyone still reading it; TrackGenerator itself
    // ignores this value now.
    [Range(0f, 1f)]
    [Tooltip("DEPRECATED - no longer drives chicane count, see chicanesNorth/East/South/West. " +
             "Kept only for logging/back-compat.")]
    public float curvatureDifficulty = 0f;
    // Per-edge ABSOLUTE chicane counts (2026-07-18), replacing the single
    // curvatureDifficulty 0-1 knob (which only ever applied to the north/top
    // edge). Lets a curriculum stage/experiment design place a specific,
    // deterministic number of chicanes on each of the 4 edges independently
    // (e.g. "stage 3 has 1 chicane on the south, 1 on the east, none on
    // north/west") instead of a single difficulty ratio applied to one edge.
    [Header("Chicanes (per-edge absolute counts)")]
    [Tooltip("Number of chicane bumps on the NORTH edge (the top edge, larger Z / larger grid Y).")]
    public int chicanesNorth = 0;
    [Tooltip("Number of chicane bumps on the EAST edge (the right edge, larger X).")]
    public int chicanesEast = 0;
    [Tooltip("Number of chicane bumps on the SOUTH edge (the bottom edge, Z=0 / grid Y=0).")]
    public int chicanesSouth = 0;
    [Tooltip("Number of chicane bumps on the WEST edge (the left edge, X=0).")]
    public int chicanesWest = 0;
    [Tooltip("Deterministic seed for chicane placement.")]
    public int seed = 12345;

    [Header("Goals (built as a gate ACROSS the road)")]
    [Tooltip("Place a Goal-N every this many tiles along the path.")]
    public int goalEveryNTiles = 1;
    [Tooltip("Place goal gate(s) along each rounded corner's arc. The gate's " +
             "inner end is flush with the outer face of the inner rail (the " +
             "Inner Point of Tangency) and its longitudinal axis is the " +
             "radial normal there (perpendicular to the arc tangent = " +
             "arc-centre-outward) at each gate's own position along the arc.")]
    public bool placeCornerTangentGoals = true;
    [Tooltip("How many goal gates to place along EACH corner's arc sweep " +
             "(evenly spaced, e.g. 3 -> 25%/50%/75% of the sweep), instead " +
             "of just one at the 45-degree apex. Added 2026-07-19: a single " +
             "apex goal is a single pursuit POINT, so a car steering " +
             "straight at it (rather than following the curve) can cut the " +
             "corner and run off the outer/inner pavement edge before or " +
             "after reaching it - more, closer waypoints make the pursuit " +
             "target track the actual curve instead of chording across it. " +
             "1 reproduces the old single-apex-goal behaviour exactly.")]
    public int goalsPerCorner = 3;
    [Tooltip("Height above the road the goal gate's CENTER sits at.")]
    public float goalHeight = 1.5f;
    // Was 14 (matched the old corridor 2*7). roadHalfWidth dropped to 5.5
    // (TrackGen-v7, see roadHalfWidth comment above) -> corridor is now
    // 2*5.5=11, so this follows suit to stay flush with the new road edges
    // instead of overhanging into the walls.
    [Tooltip("Gate width - spans ACROSS the road (make it ~ the corridor width 2*roadHalfWidth).")]
    public float goalGateWidth = 11f;
    [Tooltip("Gate vertical extent - tall enough to always catch the car driving through.")]
    public float goalTriggerHeight = 4f;
    [Tooltip("Gate thickness ALONG the road (thin, so it reads as a gate line).")]
    public float goalGateThickness = 1f;
    [Tooltip("Colour of the goal gate so it's clearly visible.")]
    public Color goalColor = new Color(1f, 0.85f, 0.1f, 1f);
    [Tooltip("Yaw added after aligning the gate across the local travel direction. " +
             "180 spans the road correctly here; tweak only if calibrating.")]
    public float goalYawOffset = 180f;

    [Header("Ground - guarantees a driving surface regardless of kit-tile colliders")]
    [Tooltip("Lay a flat ground box under the whole loop so the car always has something to drive on. " +
             "Leave on unless the kit road prefabs already provide drivable mesh colliders.")]
    public bool buildGround = true;
    [Tooltip("Extra margin (tiles) around the loop bounding box for the ground.")]
    public float groundMarginTiles = 1.5f;

    [Header("Walls (curbs) - make the track raycast-visible + crashable")]
    public bool buildWalls = true;
    [Tooltip("Show the visual geometry only for the outer rails (the side of the road facing away from " +
             "the loop centre). Inner rails are kept as invisible colliders so raycasts and crash detection " +
             "still work. ON by default for a cleaner look - turn OFF to show all rails.")]
    public bool outerWallsOnlyVisible = true;
    [Tooltip("Layer for wall colliders. MUST be included in CarController.layerMask so raycasts see them. " +
             "CarController.layerMask is baked as bit=1 (Default only) in the car prefab, so walls must be " +
             "on Default for SphereCast to detect them. Crash detection uses the object NAME 'Curb'/'Rail', " +
             "not the layer, so Default is safe here.")]
    public string wallLayerName = "Default";
    [Tooltip("Show the visual mesh on arc wall segments (the short faceted pieces at corners). " +
             "OFF (default): arc walls are collider-only - invisible but still raycast-detectable and " +
             "crash-triggering. The merged straight rails remain visible. Turning this ON reveals the " +
             "tick-mark arc segments at corners, which is useful for debugging arc geometry.")]
    public bool showArcWallGeometry = false;
    // Lowered from 7 -> 5.5 (TrackGen-v7) to widen the cornerRadius clamp
    // FLOOR (roadHalfWidth+0.5, see cornerRadius tooltip below) from 7.5 to
    // 6.0, so tight-corner curriculum stages are actually achievable instead
    // of clamping up to the same 7.5 as every other "tight" stage. Narrows
    // the drivable corridor 14m -> 11m everywhere (not just corners) - the
    // DEMO driver's corner-urgency/clearance signals are ratio-based
    // (see curvature_asymmetry in collect_expert_demos()) so they should
    // degrade gracefully, but absolute clearance readings will run smaller.
    [Tooltip("Half the drivable corridor width. Walls sit at +/- this from tile centre. Keep < tileSize/2.")]
    public float roadHalfWidth = 5.5f;
    public float wallHeight = 2f;
    public float wallThickness = 0.5f;

    [Header("Lifecycle")]
    [Tooltip("Generate automatically in Awake() (before SimController.Start()). Required for play/training.")]
    public bool generateOnAwake = true;
    [Tooltip("Editor only: rebuild the track live whenever you change any field in the Inspector, " +
             "so you can dial knobs and see the result immediately (no Generate Track / Play needed).")]
    public bool autoRegenerateInEditor = true;

    // Parent for everything we spawn, so Clear() is a single Destroy.
    private const string ROOT_NAME = "GeneratedTrack";

    // 4 cardinal grid directions (CCW order matters for turn handedness).
    private static readonly Vector2Int[] DIRS =
    {
        new Vector2Int(1, 0),   // East  (+x)
        new Vector2Int(0, 1),   // North (+z)
        new Vector2Int(-1, 0),  // West
        new Vector2Int(0, -1),  // South
    };

    void Awake()
    {
        if (generateOnAwake)
        {
            Generate();
        }
    }

#if UNITY_EDITOR
    // Auto-regenerate live in the editor whenever any Inspector field
    // changes, so you can dial knobs (e.g. Goal Yaw Offset) and watch the
    // track rebuild immediately - no context-menu / Play needed. Deferred
    // via delayCall because destroying/instantiating objects directly
    // inside OnValidate is unsafe; delayCall runs just after.
    void OnValidate()
    {
        if (Application.isPlaying) return;
        if (!autoRegenerateInEditor) return;
        UnityEditor.EditorApplication.delayCall += _DeferredEditorRegen;
    }

    private void _DeferredEditorRegen()
    {
        UnityEditor.EditorApplication.delayCall -= _DeferredEditorRegen;
        // The object may have been deleted / scene changed between the
        // OnValidate and this callback.
        if (this == null) return;
        if (Application.isPlaying) return;
        Generate();
    }
#endif

    [ContextMenu("Generate Track")]
    public void Generate()
    {
        Clear();

        var root = new GameObject(ROOT_NAME);
        root.transform.SetParent(transform, false);

        List<Vector2Int> path = BuildLoopCells();
        if (path == null || path.Count < 4)
        {
            Debug.LogError("[TrackGenerator] failed to build a valid loop; check loopWidth/Height (>=3).");
            return;
        }

        int n = path.Count;
        // Mirrors BuildLoopCells' own w so we can identify west/east (x=0 /
        // x=w-1) column cells below without changing loop shape.
        int loopW = Mathf.Max(3, loopWidthTiles);
        int wallLayer = LayerMask.NameToLayer(wallLayerName);
        if (wallLayer < 0)
        {
            Debug.LogWarning($"[TrackGenerator] layer '{wallLayerName}' not found; walls go on Default. " +
                             "Add the layer or set wallLayerName to one in CarController.layerMask.");
            wallLayer = 0;
        }
        // CarController.layerMask is baked as bit=1 (layer 0 = Default only).
        // Walls on any other layer are invisible to the car's SphereCasts.
        // This most commonly happens when wallLayerName is left as the old
        // default "Road" in a serialized scene (the script default changed to
        // "Default" but existing Inspector values are not updated automatically).
        // Auto-correct here so the scene works immediately; update the Inspector
        // field to "Default" to silence this warning.
        if (wallLayer != 0)
        {
            Debug.LogWarning(
                $"[TrackGenerator] wallLayerName='{wallLayerName}' (layer {wallLayer}) is not " +
                $"Default (layer 0). CarController.layerMask=1 only sees layer 0 — auto-correcting " +
                $"so walls are perception-visible. Update Wall Layer Name to 'Default' in the " +
                $"Inspector to silence this warning.");
            wallLayer = 0;
        }

        if (buildGround)
        {
            BuildGround(root.transform, path);
        }

        int goalCounter = 0;
        // Counts how many consecutive rounded-corner cells we have been
        // inside. Resets to 0 whenever we return to a straight cell.
        // Used below to decide which corners in a multi-corner run get a goal.
        int consecutiveCornerCount = 0;
        for (int i = 0; i < n; i++)
        {
            Vector2Int cell = path[i];
            Vector2Int inDir = Step(path[(i - 1 + n) % n], cell);
            Vector2Int outDir = Step(cell, path[(i + 1) % n]);
            Vector3 center = CellToWorld(cell);

            bool isStraight = inDir == outDir;

            // Is this corner part of a multi-corner (chicane) run, as opposed
            // to one of the 4 isolated rectangle corners? Checked directly
            // against neighbouring cells (rather than relying on
            // consecutiveCornerCount, which isn't updated until below) so
            // PlaceRoundedCorner can pick chicaneCornerRadius over
            // cornerRadius before it needs the value.
            bool isChicaneCorner = false;
            if (!isStraight)
            {
                Vector2Int prevIn = Step(path[(i - 2 + n) % n], path[(i - 1 + n) % n]);
                bool prevCellIsCorner = prevIn != inDir;
                Vector2Int nextOut2 = Step(path[(i + 1) % n], path[(i + 2) % n]);
                bool nextCellIsCorner = outDir != nextOut2;
                isChicaneCorner = prevCellIsCorner || nextCellIsCorner;
            }

            // Flat-geometry mode lays a corridor-width road (2*roadHalfWidth)
            // with flush flanking rails so the guard rails sit EXACTLY on the
            // road edges - matching the rounded corners. The old straight path
            // (full tileSize square road + walls inset at roadHalfWidth) left a
            // road lip sticking out past the rail and square nubs at corner
            // approaches; that was the "outer rails not flush" artifact.
            if (!useKitTiles)
            {
                if (isStraight)
                {
                    PlaceStraightCell(root.transform, center, inDir, wallLayer);
                }
                else if (roundedCorners)
                {
                    PlaceRoundedCorner(root.transform, center, inDir, outDir, wallLayer, ref goalCounter,
                                       isChicaneCorner ? chicaneCornerRadius : cornerRadius);
                }
                else
                {
                    // Flat sharp corner (rounded corners disabled): square road
                    // + per-side walls.
                    PlaceTile(root.transform, center, inDir, outDir, isStraight, i);
                    if (buildWalls)
                        BuildCellWalls(root.transform, cell, center, inDir, outDir, wallLayer);
                }
            }
            else
            {
                // Kit-tile mode: prefab road + per-side walls.
                PlaceTile(root.transform, center, inDir, outDir, isStraight, i);
                if (buildWalls)
                    BuildCellWalls(root.transform, cell, center, inDir, outDir, wallLayer);
            }

            // Goals along the path, every goalEveryNTiles, named in order.
            if (goalEveryNTiles < 1) goalEveryNTiles = 1;

            // Track consecutive rounded-corner run depth. Captured BEFORE
            // this cell's own update so it reflects the run-length ending at
            // the PREVIOUS cell (used below to detect "am I straight-cell
            // right after an isolated corner").
            int prevConsecutiveCornerCount = consecutiveCornerCount;
            bool isCornerWithArc = !isStraight && roundedCorners && !useKitTiles;
            if (isCornerWithArc)
                consecutiveCornerCount++;
            else
                consecutiveCornerCount = 0;

            if (i % goalEveryNTiles == 0)
            {
                // ---------------------------------------------------------
                // Goal placement rules:
                //
                //  Straight cell   → place goal (normal cadence), UNLESS it
                //                    sits immediately before/after an
                //                    ISOLATED (single-cell, non-chicane)
                //                    corner - that corner already gets its
                //                    own apex goal from PlaceRoundedCorner,
                //                    so also placing the adjacent straight
                //                    cells' goals crams 3 goals with 3
                //                    different orientations (straight-cell
                //                    cardinal -> corner-apex radial ->
                //                    straight-cell cardinal) into a very
                //                    short span - confirmed (2026-07-15) as
                //                    the cause of persistent low-speed
                //                    crashes at track corners: the target
                //                    heading whipsaws through 3 sharply
                //                    different orientations faster than any
                //                    controller can settle between them.
                //                    Chicane (multi-corner) runs are
                //                    untouched - their own apex-goal logic
                //                    below already produces a different,
                //                    sparser density there.
                //  Corner cell     → NO goal at the cell itself
                //  Chicane middle  → if this is the FIRST of a consecutive-
                //                    middle pair, place ONE apex goal at the
                //                    world-space midpoint between this cell
                //                    and the next (bottom of the U-turn).
                //
                // Corners are excluded because PlaceGoal's arc-position
                // offset moves the gate into the wall/arc junction where it
                // looks like a phantom curb. Adjacent straight cells carry
                // goals naturally, so the sequence stays dense.
                // ---------------------------------------------------------

                // Is this cell the exit corner of its consecutive-corner run?
                bool isExitCorner = isCornerWithArc &&
                    Step(path[(i + 1) % n], path[(i + 2) % n]) == outDir;

                // Look ahead two cells to classify the NEXT cell.
                Vector2Int nextOutDir  = Step(path[(i + 1) % n], path[(i + 2) % n]);
                Vector2Int nextNextOut = Step(path[(i + 2) % n], path[(i + 3) % n]);
                bool nextIsCorner     = isCornerWithArc && (outDir != nextOutDir);
                bool nextIsExitCorner = nextIsCorner && (nextOutDir == nextNextOut);

                // Standalone (not gated on THIS cell being a corner) checks
                // for "is the adjacent cell an ISOLATED corner" - used only
                // by the straight-cell goal skip below.
                bool prevWasIsolatedCorner = prevConsecutiveCornerCount == 1;
                bool nextCellIsCorner = outDir != nextOutDir;
                bool nextCellIsIsolatedCorner = nextCellIsCorner && (nextOutDir == nextNextOut);
                bool adjacentToIsolatedCorner = prevWasIsolatedCorner || nextCellIsIsolatedCorner;

                // West/east ends (the SHORT vertical edges, x=0 or x=loopW-1)
                // only have loopHeightTiles-2 straight cells to begin with -
                // too few to also lose 2 of them to the isolated-corner skip
                // below and still read as "4 goals per side" (requested
                // 2026-07-18; loopHeightTiles/track size explicitly NOT to be
                // changed for this). The skip's original whiplash-crash
                // rationale (see comment block above) was observed on the
                // long north/south edges/chicanes, which keep the skip
                // unchanged; west/east cells are exempted from it instead.
                bool isVerticalSideCell = (cell.x == 0 || cell.x == loopW - 1);

                // Goals only go on STRAIGHT cells. Corner cells are
                // never given their own goal because PlaceGoal's arc-
                // position offset for corners pulls the gate into the
                // wall/arc junction, making it look like a phantom curb.
                // Adjacent straight cells already carry goals naturally
                // (every goalEveryNTiles tile), EXCEPT the ones immediately
                // flanking an isolated corner (see comment block above) -
                // UNLESS they're on a west/east vertical edge, which is
                // exempted from that skip (see isVerticalSideCell above).
                // The one exception is the chicane apex goal placed below.
                if (!isCornerWithArc && (!adjacentToIsolatedCorner || isVerticalSideCell))
                {
                    goalCounter++;
                    PlaceGoal(root.transform, center, goalCounter, inDir, outDir);
                }

                // Place apex goal at midpoint between this and the next cell
                // when this is the first of a consecutive-middle pair.
                bool isFirstOfMiddlePair = isCornerWithArc
                    && consecutiveCornerCount >= 2
                    && !isExitCorner
                    && nextIsCorner
                    && !nextIsExitCorner;

                if (isFirstOfMiddlePair)
                {
                    Vector3 nextCellCenter = CellToWorld(path[(i + 1) % n]);
                    Vector3 apexPos = (center + nextCellCenter) * 0.5f;
                    goalCounter++;
                    // Pass outDir as both dirs (signals "straight") so
                    // PlaceGoal skips the arc-position offset and places the
                    // gate perpendicular to the direction of travel at the apex.
                    PlaceGoal(root.transform, apexPos, goalCounter, outDir, outDir);
                }
            }
        }

        // Merge collinear wall segments into single long BoxColliders.
        // PlaceStraightCell emits 2 short Curb segments per tile; a full
        // straight run of N tiles becomes 2N colliders. Merging collapses
        // that to 2 per run (one inner, one outer), shrinking the broadphase
        // from ~276 to ~8-16 colliders and dramatically reducing physics cost.
        MergeWalls(root.transform, wallLayer);

        // Static batch in play mode only. In edit mode, StaticBatchingUtility
        // kills MaterialPropertyBlock color overrides (Unity applies batching
        // per sharedMaterial, discarding per-renderer PropertyBlock values),
        // making all roads/walls appear white. In play mode PropertyBlocks
        // survive and we get the performance win.
        if (Application.isPlaying)
            StaticBatchingUtility.Combine(root);

        Debug.Log($"[TrackGenerator] generated loop: {n} tiles, {goalCounter} goals, " +
                  $"curvature={curvatureDifficulty:0.00}, seed={seed}. " +
                  $"Goal-1 findable={GameObject.Find("Goal-1") != null}.");

        // If a car is already live (e.g. an editor "Generate Track" during
        // Play, or a future in-place mid-run regenerate), tell it to drop
        // references to the goals/curbs we just destroyed and re-acquire the
        // new track - otherwise its next FixedUpdate dereferences destroyed
        // objects (MissingReferenceException). No-op when no car exists yet:
        // the first Awake-time generation (car not instantiated until
        // SimController.Start) and SimController.Restart (which destroys the
        // car BEFORE regenerating and re-instantiates it AFTER).
        if (Application.isPlaying)
        {
            var liveCar = FindObjectOfType<CarController>();
            if (liveCar != null)
                liveCar.OnTrackRegenerated();
        }
    }

    [ContextMenu("Clear Track")]
    public void Clear()
    {
        // Destroy any prior root (handles re-generate + editor preview cleanup).
        for (int c = transform.childCount - 1; c >= 0; c--)
        {
            Transform child = transform.GetChild(c);
            if (child.name == ROOT_NAME)
            {
                if (Application.isPlaying)
                {
                    // SetActive(false) is CRITICAL: Destroy() in play mode is
                    // deferred (end-of-frame), so old Goal-N objects stay alive
                    // in the scene for the rest of the current frame. When
                    // Generate() immediately creates new Goal-N objects with the
                    // same names, CarController.SetUpGoalsArray()'s
                    // GameObject.Find("Goal-N") can accidentally find the old
                    // (soon-to-be-deleted) goals. By disabling the root first,
                    // we hide ALL children (including old goals) from Find(),
                    // which only searches active objects. The new goals created
                    // in the same Generate() call ARE active and findable.
                    child.gameObject.SetActive(false);
                    Destroy(child.gameObject);
                }
                else DestroyImmediate(child.gameObject);
            }
        }
    }

    // ---- Loop construction -------------------------------------------------

    /// <summary>
    /// Build a closed rectilinear loop of grid cells (CCW). Starts as a
    /// rectangle perimeter, then splices outward chicanes onto the top edge
    /// (count scales with curvatureDifficulty) to add turns. Chicanes
    /// protrude +y and are spaced so the loop stays simple (non-self-
    /// intersecting).
    /// </summary>
    private List<Vector2Int> BuildLoopCells()
    {
        int w = Mathf.Max(3, loopWidthTiles);
        int h = Mathf.Max(3, loopHeightTiles);

        // Rectangle corners: (0,0) .. (w-1, h-1). Walk CCW:
        //   bottom(SOUTH) edge W->E, right(EAST) edge S->N,
        //   top(NORTH) edge E->W, left(WEST) edge N->S.
        // Each of the 4 straight runs below can now splice in per-edge
        // chicane bumps (2026-07-18, see BuildEdgeWithChicanes) instead of
        // only the north/top edge supporting them.
        var cells = new List<Vector2Int>();

        // Bottom (SOUTH) edge, W->E: (0,0) .. (w-1,0). Both rectangle
        // corners included (matches the original un-chicaned behaviour) -
        // interior runs x=1..w-2, bump direction is southward (y decreasing).
        cells.Add(new Vector2Int(0, 0));
        cells.AddRange(BuildEdgeWithChicanes(
            new Vector2Int(1, 0), new Vector2Int(1, 0), new Vector2Int(0, -1),
            w - 2, chicanesSouth));
        cells.Add(new Vector2Int(w - 1, 0));

        // Right (EAST) edge, S->N: excludes the SE corner (already added
        // above), includes the NE corner at the end - interior runs
        // y=1..h-2, bump direction is eastward (x increasing).
        cells.AddRange(BuildEdgeWithChicanes(
            new Vector2Int(w - 1, 1), new Vector2Int(0, 1), new Vector2Int(1, 0),
            h - 2, chicanesEast));
        cells.Add(new Vector2Int(w - 1, h - 1));

        // Top (NORTH) edge, E->W: excludes both corners (NE above, NW
        // below) - interior runs x=w-2..1, bump direction is northward
        // (y increasing).
        cells.AddRange(BuildEdgeWithChicanes(
            new Vector2Int(w - 2, h - 1), new Vector2Int(-1, 0), new Vector2Int(0, 1),
            w - 2, chicanesNorth));

        // Left (WEST) edge, N->S: includes the NW corner (missed by the top
        // edge above), excludes the SW corner (already the very first cell
        // of the whole path) - interior runs y=h-2..1, bump direction is
        // westward (x decreasing).
        cells.Add(new Vector2Int(0, h - 1));
        cells.AddRange(BuildEdgeWithChicanes(
            new Vector2Int(0, h - 2), new Vector2Int(0, -1), new Vector2Int(-1, 0),
            h - 2, chicanesWest));

        // De-dupe consecutive duplicates defensively (shouldn't happen).
        var dedup = new List<Vector2Int>();
        foreach (var c in cells)
        {
            if (dedup.Count == 0 || dedup[dedup.Count - 1] != c) dedup.Add(c);
        }
        return dedup;
    }

    /// <summary>
    /// Chooses up to <paramref name="count"/> evenly-spaced, DETERMINISTIC
    /// trigger indices (0-based, along an interior run of <paramref
    /// name="length"/> cells) at which a chicane bump should start. A bump
    /// starting at index i consumes indices i, i+1, i+2 (3 cells - see
    /// BuildEdgeWithChicanes).
    ///
    /// Reserves >=1 PLAIN buffer cell both BEFORE the first bump and AFTER
    /// the last bump (so a chicane never sits flush against the adjacent
    /// rectangle/edge corner with zero straight recovery - the vertical-edge
    /// analogue of the "no straight recovery" defect fixed within a single
    /// bump on 2026-07-18), and requires >=4 spacing between consecutive
    /// trigger starts (3 for the bump + >=1 plain gap cell), so no two
    /// bumps ever share a corner with zero straight recovery between them
    /// either. Valid trigger range is therefore [1, length-4].
    ///
    /// Deterministic (no RNG) and purely a function of (count, length) - not
    /// jittered - so a given edge's Nth chicane lands at roughly the same
    /// place regardless of which curriculum stage is active, matching the
    /// "stage 2 adds the first chicane... stage 5 adds a third" incremental
    /// framing (count=1 -> centre of the usable range, by construction).
    ///
    /// If the full requested count doesn't fit at >=4 spacing within the
    /// available range, silently drops chicanes (fewest-first is avoided -
    /// we shrink the whole even-spread, not truncate one end) rather than
    /// violating the buffer/spacing guarantees; callers should size
    /// loopWidthTiles/loopHeightTiles so the requested counts actually fit.
    /// Never throws.
    /// </summary>
    private List<int> ChooseChicaneTriggerIndices(int count, int length)
    {
        var result = new List<int>();
        if (count <= 0) return result;
        const int minSpacing = 4;
        int lo = 1;
        int hi = length - 4;
        if (hi < lo) return result; // edge too short for even 1 buffered chicane

        int span = hi - lo;
        int fitCount = count;
        while (fitCount > 1 && span < minSpacing * (fitCount - 1))
            fitCount--;
        if (fitCount <= 0) return result;

        for (int k = 0; k < fitCount; k++)
        {
            int idx = (fitCount == 1)
                ? lo + span / 2
                : lo + Mathf.RoundToInt((float) k / (fitCount - 1) * span);
            result.Add(idx);
        }
        return result;
    }

    /// <summary>
    /// Builds one straight edge's INTERIOR cells (excluding the rectangle
    /// corners at either end - callers add those separately, see
    /// BuildLoopCells), splicing in <paramref name="chicaneCount"/> outward
    /// bumps distributed via ChooseChicaneTriggerIndices.
    ///
    /// <paramref name="start"/> is the first interior cell. <paramref
    /// name="walkDir"/> is the unit step direction of travel along the edge
    /// (e.g. (-1,0) walking west). <paramref name="bumpDir"/> is the unit
    /// OUTWARD bump direction, perpendicular to walkDir (e.g. (0,1) = north,
    /// for the top edge's bump). <paramref name="length"/> is the number of
    /// interior cells along the edge (excluding both corners).
    ///
    /// A bump at walk-index i replaces the single interior cell with 5
    /// cells: turn out, straight (1 recovery tile), turn back - generalizing
    /// the north-edge bump shape confirmed navigable on 2026-07-18:
    ///   p, p+bumpDir, p+bumpDir+walkDir, p+bumpDir+2*walkDir, p+2*walkDir
    /// where p is the cell at that index. This consumes indices i, i+1, i+2
    /// (3 interior cells -> 5 path cells), so the walk advances by 3 instead
    /// of 1 whenever a bump triggers.
    /// </summary>
    private List<Vector2Int> BuildEdgeWithChicanes(
        Vector2Int start, Vector2Int walkDir, Vector2Int bumpDir,
        int length, int chicaneCount)
    {
        var edge = new List<Vector2Int>();
        if (length <= 0) return edge;

        var triggers = new HashSet<int>(ChooseChicaneTriggerIndices(chicaneCount, length));

        int i = 0;
        while (i < length)
        {
            Vector2Int p = start + walkDir * i;
            if (triggers.Contains(i) && i + 2 < length)
            {
                edge.Add(p);
                edge.Add(p + bumpDir);
                edge.Add(p + bumpDir + walkDir);
                edge.Add(p + bumpDir + walkDir * 2);
                edge.Add(p + walkDir * 2);
                i += 3;
            }
            else
            {
                edge.Add(p);
                i += 1;
            }
        }
        return edge;
    }

    // ---- Placement helpers -------------------------------------------------

    /// <summary>One flat ground box spanning the loop's bounding area so the
    /// car always has a drivable surface. Named "Road" + on Default layer.</summary>
    private void BuildGround(Transform parent, List<Vector2Int> path)
    {
        int minX = int.MaxValue, maxX = int.MinValue, minY = int.MaxValue, maxY = int.MinValue;
        foreach (var c in path)
        {
            minX = Mathf.Min(minX, c.x); maxX = Mathf.Max(maxX, c.x);
            minY = Mathf.Min(minY, c.y); maxY = Mathf.Max(maxY, c.y);
        }
        float margin = groundMarginTiles * tileSize;
        Vector3 min = new Vector3(minX * tileSize - tileSize * 0.5f - margin, 0f,
                                  minY * tileSize - tileSize * 0.5f - margin);
        Vector3 max = new Vector3(maxX * tileSize + tileSize * 0.5f + margin, 0f,
                                  maxY * tileSize + tileSize * 0.5f + margin);
        Vector3 size = max - min;
        Vector3 mid = (min + max) * 0.5f;

        var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
        ground.name = "Road"; // drivable surface; CarController treats name "Road" as track
        // Layer 2 = Ignore Raycast: car physics still rests on the ground but
        // SphereCast (layerMask=1) won't hit it, preventing spurious perception
        // hits on the road surface.
        ground.layer = 2;
        ground.transform.SetParent(parent, false);
        ground.transform.position = new Vector3(mid.x, trackY - 0.1f, mid.z);
        ground.transform.localScale = new Vector3(size.x, 0.2f, size.z);
        SetRendererColor(ground.GetComponent<Renderer>(), new Color(0.72f, 0.72f, 0.68f)); // light concrete
        DisableShadows(ground.GetComponent<Renderer>());
    }

    private Vector3 CellToWorld(Vector2Int cell)
    {
        return new Vector3(cell.x * tileSize, trackY, cell.y * tileSize);
    }

    private static Vector2Int Step(Vector2Int from, Vector2Int to)
    {
        int dx = Mathf.Clamp(to.x - from.x, -1, 1);
        int dy = Mathf.Clamp(to.y - from.y, -1, 1);
        return new Vector2Int(dx, dy);
    }

    private static float HeadingYaw(Vector2Int dir)
    {
        // World: +x = East (yaw 90), +z = North (yaw 0). Unity yaw is CW from +z.
        if (dir == new Vector2Int(1, 0)) return 90f;    // East
        if (dir == new Vector2Int(-1, 0)) return 270f;  // West
        if (dir == new Vector2Int(0, 1)) return 0f;     // North
        return 180f;                                    // South
    }

    // Reused across all tint calls so a regenerate allocates nothing.
    private static MaterialPropertyBlock _mpb;
    private static readonly int _ColorId = Shader.PropertyToID("_Color");
    private static readonly int _BaseColorId = Shader.PropertyToID("_BaseColor");

    /// <summary>
    /// Tint a renderer WITHOUT instantiating a per-object material. Uses a
    /// MaterialPropertyBlock, so it's leak-free in edit mode (renderer.material
    /// clones + leaks a material on every regenerate, which is what triggers
    /// the "Instantiating material ... during edit mode" warning) and is fine
    /// in play mode too. Sets both _Color (built-in pipeline) and _BaseColor
    /// (URP) so the tint applies regardless of render pipeline.
    /// </summary>
    private static void SetRendererColor(Renderer rend, Color color)
    {
        if (rend == null) return;
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        rend.GetPropertyBlock(_mpb);
        _mpb.SetColor(_ColorId, color);
        _mpb.SetColor(_BaseColorId, color);
        rend.SetPropertyBlock(_mpb);
        DisableShadows(rend);
    }

    /// <summary>
    /// Turn off shadow casting + receiving on a generated piece. A procedural
    /// track spawns hundreds of renderers (e.g. ~445 for a default loop); the
    /// per-object shadow pass doubles draw work and is a major FPS sink. Since
    /// the sim's ROS message rate is frame-bound, that FPS hit shows up as a
    /// reduced publish/step rate. Shadows add nothing to a top-down RL track.
    /// </summary>
    private static void DisableShadows(Renderer rend)
    {
        if (rend == null) return;
        rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        rend.receiveShadows = false;
    }

    private void PlaceTile(Transform parent, Vector3 center, Vector2Int inDir,
                           Vector2Int outDir, bool isStraight, int index)
    {
        // Default spike path: a clean flat road quad per cell. Squares are
        // rotation-symmetric so corners "just work" - no kit orientation
        // calibration. The union of per-cell squares forms the full path
        // (straights + corners), matching the walled corridor exactly.
        if (!useKitTiles)
        {
            var road = GameObject.CreatePrimitive(PrimitiveType.Cube);
            road.name = "Road";
            road.layer = 2; // Ignore Raycast — drivable but transparent to SphereCast
            road.transform.SetParent(parent, false);
            // Sit just above the ground box; full-tile footprint so adjacent
            // cells abut with no gaps.
            road.transform.position = new Vector3(center.x, trackY + 0.02f, center.z);
            road.transform.localScale = new Vector3(tileSize, 0.05f, tileSize);
            SetRendererColor(road.GetComponent<Renderer>(), roadColor);
            return;
        }

        GameObject prefab;
        float yaw;

        if (isStraight)
        {
            prefab = straightPrefab;
            yaw = HeadingYaw(inDir) + straightYawOffset;
        }
        else
        {
            // Left turn if outDir is 90deg CCW of inDir (cross product +).
            int cross = inDir.x * outDir.y - inDir.y * outDir.x; // +1 left, -1 right
            bool left = cross > 0;
            prefab = (left || curveFlippedPrefab == null) ? curvePrefab : curveFlippedPrefab;
            // Base the curve yaw on the incoming heading; calibrate via offset.
            yaw = HeadingYaw(inDir) + curveYawOffset + (left ? 0f : 0f);
        }

        if (prefab == null)
        {
            // No tile prefab assigned: drop a flat placeholder quad so the
            // loop is still visible/drivable for the pipeline test.
            var placeholder = GameObject.CreatePrimitive(PrimitiveType.Cube);
            placeholder.name = "Road";
            placeholder.layer = 2; // Ignore Raycast
            placeholder.transform.SetParent(parent, false);
            placeholder.transform.position = center;
            placeholder.transform.localScale = new Vector3(tileSize, 0.2f, tileSize);
            placeholder.transform.rotation = Quaternion.Euler(0f, isStraight ? HeadingYaw(inDir) : 0f, 0f);
            return;
        }

        GameObject tile = Instantiate(prefab, center, Quaternion.Euler(0f, yaw, 0f), parent);
        // Name road tiles "Road" so CarController.GetListOfRoads()/PlaceObstacles
        // can find them later (obstacles are off for the spike, but harmless).
        tile.name = "Road";
    }

    /// <summary>Yaw (Unity, CW from +z) of an arbitrary 2D grid direction.
    /// Used so corner goals can face the diagonal bisector, not just a
    /// cardinal heading.</summary>
    private static float HeadingYawFromVector(Vector2 dir)
    {
        if (dir.sqrMagnitude < 1e-6f) return 0f;
        return Mathf.Atan2(dir.x, dir.y) * Mathf.Rad2Deg;
    }

    // inDir/outDir are Vector2 (not Vector2Int) so callers placing goals at
    // ARBITRARY points along a corner's arc (see goalsPerCorner) can pass a
    // continuous tangent direction for orientation - grid cell callers keep
    // passing Vector2Int as before, which implicitly converts to Vector2.
    private void PlaceGoal(Transform parent, Vector3 center, int number,
                           Vector2 inDir, Vector2 outDir, float radiusOverride = -1f)
    {
        // Build a visible, ASYMMETRIC gate: wide ACROSS the road (local X),
        // thin ALONG it (local Z). The asymmetry is what makes the per-heading
        // yaw both meaningful and visible (the kit Goal.prefab is a vertical
        // pole, symmetric about Y, so its yaw was invisible - that was the
        // "goals won't rotate" gotcha).
        // Anchor position. On straights the cell centre is on the road
        // centreline, so use it directly. On a ROUNDED corner the drivable
        // road is the arc, which is offset toward the inside of the turn - the
        // cell centre sits OUTSIDE the arc, so a corner goal left at the centre
        // hangs off the outer corner. Shift it onto the arc's 45-degree
        // midpoint and anchor it so the gate's inner edge is flush with the
        // outer face of the inner rail (and, at the default gate width =
        // corridor, the outer edge meets the outer rail).
        Vector3 goalPos = center;
        if (inDir != outDir && roundedCorners && !useKitTiles)
        {
            float half = tileSize * 0.5f;
            // radiusOverride (< 0 = "not supplied") lets chicane apex goals use
            // the SAME chicaneCornerRadius as the arc PlaceRoundedCorner built
            // for that cell, instead of the instance cornerRadius field -
            // otherwise the goal would sit at the position for a DIFFERENT
            // radius than the one actually drivable there.
            float baseRadius = radiusOverride >= 0f ? radiusOverride : cornerRadius;
            float r = Mathf.Clamp(baseRadius, roadHalfWidth + 0.5f, half);
            Vector3 inW = new Vector3(inDir.x, 0f, inDir.y);
            Vector3 outW = new Vector3(outDir.x, 0f, outDir.y);
            Vector3 arcC = center + (-inW + outW) * r;
            // Outward radial direction at the arc's midpoint.
            Vector3 midUnit = (inW - outW).normalized;
            // Outer face of the inner rail = inner-rail radius + half its thickness.
            float innerEdge = (r - roadHalfWidth) + wallThickness * 0.5f;
            // Push out by half the gate width so the gate's inner edge lands there.
            goalPos = arcC + midUnit * (innerEdge + goalGateWidth * 0.5f);
        }

        GameObject goal;
        Vector3 goalWorldPos = goalPos + new Vector3(0f, goalHeight, 0f);

        if (goalPrefab != null)
        {
            // Lay the cylinder flat (X-axis 90°) and rotate on Z so its long
            // axis runs ACROSS the road (perpendicular to travel direction).
            // This maximises the cross-sectional area the car passes through.
            //
            // Implementation: use FromToRotation to take the cylinder's local
            // +Y (long axis) and point it at the "across the road" vector =
            // 90° CW from the travel bisector in the horizontal plane.
            // This is equivalent to X=90 + the Z orientation, without any
            // Euler-order gymnastics.
            Vector2 bisectorVec = new Vector2(inDir.x + outDir.x, inDir.y + outDir.y);
            if (bisectorVec.sqrMagnitude < 0.01f)
                bisectorVec = new Vector2(inDir.x, inDir.y);
            Vector3 travelDir = new Vector3(bisectorVec.x, 0f, bisectorVec.y).normalized;
            // Perpendicular to travelDir in the horizontal plane (90° CW).
            Vector3 acrossRoad = new Vector3(-travelDir.z, 0f, travelDir.x);
            Quaternion goalRot = Quaternion.FromToRotation(Vector3.up, acrossRoad);

            goal = Instantiate(goalPrefab,
                               new Vector3(goalPos.x, trackY + goalHeight, goalPos.z),
                               goalRot, parent);
        }
        else
        {
            // Fallback: asymmetric flat cube gate spanning the road width.
            goal = GameObject.CreatePrimitive(PrimitiveType.Cube);
            goal.transform.SetParent(parent, false);
            goal.transform.position = goalWorldPos;
            goal.transform.localScale = new Vector3(goalGateWidth, goalTriggerHeight, goalGateThickness);

            // Face perpendicular to bisector of in/out so the gate cuts the
            // widest road cross-section.
            Vector2 bisector = new Vector2(inDir.x + outDir.x, inDir.y + outDir.y);
            goal.transform.rotation =
                Quaternion.Euler(0f, HeadingYawFromVector(bisector) + goalYawOffset, 0f);

            SetRendererColor(goal.GetComponent<Renderer>(), goalColor);
        }

        goal.name = "Goal-" + number; // EXACT name CarController.SetUpGoalsArray expects.

        // Goals must NOT block the car's SphereCast perception rays.
        // Unity's Physics.queriesHitTriggers = true (the default) means
        // SphereCast hits trigger colliders on the raycast layer (Default=0),
        // creating a visible "wall" across the road for the fallback cube goal
        // (CreatePrimitive defaults to layer 0). Ignore Raycast (layer 2)
        // removes goals from all raycast queries while leaving OnTriggerEnter
        // (physics-based, layer-agnostic) intact for goal completion detection.
        goal.layer = 2;

        // The car detects goals by OnTriggerEnter; collider must be a trigger.
        var col = goal.GetComponent<Collider>();
        if (col != null) col.isTrigger = true;

        // SimController does goal.GetComponent<Goal>().goalComplete = false,
        // so the Goal component is REQUIRED or that line NREs.
        if (goal.GetComponent<Goal>() == null)
            goal.AddComponent<Goal>();
    }

    // ---- Rounded corners ---------------------------------------------------

    /// <summary>
    /// Build a quarter-circle rounded corner inside this cell: a short straight
    /// lead-in, a faceted arc, and a short straight lead-out. Lays the road
    /// surface plus inner + outer "Curb" walls (raycast layer) along the arc,
    /// so the perception SphereCasts get distance readings off the curve and
    /// crashing into it registers.
    /// </summary>
    private void PlaceRoundedCorner(Transform parent, Vector3 center,
                                    Vector2Int inDir, Vector2Int outDir, int wallLayer,
                                    ref int goalCounter, float radius)
    {
        float half = tileSize * 0.5f;
        float r = Mathf.Clamp(radius, roadHalfWidth + 0.5f, half);
        float corridor = roadHalfWidth * 2f;

        Vector3 inW = new Vector3(inDir.x, 0f, inDir.y);
        Vector3 outW = new Vector3(outDir.x, 0f, outDir.y);

        // Arc centre is offset from the cell centre toward the inside of the
        // turn by r along both -inDir and +outDir. Tangent points T1/T2 are r
        // back/forward from the cell centre along the entry/exit headings.
        Vector3 arcC = center + (-inW + outW) * r;
        Vector3 T1 = center - inW * r;     // arc start (on entry straight)
        Vector3 T2 = center + outW * r;    // arc end (on exit straight)
        Vector3 E1 = center - inW * half;  // entry edge midpoint
        Vector3 E2 = center + outW * half; // exit edge midpoint

        // Straight lead-in / lead-out (zero-length when r == half = pure arc).
        if ((E1 - T1).sqrMagnitude > 0.01f)
        {
            PlaceRoadSegment(parent, E1, T1, corridor);
            PlaceFlankingWalls(parent, E1, T1, wallLayer);
        }
        if ((T2 - E2).sqrMagnitude > 0.01f)
        {
            PlaceRoadSegment(parent, T2, E2, corridor);
            PlaceFlankingWalls(parent, T2, E2, wallLayer);
        }

        // Faceted arc: centreline radius r, inner r-roadHalfWidth, outer r+roadHalfWidth.
        float ri = Mathf.Max(0.1f, r - roadHalfWidth);
        float ro = r + roadHalfWidth;
        float th1 = Mathf.Atan2((T1 - arcC).x, (T1 - arcC).z) * Mathf.Rad2Deg;
        float th2 = Mathf.Atan2((T2 - arcC).x, (T2 - arcC).z) * Mathf.Rad2Deg;
        float delta = Mathf.DeltaAngle(th1, th2); // shortest signed sweep (+/-90)
        int n = Mathf.Max(2, cornerFacets);

        // Goal gate(s) along the arc, evenly spaced at goalsPerCorner points
        // between T1 and T2 (goalsPerCorner=1 -> the 45-degree apex only,
        // identical to the old single-goal behaviour). Placed directly at
        // each point's OWN position/tangent (rather than routing through
        // PlaceGoal's inDir!=outDir arc-reconstruction branch, which only
        // knows how to anchor a SINGLE apex point) so the pursuit target the
        // car steers toward advances along the curve instead of chording
        // straight across it from one apex point to the next corner's apex -
        // see the 2026-07-19 "car drifts off at corners" investigation.
        // Each gate's inDir/outDir is the arc's own tangent direction at that
        // point (a continuous Vector2, not a grid-aligned Vector2Int) so
        // PlaceGoal orients the gate perpendicular to the ACTUAL direction of
        // travel there, not the corner's overall entry/exit heading.
        if (placeCornerTangentGoals)
        {
            float innerEdge = (r - roadHalfWidth) + wallThickness * 0.5f;
            float travelSign = Mathf.Sign(delta);
            int count = Mathf.Max(1, goalsPerCorner);
            for (int k = 1; k <= count; k++)
            {
                float frac = k / (float)(count + 1);
                float thG = (th1 + delta * frac) * Mathf.Deg2Rad;
                Vector3 unitG = new Vector3(Mathf.Sin(thG), 0f, Mathf.Cos(thG));
                Vector3 goalPosG = arcC + unitG * (innerEdge + goalGateWidth * 0.5f);
                Vector3 tangent = new Vector3(Mathf.Cos(thG), 0f, -Mathf.Sin(thG)) * travelSign;
                Vector2 tangent2 = new Vector2(tangent.x, tangent.z);
                goalCounter++;
                PlaceGoal(parent, goalPosG, goalCounter, tangent2, tangent2);
            }
        }

        Vector3 prevC = Vector3.zero, prevI = Vector3.zero, prevO = Vector3.zero;
        for (int i = 0; i <= n; i++)
        {
            float th = (th1 + delta * (i / (float)n)) * Mathf.Deg2Rad;
            Vector3 unit = new Vector3(Mathf.Sin(th), 0f, Mathf.Cos(th));
            Vector3 pc = arcC + unit * r;
            Vector3 pi = arcC + unit * ri;
            Vector3 po = arcC + unit * ro;
            if (i > 0)
            {
                PlaceRoadSegment(parent, prevC, pc, corridor);
                PlaceWallSegment(parent, prevI, pi, wallLayer, isInner: true);
                PlaceWallSegment(parent, prevO, po, wallLayer, isInner: false);
            }
            prevC = pc; prevI = pi; prevO = po;
        }
    }

    /// <summary>
    /// Lay one straight cell as a corridor-width road running the full tile
    /// length along the heading, with two flush flanking "Curb" rails at
    /// +/- roadHalfWidth. The rails coincide exactly with the road edges (no
    /// overhang), and abut both the neighbouring straight cells and the
    /// rounded-corner lead-ins, so the outer + inner rails form one
    /// continuous flush line around the loop.
    /// </summary>
    private void PlaceStraightCell(Transform parent, Vector3 center, Vector2Int dir, int wallLayer)
    {
        float half = tileSize * 0.5f;
        float corridor = roadHalfWidth * 2f;
        Vector3 dW = new Vector3(dir.x, 0f, dir.y);
        Vector3 a = center - dW * half;
        Vector3 b = center + dW * half;
        PlaceRoadSegment(parent, a, b, corridor);
        if (buildWalls) PlaceFlankingWalls(parent, a, b, wallLayer);
    }

    /// <summary>Flat road quad from a to b, `width` across the travel direction.</summary>
    private void PlaceRoadSegment(Transform parent, Vector3 a, Vector3 b, float width)
    {
        Vector3 d = b - a;
        float len = d.magnitude;
        if (len < 1e-4f) return;
        Vector3 mid = (a + b) * 0.5f;
        float yaw = Mathf.Atan2(d.x, d.z) * Mathf.Rad2Deg;
        var road = GameObject.CreatePrimitive(PrimitiveType.Cube);
        road.name = "Road";
        road.layer = 2; // Ignore Raycast — drivable but transparent to SphereCast
        road.transform.SetParent(parent, false);
        road.transform.position = new Vector3(mid.x, trackY + 0.02f, mid.z);
        road.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
        road.transform.localScale = new Vector3(width, 0.05f, len + segmentOverlap);
        SetRendererColor(road.GetComponent<Renderer>(), roadColor);
    }

    /// <summary>A "Curb" wall box from a to b. Inner arc walls go on Ignore
    /// Raycast (layer 2) so they don't produce floating perception spheres —
    /// crash detection still works because OnCollisionEnter is layer-agnostic.
    /// Outer arc walls go on wallLayer (Default) so the car can sense them.</summary>
    private void PlaceWallSegment(Transform parent, Vector3 a, Vector3 b, int wallLayer, bool isInner = false)
    {
        Vector3 d = b - a;
        float len = d.magnitude;
        if (len < 1e-4f) return;
        Vector3 mid = (a + b) * 0.5f;
        float yaw = Mathf.Atan2(d.x, d.z) * Mathf.Rad2Deg;
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "Curb";
        // All arc walls go on wallLayer (Default) so the car can sense both
        // sides of the corridor through corners. Inner arc walls are invisible
        // (renderer disabled below) but still raycast-detectable.
        wall.layer = wallLayer;
        wall.transform.SetParent(parent, false);
        wall.transform.position = new Vector3(mid.x, wallHeight * 0.5f, mid.z);
        wall.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
        wall.transform.localScale = new Vector3(wallThickness, wallHeight, len + segmentOverlap);
        var wr = wall.GetComponent<Renderer>();
        if (showArcWallGeometry && !isInner)
        {
            SetRendererColor(wr, railColor);
            DisableShadows(wr);
        }
        else
        {
            wr.enabled = false;
        }
    }

    /// <summary>Two Curb walls parallel to a->b, offset +/- roadHalfWidth.</summary>
    private void PlaceFlankingWalls(Transform parent, Vector3 a, Vector3 b, int wallLayer)
    {
        Vector3 d = (b - a);
        if (d.sqrMagnitude < 1e-6f) return;
        Vector3 perp = Vector3.Cross(Vector3.up, d.normalized) * roadHalfWidth;
        PlaceWallSegment(parent, a + perp, b + perp, wallLayer);
        PlaceWallSegment(parent, a - perp, b - perp, wallLayer);
    }

    /// <summary>
    /// Post-process: find all "Curb" children that are axis-aligned (produced
    /// by PlaceStraightCell / PlaceFlankingWalls) and merge collinear,
    /// same-side runs into a single long BoxCollider, then destroy the
    /// originals. Arc-wall segments from PlaceRoundedCorner are left as-is
    /// (they're not axis-aligned so they don't merge cleanly, and there are
    /// at most 2*cornerFacets*4 of them).
    ///
    /// Reduces broadphase collider count from ~2 per straight tile (e.g. 276
    /// for a default loop) to ~2 per straight run (~8-16 total).
    /// </summary>
    private void MergeWalls(Transform parent, int wallLayer)
    {
        // Gather all Curb children that are axis-aligned (yaw ≈ 0 or 90).
        var curbs = new List<Transform>();
        foreach (Transform child in parent.GetComponentsInChildren<Transform>())
        {
            if (child.name != "Curb") continue;
            float yaw = child.eulerAngles.y % 180f;
            // Keep only walls within 1° of exactly 0° (NS) or 90° (EW).
            // The old filter (yaw > 1 && yaw < 89) only blocked the 1-89° band;
            // arc-wall endpoint segments near the corner tangent can land at
            // ~99° or ~170° (after %180), slipping through and being merged
            // with the collinear straight wall — shifting its pivot and
            // extending the BoxCollider into the corner road space (phantom barrier).
            bool isAxisAligned = yaw <= 1f || (yaw >= 89f && yaw <= 91f);
            if (!isAxisAligned) continue;
            curbs.Add(child);
        }

        // Group by (direction, lateral-offset) so only truly collinear walls merge.
        // Key: rounded yaw (0=N/S, 90=E/W) + rounded lateral position on the
        // perpendicular axis. Walls with the same key lie on the same infinite
        // line and can be merged into one.
        var groups = new Dictionary<string, List<Transform>>();
        foreach (var c in curbs)
        {
            float yaw = c.eulerAngles.y % 180f;
            bool isNS = yaw < 45f; // runs N-S (scale.z = length, x = thickness)
            float lateral = isNS
                ? Mathf.Round(c.position.x / wallThickness) // group by X
                : Mathf.Round(c.position.z / wallThickness); // group by Z
            string key = (isNS ? "NS" : "EW") + "_" + lateral;
            if (!groups.ContainsKey(key)) groups[key] = new List<Transform>();
            groups[key].Add(c);
        }

        // Find the extreme lateral key values for NS and EW groups.
        // Outer walls sit at the min/max lateral of their axis — furthest from
        // the loop centre. Everything between is an inner wall (facing the
        // interior). We use the rounded key values (not world positions) for
        // the comparison so floating-point drift doesn't misclassify.
        float nsKeyMin = float.MaxValue, nsKeyMax = float.MinValue;
        float ewKeyMin = float.MaxValue, ewKeyMax = float.MinValue;
        foreach (var kvp2 in groups)
        {
            if (kvp2.Value.Count < 2) continue;
            bool kNS = kvp2.Key.StartsWith("NS");
            float kLat = float.Parse(kvp2.Key.Substring(3),
                System.Globalization.CultureInfo.InvariantCulture);
            if (kNS) { nsKeyMin = Mathf.Min(nsKeyMin, kLat); nsKeyMax = Mathf.Max(nsKeyMax, kLat); }
            else     { ewKeyMin = Mathf.Min(ewKeyMin, kLat); ewKeyMax = Mathf.Max(ewKeyMax, kLat); }
        }

        foreach (var kvp in groups)
        {
            var group = kvp.Value;
            if (group.Count < 1) continue;

            bool isNS = kvp.Key.StartsWith("NS");

            // Sort along the run axis.
            group.Sort((a, b) =>
                isNS ? a.position.z.CompareTo(b.position.z)
                     : a.position.x.CompareTo(b.position.x));

            // Split into CONTIGUOUS RUNS. Segments that share the same
            // lateral line but have a physical gap between them (e.g. the
            // inner straight wall on both sides of a chicane opening) must
            // NOT be merged into a single BoxCollider — that would bridge
            // the chicane gap and create an impassable barrier.
            //
            // Two consecutive sorted segments are "adjacent" if the gap
            // between the trailing edge of one and the leading edge of the
            // next is less than half a tile (tileSize * 0.5). Adjacent
            // cells have wall segments that overlap (by segmentOverlap),
            // so their gap is negative. Non-adjacent cells separated by
            // corner tiles have a gap ≥ (tileSize - segmentOverlap) ≈ 19.6 m.
            float gapThreshold = tileSize * 0.5f;

            var run = new List<Transform>();
            for (int gi = 0; gi < group.Count; gi++)
            {
                run.Add(group[gi]);

                bool isLastInGroup = gi == group.Count - 1;
                bool gapToNext = false;
                if (!isLastInGroup)
                {
                    Transform cur  = group[gi];
                    Transform nxt  = group[gi + 1];
                    float curLen  = isNS ? cur.localScale.z : cur.localScale.x;
                    float nxtLen  = isNS ? nxt.localScale.z : nxt.localScale.x;
                    float curEnd  = (isNS ? cur.position.z : cur.position.x) + curLen * 0.5f;
                    float nxtStart= (isNS ? nxt.position.z : nxt.position.x) - nxtLen * 0.5f;
                    gapToNext = (nxtStart - curEnd) > gapThreshold;
                }

                if (isLastInGroup || gapToNext)
                {
                    // Emit one merged wall for this contiguous run.
                    if (run.Count >= 2)
                    {
                        Transform first = run[0], last = run[run.Count - 1];
                        float fullFirst = isNS ? first.localScale.z : first.localScale.x;
                        float fullLast  = isNS ? last.localScale.z  : last.localScale.x;
                        float start     = (isNS ? first.position.z : first.position.x) - fullFirst * 0.5f;
                        float end       = (isNS ? last.position.z  : last.position.x)  + fullLast  * 0.5f;
                        float midRun    = (start + end) * 0.5f;
                        float totalLen  = end - start;
                        float lateral   = isNS ? first.position.x : first.position.z;

                        Vector3 mergedPos = isNS
                            ? new Vector3(lateral, wallHeight * 0.5f, midRun)
                            : new Vector3(midRun,  wallHeight * 0.5f, lateral);

                        var merged = GameObject.CreatePrimitive(PrimitiveType.Cube);
                        merged.name = "Curb";
                        merged.layer = wallLayer;
                        merged.transform.SetParent(parent, false);
                        merged.transform.position = mergedPos;
                        merged.transform.rotation = Quaternion.identity;
                        merged.transform.localScale = isNS
                            ? new Vector3(wallThickness, wallHeight, totalLen)
                            : new Vector3(totalLen, wallHeight, wallThickness);

                        float lateralKey = Mathf.Round(lateral / wallThickness);
                        float keyMin = isNS ? nsKeyMin : ewKeyMin;
                        float keyMax = isNS ? nsKeyMax : ewKeyMax;
                        bool isOuter = Mathf.Approximately(lateralKey, keyMin) ||
                                       Mathf.Approximately(lateralKey, keyMax);

                        merged.layer = wallLayer;
                        var mergedRend = merged.GetComponent<Renderer>();
                        if (isOuter || !outerWallsOnlyVisible)
                        {
                            SetRendererColor(mergedRend, railColor);
                            DisableShadows(mergedRend);
                        }
                        else
                        {
                            mergedRend.enabled = false;
                        }
                    }
                    // Destroy originals in this run.
                    foreach (var c in run)
                    {
                        if (run.Count >= 2) // only destroy if actually merged
                        {
                            if (Application.isPlaying) Destroy(c.gameObject);
                            else DestroyImmediate(c.gameObject);
                        }
                    }
                    run = new List<Transform>();
                }
            }
        }
    }

    /// <summary>
    /// Wall the two sides of this tile that aren't the entry/exit edges, so
    /// the corridor is bounded on both sides (straights) / the outer corner
    /// (curves). Walls are named "Curb" (crash detection) on the raycast
    /// layer (so the agent sees them).
    /// </summary>
    private void BuildCellWalls(Transform parent, Vector2Int cell, Vector3 center,
                               Vector2Int inDir, Vector2Int outDir, int wallLayer)
    {
        // Open sides: where the car enters (-inDir) and exits (outDir).
        Vector2Int entrySide = new Vector2Int(-inDir.x, -inDir.y);
        Vector2Int exitSide = outDir;

        foreach (var d in DIRS)
        {
            if (d == entrySide || d == exitSide) continue; // open edge
            // Wall sits at the tile edge midpoint on side d.
            Vector3 edgeMid = center + new Vector3(d.x, 0f, d.y) * roadHalfWidth;
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = "Curb";
            wall.layer = wallLayer;
            wall.transform.SetParent(parent, false);
            wall.transform.position = edgeMid + new Vector3(0f, wallHeight * 0.5f, 0f);
            // Length runs perpendicular to d (along the edge); thickness along d.
            bool edgeRunsAlongX = d.y != 0; // north/south edge -> wall spans X
            float lenX = edgeRunsAlongX ? tileSize : wallThickness;
            float lenZ = edgeRunsAlongX ? wallThickness : tileSize;
            wall.transform.localScale = new Vector3(lenX, wallHeight, lenZ);
        }
    }
}
