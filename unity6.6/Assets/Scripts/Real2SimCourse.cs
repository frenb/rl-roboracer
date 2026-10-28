using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A drivable course rebuilt from a video of the physical track
/// (scripts/real2sim/video_to_layout.py writes the layout.json and mat.png it
/// reads). Builds the same objects TrackGenerator does, so CarController and
/// SimController need no changes:
///   * "Road": the mat, textured with the printed design, over a floor box
///     (Ignore Raycast, so perception rays pass through).
///   * "Curb": one wall per physical fence panel, on the raycast layer, so
///     the car senses them and hitting one counts as a crash.
///   * "Goal-1".."Goal-N": trigger gates across the road along the layout's
///     centerline, in driving order, Goal-1 nearest the layout's start pose.
///
/// The layout is in real meters with the mat's corner at this object's
/// origin; layout x maps to local X and layout y to local Z. scale converts
/// real meters to sim units; by default it is derived from the car so the
/// sim car fills the lane as the real JetRacer does (the sim car is not
/// proportioned like the real one, so width, the clearance that matters
/// between fences, is what is matched).
/// </summary>
public class Real2SimCourse : MonoBehaviour
{
    [Header("Source (from scripts/real2sim)")]
    [Tooltip("layout.json written by video_to_layout.py.")]
    public TextAsset layoutJson;
    [Tooltip("mat.png written by video_to_layout.py: the mat's printed design, seen from above.")]
    public Texture2D matTexture;

    [Header("Scale")]
    [Tooltip("Size the course to the sim car: scale = the car prefab's collider width / realCarSize.x, " +
             "so the car has the same room between the fences as the real one. Off: use scale as set.")]
    public bool scaleToCar = true;
    [Tooltip("The real car, meters (width, height, length). Waveshare JetRacer Pro: 212 x 120 x 320 mm.")]
    public Vector3 realCarSize = new Vector3(0.212f, 0.12f, 0.32f);
    [Tooltip("The car to size the course to. Empty: SimController's carPrefab.")]
    public GameObject carPrefab;
    [Tooltip("Sim units per real meter. Computed from the car when scaleToCar is on.")]
    public float scale = 16.6f;
    public float trackY = 0f;

    [Header("Walls (fence panels)")]
    [Tooltip("Must be in CarController.layerMask (Default) so the car's SphereCasts see the walls.")]
    public string wallLayerName = "Default";
    [Tooltip("Wall thickness in sim units. The real panels are millimetres thick, too thin for stable physics.")]
    public float wallThickness = 0.2f;
    [Tooltip("Panel height in real meters when the layout gives none.")]
    public float defaultPanelHeight = 0.35f;
    [Tooltip("Off hides the walls but keeps their colliders, so crashes are still detected.")]
    public bool showBarriers = true;
    public Color wallColor = new Color(0.62f, 0.62f, 0.6f);
    public Color floorColor = new Color(0.72f, 0.62f, 0.5f);

    [Header("Lane")]
    [Tooltip("Real meters to pull each side of the inner barrier ring toward its centre. The ring stays a " +
             "square-cornered rectangle; this only widens the lane so the sim car has room to turn.")]
    public float innerBarrierInset = 0.05f;
    [Tooltip("Replace the layout's centerline (traced from the printed lane markings) with one derived from " +
             "the fences: mid-lane on each side, and around each inner barrier corner a quarter ellipse " +
             "centred on that corner, so the line keeps half the lane width from the post. Needs an outer " +
             "fence and one inner barrier ring; otherwise the layout's centerline is used.")]
    public bool centerlineFromFences = true;
    [Tooltip("Inner barrier panels are those with both ends at least this far (real meters) inside the mat edge.")]
    public float innerBarrierEdgeMargin = 0.15f;

    [Header("Goals")]
    [Tooltip("Distance between goal gates along the centerline, in real meters.")]
    public float goalSpacing = 0.3f;
    [Tooltip("Height of the gate's centre above the road, in sim units (the car spawns at a gate).")]
    public float goalHeight = 1.5f;
    public float goalTriggerHeight = 4f;
    public float goalGateThickness = 0.5f;
    [Tooltip("Off hides the gates but keeps their triggers, so goals are still counted.")]
    public bool showGoals = true;
    public Color goalColor = new Color(1f, 0.85f, 0.1f, 1f);

    [Header("Lifecycle")]
    [Tooltip("Build in Awake(), before SimController.Start() looks for Goal-1. Required for play/training.")]
    public bool generateOnAwake = true;
    public bool autoRegenerateInEditor = true;

    private const string ROOT_NAME = "Real2SimTrack";

    [Serializable] public class Pt { public float x, y; }
    [Serializable] public class Mat { public float width, length; public string base_color; }
    [Serializable] public class Road { public float width; public bool closed = true; public Pt[] centerline; }
    [Serializable] public class Panel { public float x1, y1, x2, y2, height; }
    [Serializable] public class Pose { public float x, y, heading_deg; }
    [Serializable] public class Layout { public string name; public Mat mat; public Road road; public Panel[] fence_panels; public Pose start; }

    private static MaterialPropertyBlock _mpb;
    private Material _matMaterial;

    void Awake()
    {
        if (generateOnAwake) Generate();
    }

#if UNITY_EDITOR
    void OnValidate()
    {
        if (Application.isPlaying || !autoRegenerateInEditor) return;
        UnityEditor.EditorApplication.delayCall += _DeferredEditorRegen;
    }

    private void _DeferredEditorRegen()
    {
        UnityEditor.EditorApplication.delayCall -= _DeferredEditorRegen;
        if (this == null || Application.isPlaying) return;
        Generate();
    }
#endif

    public Layout Parse()
    {
        if (layoutJson == null) return null;
        var layout = JsonUtility.FromJson<Layout>(layoutJson.text);
        if (layout == null || layout.mat == null || layout.road == null || layout.road.centerline == null || layout.road.centerline.Length < 3)
        {
            Debug.LogError($"[Real2SimCourse] {layoutJson.name} is not a usable layout (needs mat, road.centerline).");
            return null;
        }
        return layout;
    }

    [ContextMenu("Generate Track")]
    public void Generate()
    {
        Clear();
        Layout layout = Parse();
        if (layout == null) return;
        if (scaleToCar) ScaleToCar();

        var root = new GameObject(ROOT_NAME);
        root.transform.SetParent(transform, false);
        int wallLayer = LayerMask.NameToLayer(wallLayerName);
        if (wallLayer < 0) wallLayer = 0;

        float w = layout.mat.width * scale, l = layout.mat.length * scale;
        BuildFloorAndMat(root.transform, w, l);

        ShapeLane(layout);

        int walls = 0;
        foreach (var p in layout.fence_panels ?? new Panel[0])
        {
            Vector3 a = Local(p.x1, p.y1), b = Local(p.x2, p.y2);
            float h = (p.height > 0f ? p.height : defaultPanelHeight) * scale;
            if (PlaceWall(root.transform, a, b, h, wallLayer)) walls++;
        }

        int goals = PlaceGoals(root.transform, layout);

        if (Application.isPlaying)
            StaticBatchingUtility.Combine(root);
        Debug.Log($"[Real2SimCourse] {layout.name}: mat {w:0.#} x {l:0.#} sim units (scale {scale}), " +
                  $"{walls} walls, {goals} goals. Goal-1 findable={GameObject.Find("Goal-1") != null}.");

        if (Application.isPlaying)
        {
            var liveCar = FindObjectOfType<CarController>();
            if (liveCar != null) liveCar.OnTrackRegenerated();
        }
    }

    [ContextMenu("Clear Track")]
    public void Clear()
    {
        for (int c = transform.childCount - 1; c >= 0; c--)
        {
            Transform child = transform.GetChild(c);
            if (child.name != ROOT_NAME) continue;
            if (Application.isPlaying)
            {
                // Destroy is deferred to the end of the frame; hide the old
                // Goal-N objects from GameObject.Find right away.
                child.gameObject.SetActive(false);
                Destroy(child.gameObject);
            }
            else DestroyImmediate(child.gameObject);
        }
    }

    private void ScaleToCar()
    {
        GameObject car = carPrefab;
        if (car == null)
        {
            var controller = FindObjectOfType<SimController>();
            if (controller != null) car = controller.carPrefab;
        }
        var box = car != null ? car.GetComponent<BoxCollider>() : null;
        if (box == null || realCarSize.x <= 0f)
        {
            Debug.LogWarning($"[Real2SimCourse] no car with a BoxCollider to size the course to; using scale {scale}.");
            return;
        }
        Vector3 s = car.transform.localScale;
        Vector3 sim = Vector3.Scale(box.size, s);
        scale = sim.x / realCarSize.x;
        Debug.Log($"[Real2SimCourse] scale {scale:0.##} from {car.name}: sim car {sim.x:0.##} x {sim.z:0.##} (W x L) " +
                  $"vs real {realCarSize.x} x {realCarSize.z} m; at this scale the real car's length is {realCarSize.z * scale:0.##}.");
    }

    private Vector3 Local(float x, float y) => new Vector3(x * scale, trackY, y * scale);

    /// <summary>
    /// Shrinks the inner barrier ring by innerBarrierInset and, when
    /// centerlineFromFences is on, rebuilds road.centerline from the fences.
    /// Edits the freshly parsed layout in place, in real meters.
    /// </summary>
    private void ShapeLane(Layout layout)
    {
        var panels = layout.fence_panels ?? new Panel[0];
        float m = innerBarrierEdgeMargin, W = layout.mat.width, L = layout.mat.length;
        bool Inside(float x, float y) => x > m && x < W - m && y > m && y < L - m;
        var inner = new List<Panel>();
        var outer = new List<Panel>();
        foreach (var p in panels)
            (Inside(p.x1, p.y1) && Inside(p.x2, p.y2) ? inner : outer).Add(p);
        if (inner.Count == 0 || outer.Count == 0) return;

        Rect ib = Bounds(inner);
        if (innerBarrierInset > 0f && innerBarrierInset * 2f < Mathf.Min(ib.width, ib.height))
        {
            const float tol = 0.02f;
            float Pull(float v, float lo, float hi) =>
                Mathf.Abs(v - lo) < tol ? v + innerBarrierInset
                : Mathf.Abs(v - hi) < tol ? v - innerBarrierInset : v;
            foreach (var p in inner)
            {
                p.x1 = Pull(p.x1, ib.xMin, ib.xMax); p.x2 = Pull(p.x2, ib.xMin, ib.xMax);
                p.y1 = Pull(p.y1, ib.yMin, ib.yMax); p.y2 = Pull(p.y2, ib.yMin, ib.yMax);
            }
            ib = Bounds(inner);
        }

        if (!centerlineFromFences) return;
        Rect ob = Bounds(outer);
        float xL = (ob.xMin + ib.xMin) * 0.5f, xR = (ob.xMax + ib.xMax) * 0.5f;
        float yB = (ob.yMin + ib.yMin) * 0.5f, yT = (ob.yMax + ib.yMax) * 0.5f;
        if (!(xL < ib.xMin && xR > ib.xMax && yB < ib.yMin && yT > ib.yMax))
        {
            Debug.LogWarning("[Real2SimCourse] inner barrier is not inside the outer fence; keeping the layout's centerline.");
            return;
        }

        // Counter-clockwise (x right, y up): a quarter ellipse around each
        // inner corner; the straights are the gaps between consecutive arcs.
        const int arcSegs = 8;
        var pts = new List<Pt>();
        void Arc(float cx, float cy, float rx, float ry, float fromDeg)
        {
            for (int s = 0; s <= arcSegs; s++)
            {
                float a = (fromDeg + 90f * s / arcSegs) * Mathf.Deg2Rad;
                pts.Add(new Pt { x = cx + rx * Mathf.Cos(a), y = cy + ry * Mathf.Sin(a) });
            }
        }
        Arc(ib.xMax, ib.yMin, xR - ib.xMax, ib.yMin - yB, -90f);
        Arc(ib.xMax, ib.yMax, xR - ib.xMax, yT - ib.yMax, 0f);
        Arc(ib.xMin, ib.yMax, ib.xMin - xL, yT - ib.yMax, 90f);
        Arc(ib.xMin, ib.yMin, ib.xMin - xL, ib.yMin - yB, 180f);

        // Keep the driving direction the layout's centerline had.
        if (layout.road == null) layout.road = new Road();
        var old = layout.road.centerline;
        if (old != null && old.Length >= 3 && SignedArea(old) < 0f) pts.Reverse();
        layout.road.centerline = pts.ToArray();
        layout.road.closed = true;
        Debug.Log($"[Real2SimCourse] lane from fences: inner barrier {ib.width:0.###} x {ib.height:0.###} m " +
                  $"(inset {innerBarrierInset} m), lane {ib.xMin - ob.xMin:0.###}/{ob.xMax - ib.xMax:0.###}/" +
                  $"{ib.yMin - ob.yMin:0.###}/{ob.yMax - ib.yMax:0.###} m wide (L/R/B/T).");
    }

    private static Rect Bounds(List<Panel> panels)
    {
        float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
        foreach (var p in panels)
        {
            x0 = Mathf.Min(x0, Mathf.Min(p.x1, p.x2)); x1 = Mathf.Max(x1, Mathf.Max(p.x1, p.x2));
            y0 = Mathf.Min(y0, Mathf.Min(p.y1, p.y2)); y1 = Mathf.Max(y1, Mathf.Max(p.y1, p.y2));
        }
        return Rect.MinMaxRect(x0, y0, x1, y1);
    }

    private static float SignedArea(Pt[] pts)
    {
        float a = 0f;
        for (int i = 0; i < pts.Length; i++)
        {
            Pt p = pts[i], q = pts[(i + 1) % pts.Length];
            a += p.x * q.y - q.x * p.y;
        }
        return a * 0.5f;
    }

    private void BuildFloorAndMat(Transform parent, float w, float l)
    {
        float margin = Mathf.Max(w, l) * 0.5f;
        var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.name = "Road";
        floor.layer = 2;
        floor.transform.SetParent(parent, false);
        floor.transform.localPosition = new Vector3(w * 0.5f, trackY - 0.1f, l * 0.5f);
        floor.transform.localScale = new Vector3(w + 2f * margin, 0.2f, l + 2f * margin);
        Tint(floor.GetComponent<Renderer>(), floorColor);

        // A quad lies in XY facing -Z with UVs along +X/+Y; tipped 90 degrees
        // about X it faces up with the image's top toward +Z (layout +y),
        // which is how video_to_layout.py draws mat.png.
        var mat = GameObject.CreatePrimitive(PrimitiveType.Quad);
        mat.name = "Road";
        mat.layer = 2;
        mat.transform.SetParent(parent, false);
        mat.transform.localPosition = new Vector3(w * 0.5f, trackY + 0.01f, l * 0.5f);
        mat.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        mat.transform.localScale = new Vector3(w, l, 1f);
        var col = mat.GetComponent<Collider>();
        if (Application.isPlaying) Destroy(col); else DestroyImmediate(col);
        var rend = mat.GetComponent<Renderer>();
        if (matTexture != null)
        {
            if (_matMaterial == null)
                _matMaterial = new Material(Shader.Find("Standard")) { name = "Real2SimMat", hideFlags = HideFlags.DontSave };
            _matMaterial.mainTexture = matTexture;
            _matMaterial.SetFloat("_Glossiness", 0.1f);
            rend.sharedMaterial = _matMaterial;
        }
        else Tint(rend, new Color(0.3f, 0.3f, 0.32f));
        rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
    }

    private bool PlaceWall(Transform parent, Vector3 a, Vector3 b, float height, int wallLayer)
    {
        Vector3 d = b - a;
        if (d.magnitude < 1e-3f) return false;
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "Curb"; // CarController counts a collision with "Curb" as a crash.
        wall.layer = wallLayer;
        wall.transform.SetParent(parent, false);
        wall.transform.localPosition = (a + b) * 0.5f + new Vector3(0f, height * 0.5f, 0f);
        wall.transform.localRotation = Quaternion.LookRotation(d.normalized, Vector3.up);
        wall.transform.localScale = new Vector3(wallThickness, height, d.magnitude);
        var rend = wall.GetComponent<Renderer>();
        if (showBarriers) Tint(rend, wallColor); else rend.enabled = false;
        return true;
    }

    private int PlaceGoals(Transform parent, Layout layout)
    {
        var pts = new List<Vector3>();
        foreach (var p in layout.road.centerline) pts.Add(Local(p.x, p.y));
        bool closed = layout.road.closed;
        if (closed && (pts[0] - pts[pts.Count - 1]).magnitude < 1e-3f) pts.RemoveAt(pts.Count - 1);

        // Evenly spaced points by arc length along the centerline.
        var gates = new List<(Vector3 pos, Vector3 dir)>();
        float step = Mathf.Max(0.05f, goalSpacing) * scale;
        int segs = closed ? pts.Count : pts.Count - 1;
        float carry = 0f;
        for (int i = 0; i < segs; i++)
        {
            Vector3 a = pts[i], b = pts[(i + 1) % pts.Count];
            Vector3 d = b - a;
            float len = d.magnitude;
            if (len < 1e-4f) continue;
            if (carry >= len) { carry -= len; continue; }
            for (float t = carry; t < len; t += step)
                gates.Add((a + d * (t / len), d / len));
            carry = step - ((len - carry) % step);
            if (carry >= step) carry = 0f;
        }
        if (gates.Count == 0) return 0;

        int first = 0;
        if (layout.start != null)
        {
            Vector3 s = Local(layout.start.x, layout.start.y);
            float best = float.MaxValue;
            for (int i = 0; i < gates.Count; i++)
            {
                float dist = (gates[i].pos - s).sqrMagnitude;
                if (dist < best) { best = dist; first = i; }
            }
        }

        float width = (layout.road.width > 0f ? layout.road.width : 0.5f) * scale;
        for (int k = 0; k < gates.Count; k++)
        {
            var g = gates[(first + k) % gates.Count];
            var goal = GameObject.CreatePrimitive(PrimitiveType.Cube);
            goal.name = "Goal-" + (k + 1); // the exact names CarController.SetUpGoalsArray looks up
            goal.layer = 6; // the goal layer TrackGenerator uses: out of the car's raycast mask
            goal.transform.SetParent(parent, false);
            goal.transform.localPosition = g.pos + new Vector3(0f, goalHeight, 0f);
            goal.transform.localRotation = Quaternion.LookRotation(g.dir, Vector3.up);
            goal.transform.localScale = new Vector3(width, goalTriggerHeight, goalGateThickness);
            goal.GetComponent<Collider>().isTrigger = true;
            var rend = goal.GetComponent<Renderer>();
            if (showGoals) Tint(rend, goalColor); else rend.enabled = false;
            if (goal.GetComponent<Goal>() == null) goal.AddComponent<Goal>();
        }
        return gates.Count;
    }

    private static void Tint(Renderer rend, Color color)
    {
        if (rend == null) return;
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        rend.GetPropertyBlock(_mpb);
        _mpb.SetColor("_Color", color);
        rend.SetPropertyBlock(_mpb);
        rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        rend.receiveShadows = false;
    }
}
