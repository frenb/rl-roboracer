using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Swaps the car prefab's visuals for the detailed JetRacer FBX without touching
// physics: the WheelColliders, BoxCollider, Rigidbody, CarController, CSI camera
// pose and RaycastOrigin stay as they are, so trained policies and demos remain valid.
//
//   Unity.exe -quit -projectPath unity6.6 -executeMethod JetRacerVisualSetup.Apply -logFile <log>
//   or menu JetRacer > Apply Visual Model
//
// Re-runnable: replaces the JetRacerVisual child it made last time.
public static class JetRacerVisualSetup
{
    const string FbxPath = "Assets/JetRacer/sim2real-JetRacer-v1.fbx";
    const string PrefabPath = "Assets/JetRacer_Physics_wcourse.prefab";
    const string ContainerName = "JetRacerVisual";
    const string LayerName = "CarVisual";
    const int LayerIndex = 8;
    // Old visuals: JetRacer_Jetson_Nano_No_Wheels.fbx and jet_racer_wheel.fbx.
    static readonly string[] OldModelGuids = { "dee89233e0d0a854184a9170c8077c4f", "1b58b4ab47b1a0e46a5fe503016beb70" };

    [MenuItem("JetRacer/Apply Visual Model")]
    public static void Apply()
    {
        bool batch = Environment.GetCommandLineArgs().Contains("-executeMethod");
        try
        {
            Run();
            if (batch) EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.LogError("JetRacerVisualSetup: FAILED " + e);
            if (batch) EditorApplication.Exit(1);
        }
    }

    static void Run()
    {
        ConfigureImporter();
        EnsureLayer();

        GameObject model = AssetDatabase.LoadAssetAtPath<GameObject>(FbxPath);
        if (model == null) throw new Exception("model not found at " + FbxPath);

        GameObject root = PrefabUtility.LoadPrefabContents(PrefabPath);
        try
        {
            Transform old = root.transform.Find(ContainerName);
            if (old != null) UnityEngine.Object.DestroyImmediate(old.gameObject);

            int hidden = HideOldVisuals(root);
            var car = root.GetComponent<CarController>();
            var box = root.GetComponent<BoxCollider>();
            float k = JetRacerPhysicsToReal.K;

            var container = new GameObject(ContainerName).transform;
            container.SetParent(root.transform, false);
            var inst = (GameObject)PrefabUtility.InstantiatePrefab(model, container);
            inst.name = "sim2real-JetRacer-v1";
            Transform t = inst.transform;

            Transform kL = Find(t, "R_knuckle_L"), kR = Find(t, "R_knuckle_R");
            Transform rl = Find(t, "R_wheel_RL"), rr = Find(t, "R_wheel_RR");

            // Orient: forward = rear axle -> front axle, right = left kingpin -> right kingpin.
            Vector3 front = Mid(container, kL, kR), rear = Mid(container, rl, rr);
            Vector3 fwd = (front - rear).normalized;
            Vector3 right = (P(container, kR) - P(container, kL)).normalized;
            Vector3 up = Vector3.Cross(fwd, right).normalized;
            Quaternion fix = Quaternion.Inverse(Quaternion.LookRotation(fwd, up));
            t.localRotation = fix * t.localRotation;
            t.localPosition = fix * t.localPosition;

            // Scale: the model's real wheelbase at the course's sim-units-per-meter.
            float wbNow = Vector3.Distance(Mid(container, kL, kR), Mid(container, rl, rr));
            float wbReal = 0.1524f;
            float factor = wbReal * k / wbNow;
            t.localScale *= factor;
            t.localPosition *= factor;

            // Place: model mid-wheelbase over the colliders' mid-wheelbase, tyres on the ground.
            Vector3 mid = (Mid(container, kL, kR) + Mid(container, rl, rr)) * 0.5f;
            float tyreBottom = LowestTyrePoint(container, t);
            t.localPosition -= new Vector3(mid.x, tyreBottom, mid.z);

            Vector3 colliderMid = Vector3.zero; float groundY = 0f; int n = 0;
            foreach (AxleInfo a in car.axleInfos)
                foreach (WheelCollider w in new[] { a.leftWheel, a.rightWheel })
                {
                    Vector3 c = root.transform.InverseTransformPoint(w.transform.TransformPoint(w.center));
                    colliderMid += c;
                    groundY += c.y - w.suspensionDistance * (1f - w.suspensionSpring.targetPosition) - w.radius;
                    n++;
                }
            colliderMid /= n; groundY /= n;
            container.localPosition = new Vector3(colliderMid.x, groundY, colliderMid.z);

            SetLayer(container, LayerIndex);
            foreach (Camera cam in root.GetComponentsInChildren<Camera>(true))
            {
                if (cam.name.IndexOf("Csi", StringComparison.OrdinalIgnoreCase) < 0) continue;
                cam.cullingMask &= ~(1 << LayerIndex);
                Debug.Log($"JetRacerVisualSetup: {cam.name} no longer renders layer {LayerName}");
            }

            var rig = container.gameObject.AddComponent<JetRacerVisualRig>();
            rig.car = car;
            rig.container = container;
            rig.knuckleL = kL; rig.knuckleR = kR;
            rig.wheelFL = Find(t, "R_wheel_FL"); rig.wheelFR = Find(t, "R_wheel_FR");
            rig.wheelRL = rl; rig.wheelRR = rr;
            rig.servoHorn = Find(t, "R_servo_horn");
            rig.dragLink = Find(t, "R_draglink"); rig.dragFrom = Find(t, "R_U_L"); rig.dragTo = Find(t, "R_B_horn");
            rig.tieRod = Find(t, "R_tierod"); rig.tieFrom = Find(t, "R_J_L"); rig.tieTo = Find(t, "R_J_R");
            rig.lidarRotor = Find(t, "R_lidar_rotor"); rig.lidarPulley = Find(t, "R_lidar_pulley_spin");

            Debug.Log($"JetRacerVisualSetup: hid {hidden} old renderers; scale {k:0.###} sim units/m, model x{factor:0.###}; " +
                      $"container at {container.localPosition}; left kingpin x = {container.InverseTransformPoint(kL.position).x:0.###} (expect < 0)");
            Verify(rig, k);

            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            Debug.Log("JetRacerVisualSetup: saved " + PrefabPath);
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(root);
        }
        RenderPreviews();
    }

    static void ConfigureImporter()
    {
        var imp = (ModelImporter)AssetImporter.GetAtPath(FbxPath);
        if (imp == null) throw new Exception("no ModelImporter for " + FbxPath);
        imp.addCollider = false;
        imp.importAnimation = false;
        imp.animationType = ModelImporterAnimationType.None;
        imp.importCameras = false;
        imp.importLights = false;
        imp.importBlendShapes = false;
        imp.SaveAndReimport();
    }

    static void EnsureLayer()
    {
        var tm = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
        SerializedProperty layers = tm.FindProperty("layers");
        SerializedProperty slot = layers.GetArrayElementAtIndex(LayerIndex);
        if (slot.stringValue == LayerName) return;
        if (!string.IsNullOrEmpty(slot.stringValue))
            throw new Exception($"layer {LayerIndex} is already '{slot.stringValue}'");
        slot.stringValue = LayerName;
        tm.ApplyModifiedProperties();
    }

    static int HideOldVisuals(GameObject root)
    {
        var oldPaths = OldModelGuids.Select(AssetDatabase.GUIDToAssetPath).ToArray();
        int count = 0;
        foreach (MeshRenderer r in root.GetComponentsInChildren<MeshRenderer>(true))
        {
            var mf = r.GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null) continue;
            if (!oldPaths.Contains(AssetDatabase.GetAssetPath(mf.sharedMesh))) continue;
            r.enabled = false;
            count++;
        }
        return count;
    }

    // Steer sweep: the drag link and tie rod are rigid on the real car, so their
    // end-to-end lengths must not change. Also confirms the knuckle yaw sign.
    static void Verify(JetRacerVisualRig rig, float k)
    {
        rig.Capture();
        float mm = 1000f / k;
        float drag0 = Vector3.Distance(rig.dragFrom.position, rig.dragTo.position) * mm;
        float tie0 = Vector3.Distance(rig.tieFrom.position, rig.tieTo.position) * mm;
        float worstDrag = 0f, worstTie = 0f;
        foreach (float steer in new[] { -17f, -12f, -6f, 0f, 6f, 12f, 17f })
        {
            rig.ApplyPose(steer);
            float drag = Vector3.Distance(rig.dragFrom.position, rig.dragTo.position) * mm;
            float tie = Vector3.Distance(rig.tieFrom.position, rig.tieTo.position) * mm;
            Vector3 wheelAxle = rig.container.InverseTransformDirection(rig.wheelFL.position - rig.knuckleL.position);
            float wheelYaw = Vector3.SignedAngle(Vector3.left, new Vector3(wheelAxle.x, 0f, wheelAxle.z), Vector3.up);
            worstDrag = Mathf.Max(worstDrag, Mathf.Abs(drag - drag0));
            worstTie = Mathf.Max(worstTie, Mathf.Abs(tie - tie0));
            Debug.Log($"JetRacerVisualSetup: VERIFY steer {steer,6:0.0} deg  drag link {drag:0.000} mm  tie rod {tie:0.000} mm  " +
                      $"left spindle yaw {wheelYaw:0.0} deg");
        }
        rig.ApplyPose(0f);
        Debug.Log($"JetRacerVisualSetup: VERIFY drag link {drag0:0.00} mm, worst change {worstDrag:0.000} mm; tie rod {tie0:0.00} mm, worst change {worstTie:0.000} mm");
    }

    static void RenderPreviews()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var car = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath));
        car.transform.position = Vector3.zero;
        car.transform.rotation = Quaternion.identity;
        var rig = car.GetComponentInChildren<JetRacerVisualRig>();
        rig.Capture();

        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.transform.position = rig.container.position;
        ground.transform.localScale = Vector3.one * 3f;
        var light = new GameObject("Sun").AddComponent<Light>();
        light.type = LightType.Directional;
        light.shadows = LightShadows.Soft;
        light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        RenderSettings.ambientLight = new Color(0.45f, 0.45f, 0.5f);

        var cam = new GameObject("PreviewCam").AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.82f, 0.85f, 0.9f);
        cam.fieldOfView = 30f;
        cam.nearClipPlane = 0.05f;
        var rt = new RenderTexture(1280, 800, 24) { antiAliasing = 8 };
        cam.targetTexture = rt;

        string outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Builds", "previews"));
        Directory.CreateDirectory(outDir);
        Vector3 focus = rig.container.position + Vector3.up * 1.2f;
        var shots = new (string name, float steer, Vector3 offset, Vector3 look)[]
        {
            ("three_quarter_left_lock", -17f, new Vector3(-9f, 6f, 10f), focus),
            ("three_quarter_right_lock", 17f, new Vector3(-9f, 6f, 10f), focus),
            ("top_right_lock", 17f, new Vector3(0f, 16f, 0.01f), focus),
            ("front_closeup_right_lock", 17f, new Vector3(-2.5f, 3.5f, 6.5f), rig.container.position + new Vector3(0f, 0.7f, 1.3f)),
        };
        foreach (var s in shots)
        {
            rig.ApplyPose(s.steer);
            cam.transform.position = rig.container.position + s.offset;
            cam.transform.LookAt(s.look);
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            string path = Path.Combine(outDir, $"jetracer_v1_{s.name}.png");
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Debug.Log("JetRacerVisualSetup: preview " + path);
        }
        RenderTexture.active = null;
    }

    static Transform Find(Transform t, string name)
    {
        if (t.name == name) return t;
        foreach (Transform c in t)
        {
            Transform f = Find(c, name);
            if (f != null) return f;
        }
        return null;
    }

    static Vector3 P(Transform space, Transform t) => space.InverseTransformPoint(t.position);
    static Vector3 Mid(Transform space, Transform a, Transform b) => (P(space, a) + P(space, b)) * 0.5f;

    static float LowestTyrePoint(Transform space, Transform model)
    {
        float min = float.MaxValue;
        foreach (MeshFilter mf in model.GetComponentsInChildren<MeshFilter>(true))
        {
            if (!mf.name.Contains("tyre") || mf.sharedMesh == null) continue;
            foreach (Vector3 v in mf.sharedMesh.vertices)
                min = Mathf.Min(min, space.InverseTransformPoint(mf.transform.TransformPoint(v)).y);
        }
        if (min == float.MaxValue) throw new Exception("no tyre meshes found");
        return min;
    }

    static void SetLayer(Transform t, int layer)
    {
        t.gameObject.layer = layer;
        foreach (Transform c in t) SetLayer(c, layer);
    }
}
