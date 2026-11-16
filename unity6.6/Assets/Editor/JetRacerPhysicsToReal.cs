using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

// Moves the car prefab's physics and sensors to the real JetRacer's geometry
// (robots/jetracer/jetracer.urdf) at the course scale: wheel colliders, box
// collider, lidar RaycastOrigin and CSI camera pose. Ground level and the box's
// fore-aft centre stay where they were, so spawn poses are unchanged.
//
//   Unity.exe -batchmode -quit -projectPath unity6.6 -executeMethod JetRacerPhysicsToReal.Apply -logFile <log>
//
// Re-runnable: every value is set absolutely. Run JetRacerVisualSetup.Apply
// afterwards so the visual model sits on the moved wheels.
public static class JetRacerPhysicsToReal
{
    const string PrefabPath = "Assets/JetRacer_Physics_wcourse.prefab";

    // Sim units per real meter. Real2SimCourse.scale is pinned to the same value.
    public const float K = 17.6f;

    // Real geometry, meters. Rear axle centre is the origin for fore-aft offsets.
    const float Wheelbase = 0.1524f;
    const float FrontTrack = 0.1524f;
    const float RearTrack = 0.170f;
    const float TyreRadius = 0.0325f;
    const float BodyLength = 0.2429f;
    const float BodyWidth = 0.195f;          // rear track + one 25 mm tyre
    const float BodyRearOfRearAxle = 0.0365f;
    const float GroundClearance = 0.025f;
    const float TopHeight = 0.1476f;         // lidar top
    const float LidarHeight = 0.1376f;       // scan plane
    const float LidarAheadOfRearAxle = 0.0966f;
    const float CameraHeight = 0.108f;       // lens centre
    const float CameraAheadOfRearAxle = 0.1701f;
    const float CameraPitchDownDeg = 9.8f;

    [MenuItem("JetRacer/Move Physics To Real Geometry")]
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
            Debug.LogError("JetRacerPhysicsToReal: FAILED " + e);
            if (batch) EditorApplication.Exit(1);
        }
    }

    static float S(float meters) => meters * K;

    static void Run()
    {
        GameObject root = PrefabUtility.LoadPrefabContents(PrefabPath);
        try
        {
            Transform rt = root.transform;
            var car = root.GetComponent<CarController>();
            var box = root.GetComponent<BoxCollider>();
            if (Mathf.Abs(rt.lossyScale.x - 1f) > 1e-4f) throw new Exception("expected unit root scale, got " + rt.lossyScale);

            float groundY = 0f; int n = 0;
            foreach (AxleInfo a in car.axleInfos)
                foreach (WheelCollider w in new[] { a.leftWheel, a.rightWheel })
                {
                    groundY += RestAxleY(rt, w) - w.radius * w.transform.lossyScale.y;
                    n++;
                }
            groundY /= n;

            float boxCenterZ = box.center.z;
            float rearAxleZ = boxCenterZ - S(BodyLength) * 0.5f + S(BodyRearOfRearAxle);
            float axleY = groundY + S(TyreRadius);

            var axles = car.axleInfos.OrderBy(a => rt.InverseTransformPoint(a.leftWheel.transform.TransformPoint(a.leftWheel.center)).z).ToList();
            PlaceAxle(rt, axles[0], rearAxleZ, S(RearTrack), axleY);
            PlaceAxle(rt, axles[axles.Count - 1], rearAxleZ + S(Wheelbase), S(FrontTrack), axleY);

            float bottom = groundY + S(GroundClearance), top = groundY + S(TopHeight);
            box.size = new Vector3(S(BodyWidth), top - bottom, S(BodyLength));
            box.center = new Vector3(0f, (top + bottom) * 0.5f, boxCenterZ);

            if (car.raycastOrigin == null) throw new Exception("CarController.raycastOrigin is not set");
            car.raycastOrigin.position = rt.TransformPoint(new Vector3(0f, groundY + S(LidarHeight), rearAxleZ + S(LidarAheadOfRearAxle)));

            Camera csi = root.GetComponentsInChildren<Camera>(true).FirstOrDefault(c => c.name.IndexOf("Csi", StringComparison.OrdinalIgnoreCase) >= 0);
            if (csi == null) throw new Exception("no CSI camera under the prefab");
            csi.transform.SetPositionAndRotation(
                rt.TransformPoint(new Vector3(0f, groundY + S(CameraHeight), rearAxleZ + S(CameraAheadOfRearAxle))),
                rt.rotation * Quaternion.Euler(CameraPitchDownDeg, 0f, 0f));

            Debug.Log($"JetRacerPhysicsToReal: ground y {groundY:0.###}, rear axle z {rearAxleZ:0.###}, box {box.size} at {box.center}, " +
                      $"lidar {rt.InverseTransformPoint(car.raycastOrigin.position)}, camera {rt.InverseTransformPoint(csi.transform.position)}");
            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            Debug.Log("JetRacerPhysicsToReal: saved " + PrefabPath);
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(root);
        }
    }

    // Unity hangs the wheel suspensionDistance * (1 - targetPosition) below the collider centre at rest.
    static float RestDrop(WheelCollider w) =>
        w.suspensionDistance * (1f - w.suspensionSpring.targetPosition) * w.transform.lossyScale.y;

    static float RestAxleY(Transform rt, WheelCollider w) =>
        rt.InverseTransformPoint(w.transform.TransformPoint(w.center)).y - RestDrop(w);

    static void PlaceAxle(Transform rt, AxleInfo a, float z, float track, float axleY)
    {
        foreach (var (w, x) in new[] { (a.leftWheel, -track * 0.5f), (a.rightWheel, track * 0.5f) })
        {
            Vector3 want = new Vector3(x, axleY + RestDrop(w), z);
            w.center = w.transform.InverseTransformPoint(rt.TransformPoint(want));
            w.radius = S(TyreRadius) / w.transform.lossyScale.y;
        }
    }
}
