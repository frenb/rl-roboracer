using System.Text;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Prints the car prefab's physics and sensor geometry in real millimetres
/// (sim units / Real2SimCourse scale), for checking against
/// robots/jetracer/jetracer.urdf. Batch: -executeMethod CarGeometryDump.Run
/// </summary>
public static class CarGeometryDump
{
    const string PrefabPath = "Assets/JetRacer_Physics_wcourse.prefab";

    [MenuItem("Tools/JetRacer/Dump Car Geometry")]
    public static void Run()
    {
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        var car = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        car.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        try { Debug.Log(Describe(car)); }
        finally { Object.DestroyImmediate(car); }
    }

    static string Describe(GameObject car)
    {
        var box = car.GetComponent<BoxCollider>();
        Vector3 boxSize = Vector3.Scale(box.size, car.transform.lossyScale);
        float k = JetRacerPhysicsToReal.K;
        var sb = new StringBuilder();
        sb.AppendLine($"[CarGeometry] k={k:0.###} sim units per m; root scale {car.transform.lossyScale}; forward {car.transform.forward}");
        string Mm(Vector3 v) => $"({v.x / k * 1000f:0.0}, {v.y / k * 1000f:0.0}, {v.z / k * 1000f:0.0}) mm";
        Vector3 boxCenter = car.transform.TransformPoint(box.center);
        sb.AppendLine($"[CarGeometry] box size {boxSize} = {Mm(boxSize)}; center {Mm(boxCenter)}");

        var rb = car.GetComponent<Rigidbody>();
        if (rb) sb.AppendLine($"[CarGeometry] rigidbody mass {rb.mass}, centerOfMass {Mm(car.transform.TransformPoint(rb.centerOfMass))}");

        foreach (var w in car.GetComponentsInChildren<WheelCollider>(true))
        {
            Vector3 c = w.transform.TransformPoint(w.center);
            float r = w.radius * w.transform.lossyScale.y;
            float susp = w.suspensionDistance * w.transform.lossyScale.y;
            Vector3 axle = c - Vector3.up * susp * (1f - w.suspensionSpring.targetPosition);
            sb.AppendLine($"[CarGeometry] wheel {w.name}: axle at rest {Mm(axle)}, radius {r / k * 1000f:0.0} mm, suspension {susp / k * 1000f:0.0} mm, " +
                          $"ground {(axle.y - r) / k * 1000f:0.0} mm, spring {w.suspensionSpring.spring}, damper {w.suspensionSpring.damper}");
        }

        var cc = car.GetComponent<CarController>();
        if (cc)
        {
            Transform o = cc.raycastOrigin != null ? cc.raycastOrigin : cc.transform;
            sb.AppendLine($"[CarGeometry] lidar ray origin '{o.name}' {Mm(o.position)}, sphere radius 0.15 = {0.15f / k * 1000f:0.0} mm");
        }

        foreach (var cam in car.GetComponentsInChildren<Camera>(true))
            sb.AppendLine($"[CarGeometry] camera '{cam.name}' {Mm(cam.transform.position)}, euler {cam.transform.eulerAngles}, vfov {cam.fieldOfView:0.#}");

        return sb.ToString();
    }
}
