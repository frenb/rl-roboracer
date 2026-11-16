using UnityEngine;

/// <summary>
/// A see-through, true-to-scale Waveshare JetRacer (robots/jetracer/jetracer.urdf)
/// riding on the sim car, so the two can be compared against the real2sim track.
/// Its rear axle sits on the sim car's rear axle and its front wheels follow the
/// sim car's steering. Real meters become sim units through Real2SimCourse.scale.
///
/// Off unless the player is started with --real-car-proxy: the proxy would
/// otherwise show up in the onboard camera's frames. It has no colliders and
/// sits on the Ignore Raycast layer, so the car's SphereCasts never hit it.
/// </summary>
public class RealCarProxy : MonoBehaviour
{
    const float WheelRadius = 0.03175f, WheelWidth = 0.0254f;
    const float Wheelbase = 0.1524f, FrontTrack = 0.1524f, RearTrack = 0.1746f;

    public Color bodyColor = new Color(1f, 0.2f, 0.8f, 0.45f);
    public Color wheelColor = new Color(0.1f, 1f, 1f, 0.6f);

    CarController car;
    Real2SimCourse course;
    Transform frontLeft, frontRight;
    Material bodyMat, wheelMat;
    Renderer[] renderers;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "--real-car-proxy") < 0) return;
        new GameObject("RealCarProxy").AddComponent<RealCarProxy>();
    }

    void Awake()
    {
        var shader = Shader.Find("Sprites/Default");
        bodyMat = new Material(shader) { color = bodyColor };
        wheelMat = new Material(shader) { color = wheelColor };
        gameObject.layer = 2;

        // URDF frame (x forward, y left, z up) -> Unity local (z forward, x right, y up),
        // origin on the ground under the middle of the rear axle.
        Part(PrimitiveType.Cube, transform, new Vector3(0f, WheelRadius + 0.023f, 0.085f), new Vector3(0.100f, 0.060f, 0.243f), bodyMat);
        Part(PrimitiveType.Cylinder, transform, new Vector3(0f, WheelRadius + 0.103f, 0.095f), new Vector3(0.070f, 0.020f, 0.070f), bodyMat);
        Part(PrimitiveType.Cube, transform, new Vector3(0f, WheelRadius + 0.078f, 0.170f), new Vector3(0.025f, 0.025f, 0.015f), bodyMat);
        Wheel(transform, new Vector3(-RearTrack / 2f, WheelRadius, 0f));
        Wheel(transform, new Vector3(RearTrack / 2f, WheelRadius, 0f));
        frontLeft = Knuckle(new Vector3(-FrontTrack / 2f, WheelRadius, Wheelbase));
        frontRight = Knuckle(new Vector3(FrontTrack / 2f, WheelRadius, Wheelbase));
        renderers = GetComponentsInChildren<Renderer>(true);
    }

    Transform Knuckle(Vector3 pos)
    {
        var k = new GameObject("knuckle").transform;
        k.SetParent(transform, false);
        k.localPosition = pos;
        k.gameObject.layer = 2;
        Wheel(k, Vector3.zero);
        return k;
    }

    void Wheel(Transform parent, Vector3 pos)
    {
        var w = Part(PrimitiveType.Cylinder, parent, pos, new Vector3(2f * WheelRadius, WheelWidth / 2f, 2f * WheelRadius), wheelMat);
        w.localRotation = Quaternion.Euler(0f, 0f, 90f);
    }

    static Transform Part(PrimitiveType type, Transform parent, Vector3 pos, Vector3 size, Material mat)
    {
        var go = GameObject.CreatePrimitive(type);
        DestroyImmediate(go.GetComponent<Collider>());
        go.layer = 2;
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        go.transform.localScale = size;
        go.GetComponent<Renderer>().sharedMaterial = mat;
        return go.transform;
    }

    void LateUpdate()
    {
        if (course == null) course = FindAnyObjectByType<Real2SimCourse>();
        if (car == null) car = FindAnyObjectByType<CarController>();
        if (car == null || course == null || car.axleInfos == null || car.axleInfos.Count == 0)
        {
            SetVisible(false);
            return;
        }

        AxleInfo rear = car.axleInfos[0];
        foreach (var a in car.axleInfos)
            if (car.transform.InverseTransformPoint(a.leftWheel.transform.TransformPoint(a.leftWheel.center)).z <
                car.transform.InverseTransformPoint(rear.leftWheel.transform.TransformPoint(rear.leftWheel.center)).z)
                rear = a;

        rear.leftWheel.GetWorldPose(out Vector3 l, out _);
        rear.rightWheel.GetWorldPose(out Vector3 r, out _);
        Vector3 up = car.transform.up;
        float radius = rear.leftWheel.radius * rear.leftWheel.transform.lossyScale.y;

        transform.SetPositionAndRotation((l + r) / 2f - up * radius, car.transform.rotation);
        transform.localScale = Vector3.one * course.scale;
        frontLeft.localRotation = frontRight.localRotation = Quaternion.Euler(0f, car.steering, 0f);
        SetVisible(true);
    }

    void SetVisible(bool on)
    {
        foreach (var rend in renderers) rend.enabled = on;
    }
}
