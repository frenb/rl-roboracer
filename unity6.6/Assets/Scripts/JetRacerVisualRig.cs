using UnityEngine;

/// <summary>
/// Poses the detailed JetRacer model (Assets/JetRacer/sim2real-JetRacer-v1.fbx)
/// from the car's WheelColliders every frame. Purely visual: it reads the
/// colliders CarController drives and never writes to physics, so the policy's
/// steering and the observations it gets are unchanged.
///
/// The FBX carries no animation (Blender drivers and constraints don't export),
/// so the steering linkage is re-solved here the way the Blender rig does it:
///   knuckles      yaw by the collider steer angle (parallel steering, both sides equal)
///   servo horn    rolls about the car's forward axis by the polynomial fitted in Blender
///   drag link     sits on the left steering arm ball (R_U_L) and aims at the horn ball (R_B_horn)
///   tie rod       sits on the left lower ball (R_J_L) and aims at the right one (R_J_R)
///   wheels        spin from collider rpm; lidar rotor and drive pulley spin constantly
///
/// "container" is the parent of the FBX instance, oriented Y up, Z forward,
/// X right. All axes below are in container space.
/// </summary>
public class JetRacerVisualRig : MonoBehaviour
{
    public CarController car;
    public Transform container;

    public Transform knuckleL, knuckleR;
    public Transform wheelFL, wheelFR, wheelRL, wheelRR;
    public Transform servoHorn;
    public Transform dragLink, dragFrom, dragTo;
    public Transform tieRod, tieFrom, tieTo;
    public Transform lidarRotor, lidarPulley;

    [Tooltip("The steer angle the Blender rig's s = ±1 maps to (the knuckle polynomial is 0.2967 rad · s).")]
    public float blenderFullLockDeg = 17f;
    [Tooltip("Lidar rotor rate, deg/s. RPLIDAR A1 at 5.5 Hz; Blender: frame * -1.4399 at 24 fps.")]
    public float lidarDegPerSec = 1980f;
    [Tooltip("Lidar drive pulley rate, deg/s (3:1 belt). Blender: frame * -4.31969 at 24 fps.")]
    public float pulleyDegPerSec = 5940f;
    [Tooltip("Keep the visual tyres on the ground under the WheelColliders as the suspension moves.")]
    public bool followSuspension = true;

    // Servo horn roll (radians, Blender right-handed about +X forward) vs Blender steer s
    // (+1 = full left), highest order first. Fitted so the drag link holds 63.48 mm.
    static readonly float[] HornPoly =
    {
        -0.000241537f, -0.000015076f, -0.001248964f, 0.000044349f,
        -0.013729326f, 0.017667333f, -0.497684554f, -0.000000202f,
    };

    Quaternion knuckleL0, knuckleR0, horn0, rotor0, pulley0, dragRot0, tieRot0;
    Quaternion wheelFL0, wheelFR0, wheelRL0, wheelRR0;
    Vector3 axleFL, axleFR, axleRL, axleRR;
    Vector3 dragAxis, tieAxis;
    float spinFL, spinFR, spinRL, spinRR, lidarAngle, pulleyAngle;
    bool ready;

    void Awake()
    {
        if (car == null) car = GetComponentInParent<CarController>();
        Capture();
    }

    /// <summary>Records the neutral pose. Call with the model at zero steer.</summary>
    public void Capture()
    {
        if (container == null || knuckleL == null || knuckleR == null) return;
        knuckleL0 = Rel(knuckleL);
        knuckleR0 = Rel(knuckleR);
        horn0 = servoHorn ? Rel(servoHorn) : Quaternion.identity;
        rotor0 = lidarRotor ? Rel(lidarRotor) : Quaternion.identity;
        pulley0 = lidarPulley ? Rel(lidarPulley) : Quaternion.identity;
        CaptureWheel(wheelFL, out wheelFL0, out axleFL);
        CaptureWheel(wheelFR, out wheelFR0, out axleFR);
        CaptureWheel(wheelRL, out wheelRL0, out axleRL);
        CaptureWheel(wheelRR, out wheelRR0, out axleRR);
        if (dragLink && dragFrom && dragTo)
            dragAxis = Quaternion.Inverse(dragLink.rotation) * (dragTo.position - dragFrom.position).normalized;
        if (tieRod && tieFrom && tieTo)
            tieAxis = Quaternion.Inverse(tieRod.rotation) * (tieTo.position - tieFrom.position).normalized;
        ready = true;
    }

    Quaternion Rel(Transform t) => Quaternion.Inverse(container.rotation) * t.rotation;

    void CaptureWheel(Transform w, out Quaternion local0, out Vector3 axleLocal)
    {
        local0 = w ? w.localRotation : Quaternion.identity;
        axleLocal = w ? (Quaternion.Inverse(w.rotation) * container.TransformDirection(Vector3.right)).normalized : Vector3.right;
    }

    void LateUpdate()
    {
        if (!ready || car == null || car.axleInfos == null) return;
        AxleInfo front = null, rear = null;
        foreach (AxleInfo a in car.axleInfos)
        {
            if (a.steering) front = a; else rear = a;
        }
        if (front == null) return;

        float dt = Time.deltaTime;
        spinFL += front.leftWheel.rpm * 6f * dt;
        spinFR += front.rightWheel.rpm * 6f * dt;
        if (rear != null)
        {
            spinRL += rear.leftWheel.rpm * 6f * dt;
            spinRR += rear.rightWheel.rpm * 6f * dt;
        }
        lidarAngle = Mathf.Repeat(lidarAngle + lidarDegPerSec * dt, 360f);
        pulleyAngle = Mathf.Repeat(pulleyAngle + pulleyDegPerSec * dt, 360f);
        spinFL %= 360f; spinFR %= 360f; spinRL %= 360f; spinRR %= 360f;

        if (followSuspension) FollowSuspension(front, rear);
        ApplyPose((front.leftWheel.steerAngle + front.rightWheel.steerAngle) * 0.5f);
    }

    /// <summary>Poses the model for a Unity steer angle in degrees (positive = right).</summary>
    public void ApplyPose(float steerDeg)
    {
        if (!ready) return;
        Quaternion c = container.rotation;
        Quaternion yaw = Quaternion.AngleAxis(steerDeg, Vector3.up);
        knuckleL.rotation = c * yaw * knuckleL0;
        knuckleR.rotation = c * yaw * knuckleR0;

        if (servoHorn)
        {
            float s = -steerDeg / blenderFullLockDeg;
            float theta = 0f;
            foreach (float k in HornPoly) theta = theta * s + k;
            // Right-handed +theta about forward in Blender is -theta in Unity's left-handed frame.
            servoHorn.rotation = c * Quaternion.AngleAxis(-theta * Mathf.Rad2Deg, Vector3.forward) * horn0;
        }

        Spin(wheelFL, wheelFL0, axleFL, spinFL);
        Spin(wheelFR, wheelFR0, axleFR, spinFR);
        Spin(wheelRL, wheelRL0, axleRL, spinRL);
        Spin(wheelRR, wheelRR0, axleRR, spinRR);

        Track(dragLink, dragFrom, dragTo, dragAxis);
        Track(tieRod, tieFrom, tieTo, tieAxis);

        if (lidarRotor) lidarRotor.rotation = c * Quaternion.AngleAxis(lidarAngle, Vector3.up) * rotor0;
        if (lidarPulley) lidarPulley.rotation = c * Quaternion.AngleAxis(pulleyAngle, Vector3.up) * pulley0;
    }

    static void Spin(Transform w, Quaternion local0, Vector3 axle, float deg)
    {
        if (w) w.localRotation = local0 * Quaternion.AngleAxis(deg, axle);
    }

    // Blender's COPY_LOCATION + DAMPED_TRACK: move onto 'from', then the smallest
    // rotation that points the link's track axis at 'to'.
    static void Track(Transform link, Transform from, Transform to, Vector3 axisLocal)
    {
        if (!link || !from || !to) return;
        link.position = from.position;
        Vector3 cur = link.rotation * axisLocal;
        Vector3 want = to.position - from.position;
        if (want.sqrMagnitude < 1e-12f) return;
        link.rotation = Quaternion.FromToRotation(cur, want) * link.rotation;
    }

    // The container's origin is the ground under the model's mid-wheelbase.
    // Keep it at the ground under the colliders: mean wheel centre minus radius.
    void FollowSuspension(AxleInfo front, AxleInfo rear)
    {
        Vector3 sum = Vector3.zero;
        float radius = 0f;
        int n = 0;
        AddWheel(front.leftWheel, ref sum, ref radius, ref n);
        AddWheel(front.rightWheel, ref sum, ref radius, ref n);
        if (rear != null)
        {
            AddWheel(rear.leftWheel, ref sum, ref radius, ref n);
            AddWheel(rear.rightWheel, ref sum, ref radius, ref n);
        }
        if (n == 0 || container.parent == null) return;
        Vector3 ground = sum / n - container.parent.up * (radius / n);
        Vector3 local = container.parent.InverseTransformPoint(ground);
        Vector3 lp = container.localPosition;
        container.localPosition = new Vector3(lp.x, local.y, lp.z);
    }

    static void AddWheel(WheelCollider w, ref Vector3 sum, ref float radius, ref int n)
    {
        if (w == null) return;
        w.GetWorldPose(out Vector3 p, out _);
        sum += p;
        radius += w.radius * w.transform.lossyScale.y;
        n++;
    }
}
