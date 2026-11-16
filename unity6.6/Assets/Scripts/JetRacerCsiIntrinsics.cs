using UnityEngine;

/// <summary>
/// Gives <c>JetRacerCsiCamera</c> the real IMX219-160's pinhole intrinsics
/// (config/camera_calibration/cam_640x480.yaml K). The Nano runs cv2.undistort
/// with that K and D, so the real policy sees a pinhole image with this K; the
/// sim renders one directly. fx and fy differ (77 deg horizontal, 48.5 deg
/// vertical), which a fieldOfView + aspect camera cannot express, so the
/// projection matrix is built from K. Pose lives on the prefab
/// (JetRacerPhysicsToReal). Sky_Dome is hidden from this camera by CameraViewSwitcher.
/// </summary>
public class JetRacerCsiIntrinsics : MonoBehaviour
{
    public const int Width = 640;
    public const int Height = 480;
    public const float Fx = 400.2333557174174f;
    public const float Fy = 533.2837800786184f;
    public const float Cx = 331.4549671721432f;
    public const float Cy = 263.3567321479512f;

    public static float VerticalFovDeg => 2f * Mathf.Atan(Height * 0.5f / Fy) * Mathf.Rad2Deg;

    const float Near = 0.05f;
    const float Far = 200f;

    Camera _cam;

    void OnEnable()
    {
        _cam = GetComponent<Camera>();
        Apply();
    }

    void LateUpdate()
    {
        Apply();
    }

    public void Apply()
    {
        if (_cam == null) _cam = GetComponent<Camera>();
        if (_cam == null) return;

        _cam.allowHDR = false;
        _cam.allowMSAA = false;
        _cam.nearClipPlane = Near;
        _cam.farClipPlane = Far;
        _cam.aspect = Width / (float)Height;
        _cam.fieldOfView = VerticalFovDeg;
        _cam.projectionMatrix = Projection();
        _cam.clearFlags = CameraClearFlags.Skybox;
        // Prefab mask 63 is layers 0–5. Kit asphalt is on layer 6 ("Road"),
        // so P-view was seeing the Default ground instead of Tarmac_c.
        //
        // Everything EXCEPT the fly-brain overlay. This camera is what
        // CsiFramePublisher reads for `camera/front`, so anything it renders
        // can reach a policy as observation. A debug overlay appearing in
        // training input would be a silent data bug, not a visible one, so the
        // overlay gets a layer and this mask excludes it.
        _cam.cullingMask = ~0 & ~FlyBrainViz.OverlayLayerMask;
    }

    // OpenGL-style projection for pixel K with a top-left image origin
    // (CsiFramePublisher flips rows to match ROS).
    static Matrix4x4 Projection()
    {
        var p = Matrix4x4.zero;
        p[0, 0] = 2f * Fx / Width;
        p[0, 2] = 1f - 2f * Cx / Width;
        p[1, 1] = 2f * Fy / Height;
        p[1, 2] = 2f * Cy / Height - 1f;
        p[2, 2] = -(Far + Near) / (Far - Near);
        p[2, 3] = -2f * Far * Near / (Far - Near);
        p[3, 2] = -1f;
        return p;
    }

    void OnDisable()
    {
        if (_cam != null) _cam.ResetProjectionMatrix();
    }
}
