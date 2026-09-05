using UnityEngine;

/// <summary>
/// Applies CSI size / clips to <c>JetRacerCsiCamera</c>. Vertical FOV is
/// opened to 80° (yaml <c>fx</c> is ~62°) so the stalk view shows more
/// road surface; pitch lives on the prefab (~5° down). No D warp.
/// Sky_Dome is hidden from this camera by CameraViewSwitcher.
/// </summary>
public class JetRacerCsiIntrinsics : MonoBehaviour
{
    public const int Width = 640;
    public const int Height = 480;
    public const float Fx = 400.2333557174174f;
    public const float Fy = 533.2837800786184f;
    // Wider than 2*atan(h/2fx) so P-view sees tarmac, not a grey ribbon.
    public const float VisualVerticalFovDeg = 80f;

    public static float VerticalFovDeg
    {
        get { return VisualVerticalFovDeg; }
    }

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
        _cam.nearClipPlane = 0.05f;
        _cam.farClipPlane = 200f;
        _cam.ResetProjectionMatrix();
        _cam.aspect = Width / (float)Height;
        _cam.fieldOfView = VerticalFovDeg;
        _cam.clearFlags = CameraClearFlags.Skybox;
        // Prefab mask 63 is layers 0–5. Kit asphalt is on layer 6 ("Road"),
        // so P-view was seeing the Default ground instead of Tarmac_c.
        _cam.cullingMask = ~0;
    }

    void OnDisable()
    {
        if (_cam != null) _cam.ResetProjectionMatrix();
    }
}
