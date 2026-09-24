using UnityEngine;

/// <summary>
/// Frames the whole course in the top-down view, at any window shape.
///
/// The overhead camera is a plain perspective camera with a fixed 60 degree
/// vertical field of view, so the world height it shows is fixed and the width
/// follows the window's aspect. On a wide window that means the course sits in
/// a band of dead black space; on a tall one it runs off the top and bottom.
/// Neither is a framing anyone chose - it is just what a fixed FOV does when
/// the window is not the shape the scene was authored for.
///
/// This pulls the camera back until the course fits, with a margin, and
/// centres it. Purely a display change: the policy's camera input comes from
/// JetRacerCsiCamera, which CsiFramePublisher renders to its own
/// RenderTexture at a fixed resolution and which this never touches.
///
/// Attached at runtime by SimController; nothing needs to be wired in a scene.
/// </summary>
[DisallowMultipleComponent]
public class OverheadCameraFit : MonoBehaviour
{
    [Tooltip("Fraction of slack around the course. 0.05 leaves a 5% border; 0 "
             + "puts the course's bounding box exactly against the edges.")]
    public float margin = 0.06f;

    [Tooltip("How often to re-measure the course, in seconds. The track can be "
             + "rebuilt under us by the curriculum, and the window can be "
             + "resized, so this is not a one-shot.")]
    public float refitInterval = 0.5f;

    [Tooltip("Never place the camera closer than this to the course plane.")]
    public float minHeight = 5f;

    [Tooltip("Turn off to go back to the camera's authored transform.")]
    public bool enableFit = true;

    Camera _cam;
    float _nextFit;
    int _lastW, _lastH;
    Bounds _bounds;
    bool _haveBounds;

    // The scene's authored orientation, captured before anything is written.
    // This component only ever moves the camera along its own view direction -
    // the scene decides which way is up on screen. Overwriting the rotation
    // with a straight-down Euler threw away the authored yaw and stood the
    // course on end.
    Quaternion _authoredRotation;
    bool _haveAuthored;
    Camera _gutterClear;

    void OnDisable()
    {
        // Give the window back rather than leaving a black stripe behind.
        if (_cam != null) { _cam.rect = new Rect(0f, 0f, 1f, 1f); _cam.ResetAspect(); }
        if (_gutterClear != null) _gutterClear.enabled = false;
        TrackScreenHeight = 0f;
    }

    void LateUpdate()
    {
        if (!enableFit) return;
        if (_cam == null)
        {
            _cam = GetComponent<Camera>();
            if (_cam == null || _cam.orthographic) return;
        }
        if (!_haveAuthored)
        {
            _authoredRotation = transform.rotation;
            _haveAuthored = true;
        }

        // Re-measure on a timer, and immediately on a resize so dragging the
        // window edge tracks rather than lagging by up to refitInterval.
        bool resized = Screen.width != _lastW || Screen.height != _lastH;
        if (resized || Time.unscaledTime >= _nextFit)
        {
            _lastW = Screen.width;
            _lastH = Screen.height;
            _nextFit = Time.unscaledTime + Mathf.Max(0.05f, refitInterval);
            _haveBounds = TryMeasureCourse(out _bounds);
        }
        ApplyGutter();
        if (!_haveBounds)
        {
            // Nothing measured means nothing to align to. Say so rather than
            // leaving the last good band published, which the overlay would
            // keep matching against a track that is no longer there.
            TrackScreenHeight = 0f;
            return;
        }

        Fit(_bounds);
    }

    /// <summary>
    /// The band of the window the track's road actually covers, vertically:
    /// its height as a fraction of window height, and the centre it is
    /// balanced on. FlyBrainViz matches its overlay to this so the brain is
    /// the same height as the track and centred on the same line.
    ///
    /// 0 height means "not measured" - the overlay falls back to its own
    /// configured viewport rather than collapsing.
    ///
    /// Published rather than derived by the overlay itself because only this
    /// component knows the fitted distance, and re-deriving it there would be
    /// a second copy of the framing maths to keep in step.
    /// </summary>
    public static float TrackScreenHeight { get; private set; }
    public static float TrackScreenCentreY { get; private set; } = 0.5f;

    /// <summary>
    /// Hand the left edge of the window to the fly-brain overlay and render
    /// the track in what is left.
    ///
    /// A viewport rect rather than a zoom-out, because the track should stay
    /// as large as the remaining space allows rather than shrink inside a
    /// full-width frame. Unity derives the camera's aspect from this rect, so
    /// Fit() then frames the course for the narrower view automatically -
    /// hence ResetAspect, which discards any aspect previously forced on the
    /// camera and puts it back under the rect's control.
    ///
    /// The width is read live, so pressing B to dismiss the overlay gives the
    /// full window back to the track on the next frame.
    /// </summary>
    void ApplyGutter()
    {
        float g = Mathf.Clamp(FlyBrainViz.TrackGutterFraction, 0f, FlyBrainViz.MaxGutter);
        Rect want = new Rect(g, 0f, 1f - g, 1f);
        if (_cam.rect != want)
        {
            _cam.rect = want;
            _cam.ResetAspect();
        }

        // Nothing else draws in the reserved column - the overlay camera
        // clears depth only - so without this the gutter keeps whatever was
        // last in the backbuffer and smears.
        EnsureGutterClear().enabled = g > 0.0001f;
    }

    Camera EnsureGutterClear()
    {
        if (_gutterClear != null) return _gutterClear;
        var go = new GameObject("OverheadGutterClear");
        go.hideFlags = HideFlags.HideAndDontSave;
        _gutterClear = go.AddComponent<Camera>();
        _gutterClear.clearFlags = CameraClearFlags.SolidColor;
        _gutterClear.backgroundColor = Color.black;
        _gutterClear.cullingMask = 0;          // clears, renders nothing
        _gutterClear.depth = -100f;            // before every other camera
        _gutterClear.orthographic = true;
        _gutterClear.allowHDR = false;
        _gutterClear.allowMSAA = false;
        _gutterClear.useOcclusionCulling = false;
        return _gutterClear;
    }

    /// <summary>
    /// Back the camera away along its own view direction until the course
    /// fits, keeping the scene's authored orientation.
    ///
    /// Distance is checked against both screen axes and the larger wins, which
    /// is what makes this work on a window of any shape: on a wide window the
    /// vertical extent binds, on a narrow one the horizontal does.
    ///
    /// The extents are measured along the camera's own right and up vectors
    /// rather than world X and Z, because the overhead camera is yawed - the
    /// course's world X is not necessarily the screen's horizontal. Using
    /// world axes would fit the wrong extent to the wrong screen axis on any
    /// camera that is not axis-aligned.
    /// </summary>
    void Fit(Bounds b)
    {
        float halfV = Mathf.Tan(_cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        if (halfV <= 1e-4f) return;
        float aspect = _cam.aspect;
        if (aspect <= 1e-4f) return;

        Quaternion rot = _authoredRotation;
        Vector3 fwd = rot * Vector3.forward;
        Vector3 right = rot * Vector3.right;
        Vector3 up = rot * Vector3.up;

        // Raw extents are kept alongside the padded ones: the padding is
        // breathing room around the track, not part of it, so the band
        // published for the overlay to match has to be measured without it or
        // the brain comes out a margin taller than the road it sits beside.
        float halfRightRaw = ExtentAlong(b.extents, right);
        float halfUpRaw = ExtentAlong(b.extents, up);
        float halfRight = halfRightRaw * (1f + margin);
        float halfUp = halfUpRaw * (1f + margin);
        float halfFwd = ExtentAlong(b.extents, fwd);

        float dist = Mathf.Max(minHeight,
                               Mathf.Max(halfUp / halfV, halfRight / (halfV * aspect)));

        // What the road ends up covering on screen. The camera is aimed at
        // b.center, so the box lands centred in the camera's rect whatever
        // the fit distance - the centre is the rect's, and only the height
        // has to be worked out. minHeight can make dist larger than the fit
        // asked for, in which case the track is smaller than the padding
        // implies and this picks that up for free.
        float visibleHalfUp = dist * halfV;
        TrackScreenHeight = visibleHalfUp > 1e-4f
            ? Mathf.Clamp01(halfUpRaw / visibleHalfUp) * _cam.rect.height
            : 0f;
        TrackScreenCentreY = _cam.rect.y + _cam.rect.height * 0.5f;

        // Pulled back past the box's own depth as well, so the near side of a
        // course with height does not end up behind the camera.
        _cam.transform.rotation = rot;
        _cam.transform.position = b.center - fwd * (dist + halfFwd);

        // Clip planes follow the distance, or a course that needed a long pull
        // back clips against a far plane that was fine when authored.
        _cam.nearClipPlane = Mathf.Max(0.1f, dist * 0.01f);
        _cam.farClipPlane = dist + halfFwd * 2f + 100f;
    }

    /// <summary>
    /// Half-width of an axis-aligned box along an arbitrary direction - the
    /// box's support function, which is the sum of each extent times that
    /// axis's contribution to the direction.
    /// </summary>
    static float ExtentAlong(Vector3 extents, Vector3 dir)
    {
        return Mathf.Abs(dir.x) * extents.x
             + Mathf.Abs(dir.y) * extents.y
             + Mathf.Abs(dir.z) * extents.z;
    }

    /// <summary>
    /// The course's extent, from the renderers that make up the road surface.
    ///
    /// Road first, because that is what should fill the frame - measuring
    /// every renderer would include the ground plane, which extends well past
    /// the track and would frame mostly grass. TrackGenerator names its tiles
    /// "Road" and the kit puts asphalt on the "Road" layer, so both are
    /// accepted; a generated course and an authored one are then handled the
    /// same way.
    /// </summary>
    bool TryMeasureCourse(out Bounds bounds)
    {
        bounds = new Bounds();
        bool any = false;
        int roadLayer = LayerMask.NameToLayer("Road");

        var renderers = Object.FindObjectsOfType<Renderer>();
        for (int i = 0; i < renderers.Length; i++)
        {
            var r = renderers[i];
            if (r == null || !r.enabled) continue;
            var go = r.gameObject;
            if (!go.activeInHierarchy) continue;
            if (go.layer == FlyBrainViz.OverlayLayer) continue;

            bool isRoad = (roadLayer >= 0 && go.layer == roadLayer)
                          || go.name == "Road";
            if (!isRoad) continue;

            if (!any) { bounds = r.bounds; any = true; }
            else bounds.Encapsulate(r.bounds);
        }

        // A course with no road is not a course; leaving the camera alone is
        // better than framing an empty bounding box at the origin.
        return any && bounds.size.x > 0.01f && bounds.size.z > 0.01f;
    }
}
