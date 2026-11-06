using UnityEngine;

/// <summary>
/// One scale factor for every on-screen overlay, so the HUD, the curriculum
/// buttons and the fly-brain panels stay the same size *relative to the
/// window* instead of the same size in pixels.
///
/// IMGUI draws in raw pixels and has no equivalent of uGUI's CanvasScaler, so
/// a panel authored at 1080p is half the apparent size on a 4K monitor and
/// swamps a small window. Every overlay in this project was authored in fixed
/// pixels, which is why the text is unreadable on a large display and
/// oversized on a short one.
///
/// The fix is the CanvasScaler approach: pick a reference height, derive a
/// scale from the real one, and push it through GUI.matrix so existing layout
/// arithmetic keeps working unchanged. Callers then lay out in *logical*
/// pixels - <see cref="LogicalWidth"/> / <see cref="LogicalHeight"/> rather
/// than Screen.width / Screen.height - and everything else about their code
/// stays as it was.
///
/// Height, not width or diagonal: vertical space is what panels compete for,
/// and scaling off width would blow the overlays up on an ultrawide window
/// that has no more room for them vertically than a 16:9 one.
/// </summary>
public static class OverlayUi
{
    /// <summary>The resolution the overlays' pixel sizes were authored at.</summary>
    public const float ReferenceHeight = 1080f;

    // Clamped at both ends. Below the floor an overlay stops being legible at
    // all and it is better to let it take a larger share of a small window;
    // above the ceiling it starts eating a display big enough not to need the
    // help.
    public const float MinScale = 0.55f;
    public const float MaxScale = 3.0f;

    public static float Scale
    {
        get
        {
            float h = Screen.height;
            if (h < 1f) return 1f;
            return Mathf.Clamp(h / ReferenceHeight, MinScale, MaxScale);
        }
    }

    /// <summary>Window width in the units callers should lay out in.</summary>
    public static float LogicalWidth { get { return Screen.width / Scale; } }

    /// <summary>Window height in the units callers should lay out in.</summary>
    public static float LogicalHeight { get { return Screen.height / Scale; } }

    /// <summary>
    /// Scale pixels from screen space into layout space. Needed for anything
    /// sourced outside GUI - a mouse position, say - since GUI.matrix does not
    /// apply to those.
    /// </summary>
    public static Vector2 ToLogical(Vector2 screenPixels)
    {
        return screenPixels / Scale;
    }

    /// <summary>
    /// Start drawing in logical pixels. Returns the previous matrix, which the
    /// caller must hand back to <see cref="End"/> - IMGUI state is global, so
    /// leaving it set would silently rescale every overlay that draws after
    /// this one.
    /// </summary>
    public static Matrix4x4 Begin()
    {
        Matrix4x4 prev = GUI.matrix;
        float s = Scale;
        GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity,
                                   new Vector3(s, s, 1f));
        return prev;
    }

    public static void End(Matrix4x4 previous)
    {
        GUI.matrix = previous;
    }
}
