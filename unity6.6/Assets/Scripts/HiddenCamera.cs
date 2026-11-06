using UnityEngine;

/// <summary>
/// Helper cameras created at runtime with HideAndDontSave. Those flags keep
/// them out of the hierarchy and the saved scene, but also mean nothing
/// cleans them up when Play mode stops: each run in the Editor left another
/// copy behind, still enabled and still clearing its part of the screen.
/// Owners destroy theirs in OnDestroy; Create also sweeps up any copies an
/// earlier run left behind.
/// </summary>
public static class HiddenCamera
{
    public static Camera Create(string name)
    {
        DestroyAll(name);
        var go = new GameObject(name);
        go.hideFlags = HideFlags.HideAndDontSave;
        return go.AddComponent<Camera>();
    }

    public static void Destroy(Camera cam)
    {
        if (cam == null) return;
        if (Application.isPlaying) Object.Destroy(cam.gameObject);
        else Object.DestroyImmediate(cam.gameObject);
    }

    static void DestroyAll(string name)
    {
        foreach (var c in Resources.FindObjectsOfTypeAll<Camera>())
        {
            if (c == null || c.name != name) continue;
            if ((c.gameObject.hideFlags & HideFlags.DontSave) == 0) continue;
            Destroy(c);
        }
    }
}
