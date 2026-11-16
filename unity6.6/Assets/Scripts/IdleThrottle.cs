using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Caps the frame rate while no trainer is driving this client, so an idle
/// gym window does not keep the GPU busy while another robot's job (e.g. a
/// microduck run in duck-trainer) holds the shared job queue.
///
/// "Idle" means no SimCommand and no keyboard or mouse input (while the
/// window has focus) for
/// <see cref="IdleAfterSeconds"/> of real time. Any of those restores the
/// uncapped rate RosBootstrap sets; the first command after a long idle waits
/// at most one idle frame (1 / idle fps) before it is processed.
///
///   --idle-fps N   frame rate while idle (default 2; 0 never throttles)
/// </summary>
public class IdleThrottle : MonoBehaviour
{
    const float IdleAfterSeconds = 60f;
    const int DefaultIdleFps = 2;

    static IdleThrottle _instance;

    int idleFps = DefaultIdleFps;
    float lastActivity;
    bool idle;

    /// <summary>Mark trainer activity; called for every SimCommand.</summary>
    public static void Wake()
    {
        if (_instance != null) _instance.MarkActive();
    }

    void Awake()
    {
        _instance = this;
        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++)
        {
            if (args[i] == "--idle-fps" && int.TryParse(args[i + 1], out var fps) && fps >= 0)
                idleFps = fps;
        }
        lastActivity = Time.unscaledTime;
    }

    void Update()
    {
        if (idleFps == 0) return;
        if (UserInput())
        {
            MarkActive();
        }
        else if (!idle && Time.unscaledTime - lastActivity > IdleAfterSeconds)
        {
            idle = true;
            Application.targetFrameRate = idleFps;
            Debug.Log($"[IdleThrottle] no sim commands for {IdleAfterSeconds:0}s; capping at {idleFps} fps");
        }
    }

    void MarkActive()
    {
        lastActivity = Time.unscaledTime;
        if (!idle) return;
        idle = false;
        Application.targetFrameRate = -1;
        Debug.Log("[IdleThrottle] active again; frame rate uncapped");
    }

    static bool UserInput()
    {
        // Unfocused, the Input System still reports mouse movement made in
        // other windows, which would keep a background client awake.
        if (!Application.isFocused) return false;
        var kb = Keyboard.current;
        if (kb != null && kb.anyKey.isPressed) return true;
        var mouse = Mouse.current;
        return mouse != null
            && (mouse.delta.ReadValue().sqrMagnitude > 0f
                || mouse.leftButton.isPressed
                || mouse.rightButton.isPressed
                || mouse.scroll.ReadValue().sqrMagnitude > 0f);
    }

    void OnDestroy()
    {
        if (_instance == this) _instance = null;
    }
}
