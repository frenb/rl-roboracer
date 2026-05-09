using UnityEngine;
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
using System;
using System.Runtime.InteropServices;
#endif

/// <summary>
/// Apply --ros-ip / --ros-port / --unity-port command-line overrides to
/// the ROSConnection singleton before any SimController.Start() runs, so
/// multiple Unity instances on the same machine can each be pointed at a
/// different ros-server endpoint (one per training actor).
///
/// Usage (multi-actor):
///   "robotaxi gym level 1.exe" --ros-ip host.docker.internal --ros-port 10001 --unity-port 5006
///
/// --ros-ip / --ros-port pin the OUTBOUND endpoint (Unity dials ros-server).
/// --unity-port pins the INBOUND port that this Unity instance opens for
/// ros-server to push messages back to. ROS-TCP-Connector's protocol is
/// bidirectional: Unity opens a TcpListener on unityPort and tells the
/// matching ros-server where to find it via the handshake. Without
/// --unity-port, every Unity on the host tries to bind the same default
/// (5005), so only the first instance wins; the rest throw
/// "Address already in use" out of ROSConnection.StartMessageServer and
/// silently never connect to ros-server at all (you get N Unity windows
/// that look alive but only one of them is wired up to anything). Pair
/// each instance with a unique unityPort, e.g. 5005 + actor_index.
///
/// With no flags, the build keeps whatever endpoint was configured in
/// Unity's ROS Settings window at edit time, preserving the single-client
/// workflow.
///
/// Also drops to the lowest QualitySettings level and uncaps the frame
/// rate so per-instance rendering is cheaper when several actors share
/// one GPU. (Cannot run with -nographics: physics + raycasting depend on
/// the renderer being active in this project.)
///
/// On Windows standalone builds the player window is forced resizable at
/// runtime via Win32 (WS_THICKFRAME + min/max box). That bypasses the
/// build-time Player Settings -> Resizable Window flag, so a single build
/// can serve both "tiny popup tile" multi-actor runs (when launched with
/// -popupwindow) and "drag the corner to make it bigger" single-client
/// runs without rebuilding.
/// </summary>
public static class RosBootstrap
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Apply()
    {
        ApplyCommandLineOverrides();
        TunePerformance();
        EnableWindowResize();
    }

    static void ApplyCommandLineOverrides()
    {
        string ip = null;
        int? port = null;
        int? unityPort = null;

        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++)
        {
            switch (args[i])
            {
                case "--ros-ip":
                    ip = args[i + 1];
                    break;
                case "--ros-port":
                    if (int.TryParse(args[i + 1], out var p))
                    {
                        port = p;
                    }
                    else
                    {
                        Debug.LogWarning($"[RosBootstrap] Could not parse --ros-port '{args[i + 1]}', leaving default.");
                    }
                    break;
                case "--unity-port":
                    if (int.TryParse(args[i + 1], out var up))
                    {
                        unityPort = up;
                    }
                    else
                    {
                        Debug.LogWarning($"[RosBootstrap] Could not parse --unity-port '{args[i + 1]}', leaving default.");
                    }
                    break;
            }
        }

        var ros = ROSConnection.instance;
        if (ros == null)
        {
            if (ip != null || port.HasValue || unityPort.HasValue)
            {
                Debug.LogWarning("[RosBootstrap] No ROSConnection found in scene; --ros-ip / --ros-port / --unity-port ignored.");
            }
            return;
        }

        if (ip != null)            ros.rosIPAddress = ip;
        if (port.HasValue)         ros.rosPort = port.Value;
        if (unityPort.HasValue)    ros.unityPort = unityPort.Value;

        Debug.Log($"[RosBootstrap] ROS endpoint: {ros.rosIPAddress}:{ros.rosPort}, unityPort: {ros.unityPort}");
    }

    static void TunePerformance()
    {
        QualitySettings.SetQualityLevel(0, applyExpensiveChanges: true);
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
    }

    static void EnableWindowResize()
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        // -popupwindow uses WS_POPUP and intentionally has no chrome; leave it alone.
        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-popupwindow") return;
        }

        try
        {
            var hwnd = Win32.GetActiveWindow();
            if (hwnd == IntPtr.Zero) return;

            int style = Win32.GetWindowLong(hwnd, Win32.GWL_STYLE);
            int newStyle = style
                | Win32.WS_THICKFRAME
                | Win32.WS_MAXIMIZEBOX
                | Win32.WS_MINIMIZEBOX
                | Win32.WS_SYSMENU
                | Win32.WS_CAPTION;
            if (newStyle == style) return;

            Win32.SetWindowLong(hwnd, Win32.GWL_STYLE, newStyle);
            Win32.SetWindowPos(
                hwnd, IntPtr.Zero, 0, 0, 0, 0,
                Win32.SWP_NOMOVE | Win32.SWP_NOSIZE | Win32.SWP_NOZORDER | Win32.SWP_FRAMECHANGED);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[RosBootstrap] Could not enable resizable window: {e.Message}");
        }
#endif
    }

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
    static class Win32
    {
        public const int GWL_STYLE       = -16;
        public const int WS_CAPTION      = 0x00C00000;
        public const int WS_SYSMENU      = 0x00080000;
        public const int WS_THICKFRAME   = 0x00040000;
        public const int WS_MINIMIZEBOX  = 0x00020000;
        public const int WS_MAXIMIZEBOX  = 0x00010000;

        public const uint SWP_NOSIZE       = 0x0001;
        public const uint SWP_NOMOVE       = 0x0002;
        public const uint SWP_NOZORDER     = 0x0004;
        public const uint SWP_FRAMECHANGED = 0x0020;

        [DllImport("user32.dll")]
        public static extern IntPtr GetActiveWindow();

        [DllImport("user32.dll", SetLastError = true)]
        public static extern int GetWindowLong(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool SetWindowPos(
            IntPtr hWnd, IntPtr hWndInsertAfter,
            int X, int Y, int cx, int cy, uint uFlags);
    }
#endif
}
