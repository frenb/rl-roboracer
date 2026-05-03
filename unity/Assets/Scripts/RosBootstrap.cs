using UnityEngine;

/// <summary>
/// Apply --ros-ip / --ros-port command-line overrides to the ROSConnection
/// singleton before any SimController.Start() runs, so multiple Unity
/// instances on the same machine can each be pointed at a different
/// ros-server endpoint (one per training actor).
///
/// Usage (multi-actor):
///   "robotaxi gym level 1.exe" --ros-ip host.docker.internal --ros-port 10001
///
/// With no flags, the build keeps whatever endpoint was configured in
/// Unity's ROS Settings window at edit time, preserving the single-client
/// workflow.
///
/// Also drops to the lowest QualitySettings level and uncaps the frame
/// rate so per-instance rendering is cheaper when several actors share
/// one GPU. (Cannot run with -nographics: physics + raycasting depend on
/// the renderer being active in this project.)
/// </summary>
public static class RosBootstrap
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Apply()
    {
        ApplyCommandLineOverrides();
        TunePerformance();
    }

    static void ApplyCommandLineOverrides()
    {
        string ip = null;
        int? port = null;

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
            }
        }

        var ros = ROSConnection.instance;
        if (ros == null)
        {
            if (ip != null || port.HasValue)
            {
                Debug.LogWarning("[RosBootstrap] No ROSConnection found in scene; --ros-ip / --ros-port ignored.");
            }
            return;
        }

        if (ip != null)        ros.rosIPAddress = ip;
        if (port.HasValue)     ros.rosPort = port.Value;

        Debug.Log($"[RosBootstrap] ROS endpoint: {ros.rosIPAddress}:{ros.rosPort}");
    }

    static void TunePerformance()
    {
        QualitySettings.SetQualityLevel(0, applyExpensiveChanges: true);
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
    }
}
