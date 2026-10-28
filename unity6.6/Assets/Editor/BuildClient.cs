using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

// Entry point for scripts/Build-UnityClient.ps1:
//   Unity.exe -quit -projectPath unity -executeMethod BuildClient.Build -buildOutput <dir>
// Builds the enabled scenes in Build Settings as a Windows 64-bit player into <dir>.
// The editor is run windowed, not with -batchmode: a Unity Personal licence on
// 2020.3 is refused in batch mode ("Missing or bad username or password").
public static class BuildClient
{
    public static void Build()
    {
        string output = ArgValue("-buildOutput");
        if (string.IsNullOrEmpty(output))
        {
            Debug.LogError("BuildClient: -buildOutput <dir> is required");
            EditorApplication.Exit(2);
            return;
        }

        string[] scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
        if (scenes.Length == 0)
        {
            Debug.LogError("BuildClient: no scenes are enabled in Build Settings");
            EditorApplication.Exit(2);
            return;
        }

        Directory.CreateDirectory(output);
        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = Path.Combine(output, PlayerSettings.productName + ".exe"),
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None,
        };
        Debug.Log($"BuildClient: building {string.Join(", ", scenes)} -> {options.locationPathName}");

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;
        Debug.Log($"BuildClient: {summary.result}, {summary.totalErrors} error(s), {summary.totalSize / (1024 * 1024)} MB, {summary.totalTime}");
        EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }

    static string ArgValue(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
