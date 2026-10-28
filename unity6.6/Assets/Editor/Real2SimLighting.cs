using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

// Unity 6 players render skybox ambient as black when the scene has no
// generated lighting data; Unity 2020 players computed it at load. The camera
// policy was trained on 2020 frames, so shaded wall faces must stay grey.
// Generating with baked and realtime GI off writes only the ambient probe and
// the default reflection cubemap: no lightmaps. A Baked light only acts as
// realtime while the scene has no lighting data, so lights are made Realtime.
public static class Real2SimLighting
{
    const string ScenePath = "Assets/Scenes/real2sim-home-track.unity";
    const string SettingsPath = "Assets/Scenes/real2sim-home-track/LightingSettings.lighting";

    [MenuItem("Real2Sim/Generate Ambient Lighting")]
    public static void GenerateAmbientLighting()
    {
        var scene = EditorSceneManager.GetActiveScene();
        if (scene.path != ScenePath)
            scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

        if (!Lightmapping.TryGetLightingSettings(out LightingSettings settings))
        {
            settings = new LightingSettings { name = "LightingSettings" };
            Directory.CreateDirectory(Path.GetDirectoryName(SettingsPath));
            AssetDatabase.CreateAsset(settings, SettingsPath);
            Lightmapping.lightingSettings = settings;
        }
        settings.bakedGI = false;
        settings.realtimeGI = false;
        EditorUtility.SetDirty(settings);

        foreach (var light in Object.FindObjectsByType<Light>(FindObjectsInactive.Include))
        {
            if (light.lightmapBakeType == LightmapBakeType.Realtime) continue;
            light.lightmapBakeType = LightmapBakeType.Realtime;
            EditorUtility.SetDirty(light);
        }
        RenderSettings.reflectionIntensity = 1f;

        bool ok = Lightmapping.Bake();
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        AssetDatabase.SaveAssets();
        Debug.Log($"[Real2Sim] ambient lighting generated: ok={ok}, " +
                  $"lightingDataAsset={Lightmapping.lightingDataAsset}, " +
                  $"ambientProbe[0]={RenderSettings.ambientProbe[0, 0]:F3}");
    }

    [MenuItem("Real2Sim/Build Gym (Unity6.6 v3)")]
    public static void BuildGym()
    {
        string output = Path.GetFullPath(Path.Combine(Application.dataPath,
            "../../../UnityBinary/real2sim2026.10.27-fences-Unity6.6-v3"));
        Directory.CreateDirectory(output);
        var options = new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = Path.Combine(output, PlayerSettings.productName + ".exe"),
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None,
        };
        BuildSummary s = BuildPipeline.BuildPlayer(options).summary;
        Debug.Log($"[Real2Sim] build {s.result} -> {options.locationPathName} " +
                  $"({s.totalErrors} errors, {s.totalTime})");
    }
}
