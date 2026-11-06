using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Makes a training scene for a real-to-sim course: a copy of
/// generated-course.unity (car, SimController, cameras, ROS wiring) with its
/// procedural TrackGenerator track swapped for a Real2SimCourse built from
/// Assets/Real2Sim/&lt;name&gt;/layout.json and mat.png. Saved as
/// Assets/Scenes/real2sim-&lt;name&gt;.unity and listed (disabled) in Build Settings.
///
/// Batch mode:
///   Unity.exe -batchmode -quit -projectPath unity -executeMethod Real2SimSceneBuilder.BuildFromCommandLine -real2simFolder Assets/Real2Sim/home-track
/// </summary>
public static class Real2SimSceneBuilder
{
    private const string TemplateScene = "Assets/Scenes/generated-course.unity";

    [MenuItem("Real2Sim/Create Course Scene From Layout...")]
    public static void BuildFromMenu()
    {
        string abs = EditorUtility.OpenFolderPanel("Real2Sim layout folder (with layout.json)", "Assets/Real2Sim", "");
        if (string.IsNullOrEmpty(abs)) return;
        string folder = ToAssetPath(abs);
        if (folder == null) { EditorUtility.DisplayDialog("Real2Sim", "Pick a folder inside this project's Assets.", "OK"); return; }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        string scene = Build(folder);
        if (scene != null) EditorSceneManager.OpenScene(scene);
    }

    public static void BuildFromCommandLine()
    {
        var args = System.Environment.GetCommandLineArgs();
        int i = System.Array.IndexOf(args, "-real2simFolder");
        if (i < 0 || i + 1 >= args.Length) { Debug.LogError("[Real2Sim] pass -real2simFolder Assets/Real2Sim/<name>"); EditorApplication.Exit(1); return; }
        string scene = Build(args[i + 1].Replace('\\', '/').TrimEnd('/'));
        EditorApplication.Exit(scene == null ? 1 : 0);
    }

    public static string Build(string folder)
    {
        var layout = AssetDatabase.LoadAssetAtPath<TextAsset>(folder + "/layout.json");
        if (layout == null) { Debug.LogError($"[Real2Sim] no layout.json in {folder}"); return null; }
        var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(folder + "/mat.png");
        string name = Path.GetFileName(folder);

        var scene = EditorSceneManager.OpenScene(TemplateScene, OpenSceneMode.Single);
        foreach (var tg in Object.FindObjectsOfType<TrackGenerator>())
            Object.DestroyImmediate(tg.gameObject);
        // Goal-N objects anywhere else would be found by CarController.SetUpGoalsArray too.
        foreach (var go in scene.GetRootGameObjects().Where(g => g.name.StartsWith("Goal-")))
            Object.DestroyImmediate(go);

        var host = new GameObject("Real2SimCourse");
        var course = host.AddComponent<Real2SimCourse>();
        course.layoutJson = layout;
        course.matTexture = texture;
        course.Generate();

        var sim = Object.FindObjectOfType<SimController>();
        if (sim != null) sim.trackGenerator = null;

        // Frame the course for the main camera: above the mat's far edge, looking across it.
        var parsed = course.Parse();
        var cam = Camera.main;
        if (cam != null && parsed != null)
        {
            Vector3 c = new Vector3(parsed.mat.width, 0f, parsed.mat.length) * (course.scale * 0.5f);
            float size = Mathf.Max(parsed.mat.width, parsed.mat.length) * course.scale;
            cam.transform.position = c + new Vector3(0f, size * 1.1f, -size * 0.6f);
            cam.transform.LookAt(c);
        }

        string path = $"Assets/Scenes/real2sim-{name}.unity";
        if (!EditorSceneManager.SaveScene(scene, path)) { Debug.LogError($"[Real2Sim] could not save {path}"); return null; }
        var list = EditorBuildSettings.scenes.ToList();
        if (!list.Any(s => s.path == path))
        {
            list.Add(new EditorBuildSettingsScene(path, false));
            EditorBuildSettings.scenes = list.ToArray();
        }
        Debug.Log($"[Real2Sim] saved {path} ({GameObject.FindObjectsOfType<Goal>().Length} goals)");
        return path;
    }

    private static string ToAssetPath(string abs)
    {
        string root = Path.GetFullPath(Application.dataPath).Replace('\\', '/');
        string full = Path.GetFullPath(abs).Replace('\\', '/');
        return full.StartsWith(root) ? "Assets" + full.Substring(root.Length) : null;
    }
}
