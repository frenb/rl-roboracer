using UnityEngine;
using UnityEditor;

public class MeshSerializer : MonoBehaviour
{
    [MenuItem("GameObject/Serialize Meshes", false, 0)]
    static void SerializeMesh(MenuCommand command)
    {
        var gameObj = (GameObject)command.context;

        int numbMeshes = 0;

        foreach (var collider in gameObj.GetComponentsInChildren<MeshCollider>())
        {
            var mesh = collider.sharedMesh;
            string filePath = string.Format("Assets/URDFColliders/{0:D3}.asset", numbMeshes++);
            AssetDatabase.CreateAsset(mesh, filePath);
        }
    }

    /*
    public static string GetGameObjectPath(GameObject obj)
    {
        string path = obj.name;
        while (obj.transform.parent != null)
        {
            obj = obj.transform.parent.gameObject;
            path = obj.name + "." + path;
        }
        return path;
    }
    */
}