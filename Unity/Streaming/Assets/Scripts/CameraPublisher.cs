using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class CameraPublisher : MonoBehaviour
{
    private ROSConnection ros;
    public Camera camera;
    public string topic;
    private RenderTexture renderTexture;

    private int renderTextureHeight = 400;
    private int renderTextureWidth = 400;
    private int renderTextureDepth = 0;
    private RenderTextureFormat renderTextureFormat = RenderTextureFormat.ARGB32;
    private byte[] rawBytes = new byte[640000];

    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.instance;
        renderTexture = new RenderTexture(renderTextureWidth, renderTextureHeight, renderTextureDepth, renderTextureFormat);
        camera.targetTexture = renderTexture;

        StartCoroutine(DoPublish());
    }

    private IEnumerator DoPublish()
    {
        while (true)
        {
            yield return StartCoroutine(ReadPixels());
            Publish();
            yield return new WaitForSeconds(0.5f); // 2Hz
        }
    }


    private IEnumerator ReadPixels()
    {
        var request = AsyncGPUReadback.Request(renderTexture, 0);

        while (!request.done)
        {
            yield return new WaitForEndOfFrame();
        }

        request.GetData<byte>().CopyTo(rawBytes);
    }

    private void Publish()
    {
        var frame = new RosMessageTypes.NiryoMoveit.Camera();
        frame.frame.height = (uint) renderTextureHeight;
        frame.frame.width = (uint) renderTextureWidth;
        frame.frame.data = rawBytes;
        ros.Send(topic, frame);
    }
}
