using System.Collections;
using System.Collections.Generic;
using ROSGeometry;
using UnityEngine;
using UnityEngine.Rendering;

public class CameraPublisher : MonoBehaviour
{
    private ROSConnection ros;
    public Camera camera;
    public string topic;
    private RenderTexture renderTexture;
    private uint frame_sequence = 0;

    private int renderTextureHeight = 800;
    private int renderTextureWidth = 800;
    private int renderTextureDepth = 0;
    private RenderTextureFormat renderTextureFormat = RenderTextureFormat.ARGB32;
    private byte[] rawBytes = new byte[2560000];

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
            yield return new WaitForSeconds(1.0f); // 1Hz
        }
    }


    private IEnumerator ReadPixels()
    {
        var request = AsyncGPUReadback.Request(renderTexture, 0);

        while (!request.done)
        {
            yield return new WaitForEndOfFrame();
        }

        if (!request.hasError)
        {
            request.GetData<byte>().CopyTo(rawBytes);
        }
    }

    private void Publish()
    {
        var frame = new RosMessageTypes.NiryoMoveit.Camera();
        frame.frame.height = (uint) renderTextureHeight;
        frame.frame.width = (uint) renderTextureWidth;
        frame.frame.data = rawBytes;
        frame.frame.header.seq = frame_sequence++;
        ros.Send(topic, frame);
    }
}
