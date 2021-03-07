using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.RenderStreaming;
public class RemoteController : MonoBehaviour
{
    private SimpleCameraController remoteCameraController;
    // Start is calle<<d before the first frame update
    private RosSharp.Control.Controller localController;
    private ActiveController controller;
    void Start()
    {
        localController = GetComponent<RosSharp.Control.Controller>();
        controller = ActiveController.REMOTE;
        remoteCameraController = SimController.instance.streamCamera.GetComponent<SimpleCameraController>();
    }
    private enum ActiveController
    {
        LOCAL,
        REMOTE,
        NONE
    }
    // Update is called once per frame
    void Update()
    {
        if(remoteCameraController.remoteKeyboard.rightArrowKey.isPressed)
        {
            localController.UpdateDirection(false);
        }

        if(remoteCameraController.remoteKeyboard.leftArrowKey.isPressed)
        {
            localController.UpdateDirection(true);
        }
    }
}
