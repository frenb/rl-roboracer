using System.Collections;
using System.Collections.Generic;
using Unity.RenderStreaming;
using UnityEngine;
using ROSGeometry;

public class FruitSpawner : MonoBehaviour
{
    public GameObject bananaPrefab;
    private bool mouseState;

    private SimpleCameraController remoteCameraController;


    private enum ActiveController
    {
        LOCAL,
        REMOTE,
        NONE
    }


    // Start is called before the first frame update
    void Start()
    {
        remoteCameraController = SimController.instance.streamCamera.GetComponent<SimpleCameraController>();
    }

    // Update is called once per frame
    void Update()
    {
        ActiveController controller = ActiveController.NONE;
        float screenX = 0;
        float screenY = 0;

        if (Input.GetButton("Fire1"))
        {
            controller = ActiveController.LOCAL;
            screenX = Input.mousePosition.x;
            screenY = Input.mousePosition.y;
        }

        else if (remoteCameraController.remoteMouse != null && remoteCameraController.remoteMouse.leftButton.isPressed)
        {
            controller = ActiveController.REMOTE;
            screenX = remoteCameraController.remoteMouse.position.x.ReadValue();
            screenY = remoteCameraController.remoteMouse.position.y.ReadValue();
        }

        // Debug.Log(_mouseState);
        if (controller != ActiveController.NONE && !mouseState)
        {
            mouseState = true;

            var ray = Camera.main.ScreenPointToRay(new Vector3(screenX, screenY, 0));
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                var obj = Instantiate(bananaPrefab, hit.point + Vector3.up * 0.1f, Quaternion.identity);
                obj.transform.Rotate(new Vector3(0, 180, 0));
            }
        }

        if (controller == ActiveController.NONE)
        {
            mouseState = false;
        }
    }
}
