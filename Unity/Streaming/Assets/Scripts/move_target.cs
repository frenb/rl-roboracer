using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using Unity.RenderStreaming;
public class move_target : MonoBehaviour
{
    private bool _mouseState;
    public GameObject target;
    public Vector3 screenSpace;
    public Vector3 offset;
    public float tableTopSurfaceY;

    private SimpleCameraController remoteCameraController;


    // Use this for initialization
    void Start()
    {
        tableTopSurfaceY = this.transform.position.y;
        Debug.Log("tableTopSurfaceY " + tableTopSurfaceY);
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
        if (controller != ActiveController.NONE && !_mouseState)
        {
            Debug.Log("clicked " + target.gameObject.name);
            //target = GetClickedObject(out hitInfo);
            if (target.gameObject.name == this.name)
            {
                
                _mouseState = true;
                screenSpace = Camera.main.WorldToScreenPoint(target.transform.position);
                offset = target.transform.position - Camera.main.ScreenToWorldPoint(new Vector3(screenX, screenY, screenSpace.z));
            }
        }
        if (controller == ActiveController.NONE)
        {
            _mouseState = false;
        }
        if (_mouseState)
        {
            //keep track of the mouse position
            var curScreenSpace = new Vector3(screenX, screenY, screenSpace.z);

            //convert the screen mouse position to world point and adjust with offset
            var curPosition = Camera.main.ScreenToWorldPoint(curScreenSpace) + offset;
            curPosition.y = tableTopSurfaceY;
            //update the position of the object in the world
            target.transform.position = curPosition;
        }
    }


    GameObject GetClickedObject(out RaycastHit hit)
    {
        GameObject target = null;
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        if (Physics.Raycast(ray.origin, ray.direction * 10, out hit))
        {
            target = hit.collider.gameObject;
        }

        return target;
    }
}
