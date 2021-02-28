using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
public class move_target : MonoBehaviour
{
    private bool _mouseState;
    public GameObject target;
    public Vector3 screenSpace;
    public Vector3 offset;
    public float tableTopSurfaceY;
    // Use this for initialization
    void Start()
    {
        tableTopSurfaceY = this.transform.position.y;
        Debug.Log("tableTopSurfaceY " + tableTopSurfaceY);
    }

    // Update is called once per frame
    void Update()
    {
            // Debug.Log(_mouseState);
            if (Input.GetMouseButtonDown(0))
        {
            Debug.Log("clicked " + target.gameObject.name);
            //target = GetClickedObject(out hitInfo);
            if (target.gameObject.name == this.name)
            {
                
                _mouseState = true;
                screenSpace = Camera.main.WorldToScreenPoint(target.transform.position);
                offset = target.transform.position - Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenSpace.z));
            }
        }
        if (Input.GetMouseButtonUp(0))
        {
            _mouseState = false;
        }
        if (_mouseState)
        {
            //keep track of the mouse position
            var curScreenSpace = new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenSpace.z);

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
