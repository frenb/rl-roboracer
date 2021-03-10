using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoleController : MonoBehaviour
{
    // Start is called before the first frame update
    int fixedUpdateCount=0;


    public bool isUpright  {get; private set;}

    void Start()
    {
        isUpright = true;   
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if(isUpright){
            fixedUpdateCount++;
        }
        //Debug.Log("Seconds upright: " + GetSecondsUpright());
    }

    public float GetSecondsUpright(){
        return fixedUpdateCount / 50f;
    }

     void OnCollisionStay(Collision collisionInfo)
    {
        //Debug.Log("collisionInfo.gameObject.name:" + collisionInfo.gameObject.name);
        if (collisionInfo.gameObject.name.Contains("gripper")
            || collisionInfo.gameObject.name.Contains("Cart"))
        {
            isUpright = true;
            //Debug.Log("Gripper or Cart");
            
        } else {
            isUpright = false;
            //Debug.Log("Pole Fell");
        }
    }
}
