using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Goal : MonoBehaviour
{
    // Start is called before the first frame update
    public bool goalComplete = false;
    void Start()
    {
        goalComplete = false;
    }

    // Update is called 50 times a second
    void FixedUpdate()
    {
        if (goalComplete)
        {
            StartCoroutine(SetGoalFalse());
        }
    }

    IEnumerator SetGoalFalse(){
        yield return new WaitForSeconds(0.25f);
        goalComplete=false;
    }

    void OnTriggerEnter(Collider hit)
    {
        Debug.Log("collided with " + hit.transform.gameObject.name);
        if(hit.transform.gameObject.name.Contains("RiggedWaymo"))
        {
           Debug.Log("set goal zz complete to true");
           goalComplete=true;
        }
    }
}
