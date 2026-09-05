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

    static bool IsCar(Collider hit)
    {
        if (hit == null) return false;
        // Wheel / child colliders fire OnTriggerEnter with their own name,
        // so walk parents and also accept the attached rigidbody root.
        for (Transform t = hit.transform; t != null; t = t.parent)
        {
            if (NameIsCar(t.name)) return true;
        }
        if (hit.attachedRigidbody != null && NameIsCar(hit.attachedRigidbody.gameObject.name))
            return true;
        return false;
    }

    static bool NameIsCar(string name)
    {
        return name.Contains("RiggedWaymo") || name.Contains("JetRacer_Physics");
    }

    void OnTriggerEnter(Collider hit)
    {
        Debug.Log("collided with " + hit.transform.gameObject.name);
        if (IsCar(hit))
        {
           Debug.Log("set goal zz complete to true");
           goalComplete=true;
        }
    }
}
