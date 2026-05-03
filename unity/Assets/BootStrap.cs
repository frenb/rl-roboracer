using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BootStrap : MonoBehaviour
{
    public bool manualControlEnabled;
    public GameObject carProxy;
    public GameObject carPrefab;
    public GameObject spherePrefab;
    public GameObject car;
    public GameObject sphere;
    public float steeringAngle;
    public float angleRatio;
    public bool goalComplete;
    public int numObstacles;
    public GameObject currentGoal;

    public float nextGoalAngle;

    // Start is called before the first frame update
    void Start()
    {         
        Time.timeScale = 5;
        if (carPrefab != null)
        {
            car = Instantiate<GameObject>(carPrefab);
            car.name = "Car";
            if (carProxy)
            {
                car.transform.position = carProxy.transform.position;
                car.transform.rotation = carProxy.transform.rotation;
            }

            CarController cc = car.GetComponent<CarController>();
            cc.manualControlEnabled = false;
        }
    }
    public Vector3 GetClosestPointOnCollider(GameObject car, GameObject other)
    {
        CarController cc = car.GetComponent<CarController>();
        Vector3 closestPoint = cc.GetClosestPointOnCollider(car, other);
        if(!sphere)
        {
            sphere = Instantiate(spherePrefab);
        }
        sphere.transform.position = closestPoint;  
        sphere.transform.localScale = new Vector3(2,2,2);    
        return closestPoint;
    }
     private float GetAngleToGoal(){
        CarController cc = car.GetComponent<CarController>();
        Vector3 closestPointOnGoal = GetClosestPointOnCollider(car, cc.GetNextGoal());
        Vector3 targetDir = closestPointOnGoal - car.transform.position;
        float angle = Vector3.SignedAngle(
            car.transform.forward, 
            targetDir,
            Vector3.up);
        return angle;
    }
    //  private float GetAngleToGoal(){
    //     CarController cc = car.GetComponent<CarController>();
    //     if(currentGoal == null )
    //         return 0f;
        
    //     Vector3 closestPointOnGoal = GetClosestPointOnCollider(car, currentGoal);
    //     Vector3 targetDir = closestPointOnGoal - car.transform.position;
    //     float angle = Vector3.SignedAngle(
    //         car.transform.forward, 
    //         targetDir,
    //         Vector3.up);
    //     return angle;
    //     // float normal = Mathf.InverseLerp(-180, 180, angle);
    //     // float bValue = Mathf.Lerp(0, 1, normal);
    // }
    // Update is called once per frame
    void FixedUpdate()
    {
        CarController cc = car.GetComponent<CarController>();
        cc.numObstacles=numObstacles;
        if(cc.goals.Count == 0)
            return;
        nextGoalAngle = GetAngleToGoal()/180;
        IsGoalComplete();
        currentGoal = cc.GetNextGoal();
        steeringAngle = cc.steering;
        angleRatio = steeringAngle / nextGoalAngle;
        StartCoroutine(
            cc.ApplyForce(
                    nextGoalAngle, 
                    0.5f)
        );
    }
    private bool IsGoalComplete(){
        //Debug.Log(goal);
        CarController cc = car.GetComponent<CarController>();
        if(currentGoal == null )
            return false;
        bool isComplete = currentGoal.GetComponent<Goal>().goalComplete;
        if(isComplete){
            goalComplete = isComplete;
            Debug.Log("goal zz complete " + isComplete + " " + currentGoal.GetComponent<Goal>());
            return true;
        }else{
             Debug.Log("goal zz complete " + isComplete + " " + currentGoal.GetComponent<Goal>());
            return false;
        }    
    }
}
