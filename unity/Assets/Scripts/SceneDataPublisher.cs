using RosMessageTypes.NiryoMoveit;
using UnityEngine;
using ROSGeometry;
using System.Collections;
using System.Collections.Generic;

using Transform = UnityEngine.Transform;
using Quaternion = UnityEngine.Quaternion;

public class SceneDataPublisher : MonoBehaviour, IRosComponent
{
    // ROS Connector
    private ROSConnection ros;
    private string topicName = "car_scene_data";
    private GameObject car;
    private GameObject idealTrajectory;
    private GameObject goal;
    private GameObject debugSphere;
    private GameObject debugSpherePrefab;
    private Dictionary<int,Vector3> targetOnNextGoal;

    void Start()
    {
        // Get ROS connection static instance
        targetOnNextGoal = new Dictionary<int, Vector3>();
        ros = ROSConnection.instance;
        UpdateWorldRefs();
        StartCoroutine(DoPublish());
    }

    public void UpdateWorldRefs(ApplyForce af)
    {
        
        UpdateWorldRefs();
        SimController.instance.car.gameObject.GetComponent<CarController>().cmd_id = af.cmd_id;
    }
    public void UpdateWorldRefs()
    {
        if (SimController.instance.car != null)
        {
            car = SimController.instance.car.gameObject;
        }
        if (SimController.instance.goal != null)
        {
            goal = SimController.instance.goal.gameObject;
        }
        debugSphere = SimController.instance.debugSphere;
        if(SimController.instance.idealTrajectoryPrefab)
        {
            idealTrajectory = SimController.instance.idealTrajectory.gameObject;
        }
        // So one fresh scene_data before reset done signal sent.
        Publish();
    }

    private IEnumerator DoPublish()
    {
        while (true)
        {
            Publish();
            // GameObject.Find("MainCamera").GetComponent<UnityEngine.Camera>().Render();
            // yield return new WaitForSeconds(0.01f); // 50Hz
            yield return new WaitForSeconds(0.1f); // 20Hz
            //yield return new WaitForSeconds(0.2f); // 10Hz
        }
    }

    private void Publish()
    {
        CarSceneData sceneCarDataMessage = new CarSceneData();
        SceneData sceneDataMessage = new SceneData();
        if (car != null) {
            CarController cc = car.GetComponent<CarController>();
            sceneCarDataMessage.car.location_x = car.transform.position.x;
            sceneCarDataMessage.car.location_y = car.transform.position.y;
            sceneCarDataMessage.car.location_z = car.transform.position.z;
            sceneCarDataMessage.car.speed = car.GetComponent<CarController>().GetSpeed();
            sceneCarDataMessage.car.cost = GetDistanceFromTraj();          
            sceneCarDataMessage.car.dist_from_goal = 
                Vector3.Distance(
                    car.transform.position, 
                    cc.GetNextGoal().transform.position);
                
            sceneCarDataMessage.car.dist_from_traj = GetAngleToGoal()/180;
            sceneCarDataMessage.car.has_reached_goal = IsGoalComplete();
            sceneCarDataMessage.car.has_crashed = cc.HasCrashed();
            sceneCarDataMessage.car.current_goal = GetCurrentGoalName();
            sceneCarDataMessage.car.last_goal_reached = GetLastGoalCompletedName();
            sceneCarDataMessage.car.rotation_z = car.transform.eulerAngles.z;
            sceneCarDataMessage.last_executed_cmd_id = cc.cmd_id;
            sceneCarDataMessage.car.left = cc.distToClosestObjects[0];
            sceneCarDataMessage.car.forward_left = cc.distToClosestObjects[1];
            sceneCarDataMessage.car.forward_left_left = cc.distToClosestObjects[2];
            sceneCarDataMessage.car.n_27_50 = cc.distToClosestObjects[3];
            sceneCarDataMessage.car.n_25_00 = cc.distToClosestObjects[4];
            sceneCarDataMessage.car.n_22_50 = cc.distToClosestObjects[5];
            sceneCarDataMessage.car.n_20_00 = cc.distToClosestObjects[6];
            sceneCarDataMessage.car.n_17_50 = cc.distToClosestObjects[7];
            sceneCarDataMessage.car.n_15_00 = cc.distToClosestObjects[8];
            sceneCarDataMessage.car.n_12_50 = cc.distToClosestObjects[9];
            sceneCarDataMessage.car.n_10_00 = cc.distToClosestObjects[10];
            sceneCarDataMessage.car.n_07_50 = cc.distToClosestObjects[11];
            sceneCarDataMessage.car.n_05_00 = cc.distToClosestObjects[12];
            sceneCarDataMessage.car.n_02_50 = cc.distToClosestObjects[13];
            sceneCarDataMessage.car.forward = cc.distToClosestObjects[14];
            sceneCarDataMessage.car.p_02_50 = cc.distToClosestObjects[15];
            sceneCarDataMessage.car.p_05_00 = cc.distToClosestObjects[16];
            sceneCarDataMessage.car.p_07_50 = cc.distToClosestObjects[17];
            sceneCarDataMessage.car.p_10_00 = cc.distToClosestObjects[18];
            sceneCarDataMessage.car.p_12_50 = cc.distToClosestObjects[19];
            sceneCarDataMessage.car.p_15_00 = cc.distToClosestObjects[20];
            sceneCarDataMessage.car.p_17_50 = cc.distToClosestObjects[21];
            sceneCarDataMessage.car.p_20_00 = cc.distToClosestObjects[22];
            sceneCarDataMessage.car.p_22_50 = cc.distToClosestObjects[23];
            sceneCarDataMessage.car.p_25_00 = cc.distToClosestObjects[24];
            sceneCarDataMessage.car.p_27_50 = cc.distToClosestObjects[25];
            sceneCarDataMessage.car.forward_right_right = cc.distToClosestObjects[26];
            sceneCarDataMessage.car.forward_right = cc.distToClosestObjects[27];
            sceneCarDataMessage.car.right = cc.distToClosestObjects[28];
            sceneCarDataMessage.car.angular_velocity = cc.GetAngularVelocity();
            sceneCarDataMessage.car.goal_1 = GetAllGoalCount();
            sceneCarDataMessage.car.goal_2 = GetVelocityCarAngleDiff()/180;
            sceneCarDataMessage.car.goal_3 =  GetGoalCount("Goal-3");
            sceneCarDataMessage.car.goal_4 =  GetGoalCount("Goal-4");
            sceneCarDataMessage.car.acceleration = cc.GetAcceleration();
            sceneDataMessage.object_location.x = car.transform.position.x;
            sceneDataMessage.object_location.y = car.transform.position.y;
            sceneDataMessage.object_location.z = car.transform.position.z;
            sceneDataMessage.pole_cart.pole_angular_speed = sceneCarDataMessage.car.speed;
            sceneDataMessage.pole_cart.upright = sceneCarDataMessage.car.has_reached_goal;
            sceneDataMessage.last_executed_cmd_id = sceneCarDataMessage.last_executed_cmd_id;  
        }
        Debug.Log(topicName + "->"+ sceneCarDataMessage);
        ros.Send(topicName, sceneCarDataMessage);
    }

    private float GetAllGoalCount(){
        float i = 1;
        float allGoalCount = 0;
        for(i=1;i<32;i++){
            allGoalCount+=GetGoalCount("Goal-" + i);
        }
        return allGoalCount;
    }
    private float GetGoalCount(string name){
        string goal; 
        string goalName = "trig: " + name;
        if(car.GetComponent<CarController>().stats.TryGetValue(goalName, out goal))
            return float.Parse(goal);
        else
            return 0;
    }

    private string GetCurrentGoalName(){
        CarController cc = car.GetComponent<CarController>();
        string goalName = cc.goals[(cc.goalIndex + 1) % cc.goals.Count].name;
        return goalName;
    }

    private string GetLastGoalCompletedName(){
        CarController cc = car.GetComponent<CarController>();
        if(cc.goalIndex == 0)
            return "";
        
        string lastGoalName = cc.goals[(cc.goalIndex) % cc.goals.Count].name;
        return lastGoalName;
    } 
    
    private float GetVelocityCarAngleDiff()
    {
        Rigidbody rb = car.GetComponent<Rigidbody>();
        Vector3 velocity = rb.velocity;
        float angle = Vector3.SignedAngle(
            car.transform.forward, 
            velocity,
            Vector3.up);
        return angle;
    }

    private float GetDistanceFromGoal(){
        Collider coll = goal.GetComponent<Collider>();
        Vector3 closestPoint = coll.ClosestPointOnBounds(car.transform.position);
        float distance = Vector3.Distance(closestPoint, car.transform.position);
        return distance;
    }
    public float AngleDir(GameObject car, Vector3 trajectory)
    {
       Vector3 DirectionFromCarToTrajectory = car.transform.InverseTransformPoint(trajectory);

        if (DirectionFromCarToTrajectory.x < 0)
        {
            return -1f;
        }
        else if (DirectionFromCarToTrajectory.x > 0)
        {
            return 1f;
        }
        
        return 0f;
    }  

    private void MoveDebugSphere(Vector3 pos, Vector3 carPos)
    {
        if(debugSphere == null){
            throw new MissingReferenceException ("no debugsphere");
        }
        debugSphere.transform.SetPositionAndRotation(
            new Vector3(pos.x,pos.y,carPos.z), 
            debugSphere.transform.rotation);
    }
    
    public Vector3 GetClosestPointOnCollider(GameObject first, GameObject second)
    {
        CarController cc = car.GetComponent<CarController>();
        Vector3 closestPoint;
        if(targetOnNextGoal.TryGetValue(cc.goalIndex, out closestPoint))
        {
            closestPoint = targetOnNextGoal[cc.goalIndex];
        }
        else
        {
            closestPoint = cc.GetClosestPointOnCollider(first, second);
            targetOnNextGoal[cc.goalIndex]=closestPoint;
        }

        if(!cc.sphere)
            cc.sphere = Instantiate(cc.spherePrefab);
        cc.sphere.transform.position = closestPoint;
        int road = LayerMask.NameToLayer("Road");
        cc.sphere.layer = road;
        cc.sphere.transform.localScale = new Vector3(2,2,2);     
        return closestPoint;
    }
   private float GetAngleToGoal(){
        CarController cc = car.GetComponent<CarController>();
        GameObject currentGoal = cc.goals[cc.goalIndex % cc.goals.Count];
        GameObject nextGoal = cc.GetNextGoal();
        Vector3 closestPointOnGoal = GetClosestPointOnCollider(currentGoal, nextGoal);
        Vector3 targetDir = closestPointOnGoal - car.transform.position;
        float angle = Vector3.SignedAngle(
            car.transform.forward, 
            targetDir,
            Vector3.up);
        return angle;
    }
    private float GetDistanceFromTraj(){
        if(!SimController.instance.idealTrajectoryPrefab)
            return 0f;
        Collider coll = idealTrajectory.GetComponent<Collider>();
        Vector3 closestPoint = Physics.ClosestPoint(
            car.transform.position, 
            coll,
            idealTrajectory.transform.position,
            idealTrajectory.transform.rotation);
        float distance = Vector3.Distance(closestPoint, car.transform.position);
        MoveDebugSphere(closestPoint, closestPoint);//car.transform.position);
        float direction = AngleDir(car, closestPoint);
        Debug.Log("anglebetween: " + direction + " " + car.transform.position );
        return direction * distance;
    }
    private bool IsGoalComplete(){
        //Debug.Log(goal);
        CarController cc = car.GetComponent<CarController>();
        bool isGoalTrue=false;
        foreach(GameObject g in cc.goals){
            isGoalTrue = g.GetComponent<Goal>().goalComplete;
            if(isGoalTrue)
                break;
        } 
        bool isComplete = goal.GetComponent<Goal>().goalComplete || isGoalTrue;
        if(isComplete){
            SimController.instance.goal = goal = cc.GetNextGoal();
            return true;
        }else{
            return false;
        }    
    }
}
