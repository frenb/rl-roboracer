using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using System.Linq;
using System;

public class CarController : MonoBehaviour {
    public List<AxleInfo> axleInfos; // the information about each individual axle
    public float maxMotorTorque; // maximum torque the motor can apply to wheel
    public float maxSteeringAngle; // maximum steer angle the wheel can have
     // finds the corresponding visual wheel
    // correctly applies the transform
    public GameObject spherePrefab;
    public GameObject statsCanvas;
    public GameObject statsCanvasPrefab;
    public float angle;
    public float acceleration;
    public int cmd_id;
    private float[] velocities = new float[3];
    private Vector3[] positions = new Vector3[3];
    public bool applyForceDone = false;
    public Rigidbody rb;
    public RaycastHit [] distToClosestObjectsRHs = new RaycastHit [29]; 
    public float [] distToClosestObjects = new float [29]; 
    public bool [] distToClosestObjectsBools = new bool [29]; 
    public GameObject [] distToClosestObjectsDebugSpheres = new GameObject [29]; 
    public LayerMask layerMask;
    public bool rayCastEnabled = false;
    public bool manualControlEnabled = false;
    public  Dictionary<string, string> stats = new Dictionary<string, string>();
    public int numCollisions=0;
    public int numStats=0;
    public List<Collision> collisions = new List<Collision>();
    public List<Collider> triggers = new List<Collider>();
    public GameObject[] goals2 = new GameObject[4];
    public List<GameObject> goals = new List<GameObject>();
    public int goalIndex = 0;
    public float uiOffsetX = 0;
    public float uiOffsetY = 0;
    public bool showUI = false;
    public Camera Cam2;
    public GameObject sphere;
    public float steering;
    public float motor;
    public GameObject rockPrefab;
    public int numObstacles=0;
    Vector3 fwd;
    Vector3 left;
    Vector3 right;
    Dictionary<directions,float> DirectionToAngle = new Dictionary<directions, float>();
    public enum directions{
        LEFT,
        FORWARD_LEFT,
        FORWARD_LEFT_LEFT,
        n_27_50,
        n_25_00,
        n_22_50,
        n_20_00,
        n_17_50,
        n_15_00,
        n_12_50,
        n_10_00,
        n_07_50,
        n_05_00,
        n_02_50,
        FORWARD,
        p_02_50,
        p_05_00,
        p_07_50,
        p_10_00,
        p_12_50,
        p_15_00,
        p_17_50,
        p_20_00,
        p_22_50,
        p_25_00,
        p_27_50,
        FORWARD_RIGHT_RIGHT,
        FORWARD_RIGHT,
        RIGHT
    } 
    
    public void Start(){
        SetUpGoalsArray();
        SetDirections();
        SetUpDirectionToAngle();
        PlaceObstacles();
        // Cam2 = GameObject.Find ("WaymoDriverCamera").GetComponent<Camera> ();
        // if (Cam2) {
        //      Debug.Log ("found Cam2");
        //      Cam2.rect = new Rect (0.5f, 0, 0.5f, 1);
        //      Camera.main.rect = new Rect (0, 0, 0.5f, 1);
        //  } else {
        //      Camera.main.rect = new Rect (0, 0, 1, 1);
        //  }
    }
    public void SetUpDirectionToAngle(){
        DirectionToAngle[directions.LEFT] = -90;
        DirectionToAngle[directions.FORWARD_LEFT_LEFT] = -60;
        DirectionToAngle[directions.FORWARD_LEFT] = -30;
        DirectionToAngle[directions.n_27_50] = -27.50f;
        DirectionToAngle[directions.n_25_00] = -25.00f;
        DirectionToAngle[directions.n_22_50] = -22.50f;
        DirectionToAngle[directions.n_20_00] = -20.00f;
        DirectionToAngle[directions.n_17_50] = -17.50f;
        DirectionToAngle[directions.n_15_00] = -15.00f;
        DirectionToAngle[directions.n_12_50] = -12.50f;
        DirectionToAngle[directions.n_10_00] = -10.00f;
        DirectionToAngle[directions.n_07_50] = -7.50f;
        DirectionToAngle[directions.n_05_00] = -5.00f;
        DirectionToAngle[directions.n_02_50] = -2.50f;
        DirectionToAngle[directions.FORWARD] = 0.00f;
        DirectionToAngle[directions.p_27_50] = 27.50f;
        DirectionToAngle[directions.p_25_00] = 25.00f;
        DirectionToAngle[directions.p_22_50] = 22.50f;
        DirectionToAngle[directions.p_20_00] = 20.00f;
        DirectionToAngle[directions.p_17_50] = 17.50f;
        DirectionToAngle[directions.p_15_00] = 15.00f;
        DirectionToAngle[directions.p_12_50] = 12.50f;
        DirectionToAngle[directions.p_10_00] = 10.00f;
        DirectionToAngle[directions.p_07_50] = 7.50f;
        DirectionToAngle[directions.p_05_00] = 5.00f;
        DirectionToAngle[directions.p_02_50] = 2.50f;
        DirectionToAngle[directions.FORWARD_RIGHT] = 30f;
        DirectionToAngle[directions.FORWARD_RIGHT_RIGHT] = 60f;
        DirectionToAngle[directions.RIGHT] = 90f;

    }
    public void SetDirections()
    {
        fwd = transform.TransformDirection(Vector3.forward);
        left = transform.TransformDirection(Vector3.left);
        right = transform.TransformDirection(Vector3.right);
    }
    public void SetUpGoalsArray()
    {
        goals = new List<GameObject>();
        int i=1;
        while(true){
            GameObject obj = GameObject.Find("Goal-" + i); 
            if(obj == null)
                break;
            goals.Add(obj);
            i++;
        }
    }
    
    public bool IsNextGoal(string goalName){
        if (goals.Count>0 && goals[(goalIndex + 1) % goals.Count].name == goalName)
            return true;
        else 
            return false;
    }
    public GameObject GetNextGoal(){
        if(goals == null)
        {
            Debug.Log("goals is null");
            return null;
        }
        if(goals.Count ==0)
            return null;

        return goals[(goalIndex + 1) % goals.Count];
    }
    public Vector3 GetClosestPointOnCollider(GameObject car, GameObject other)
    {
        return other.transform.position;
        // Collider coll = other.GetComponent<Collider>();
        // Vector3 closestPoint = coll.ClosestPoint(car.transform.position);
        // Vector3 closestPoint = Physics.ClosestPoint(
        //     car.transform.position, 
        //     coll,
        //     other.transform.position,
        //     other.transform.rotation);
        // return new Vector3(closestPoint.x,car.transform.position.y, closestPoint.z);
        // return closestPoint;
    }
    public void UpdateGoalStates()
    {
        foreach(GameObject go in goals)
        {
            if (IsNextGoal(go.name))
            {
                Renderer r = go.GetComponent<Renderer>();
                //Call SetColor using the shader property name "_Color" and setting the color to red
                if(r==null)
                    return;
                r.material.SetColor("_Color", Color.green);
            }
            else
            {
                Renderer r = go.GetComponent<Renderer>();
                //Call SetColor using the shader property name "_Color" and setting the color to red
                if(r==null)
                    return;
                r.material.SetColor("_Color", Color.grey);
            }

        }
    }

    public void ApplyLocalPositionToVisuals(WheelCollider collider)
    {
        if (collider.transform.childCount == 0) {
            return;
        }
     
        Transform visualWheel = collider.transform.GetChild(0);
     
        Vector3 position;
        Quaternion rotation;
        collider.GetWorldPose(out position, out rotation);
     
        //visualWheel.transform.position = position;
        visualWheel.transform.rotation = rotation;
    }
    private string setText(string Key, string Value, int i){
        string output = "i: " + i + " "; 
        output += Key; 
        output += ": " + Value;
        return output;
    }

    private void setPositionOfText(GameObject t_parent, Text t, int i){
        t_parent.transform.parent = statsCanvas.gameObject.transform;
        RectTransform rt = t_parent.GetComponent<RectTransform>();
        RectTransform rtCanvas = statsCanvas.GetComponent<RectTransform>();
        rt.anchoredPosition.Set(0, 0);
        rt.anchorMax = new Vector2(0,1);
        rt.anchorMin = new Vector2(0,1);
        rt.transform.localPosition = new Vector3(
            uiOffsetX-175,
            uiOffsetY +225-15*i,
            0);
        t.GetComponent< RectTransform >( ).SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, 400);
    }
    private void UpdateUI()
    {
        GetStats();        
        numStats = stats.Count;
        if(!showUI)
            return;
        for (int i = 0; i < numStats; i++) 
        {
            KeyValuePair<string, string> stat = stats.ElementAt(i);
            Text t;
            GameObject t_parent = GameObject.Find(stat.Key);
            Debug.Log("t_parent: " + t_parent);
            if (!t_parent){ 
                t_parent = new GameObject(stat.Key);
                t = t_parent.AddComponent<Text>();
                t.font = Resources.GetBuiltinResource(typeof(Font), "Arial.ttf") as Font;
            } else{
                t = t_parent.GetComponent<Text>();
            }
            t.text = setText(stat.Key, stat.Value,i);
            setPositionOfText(t_parent, t, i);
            Debug.Log("xxxxxx " + t.text + " " + t_parent.transform.position);
        }
    }
    public float GetAcceleration(){
        float mph1 = velocities[2]-velocities[1];
        float mph2 = velocities[1]-velocities[0];
        return (mph1 - mph2) / 0.02f;
    }

    public void AddObjectDetections(directions d, Dictionary<string, string> stats)
    {
        if(distToClosestObjectsBools[(int) d])
        {
            stats[""+d] = "" + distToClosestObjects[(int) d] + " name: " + distToClosestObjectsRHs[(int) d].collider.name;
            addDebugSphere(d);
        }   
        else
        {
            stats[""+d] = "-1";
        }
    }

    public void addDebugSphere(directions d)
    {
        RaycastHit obj = distToClosestObjectsRHs[(int) d];
        string colliderName = obj.collider.name;
        try{
            bool isSphere = colliderName.Substring(0,Mathf.Min(6,colliderName.Length)) != "Sphere";
            if(isSphere){
                GameObject s = Instantiate(spherePrefab);
                s.name = "Sphere";
                s.transform.parent = GameObject.Find("PerceptionObjects").transform;
                s.GetComponent<SphereController>().animate = true;
                distToClosestObjectsDebugSpheres[(int) d] = s;
                distToClosestObjectsDebugSpheres[(int)d].transform.position = distToClosestObjectsRHs[(int) d].point;
                distToClosestObjectsDebugSpheres[(int)d].name="Sphere"+ distToClosestObjectsRHs[(int) d].point;
                distToClosestObjectsDebugSpheres[(int) d].layer = LayerMask.NameToLayer("Default");
            }
        } catch(System.Exception){
            Debug.Log("caught it");
        }
    }

    public void addKinematics(){
        Rigidbody rb = GetComponent<Rigidbody>();
        stats["speed"] = "" + GetSpeed();
        stats["angular_velocity"] = "" + GetAngularVelocity();
        stats["acceleration"] = "" + GetAcceleration();
        stats["LayerMaskId"] = "" + LayerMask.NameToLayer("Default");   
    }

    
    public float GetSpeed(){
        Rigidbody rb = GetComponent<Rigidbody>();
        return rb.velocity.magnitude;
    }
    public float GetAngularVelocity()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        return rb.angularVelocity.magnitude;
    }
    public void addPerception(){
        foreach(directions d in directions.GetValues(typeof(directions))){
            AddObjectDetections(d, stats);
        }
    }

    public void addCollisions(){
        foreach(Collision c in collisions){
            if(!c.gameObject.name.Contains("Curb") 
                && !c.gameObject.name.Contains("Rail"))
                continue;
            string name = "coll: " + c.gameObject.name;
            stats[name] = "" + c.collider.ClosestPoint(transform.position);
            numStats = stats.Count;
        }

        foreach(Collider c in triggers){
            string name = "trig: " + c.gameObject.name;
            string value;
            if (!stats.TryGetValue(name, out value))
                stats[name] = "" + 1;
            
            numStats = stats.Count;
        }
    }
    public void setUpCanvas(){
        statsCanvas = GameObject.Find("stats");
        if(!statsCanvas && statsCanvasPrefab)
        {
            statsCanvas = Instantiate(statsCanvasPrefab);
            statsCanvas.name = "stats";
            RectTransform rt = statsCanvas.GetComponent<RectTransform>();
            rt.anchoredPosition.Set(0, 0);
            rt.anchorMax = new Vector2(0,1);
            rt.anchorMin = new Vector2(0,1);
        }
    }
 
    public void GetStats(){
        //setUpCanvas();
        //addKinematics();
        addPerception();
        addCollisions();  
    }
    
    void DrawLine(Vector3 start, Vector3 end, Color color, float duration = 0.2f)
    {
        GameObject myLine = new GameObject("Line");
        myLine.transform.position = start;
        myLine.AddComponent<LineRenderer>();
        LineRenderer lr = myLine.GetComponent<LineRenderer>();
        lr.startColor=color;
        lr.endColor=color;
        lr.startWidth=0.1f;
        lr.endWidth=0.1f;
        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        myLine.gameObject.name = "Line-" + end;
        GameObject.Destroy(myLine, duration);
    }
    public bool DrawRay(Vector3 start, Vector3 direction, float distance, Color color, directions d)
    {
        RaycastHit hitData = new RaycastHit();
        Vector3 dwn = transform.TransformDirection(Vector3.down);
        bool foundObject = false;
        int i;
        float radius = 0.15f;
        for(i=-5;i<5;i++){
            int myD = (int)d;
            foundObject 
            = distToClosestObjectsBools[(int) d]
            = Physics.SphereCast(start+dwn*i*0.001f, radius, direction, out  hitData, 100000000, layerMask);
            
            Debug.DrawRay(
                start + dwn*i*0.01f,
                direction.normalized*Math.Min(distance, hitData.distance),
                foundObject ? Color.red: color, 
                0.02f,
                true);
            
            DrawLine(
                start + dwn*i*0.01f, 
                start + dwn*i*0.01f + direction*Math.Min(distance, hitData.distance),
                foundObject ? Color.red: color,
                0.02f
            );

            if(foundObject)
                break; 
        }
        
        if(distToClosestObjectsBools[(int) d])
        {
            distToClosestObjectsRHs[(int) d] = hitData;
            distToClosestObjects[(int) d] = hitData.distance;
            return true;
        } 
        else {
            Debug.Log("nothing in this direction " + d + " old data: " + distToClosestObjects[(int) d]);
            // only override distance to 100 for directions other than right or left
            // otherwise use the last value
            // if (d == directions.LEFT || d == directions.RIGHT)
            //     return false;
            
            // distToClosestObjectsRHs[(int) d] = new RaycastHit();
            // distToClosestObjects[(int) d] = 100;
            return false;
        }
        //      distToClosestObjectsRHs[(int) d] = new RaycastHit();
        //      distToClosestObjects[(int) d] = 100;
        // }
    }
    float [] rotations = {0f}; //, 1f,2f,3f,4f,5f,-1f,-2f,-3f,-4f,-5f};
    float d = 50f;
    public bool DrawRayWrapper(float angle, directions direction, float distanceMultiple)
    {
        bool hitSomething = false;
          //forward
        foreach(float r in rotations)
        {
            hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(angle+r, Vector3.up) * fwd, d*distanceMultiple, Color.green, direction);
            if(hitSomething)
                break;
        }
        return hitSomething;
    }
    public void GetIntersectionDistance()
    {
        if(!rayCastEnabled)
            return; 
        
        bool hitSomething = false;
        Dictionary<directions, bool> hitNothings = new Dictionary<directions, bool>();
        
        // bool leftHitNothing = false;
        // bool rightHitNothing = false;
        // bool forwardHitNothing = false;
        // bool forwardLeftHitNothing = false;
        // bool forwardLeftLeftHitNothing = false;
        // bool forwardRightHitNothing = false;
        // bool forwardRightRightHitNothing = false;
        //forward
        // foreach(float r in rotations)
        // {
        //     hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(0+r, Vector3.up) * fwd, d*2, Color.green, directions.FORWARD);
        //     if(hitSomething)
        //         break;
        // }
        foreach (directions direction in Enum.GetValues(typeof(directions)))
        {
            float multiple = 1;
            if(direction == directions.FORWARD)
            {
                multiple = 2;
            } else{
                multiple = 1;
            }

            hitSomething = DrawRayWrapper(DirectionToAngle[direction], direction, multiple);
            if(!hitSomething)
                hitNothings[direction]=true;
            hitSomething=false;
        }
        
        
        // //forward right
        // foreach(float r in rotations)
        // {
        //         hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(30+r, Vector3.up) * (fwd), d, Color.green, directions.FORWARD_RIGHT);
        //     if(hitSomething)
        //         break;
        // }
        // if(!hitSomething)
        //     forwardRightHitNothing=true;
        // hitSomething=false;
        
        // //forward right right
        // foreach(float r in rotations)
        // {
        //     hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(60+r, Vector3.up) * (fwd), d, Color.green, directions.FORWARD_RIGHT_RIGHT);
        //     if(hitSomething)
        //         break;
        // }
        // if(!hitSomething)
        //     forwardRightRightHitNothing=true;
        // hitSomething=false;
        // //forward left
        // foreach(float r in rotations)
        // {
        //     hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(-30+r, Vector3.up) * (fwd), d, Color.green, directions.FORWARD_LEFT);
        //     if(hitSomething)
        //         break;    
        // }
        // if(!hitSomething)
        //     forwardLeftHitNothing=true;
        // hitSomething=false;
        //    //forward left
        // foreach(float r in rotations)
        // {
        //     hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(-60+r, Vector3.up) * (fwd), d, Color.green, directions.FORWARD_LEFT_LEFT);
        //     if(hitSomething)
        //         break;    
        // }
        // if(!hitSomething)
        //     forwardLeftLeftHitNothing=true;
        // hitSomething=false;
        // //left
        // foreach(float r in rotations)
        // {
        //     hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(-90+r, Vector3.up) * (fwd), d, Color.green, directions.LEFT);
        //     if(hitSomething)
        //         break;
        // }
        // if(!hitSomething)
        //     leftHitNothing=true;
        // hitSomething=false;
        // //right
        // foreach(float r in rotations)
        // {
        //     hitSomething = DrawRay(transform.position, Quaternion.AngleAxis(90+r, Vector3.up) * (fwd), d, Color.green, directions.RIGHT);
        //     if(hitSomething)
        //         break;
        // }
        // if(!hitSomething)
        //     rightHitNothing=true;
        // hitSomething=false;
        if(hitNothings.Count == 0)
            return;
        bool anyHitNothing = false;
        string message = "";
        foreach (directions direction in Enum.GetValues(typeof(directions)))
        {
            bool b; 
            hitNothings.TryGetValue(direction,out b);
            if(b){
                anyHitNothing = true;
                message += direction + " " + b + " ";
            }
        }
        if(anyHitNothing)
        {
            Debug.Log(message);
        }
        // if (
        //     forwardHitNothing
        //     || forwardLeftHitNothing || forwardLeftLeftHitNothing 
        //     || forwardRightHitNothing || forwardRightRightHitNothing 
        //     || leftHitNothing || rightHitNothing )
        // {
        //     Debug.Log(
        //         "I am still missing things ahhhhhhhhh fwd: " + forwardHitNothing 
        //         + " fwd_left: " + forwardLeftHitNothing
        //         + " fwd_left_left: " + forwardLeftLeftHitNothing
        //         + " fwd_right_right: " + forwardRightRightHitNothing
        //         + " fwd_right: " + forwardRightHitNothing);
        // }
    }
    
    public void FixedUpdate()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        SetDirections();
        GetIntersectionDistance();
        UpdateGoalStates();
        velocities[1]=velocities[2];
        velocities[0]=velocities[1]; 
        velocities[2]=rb.velocity.magnitude;

        positions[1]=positions[2];
        positions[0]=positions[1]; 
        positions[2]=transform.position;

        UpdateUI();
        
        if(manualControlEnabled){
            acceleration = Input.GetAxis("Vertical");
            angle = Input.GetAxis("Horizontal");
            steering = maxSteeringAngle * angle;
            motor = maxMotorTorque * acceleration;
        }
        if(!applyForceDone)
        {
            motor = maxMotorTorque * acceleration;
            steering = maxSteeringAngle * angle;
       

            foreach (AxleInfo axleInfo in axleInfos) {
                if (axleInfo.steering) {
                    axleInfo.leftWheel.steerAngle = steering;
                    axleInfo.rightWheel.steerAngle = steering;
                }
                if (axleInfo.motor) {
                    axleInfo.leftWheel.motorTorque = motor;
                    axleInfo.rightWheel.motorTorque = motor;
                }
                ApplyLocalPositionToVisuals(axleInfo.leftWheel);
                ApplyLocalPositionToVisuals(axleInfo.rightWheel);
            }
         }
    }
    public IEnumerator ApplyForce(float angle, float accceleration) {
        Steer(angle);
        Accelerate(accceleration);
        applyForceDone=false;
        Debug.Log("applyForceDone: " + applyForceDone + " cmd_id:" + cmd_id);
        yield return new WaitForSeconds(0.0f);
        applyForceDone=true;
        Debug.Log("applyForceDone: " + applyForceDone + " cmd_id:" + cmd_id);
    }
    public void Steer(float angle)
    {
        this.angle = angle;
        float steerAngle = maxSteeringAngle * angle;
        // foreach (AxleInfo axleInfo in axleInfos) {
        //     if (axleInfo.steering) {
        //         axleInfo.leftWheel.steerAngle = steerAngle;
        //         axleInfo.rightWheel.steerAngle = steerAngle;
        //     }
        // }
    }
    
    public void Accelerate(float motor)
    {
        this.acceleration = motor;
        float motorTorque = maxMotorTorque * motor;
        // foreach (AxleInfo axleInfo in axleInfos) {
        //     if (axleInfo.motor) {
        //         axleInfo.leftWheel.motorTorque = motorTorque;
        //         axleInfo.rightWheel.motorTorque = motorTorque;
        //     }
        // }
    }
    public bool HasCrashed()
    {
        return collisions.Count > 0;
    }
   
    public Vector3 GetPointOnMesh(Vector3[] meshPoints, GameObject road){
        int[] tris = road.transform.GetComponent<MeshFilter>().sharedMesh.triangles;
        int triStart = UnityEngine.Random.Range(0, meshPoints.Length / 3) * 3; // get first index of each triangle
 
        float a = UnityEngine.Random.value;
        float b = UnityEngine.Random.value;
         
        if(a + b >= 1){ // reflect back if > 1
            a = 1 - a;
            b = 1 - b;
        }
 
        Vector3 newPointOnMesh = meshPoints[triStart] + (a * (meshPoints[triStart + 1] - meshPoints[triStart])) + (b * (meshPoints[triStart + 2] - meshPoints[triStart])); // apply formula to get new random point inside triangle
        return newPointOnMesh;
    }
    public void PlaceObstacles()
    {
        int MaxObstacles = numObstacles;
        Debug.Log("PlaceObstacles: " + MaxObstacles);
        GameObject[] roads = GetListOfRoads();
        List<GameObject> rocks = new List<GameObject>();
        for (int i = 0; i < MaxObstacles; i++)
        {
            int roadIndex = UnityEngine.Random.Range(0, roads.Length);
            GameObject r = roads[roadIndex];
            Debug.Log("Road " + r.transform.position);
            Mesh mesh = r.GetComponent<MeshFilter>().sharedMesh;
            Vector3[] vertices = mesh.vertices;
            var q = UnityEngine.Random.Range(0, vertices.Length);
            Vector3 spawnPoint = r.transform.TransformPoint(GetPointOnMesh(vertices, r));
            if(IsTooClose(spawnPoint, rocks))
            {
                i--;
                continue;
            }
            GameObject newRock = Instantiate(rockPrefab);
            newRock.name = "Rock" + i;
            newRock.transform.parent = GameObject.Find("Obstacles").transform;
            newRock.transform.localPosition = spawnPoint;
            rocks.Add(newRock);
        }
    }

    public bool IsTooClose(Vector3 newRockPosition, List<GameObject> existingRocks)
    {
        float minDistance = 10;
        foreach(GameObject r in existingRocks)
        {
            float distance2Object = Vector3.Distance(newRockPosition, r.transform.position);
            if(distance2Object < minDistance)
                return true;
        }
        return false;
    }

    private GameObject[] GetListOfRoads()
    {
       return Resources.FindObjectsOfTypeAll<GameObject>().Where(obj => obj.name == "Road").ToArray<GameObject>(); 
    }

    void OnCollisionEnter(Collision collision)
    {
        string name = collision.gameObject.name;
        bool isCurb = name.Contains("Curb");
        bool isRail = name.Contains("Rail");
        bool isRock = name.Contains("rock");
        if(!isCurb && !isRail && !isRock)
            return;
        collisions.Add(collision);
        numCollisions = collisions.Count;
    }

     void OnTriggerEnter(Collider c)
    {
        string name = "trig: " + c.gameObject.name;
        bool isGoal = name.Contains("Goal");
        if(!isGoal)
            return;
        if(IsNextGoal(c.gameObject.name))
            goalIndex++;
        triggers.Add(c);
        string value;
        if(stats.TryGetValue(name, out value))
            stats[name]=""+ (int.Parse(value)+1);
        numCollisions = collisions.Count;
    }
}

[System.Serializable]
public class AxleInfo {
    public WheelCollider leftWheel;
    public WheelCollider rightWheel;
    public bool motor; // is this wheel attached to motor?
    public bool steering; // does this wheel apply steer angle?
}