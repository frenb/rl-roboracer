using RosMessageTypes.NiryoMoveit;
using UnityEngine;

public class SimController : MonoBehaviour
{
    public string simCommandTopic = "sim_command";
    public string simStatusTopic = "sim_status";
    public static SimController instance
    {
        get
        {
            return _instance;
        }
    }
    // public GameObject niryoOnePrefab;
    // public GameObject targetPrefab;
    // public GameObject targetPlacementPrefab;
    public GameObject carPrefab;
    public GameObject idealTrajectoryPrefab;
    public GameObject spherePrefab;
    public GameObject goalObject;

    // public GameObject streamCamera;
    // public GameObject publishedCamera;
    // public string publishedCameraTopic = "camera/overhead";
    // public GameObject niryoOne { get; private set; }
    // public GameObject target { get; private set; }
    // public GameObject targetPlacement { get; private set; 

    public GameObject car { get; private set; }
    public GameObject carProxy;
    public GameObject goal;
    public GameObject idealTrajectory { get; private set; }
    public GameObject debugSphere { get; private set; }
    // public GameObject goal { get; private set; }
    private ROSConnection ros;
    private bool sentStarted = false;
    private static SimController _instance = null;
    private MoveService moveService;
    private SceneDataPublisher sceneDataPublisher;
    // private CameraPublisher cameraPublisher;
    private enum Command
    {
        RESTART = 0,
        APPLY_FORCE = 1
    };

    private enum Status
    {
        STARTED = 0,
        RESTARTED = 1,
        FORCE_APPLIED = 2
    };

    public SimController()
    {
        _instance = this;
    }

    // Frames to wait after reseting declaring ready.
    private const int WAIT_FRAMES = 5;

    private int currentWait = 0;

    private int command = -1;

    private int applyForceCommandId;

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("starting sim controller");
        Time.timeScale = 3;
        ApplyForce af = new ApplyForce();
        af.num_obstacles=0;
        InstantiateObjects(af);
        ros = ROSConnection.instance;
        // // Ros nodes instatiates here after world created.
        moveService = gameObject.AddComponent(typeof(MoveService)) as MoveService;
        sceneDataPublisher = gameObject.AddComponent(typeof(SceneDataPublisher)) as SceneDataPublisher;
        // // Publish camera frames for computer vision.
        // if (publishedCamera != null)
        // {
        //     cameraPublisher = gameObject.AddComponent(typeof(CameraPublisher)) as CameraPublisher;
        //     cameraPublisher.camera = publishedCamera.GetComponent<UnityEngine.Camera>();
        //     cameraPublisher.topic = publishedCameraTopic;
        // }
        Debug.Log("ros.rosIPAddress=" + ros.rosIPAddress);
        Debug.Log("ros.overrideUnityIP=" + ros.overrideUnityIP);
        ros.Subscribe<SimCommand>(simCommandTopic, onCommand);
    }

    private void DestroyObjects()
    {
        if (car != null)
        {
            DestroyImmediate(car);
        }
        if (idealTrajectory != null)
            DestroyImmediate(idealTrajectory);
        if(debugSphere != null)
            DestroyImmediate(debugSphere);
        GameObject spherePrefabClone = GameObject.Find("SpherePrefab(Clone)");
        if(spherePrefabClone != null)
            DestroyImmediate(spherePrefabClone);
        DestroyPerceptionObject();
        DestroyObstacles();
        //DestroyGoals();
        GameObject g = GameObject.Find("g");
        DestroyImmediate(g);
    }
    private void DestroyGoals()
    {
        GameObject po = GameObject.Find("Goals");
        if(po == null)
            return;
        foreach (Transform child in po.transform) {
            GameObject.Destroy(child.gameObject);
        }
    }
    private void DestroyPerceptionObject()
    {
        GameObject sphere = GameObject.Find("Sphere(Clone)");
        if(sphere != null)
            GameObject.Destroy(sphere);
        GameObject po = GameObject.Find("PerceptionObjects");
        if(po == null)
            return;
        foreach (Transform child in po.transform) {
            GameObject.Destroy(child.gameObject);
        }
    }
    private void DestroyObstacles()
    {
        GameObject po = GameObject.Find("Obstacles");
        if(po == null)
            return;
        foreach (Transform child in po.transform) {
            GameObject.Destroy(child.gameObject);
        }
    }

    private void InstantiateObjects(ApplyForce af)
    {
        int index = 1;
        Debug.Log("Instantiating objects");
        if(idealTrajectoryPrefab != null)
        {
            idealTrajectory = Instantiate(idealTrajectoryPrefab);
            //idealTrajectory.name = "idealTrajectory";
            RandomRotateTrajectoryObject();
        }
        
        if (carPrefab != null)
        {
            car = Instantiate(carPrefab);
            //car.name = "car";
            car.GetComponent<CarController>().numObstacles = (int) af.num_obstacles;
            car.GetComponent<CarController>().SetUpGoalsArray();
            System.Random rand = new System.Random();
            int maxRange =  car.GetComponent<CarController>().goals.Count;
            index = rand.Next(0, maxRange - 1);
            Vector3 carPos = getCarStartPosition(index);
            Quaternion carRot =  getCarStartRotation(index);
            car.transform.position = carPos;
            car.transform.rotation = carRot;
            car.transform.position += car.transform.forward*4;
        }
        //DestroyGoals();
        
        goal = car.GetComponent<CarController>().goals[index];
        car.GetComponent<CarController>().goalIndex = index;
        debugSphere = Instantiate(spherePrefab);
        debugSphere.name = "debugSphere";
        goal.GetComponent<Goal>().goalComplete = false;

        GameObject ngo = GameObject.Find("New Game Object");
    }

    private Vector3 getCarStartPosition(int index){
        
        GameObject goal = car.GetComponent<CarController>().goals[index];
        if(goal){
            return goal.transform.position;
        }
        if (carProxy)
        {
            return carProxy.transform.position;
        }

        return new Vector3();
    }

    private Quaternion getCarStartRotation(int index){
        int countGoals = car.GetComponent<CarController>().goals.Count;
        GameObject goal = car.GetComponent<CarController>().goals[index];
        GameObject g = new GameObject("g");
        g.transform.position= goal.transform.position;
        GameObject nextGoal = 
            car.GetComponent<CarController>().goals[(index + 1) % countGoals];
        g.transform.LookAt(nextGoal.transform.position);
        Quaternion q = g.transform.rotation;
        if(goal){
            return q;
        }
        if (carProxy)
        {
            return carProxy.transform.rotation;
        }

        return new Quaternion();
    }

    private void RandomRotateTrajectoryObject(){ 
        Vector3 rotatePosition = GameObject.Find("rotatePoint").transform.position;
        //for debugging:
        //GameObject sphere = Instantiate(spherePrefab);
        //sphere.transform.position = rotatePosition;
        float rotateAngle = Random.Range(-30f,30f);
        Debug.Log("rotateangle: " + rotateAngle);
        idealTrajectory.transform.RotateAround(rotatePosition, Vector3.up, rotateAngle);
    }

    private void onCommand(SimCommand cmd)
    {
        ApplyForce af = (ApplyForce) cmd.ApplyForce;
        switch ((Command) cmd.cmd )
        {
            case Command.RESTART:
                Debug.Log("RESTARTING");
                Restart(af);
                command = 0;
                break;
            case Command.APPLY_FORCE:
                Debug.Log("APPLY FORCE MESSAGE");
                ApplyForce(af);
                command = 1;
                applyForceCommandId = af.cmd_id;
                break;
            default:
                Debug.LogError("Unrecognized Sim Command: " + cmd.cmd);
                break;
        }
    }

    public void Restart(ApplyForce af)
    {
        Debug.Log("SimController::Restarting");
        DestroyObjects();
        InstantiateObjects(af);
        sceneDataPublisher.UpdateWorldRefs();
        currentWait = WAIT_FRAMES;
    }
    public void ApplyForce(ApplyForce af)
    {
        Debug.Log("SimController::Applying Force " + af);
        // moveService.UpdateWorldRefs();
        // car.transform.RotateAround(
        //     transform.position,
        //     transform.up, 
        //     (float) af.steering_angle);
        // Rigidbody m_Rigidbody = car.GetComponent<Rigidbody>();
        // m_Rigidbody.AddForce(transform.forward * (float) af.acceleration);
        // car.GetComponent<CarController>().Steer((float) af.steering_angle);
        // car.GetComponent<CarController>().Accelerate((float) af.acceleration);
        StartCoroutine(
            car.GetComponent<CarController>()
                .ApplyForce(
                    (float)af.steering_angle, 
                    (float) af.acceleration));
        car.GetComponent<CarController>().numObstacles = (int) af.num_obstacles;
        sceneDataPublisher.UpdateWorldRefs(af);
        currentWait = WAIT_FRAMES;
    }

    // Update is called once per frame
    //TODO: change back to void Update() if broken
    void FixedUpdate()
    {
        if (!sentStarted)
        {
            ros.Send(simStatusTopic, new SimStatus((int)Status.STARTED));
            sentStarted = true;
        }

        if (currentWait > 0)
        {
            currentWait--;
            if (currentWait == 0 && command == 0)
            {
                ros.Send(simStatusTopic, new SimStatus((int)Status.RESTARTED));
                command = -1;
                return;
            }
            // if (currentWait == 0 
            //         && command == 1 
            //         && car.GetComponent<CarController>().applyForceDone){
            //     ros.Send(simStatusTopic, new SimStatus((int)Status.FORCE_APPLIED));
            //     command = -1;
            //     return;
            // }
        }
         
        if ( command == 1 
                && car.GetComponent<CarController>().applyForceDone
                && applyForceCommandId
                    == car.GetComponent<CarController>().cmd_id){
                Debug.Log(
                    car.GetComponent<CarController>().cmd_id
                    + " ********************************************************** "
                    + applyForceCommandId);
                ros.Send(simStatusTopic, new SimStatus((int)Status.FORCE_APPLIED));
                command = -1;
                return;
        }
    }

    private void OnApplicationQuit()
    {
        Debug.Log("SimController::OnApplicationQuit");
    }
}
