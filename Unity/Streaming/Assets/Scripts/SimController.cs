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
    public GameObject niryoOnePrefab;
    public GameObject targetPrefab;
    public GameObject targetPlacementPrefab;
    public GameObject poleCartPrefab;

    public GameObject streamCamera;
    public GameObject niryoOne { get; private set; }
    public GameObject target { get; private set; }
    public GameObject targetPlacement { get; private set; }
    public GameObject poleCart { get; private set; }

    private ROSConnection ros;
    private bool sentStarted = false;
    private static SimController _instance = null;
    private MoveService moveService;
    private SceneDataPublisher sceneDataPublisher;

    private enum Command
    {
        RESTART = 0
    };

    private enum Status
    {
        STARTED = 0
    };

    public SimController()
    {
        _instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {
        InstantiateObjects();

        ros = ROSConnection.instance;

        // Ros nodes instatiates here after world created.
        moveService = gameObject.AddComponent(typeof(MoveService)) as MoveService;
        sceneDataPublisher = gameObject.AddComponent(typeof(SceneDataPublisher)) as SceneDataPublisher;

        ros.Subscribe<SimCommand>(simCommandTopic, onCommand);
    }

    private void DestroyObjects()
    {
        Destroy(niryoOne);
        Destroy(target);
        Destroy(targetPlacement);
        if (poleCart != null)
        {
            Destroy(poleCart);
        }
    }

    private void InstantiateObjects()
    {
        niryoOne = Instantiate(niryoOnePrefab);
        target = Instantiate(targetPrefab);
        targetPlacement = Instantiate(targetPlacementPrefab);
        if (poleCartPrefab != null)
        {
            poleCart = Instantiate(poleCartPrefab);
        }
    }

    private void onCommand(SimCommand cmd)
    {
        switch ((Command) cmd.cmd )
        {
            case Command.RESTART:
                Restart();
                break;
            default:
                Debug.LogError("Unrecognized Sim Command: " + cmd.cmd);
                break;
        }
    }

    public void Restart()
    {
        Debug.Log("SimController::Restarting");
        DestroyObjects();
        InstantiateObjects();
        moveService.UpdateWorldRefs();
        sceneDataPublisher.UpdateWorldRefs();
    }

    // Update is called once per frame
    void Update()
    {
        if (!sentStarted)
        {
            ros.Send(simStatusTopic, new SimStatus((int)Status.STARTED));
            sentStarted = true;
        }
    }

    private void OnApplicationQuit()
    {
        Debug.Log("SimController::OnApplicationQuit");
    }
}
