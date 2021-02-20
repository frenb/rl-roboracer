using UnityEngine;
using System.Collections;
using System.Linq;
using RosMessageTypes.NiryoMoveit;
using RosMessageGeneration;
using RosMessageTypes.Moveit;

public class MoveExecutorService : MonoBehaviour
{
    // ROS Connector
    private ROSConnection ros;

    // Hardcoded variables 
    private int numRobotJoints = 6;
    private readonly float jointAssignmentWait = 0.1f;

    // Articulation Bodies
    private ArticulationBody[] jointArticulationBodies;


    public string service_name = "move_executor";
    public GameObject niryoOne;

    // Use this for initialization
    void Start()
    {
        ros = ROSConnection.instance;

        jointArticulationBodies = new ArticulationBody[numRobotJoints];
        string shoulder_link = "world/base_link/shoulder_link";
        jointArticulationBodies[0] = niryoOne.transform.Find(shoulder_link).GetComponent<ArticulationBody>();

        string arm_link = shoulder_link + "/arm_link";
        jointArticulationBodies[1] = niryoOne.transform.Find(arm_link).GetComponent<ArticulationBody>();

        string elbow_link = arm_link + "/elbow_link";
        jointArticulationBodies[2] = niryoOne.transform.Find(elbow_link).GetComponent<ArticulationBody>();

        string forearm_link = elbow_link + "/forearm_link";
        jointArticulationBodies[3] = niryoOne.transform.Find(forearm_link).GetComponent<ArticulationBody>();

        string wrist_link = forearm_link + "/wrist_link";
        jointArticulationBodies[4] = niryoOne.transform.Find(wrist_link).GetComponent<ArticulationBody>();

        string hand_link = wrist_link + "/hand_link";
        jointArticulationBodies[5] = niryoOne.transform.Find(hand_link).GetComponent<ArticulationBody>();

        ros.ImplementService<MoveExecutorServiceRequest>(service_name, serviceCallback);
    }

    // Update is called once per frame
    void Update()
    {

    }

    private MoveExecutorServiceResponse serviceCallback(MoveExecutorServiceRequest moveReq)
    {
        // TODO: Allow tracking completion.
        Debug.Log("Received move request");
        StartCoroutine(ExecuteTrajectories(moveReq.trajectory));
        var response = new MoveExecutorServiceResponse(/* accepted */ true);
        return response;
    }

    private IEnumerator ExecuteTrajectories(RobotTrajectory trajectory)
    {
        // For every robot pose in trajectory plan
        foreach(var point in trajectory.joint_trajectory.points)
        {
            float[] jointAngles = point.positions.Select(r => (float)r * Mathf.Rad2Deg).ToArray();
            // Set the joint values for every joint
            for (int joint = 0; joint < jointArticulationBodies.Length; joint++)
            {
                var joint1XDrive = jointArticulationBodies[joint].xDrive;
                joint1XDrive.target = jointAngles[joint];
                jointArticulationBodies[joint].xDrive = joint1XDrive;
            }
            // Wait for robot to achieve pose for all joint assignments
            yield return new WaitForSeconds(jointAssignmentWait);

        }
    }
}
