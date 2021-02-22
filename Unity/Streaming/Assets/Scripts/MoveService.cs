using UnityEngine;
using System.Collections;
using System.Linq;
using System.Threading.Tasks;
using RosMessageTypes.NiryoMoveit;
using RosMessageTypes.Moveit;
using System;

public class MoveService : MonoBehaviour
{
    // ROS Connector
    private ROSConnection ros;

    // Hardcoded variables 
    private int numRobotJoints = 6;
    private readonly int jointAssingmentWaitMillis = 100;

    // Articulation Bodies
    private ArticulationBody[] jointArticulationBodies;

    public string goalTopic = "move_action/goal";
    public string resultTopic = "move_action/result";
    public string feedbackTopic = "move_action/feedback";
    public GameObject niryoOne;

    private MoveActionGoal activeGoal = null;

    enum CommandType
    {
        TRAJECTORY = 1
    };

    enum Result
    {
        SUCCESS = 0,
        ERROR = 1,
    };

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

        ros.Subscribe<MoveActionGoal>(goalTopic, onGoal);
    }

    private void sendResult(MoveActionResult result)
    {
        ros.Send(resultTopic, result);
    }

    private void sendFeedback(MoveActionFeedback feedback)
    {
        ros.Send(feedbackTopic, feedback);
    }

    private void onGoal(MoveActionGoal goal)
    {
        Debug.Log("onGoal: " + goal.ToString());
        // TODO: Currently we ignore new goals if one is received while another
        // is executing. In future: behave more like an actionlib server which
        // will pre-empt currently running goal when a new one is received.
        if (activeGoal != null)
        {
            Debug.LogWarning("onGoal: Ignoring new goal because one is already running");
            return;
        }

        switch ((CommandType) goal.cmd.cmd_type)
        {
            case CommandType.TRAJECTORY:
                processTrajectoryGoal(goal);
                break;
            default:
                Debug.LogWarning("onGoal: unknown command type " + goal.cmd.cmd_type);
                break;
        }
    }

    private async void processTrajectoryGoal(MoveActionGoal goal)
    {
        Debug.Log("accepting trajectory goal");
        activeGoal = goal;
        var result = new MoveActionResult((int)Result.ERROR);
        try {
            result = await executeTrajectories(goal.cmd.trajectory);
        } 
        finally
        {
            Debug.Log("Goal complete; publishing result");
            sendResult(result);
            activeGoal = null;
        }
    }


    private async Task<MoveActionResult> executeTrajectories(RobotTrajectory trajectory)
    {
        // For every robot pose in trajectory plan
        int step = 0;
        var totalPoints = trajectory.joint_trajectory.points.Length;
        Debug.Log("Executing trajectory with " + totalPoints + " points");
        foreach (var point in trajectory.joint_trajectory.points)
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
            await Task.Delay(jointAssingmentWaitMillis);

            // Publish progress.
            double progress = (double)++step / totalPoints;
            sendFeedback(new MoveActionFeedback(progress));
        }
        return new MoveActionResult((int)Result.SUCCESS);
    }

        // Update is called once per frame
    void Update()
    {

    }
}
