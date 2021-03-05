using UnityEngine;
using System.Collections;
using System.Linq;
using System.Threading.Tasks;
using RosMessageTypes.NiryoMoveit;
using RosMessageTypes.Moveit;
using System;

public class MoveService : MonoBehaviour, IRosComponent
{
    // ROS Connector
    private ROSConnection ros;

    // Hardcoded variables 
    private int numRobotJoints = 6;
    private readonly int jointAssingmentWaitMillis = 100;

    // Articulation Bodies
    private ArticulationBody[] jointArticulationBodies;
    private ArticulationBody leftGripper;
    private ArticulationBody rightGripper;

    private Transform gripperBase;
    private Transform leftGripperGameObject;
    private Transform rightGripperGameObject;

    public string goalTopic = "move_action/goal";
    public string resultTopic = "move_action/result";
    public string feedbackTopic = "move_action/feedback";
    private GameObject niryoOne;

    private MoveActionGoal activeGoal = null;

    enum CommandType
    {
        TRAJECTORY = 1,
        OPEN_GRIPPER = 2,
        CLOSE_GRIPPER = 3,
        POSITIONS = 4
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
        UpdateWorldRefs();
        ros.Subscribe<MoveActionGoal>(goalTopic, onGoal);
    }

    public void UpdateWorldRefs()
    {
        niryoOne = SimController.instance.niryoOne;

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

        // Find left and right fingers
        string right_gripper = hand_link + "/tool_link/gripper_base/servo_head/control_rod_right/right_gripper";
        string left_gripper = hand_link + "/tool_link/gripper_base/servo_head/control_rod_left/left_gripper";
        string gripper_base = hand_link + "/tool_link/gripper_base/Collisions/unnamed";

        gripperBase = niryoOne.transform.Find(gripper_base);
        leftGripperGameObject = niryoOne.transform.Find(left_gripper);
        rightGripperGameObject = niryoOne.transform.Find(right_gripper);

        rightGripper = rightGripperGameObject.GetComponent<ArticulationBody>();
        leftGripper = leftGripperGameObject.GetComponent<ArticulationBody>();
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
            case CommandType.OPEN_GRIPPER:
            case CommandType.CLOSE_GRIPPER:
                processGripperGoal(goal);
                break;
            case CommandType.POSITIONS:
                processPositionsGoal(goal);
                break;
            default:
                Debug.LogWarning("onGoal: unknown command type " + goal.cmd.cmd_type);
                break;
        }
    }

    private async void processPositionsGoal(MoveActionGoal goal)
    {
        Debug.Log("accepting position goal");
        activeGoal = goal;
        var result = new MoveActionResult((int)Result.ERROR);
        try
        {
            result = await executePosition(goal.cmd.positions);
        }
        finally
        {
            Debug.Log("Goal complete; publishing result");
            sendResult(result);
            activeGoal = null;
        }

    }

    private async Task<MoveActionResult> executePosition(JointPositions positions)
    {
        double[] anglesRad = new double[]
        {
            positions.joint_00,
            positions.joint_01,
            positions.joint_02,
            positions.joint_03,
            positions.joint_04,
            positions.joint_05,
        };

        sendFeedback(new MoveActionFeedback(0.0f));

        float[] jointAngles = anglesRad.Select(r => (float)r * Mathf.Rad2Deg).ToArray();
        // Set the joint values for every joint
        for (int joint = 0; joint < jointArticulationBodies.Length; joint++)
        {
            var joint1XDrive = jointArticulationBodies[joint].xDrive;
            joint1XDrive.target = jointAngles[joint];
            jointArticulationBodies[joint].xDrive = joint1XDrive;
        }
        // Wait for robot to achieve pose for all joint assignments
        await Task.Delay(jointAssingmentWaitMillis);

        sendFeedback(new MoveActionFeedback(1.0f));
        return new MoveActionResult((int)Result.SUCCESS);
    }


    private async void processGripperGoal(MoveActionGoal goal)
    {
        Debug.Log("accepting gripper goal");
        activeGoal = goal;
        var result = new MoveActionResult((int)Result.ERROR);
        try
        {
            result = await executeGripperOpen(goal.cmd.cmd_type == (int) CommandType.OPEN_GRIPPER);
        }
        finally
        {
            Debug.Log("Goal complete; publishing result");
            sendResult(result);
            activeGoal = null;
        }
    }

    private async Task<MoveActionResult> executeGripperOpen(bool open)
    {
        sendFeedback(new MoveActionFeedback(0.0));


        float leftCurrent = leftGripper.xDrive.target;
        float rightCurrent = rightGripper.xDrive.target;
        float leftTarget;
        float rightTarget;

        if (open)
        {
            leftTarget = 0.01f;
            rightTarget = -0.01f;
        } else
        {
            leftTarget = -0.01f;
            rightTarget = 0.01f;
        }

        int steps = 20;
        for (int i = 0; i < steps; i++)
        {
            var leftDrive = leftGripper.xDrive;
            var rightDrive = rightGripper.xDrive;

            leftDrive.target += (leftTarget - leftCurrent) / steps;
            rightDrive.target += (rightTarget - rightCurrent) / steps;
            leftGripper.xDrive = leftDrive;
            rightGripper.xDrive = rightDrive;


            sendFeedback(new MoveActionFeedback((double) (i + 1) / steps));
            await Task.Delay(jointAssingmentWaitMillis);

        }

        return new MoveActionResult((int)Result.SUCCESS);

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
