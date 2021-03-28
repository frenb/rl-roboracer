using RosMessageTypes.NiryoMoveit;
using UnityEngine;
using ROSGeometry;
using System.Collections;

using Transform = UnityEngine.Transform;
using Quaternion = UnityEngine.Quaternion;

public class SceneDataPublisher : MonoBehaviour, IRosComponent
{
    // ROS Connector
    private ROSConnection ros;

    private string topicName = "scene_data";

    private GameObject niryoOne;
    private GameObject target;
    private GameObject targetPlacement;
    private GameObject pole;

    private int numRobotJoints = 6;

    // Articulation Bodies
    private ArticulationBody[] jointArticulationBodies;

    private Transform gripperBase;

    /// <summary>
    /// 
    /// </summary>
    void Start()
    {
        // Get ROS connection static instance
        ros = ROSConnection.instance;

        UpdateWorldRefs();

        StartCoroutine(DoPublish());
    }


    public void UpdateWorldRefs()
    {
        // Targets
        target = SimController.instance.target;
        targetPlacement = SimController.instance.targetPlacement;

        // Robot
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

        string gripper_base = hand_link + "/tool_link/gripper_base/Collisions/unnamed";
        gripperBase = niryoOne.transform.Find(gripper_base);

        // Pole
        if (SimController.instance.poleCart != null)
        {
            pole = SimController.instance.poleCart.transform.Find("Pole").gameObject;
        }

        // So one fresh scene_data before reset done signal sent.
        Publish();
    }

    private IEnumerator DoPublish()
    {
        while (true)
        {
            Publish();
            yield return new WaitForSeconds(0.01f); // 50Hz
        }
    }

    private void Publish()
    {
        SceneData sceneDataMessage = new SceneData();

        sceneDataMessage.joint_00 = Mathf.Deg2Rad * jointArticulationBodies[0].xDrive.target;
        sceneDataMessage.joint_01 = Mathf.Deg2Rad * jointArticulationBodies[1].xDrive.target;
        sceneDataMessage.joint_02 = Mathf.Deg2Rad * jointArticulationBodies[2].xDrive.target;
        sceneDataMessage.joint_03 = Mathf.Deg2Rad * jointArticulationBodies[3].xDrive.target;
        sceneDataMessage.joint_04 = Mathf.Deg2Rad * jointArticulationBodies[4].xDrive.target;
        sceneDataMessage.joint_05 = Mathf.Deg2Rad * jointArticulationBodies[5].xDrive.target;

        // Object & Target
        if (target != null)
        {
            sceneDataMessage.object_location = target.transform.position.To<FLU>();
        }
        if (targetPlacement != null)
        {
            sceneDataMessage.target_location = targetPlacement.transform.position.To<FLU>();
        }

        // Effector Pose.
        sceneDataMessage.effector_pose.position = gripperBase.transform.position.To<FLU>();

        // TODO: orientation of gripperBase object in unity needs to be rotated to match
        // axis used by ROS. Don't totally understand the conversion, but this rotation seems
        // to work.
        var gribber_rotation = gripperBase.transform.rotation;
        gribber_rotation *= new Quaternion(0.5f, -0.5f, 0.5f, 0.5f);
        sceneDataMessage.effector_pose.orientation = gribber_rotation.To<FLU>();

        // Last command executed in this scene.
        sceneDataMessage.last_executed_cmd_id = MoveService.instance.lastExecutedCommand;

        // Pole position & orientation.
        if (pole != null) {

            // Calculate the effector speed.
            var tangent = jointArticulationBodies[5].transform.forward;
            var handSpeed = Vector3.Dot(jointArticulationBodies[5].velocity, tangent);

            // Calculate angle between effector normal & pole.
            var handRedAxis = jointArticulationBodies[5].transform.right;
            var handBlueAxis = jointArticulationBodies[5].transform.forward;
            var handGreenAxis = jointArticulationBodies[5].transform.up;
            var poleGreenAxis = pole.transform.up;


            var deltaOnHandBlueAxis = poleGreenAxis - Vector3.Dot(poleGreenAxis, handGreenAxis) * handGreenAxis;
            var poleToHandAngle = Vector3.Angle(-handRedAxis, deltaOnHandBlueAxis);
            Debug.DrawRay(pole.transform.position, deltaOnHandBlueAxis, Color.red);

            var deltaOnHandGreenAxis = poleGreenAxis - Vector3.Dot(poleGreenAxis, handBlueAxis) * handBlueAxis;
            var poleToHandAngleB = Vector3.Angle(-handRedAxis, deltaOnHandGreenAxis);
            Debug.DrawRay(pole.transform.position, deltaOnHandGreenAxis, Color.blue);


            // Calculate angular speed of pole normal to the circle.
            var poleAngularSpeed = Vector3.Dot(pole.GetComponent<Rigidbody>().angularVelocity, pole.transform.forward);
            var poleAngularSpeedB = Vector3.Dot(pole.GetComponent<Rigidbody>().angularVelocity, pole.transform.right);

            sceneDataMessage.pole_cart.pole_hand_angle = poleToHandAngle;
            sceneDataMessage.pole_cart.pole_hand_angle_b = poleToHandAngleB;
            sceneDataMessage.pole_cart.hand_tangent_speed = handSpeed;
            sceneDataMessage.pole_cart.pole_angular_speed = poleAngularSpeed;
            sceneDataMessage.pole_cart.pole_angular_speed_b = poleAngularSpeedB;
            sceneDataMessage.pole_cart.upright = (pole.transform.position.y > (jointArticulationBodies[5].transform.position.y) * 0.7)
                && poleToHandAngle < 40 && poleToHandAngleB < 40;
            //sceneDataMessage.pole_cart.upright = pole.GetComponent<PoleController>().isUpright;

            //Debug.Log(string.Format("G_CHECK poleToHandAngle = {0:f3}, poleToHandAngleB = {1:f3}, poleAngularSpeed: {2:f3}, poleAngularSpeedB: {3:f3}",
            //    poleToHandAngle, poleToHandAngleB, poleAngularSpeed, poleAngularSpeedB));
        }


        ros.Send(topicName, sceneDataMessage);
    }
}
