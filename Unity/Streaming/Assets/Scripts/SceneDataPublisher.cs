using RosMessageTypes.Geometry;
using RosMessageTypes.NiryoMoveit;
using UnityEngine;
using ROSGeometry;
using System.Collections;

public class SceneDataPublisher : MonoBehaviour
{
    // ROS Connector
    private ROSConnection ros;

    private string topicName = "SceneData_input";

    public GameObject niryoOne;
    public GameObject target;
    public GameObject targetPlacement;

    private int numRobotJoints = 6;

    // Articulation Bodies
    private ArticulationBody[] jointArticulationBodies;

    /// <summary>
    /// 
    /// </summary>
    void Start()
    {
        // Get ROS connection static instance
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

        StartCoroutine(DoPublish());

    }

    private IEnumerator DoPublish()
    {
        while (true)
        {
            Publish();
            yield return new WaitForSeconds(0.1f); // 10Hz
        }
    }

    private void Publish()
    {
        SceneData sceneDataMessage = new SceneData();

        sceneDataMessage.joint_00 = jointArticulationBodies[0].xDrive.target;
        sceneDataMessage.joint_01 = jointArticulationBodies[1].xDrive.target;
        sceneDataMessage.joint_02 = jointArticulationBodies[2].xDrive.target;
        sceneDataMessage.joint_03 = jointArticulationBodies[3].xDrive.target;
        sceneDataMessage.joint_04 = jointArticulationBodies[4].xDrive.target;
        sceneDataMessage.joint_05 = jointArticulationBodies[5].xDrive.target;

        // Object


        // Target
        sceneDataMessage.object_location = target.transform.position.To<FLU>();
        sceneDataMessage.target_location = targetPlacement.transform.position.To<FLU>();

        ros.Send(topicName, sceneDataMessage);
    }
}
