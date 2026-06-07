#!/usr/bin/env python

import rospy

from ros_tcp_endpoint import TcpServer, RosPublisher, RosSubscriber, RosService, UnityService
from niryo_moveit.msg import SceneData, CarSceneData
from niryo_moveit.msg import MoveActionGoal, MoveActionResult, MoveActionFeedback, SimCommand, SimStatus, Camera, Sphere, ApplyForce
from std_msgs.msg import String


def main():
    print("in unity_node.py")
    ros_node_name = rospy.get_param("/TCP_NODE_NAME", 'TCPServer')
    tcp_server = TcpServer(ros_node_name)
    rospy.init_node(ros_node_name, anonymous=True)

    # Start the Server Endpoint with a ROS communication objects dictionary for routing messages
    tcp_server.start({
        'scene_data': RosPublisher('scene_data', SceneData),
        'car_scene_data': RosPublisher('car_scene_data', CarSceneData),
        'move_action/goal': RosSubscriber('move_action/goal', MoveActionGoal, tcp_server),
        'move_action/result': RosPublisher('move_action/result', MoveActionResult),
        'move_action/feedback': RosPublisher('move_action/feedback', MoveActionFeedback),
        'sim_command': RosSubscriber('sim_command', SimCommand, tcp_server),
        'sim_status': RosPublisher('sim_status', SimStatus),
        'camera/overhead': RosPublisher('camera/overhead', Camera),
        # Trajectory-rollout viz: the trainer publishes a JSON payload as a
        # std_msgs/String on `policy_rollouts` (see rl_agent/rollout_viz.py).
        # This endpoint's routing table is STATIC - the embedded ROS-TCP
        # connector does not send dynamic RegisterSubscriber syscommands - so
        # a topic must be listed here as a RosSubscriber (ROS -> Unity) for
        # Unity's TrajectoryRolloutViz to ever receive it.
        'policy_rollouts': RosSubscriber('policy_rollouts', String, tcp_server),
    })
    
    rospy.spin()


if __name__ == "__main__":
    main()
