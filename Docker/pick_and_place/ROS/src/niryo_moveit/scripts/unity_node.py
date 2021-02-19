#!/usr/bin/env python

import rospy

from ros_tcp_endpoint import TcpServer, RosPublisher, RosSubscriber, RosService, UnityService
from niryo_moveit.msg import SceneData
from niryo_moveit.srv import MoveExecutorService 


def main():
    ros_node_name = rospy.get_param("/TCP_NODE_NAME", 'TCPServer')
    tcp_server = TcpServer(ros_node_name)
    rospy.init_node(ros_node_name, anonymous=True)

    # Start the Server Endpoint with a ROS communication objects dictionary for routing messages
    tcp_server.start({
        'SceneData_input': RosPublisher('scene_data', SceneData),
        'move_executor_service': UnityService('move_executor', MoveExecutorService, tcp_server)
    })
    
    rospy.spin()


if __name__ == "__main__":
    main()
