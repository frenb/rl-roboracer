#!/usr/bin/env python

from __future__ import print_function

import rospy

import sys
import copy
import math
import asyncio
import moveit_commander

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from niryo_moveit.msg import SceneData
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from niryo_moveit.srv import MoverService, MoverServiceRequest, MoverServiceResponse
from virtual_endpoint import VirtualNode

def remote_logic():
    rospy.loginfo(rospy.get_caller_id() + "I Logic Started")
    rospy.init_node('remote_logic', anonymous=True)
    v_node = VirtualNode()
    v_node.register_msg_type('niryo_moveit/SceneData', SceneData)
    asyncio.run(v_node.main())

if __name__ == "__main__":
    remote_logic()