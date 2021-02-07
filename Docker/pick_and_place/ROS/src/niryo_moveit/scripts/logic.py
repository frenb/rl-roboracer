#!/usr/bin/env python

from __future__ import print_function

import rospy

import sys
import copy
import math
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

def callback(scene_data):
    rospy.loginfo(rospy.get_caller_id() + "I Logic Received:\n%s", scene_data)

def logic():
    rospy.loginfo(rospy.get_caller_id() + "I Logic Started")
    rospy.init_node('logic', anonymous=True)
    rospy.Subscriber("SceneData", SceneData, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == "__main__":
    logic()