#!/usr/bin/env python

from __future__ import print_function

import rospy

import sys
import copy
import math
import moveit_commander

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from niryo_moveit.srv import PoseExecutorService, PoseExecutorServiceRequest, PoseExecutorServiceResponse
from niryo_moveit.msg import MoveActionGoal, MoveActionResult, MoveActionFeedback
from niryo_moveit.msg import SceneData, MoveCommand

from niryo_moveit.move_command_type import MoveCommandType 

# Node Globals
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
latest_joint_angles = []
move_goal_publisher = None

# Between Melodic and Noetic, the return type of plan() changed. moveit_commander has no __version__ variable, so checking the python version as a proxy
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan

def plan_trajectory(move_group, destination_pose, start_joint_angles):
    """
        Given the start angles of the robot, plan a trajectory that ends at the destination pose.
    """ 
    current_joint_state = JointState()
    current_joint_state.name = joint_names
    current_joint_state.position = start_joint_angles

    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = current_joint_state
    move_group.set_start_state(moveit_robot_state)

    move_group.set_pose_target(destination_pose)
    plan = move_group.plan()

    if not plan:
        exception_str = """
            Trajectory could not be planned for a destination of {} with starting joint angles {}.
            Please make sure target and destination are reachable by the robot.
        """.format(destination_pose, destination_pose)
        raise Exception(exception_str)

    return planCompat(plan)

def service_handler(req):
    rospy.loginfo(rospy.get_caller_id() + " I Received pose request")

    # Plan trajectory based on current joint angles & desired end-effector pose.
    group_name = "arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    angles = latest_joint_angles
    trajectory = plan_trajectory(move_group, req.pose, angles)

    # Send trajectories to robot to execute.
    goal = MoveActionGoal()
    goal.cmd = MoveCommand()
    goal.cmd.cmd_type = MoveCommandType.TRAJECTORY
    goal.cmd.trajectory = trajectory

    rospy.loginfo(rospy.get_caller_id() + " I Sending trajectory to execute: " + str(goal))
    move_goal_publisher.publish(goal)

    return PoseExecutorServiceResponse(True)

def scene_data_handler(scene_data):
    angles = []
    angles.append(math.radians(scene_data.joint_00))
    angles.append(math.radians(scene_data.joint_01))
    angles.append(math.radians(scene_data.joint_02))
    angles.append(math.radians(scene_data.joint_03))
    angles.append(math.radians(scene_data.joint_04))
    angles.append(math.radians(scene_data.joint_05))
    
    global latest_joint_angles
    latest_joint_angles = angles
    


def pose_executor_main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('pose_executor')
    
    # Block on robot's move executor service becoming available.
    global move_goal_publisher
    move_goal_publisher = rospy.Publisher('move_action/goal', MoveActionGoal)

    service = rospy.Service('pose_executor', PoseExecutorService, service_handler)
    subscriber = rospy.Subscriber('scene_data', SceneData, scene_data_handler)
    rospy.loginfo(rospy.get_caller_id() + " I Ready for pose requests")
    rospy.spin()


if __name__ == "__main__":
    pose_executor_main()