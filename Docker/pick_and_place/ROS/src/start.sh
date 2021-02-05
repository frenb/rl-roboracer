#!/bin/bash

source ./devel/setup.bash
echo "ROS_IP: $(hostname -I)" > src/niryo_moveit/config/params.yaml
roslaunch niryo_moveit part_3.launch
