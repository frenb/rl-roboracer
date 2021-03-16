#!/bin/bash

source ./devel/setup.bash
echo "ROS_IP: $(hostname -I)" > src/niryo_moveit/config/params.yaml

# Kill Background Jobs on exit.
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Tail server for streaming logs.
tmpfile=$(mktemp /tmp/abc-script.XXXXXX)
echo "log file at $tmpfile"
./src/ros_log_tail_server.py 60061 $tmpfile &

# Server for executing python workspace
./src/python_workspace_server.py 60062 /python_ws/ &

# Launch ROS
export PYTHONUNBUFFERED=1
roslaunch niryo_moveit part_3.launch 2>&1 | tee $tmpfile
