#!/bin/bash

source ./devel/setup.bash
# Bind ROS-TCP on all interfaces. hostname -I is the Docker bridge IP
# (e.g. 172.18.0.5); listening only there breaks Docker Desktop's
# published 127.0.0.1:10000 after a recreate (Unity SocketException).
# Reverse Unity connections still use UNITY_MACHINE_IP (host.docker.internal).
echo "ROS_IP: 0.0.0.0" > src/niryo_moveit/config/params.yaml

# Kill Background Jobs on exit.
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Tail server for streaming logs.
tmpfile=$(mktemp /tmp/abc-script.XXXXXX)
echo "log file at $tmpfile"
#./src/ros_log_tail_server.py 60061 $tmpfile &

# Server for executing python workspace
./src/python_workspace_server.py 60062 /python_ws/ &

# Tensorboard
#tensorboard --bind_all --logdir /tmp --reload_multifile true &

# Launch ROS
export PYTHONUNBUFFERED=1
while true
do    
    roslaunch niryo_moveit part_3.launch 2>&1 | tee $tmpfile
    echo "roslaunch exited..."
sleep 1
done