#!/bin/bash

source ./devel/setup.bash
# Bind ROS-TCP on all interfaces. hostname -I is the Docker bridge IP
# (e.g. 172.18.0.5); listening only there breaks Docker Desktop's
# published 127.0.0.1:10000 after a recreate (Unity SocketException).
# Reverse Unity connections still use UNITY_MACHINE_IP (host.docker.internal).
echo "ROS_IP: 0.0.0.0" > src/niryo_moveit/config/params.yaml

# Launch ROS
export PYTHONUNBUFFERED=1
while true
do    
    roslaunch niryo_moveit part_3.launch 2>&1
    echo "roslaunch exited..."
sleep 1
done
