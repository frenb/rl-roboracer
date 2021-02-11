FROM ros:noetic-ros-base

RUN sudo apt-get update && sudo apt-get install -y vim iputils-ping net-tools python3-pip ros-noetic-robot-state-publisher ros-noetic-moveit ros-noetic-rosbridge-suite ros-noetic-joy ros-noetic-ros-control ros-noetic-ros-controllers dos2unix
RUN sudo -H pip3 install rospkg jsonpickle
RUN sudo ln -s /usr/bin/python3 /usr/bin/python
RUN sudo python -m pip install grpcio google-api-python-client

ENV ROS_WORKSPACE=/catkin_ws

# Copy each directory explicitly to avoid workspace cruft
COPY ./ROS/src/moveit_msgs $ROS_WORKSPACE/src/moveit_msgs
COPY ./ROS/src/niryo_moveit $ROS_WORKSPACE/src/niryo_moveit
COPY ./ROS/src/niryo_one_ros $ROS_WORKSPACE/src/niryo_one_ros
COPY ./ROS/src/niryo_one_urdf $ROS_WORKSPACE/src/niryo_one_urdf
COPY ./ROS/src/ros_tcp_endpoint $ROS_WORKSPACE/src/ros_tcp_endpoint
COPY ./ROS/src/virtual_endpoint $ROS_WORKSPACE/src/virtual_endpoint
COPY ./ROS/src/start.sh $ROS_WORKSPACE/src/start.sh

COPY ./docker/set-up-workspace /setup.sh
COPY docker/tutorial /

RUN /bin/bash -c "find $ROS_WORKSPACE -type f -print0 | xargs -0 dos2unix"

RUN dos2unix /tutorial && dos2unix /setup.sh && chmod +x /setup.sh && /setup.sh && rm /setup.sh

WORKDIR $ROS_WORKSPACE

ENTRYPOINT ["/tutorial"]
