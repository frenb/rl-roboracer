# Build context: repo root.
# Build command (from repo root):
#   docker build -t sim_controller:latest -f sim_controller/Dockerfile .
# or simply:
#   docker compose build sim-controller
#
# Recovered from the pre-refactor Docker/pick_and_place/docker/DockerfileController;
# COPY paths updated for the new layout (rl_agent/ and ros_server/).

FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && apt-get install -y vim iputils-ping net-tools python3-pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y ros-noetic-robot-state-publisher
RUN apt-get install -y ros-noetic-moveit
RUN apt-get install -y ros-noetic-rosbridge-suite
RUN apt-get install -y ros-noetic-joy
RUN apt-get install -y ros-noetic-ros-control
RUN apt-get install -y ros-noetic-ros-controllers
RUN apt-get install -y dos2unix python3-venv
RUN apt-get reinstall -y lsb-release
RUN apt-get install apt-transport-https curl gnupg -y

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install grpcio google-api-python-client
RUN pip3 install tensorboard
RUN pip3 install -U tensorboard-plugin-profile
RUN pip3 install pymongo
RUN pip3 install numpy==1.23.1
RUN pip3 install scipy
RUN pip install --upgrade pip

ENV PYTHON_WORKSPACE=/python_ws
ENV ROS_WORKSPACE=/catkin_ws

COPY ./docker/ros_server/python_ws $PYTHON_WORKSPACE

COPY ./docker/ros_server/ROS/src/moveit_msgs              $ROS_WORKSPACE/src/moveit_msgs
COPY ./docker/ros_server/ROS/src/niryo_moveit             $ROS_WORKSPACE/src/niryo_moveit
COPY ./docker/ros_server/ROS/src/niryo_one_ros            $ROS_WORKSPACE/src/niryo_one_ros
COPY ./docker/ros_server/ROS/src/niryo_one_urdf           $ROS_WORKSPACE/src/niryo_one_urdf
COPY ./docker/ros_server/ROS/src/ros_tcp_endpoint         $ROS_WORKSPACE/src/ros_tcp_endpoint
COPY ./docker/ros_server/ROS/src/virtual_endpoint         $ROS_WORKSPACE/src/virtual_endpoint
COPY ./docker/ros_server/ROS/src/rospy_message_converter  $ROS_WORKSPACE/src/rospy_message_converter
COPY ./docker/ros_server/ROS/src/start.sh                 $ROS_WORKSPACE/src/start.sh
COPY ./docker/ros_server/ROS/src/ros_log_tail_server.py   $ROS_WORKSPACE/src/ros_log_tail_server.py
COPY ./docker/ros_server/ROS/src/python_workspace_server.py $ROS_WORKSPACE/src/python_workspace_server.py
COPY ./docker/ros_server/python_ws                        $PYTHON_WORKSPACE

COPY ./rl_agent/robotaxi.py $PYTHON_WORKSPACE/src/robotaxi.py
COPY ./rl_agent/api.py      $PYTHON_WORKSPACE/src/api.py

COPY ./docker/ros_server/set-up-workspace /setup.sh
COPY ./docker/ros_server/tutorial /

RUN /bin/bash -c "find $ROS_WORKSPACE -type f -print0 | xargs -0 dos2unix"
RUN dos2unix /tutorial && dos2unix /setup.sh && chmod +x /setup.sh && /setup.sh && rm /setup.sh
RUN /bin/bash -c "cd /python_ws && pip install -r requirements.txt"

RUN ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11
RUN apt-get install -y libcudnn8
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib:/usr/local/cuda/lib64
