# Build context: ./ros_server/
# Build command (from repo root): docker buildx build -f ros_server/Dockerfile ros_server/
FROM ros:noetic-ros-base-focal

RUN sudo apt-get update && sudo apt-get install -y vim iputils-ping net-tools python3-pip ros-noetic-robot-state-publisher ros-noetic-moveit ros-noetic-rosbridge-suite ros-noetic-joy ros-noetic-ros-control ros-noetic-ros-controllers dos2unix python3-venv
RUN sudo -H pip3 install rospkg jsonpickle
RUN sudo ln -s /usr/bin/python3 /usr/bin/python
# Dependency-rot fix for the focal base image. The system pyOpenSSL (19.x)
# references legacy OpenSSL bindings (X509_V_FLAG_NOTIFY_POLICY,
# X509_V_FLAG_CB_ISSUER_CHECK, ...) that modern cryptography removed. Once
# google-api-python-client/google-auth pull cryptography>=38.0.3, the next
# pip3 call (whose vendored urllib3 imports the system pyOpenSSL) crashes with
# "module 'lib' has no attribute 'X509_V_FLAG_*'". google-auth needs
# cryptography>=38.0.3 while pyOpenSSL 19 needs cryptography<35, so pinning
# cryptography alone can't satisfy both. Fix: install a MATCHED, modern
# pyOpenSSL + cryptography pair into /usr/local (which shadows the focal system
# pyOpenSSL on sys.path). pyOpenSSL 23.3.0 requires cryptography>=41.0.5,<42,
# so 41.0.7 satisfies both it and google-auth, and 23.3.0 no longer touches the
# removed bindings. This step runs via `python -m pip` while crypto is still
# the stock 2.8 (so pip itself still works); later pip3 calls see the new pair.
RUN sudo python -m pip install grpcio google-api-python-client "cryptography==41.0.7" "pyopenssl==23.3.0"
RUN pip3 install tensorboard
RUN pip3 install pymongo
RUN pip3 install numpy --upgrade
RUN pip3 install scipy
RUN pip3 install -U tensorboard-plugin-profile
ENV ROS_WORKSPACE=/catkin_ws
ENV PYTHON_WORKSPACE=/python_ws

# Copy each directory explicitly to avoid workspace cruft
COPY ./ROS/src/moveit_msgs $ROS_WORKSPACE/src/moveit_msgs
COPY ./ROS/src/niryo_moveit $ROS_WORKSPACE/src/niryo_moveit
COPY ./ROS/src/niryo_one_ros $ROS_WORKSPACE/src/niryo_one_ros
COPY ./ROS/src/niryo_one_urdf $ROS_WORKSPACE/src/niryo_one_urdf
COPY ./ROS/src/ros_tcp_endpoint $ROS_WORKSPACE/src/ros_tcp_endpoint
COPY ./ROS/src/virtual_endpoint $ROS_WORKSPACE/src/virtual_endpoint
COPY ./ROS/src/rospy_message_converter $ROS_WORKSPACE/src/rospy_message_converter
COPY ./ROS/src/start.sh $ROS_WORKSPACE/src/start.sh
COPY ./ROS/src/ros_log_tail_server.py $ROS_WORKSPACE/src/ros_log_tail_server.py
COPY ./ROS/src/python_workspace_server.py $ROS_WORKSPACE/src/python_workspace_server.py
COPY ./python_ws $PYTHON_WORKSPACE

VOLUME $PYTHON_WORKSPACE/src

COPY ./set-up-workspace /setup.sh
COPY ./tutorial /

RUN /bin/bash -c "find $ROS_WORKSPACE -type f -print0 | xargs -0 dos2unix"

RUN dos2unix /tutorial && dos2unix /setup.sh && chmod +x /setup.sh && /setup.sh && rm /setup.sh

RUN /bin/bash -c "cd $PYTHON_WORKSPACE &&  python3 -m venv --system-site-packages ./venv && source ./venv/bin/activate && pip install -r requirements.txt && deactivate"
RUN /bin/bash -c "cd /python_ws && pip install -r requirements.txt"
WORKDIR $ROS_WORKSPACE

ENTRYPOINT ["/tutorial"]
