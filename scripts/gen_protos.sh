#!/bin/bash
# Run from repo root: ./scripts/gen_protos.sh

# ROS server (gRPC virtual endpoint)
python -m grpc_tools.protoc -Iprotos \
    --python_out=ros_server/ROS/src/virtual_endpoint/src/ \
    --grpc_python_out=ros_server/ROS/src/virtual_endpoint/src/ \
    virtual_endpoint/proto/ros_service.proto

# RL agent (Python client)
python -m grpc_tools.protoc -Iprotos \
    --python_out=rl_agent/ \
    --grpc_python_out=rl_agent/ \
    virtual_endpoint/proto/ros_service.proto
