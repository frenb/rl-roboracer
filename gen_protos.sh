#!/bin/bash

# Docker image.
python -m grpc_tools.protoc -Iprotos --python_out=Docker/pick_and_place/ROS/src/virtual_endpoint/src/ --grpc_python_out=Docker/pick_and_place/ROS/src/virtual_endpoint/src/ virtual_endpoint/proto/ros_service.proto

# Web Frontend
protoc-gen-grpc --proto_path=protos --js_out=import_style=commonjs,binary:./Frontend/proto --grpc_out=./Frontend/proto virtual_endpoint/proto/ros_service.proto
