// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('@grpc/grpc-js');
var virtual_endpoint_proto_ros_service_pb = require('../../virtual_endpoint/proto/ros_service_pb.js');

function serialize_virtual_endpoint_PublishRequest(arg) {
  if (!(arg instanceof virtual_endpoint_proto_ros_service_pb.PublishRequest)) {
    throw new Error('Expected argument of type virtual_endpoint.PublishRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_virtual_endpoint_PublishRequest(buffer_arg) {
  return virtual_endpoint_proto_ros_service_pb.PublishRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_virtual_endpoint_PublishResponse(arg) {
  if (!(arg instanceof virtual_endpoint_proto_ros_service_pb.PublishResponse)) {
    throw new Error('Expected argument of type virtual_endpoint.PublishResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_virtual_endpoint_PublishResponse(buffer_arg) {
  return virtual_endpoint_proto_ros_service_pb.PublishResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_virtual_endpoint_ServiceRequest(arg) {
  if (!(arg instanceof virtual_endpoint_proto_ros_service_pb.ServiceRequest)) {
    throw new Error('Expected argument of type virtual_endpoint.ServiceRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_virtual_endpoint_ServiceRequest(buffer_arg) {
  return virtual_endpoint_proto_ros_service_pb.ServiceRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_virtual_endpoint_ServiceResponse(arg) {
  if (!(arg instanceof virtual_endpoint_proto_ros_service_pb.ServiceResponse)) {
    throw new Error('Expected argument of type virtual_endpoint.ServiceResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_virtual_endpoint_ServiceResponse(buffer_arg) {
  return virtual_endpoint_proto_ros_service_pb.ServiceResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_virtual_endpoint_SubscribeRequest(arg) {
  if (!(arg instanceof virtual_endpoint_proto_ros_service_pb.SubscribeRequest)) {
    throw new Error('Expected argument of type virtual_endpoint.SubscribeRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_virtual_endpoint_SubscribeRequest(buffer_arg) {
  return virtual_endpoint_proto_ros_service_pb.SubscribeRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_virtual_endpoint_TopicMessage(arg) {
  if (!(arg instanceof virtual_endpoint_proto_ros_service_pb.TopicMessage)) {
    throw new Error('Expected argument of type virtual_endpoint.TopicMessage');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_virtual_endpoint_TopicMessage(buffer_arg) {
  return virtual_endpoint_proto_ros_service_pb.TopicMessage.deserializeBinary(new Uint8Array(buffer_arg));
}


var RosNodeService = exports.RosNodeService = {
  // Subscribe to a ROS topic.
  subscribe: {
    path: '/virtual_endpoint.RosNode/Subscribe',
    requestStream: false,
    responseStream: true,
    requestType: virtual_endpoint_proto_ros_service_pb.SubscribeRequest,
    responseType: virtual_endpoint_proto_ros_service_pb.TopicMessage,
    requestSerialize: serialize_virtual_endpoint_SubscribeRequest,
    requestDeserialize: deserialize_virtual_endpoint_SubscribeRequest,
    responseSerialize: serialize_virtual_endpoint_TopicMessage,
    responseDeserialize: deserialize_virtual_endpoint_TopicMessage,
  },
  // Publish a message to a ROS topic.
  publish: {
    path: '/virtual_endpoint.RosNode/Publish',
    requestStream: false,
    responseStream: false,
    requestType: virtual_endpoint_proto_ros_service_pb.PublishRequest,
    responseType: virtual_endpoint_proto_ros_service_pb.PublishResponse,
    requestSerialize: serialize_virtual_endpoint_PublishRequest,
    requestDeserialize: deserialize_virtual_endpoint_PublishRequest,
    responseSerialize: serialize_virtual_endpoint_PublishResponse,
    responseDeserialize: deserialize_virtual_endpoint_PublishResponse,
  },
  // Call a ROS service.
  callService: {
    path: '/virtual_endpoint.RosNode/CallService',
    requestStream: false,
    responseStream: false,
    requestType: virtual_endpoint_proto_ros_service_pb.ServiceRequest,
    responseType: virtual_endpoint_proto_ros_service_pb.ServiceResponse,
    requestSerialize: serialize_virtual_endpoint_ServiceRequest,
    requestDeserialize: deserialize_virtual_endpoint_ServiceRequest,
    responseSerialize: serialize_virtual_endpoint_ServiceResponse,
    responseDeserialize: deserialize_virtual_endpoint_ServiceResponse,
  },
};

exports.RosNodeClient = grpc.makeGenericClientConstructor(RosNodeService);
