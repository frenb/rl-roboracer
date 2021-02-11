// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('@grpc/grpc-js');
var virtual_endpoint_proto_ros_service_pb = require('../../virtual_endpoint/proto/ros_service_pb.js');

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
};

exports.RosNodeClient = grpc.makeGenericClientConstructor(RosNodeService);
