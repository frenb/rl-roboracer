import threading
import time
import grpc
import asyncio
import rospy
import concurrent
import roslib.message

from virtual_endpoint.proto import ros_service_pb2_grpc
from virtual_endpoint.proto import ros_service_pb2

from rospy_message_converter import json_message_converter

from typing import AsyncIterable, Iterable
from collections import defaultdict

class RpcSubscriber:

    def __init__(self, topic, rpc_server):
        self.topic = topic
        self.rpc_server = rpc_server
    
    def callback(self, in_data):
        json_str = json_message_converter.convert_ros_message_to_json(in_data)
        message = ros_service_pb2.TopicMessage(data=json_str)
        self.rpc_server.put_topic_message_thread_safe(self.topic, message)


class RpcServer(ros_service_pb2_grpc.RosNodeServicer):
    
    def __init__(self):
        self.topics = {}
        self.subscribers = defaultdict(list)
        # One per topic ever requested.
        self.publishers = {}
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


    def put_topic_message_thread_safe(self, topic, message):
        self.loop.call_soon_threadsafe(self.put_topic_message, topic, message)

    def put_topic_message(self, topic, message):
        self._get_topic(topic).put_nowait(message)

    def _get_topic(self, topic):
        if topic not in self.topics:
            self.topics[topic] = asyncio.Queue()
        return self.topics[topic]

    def _new_subscriber(self, topic, msg_type):
        rpc_subscriber = RpcSubscriber(topic, self)
        msg_class = roslib.message.get_message_class(msg_type)
        rospy.Subscriber(topic, msg_class, rpc_subscriber.callback)
        return rpc_subscriber

    async def _wait_for_service_non_blocking(self, service_name):
        await self.loop.run_in_executor(self.executor, rospy.wait_for_service, service_name)

    async def _call_proxy_non_blocking(self, proxy, request):
        result = await self.loop.run_in_executor(self.executor, proxy, request)
        return result

    async def Subscribe(self, request, unused_context):
        self.subscribers[request.topic].append(self._new_subscriber(request.topic, request.msg_type))
        topic_queue = self._get_topic(request.topic)
        while True:
            message = await topic_queue.get()
            yield message

    def Publish(self, request, unused_context):
        topic = request.topic

        if topic not in self.publishers:
            msg_class = roslib.message.get_service_class(request.msg_type)
            self.publishers[topic] = rospy.Publisher(topic, msg_class, queue_size=10)
        publisher = self.publishers[topic]
        
        # Convert json to message
        message = json_message_converter.convert_json_to_ros_message(request.msg_type, request.data)
        publisher.publish(message)

        return ros_service_pb2.PublishResponse()



    async def CallService(self, request, unused_context):
        await self._wait_for_service_non_blocking(request.service_name)
        service_class = roslib.message.get_service_class(request.service_type)
        proxy = rospy.ServiceProxy(
            request.service_name,
            service_class)

        request_obj = json_message_converter.convert_json_to_ros_message(request.service_type, request.request, kind='request')
        rospy.loginfo(rospy.get_caller_id() + " I Received request for pose " + str(request_obj))
        
        response_obj = await self._call_proxy_non_blocking(proxy, request_obj)
        
        response_serialized = json_message_converter.convert_ros_message_to_json(response_obj)
        return ros_service_pb2.ServiceResponse(response=response_serialized)
