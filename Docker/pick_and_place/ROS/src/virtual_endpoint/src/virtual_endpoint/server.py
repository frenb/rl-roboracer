import threading
import time
import grpc
import asyncio
import rospy

from virtual_endpoint.proto import ros_service_pb2_grpc
from virtual_endpoint.proto import ros_service_pb2

from typing import AsyncIterable, Iterable
from collections import defaultdict

class RpcSubscriber:

    def __init__(self, topic, rpc_server):
        self.topic = topic
        self.rpc_server = rpc_server
    
    def callback(self, in_data):
        # TODO: testing only - generically serialize.
        message = ros_service_pb2.TopicMessage(data="joint_00 = " + str(in_data.joint_00))
        self.rpc_server.put_topic_message_thread_safe(self.topic, message)


class RpcServer(ros_service_pb2_grpc.RosNodeServicer):
    
    def __init__(self, msg_types):
        self.topics = {}
        self.subscribers = defaultdict(list)
        self.loop = asyncio.get_running_loop()
        self.msg_types = msg_types

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
        rospy.Subscriber(topic, self.msg_types[msg_type], rpc_subscriber.callback)
        return rpc_subscriber

    async def Subscribe(self, request, unused_context):
        self.subscribers[request.topic].append(self._new_subscriber(request.topic, request.msg_type))
        topic_queue = self._get_topic(request.topic)
        while True:
            message = await topic_queue.get()
            yield message