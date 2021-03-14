import json
import pprint
import asyncio
import time
from grpc import aio 

from virtual_endpoint.proto import ros_service_pb2_grpc
from virtual_endpoint.proto import ros_service_pb2


class RpcClient:
    def __init__(self, addr):
        self.channel = aio.insecure_channel(addr)
        self.stub = ros_service_pb2_grpc.RosNodeStub(self.channel)

    async def Subscribe(self, topic, msg_type, on_message):
        call = self.stub.Subscribe(ros_service_pb2.SubscribeRequest(topic=topic, msg_type=msg_type))
        async for topic_message in call:
            on_message(json.loads(topic_message.data))

    async def Publish(self, topic, msg_type, data):
        req = ros_service_pb2.PublishRequest(topic=topic, msg_type=msg_type, data=json.dumps(data))
        await self.stub.Publish(req)


class RobotApi:

    def __init__(self, addr='localhost:50051'):
        self.rpc_client = RpcClient(addr)
        self.pp = pprint.PrettyPrinter(indent=4)
        self.reset_event = asyncio.Event()
        self.move_events = {}
        self.scene_data_events = {}
        self.have_scene_data = asyncio.Event()
        self.next_id = 0
        self.latest_scene_data = None

    def _next_id(self):
        ret = self.next_id
        self.next_id += 1
        return ret


    async def Initialize(self):
        # Set up subscribers
        loop = asyncio.get_event_loop()
        loop.create_task(self.rpc_client.Subscribe('scene_data', 'niryo_moveit/SceneData', self._on_scene_data))
        loop.create_task(self.rpc_client.Subscribe('sim_status', 'niryo_moveit/SimStatus', self._on_sim_status))
        loop.create_task(self.rpc_client.Subscribe('move_action/result', 'niryo_moveit/MoveActionResult', self._on_move_action_result))

    def _on_sim_status(self, sim_status):
        if (sim_status['status'] == 1):
            self.reset_event.set()

    def _on_scene_data(self, scene_data):
        self.latest_scene_data = scene_data
        self.have_scene_data.set()
        # Check if there are command waiting on this scene data
        if scene_data['last_executed_cmd_id'] in self.scene_data_events:
            self.scene_data_events[scene_data['last_executed_cmd_id']].set()


    def _on_move_action_result(self, result):
        if result['cmd_id'] in self.move_events:
            self.move_events[result['cmd_id']].set()

    async def _do_sim_command(self, command):
        await self.rpc_client.Publish('sim_command', 'niryo_moveit/SimCommand', command)


    async def DoReset(self):
        self.reset_event.clear()
        await self._do_sim_command( { 'cmd' : 0 } )
        await self.reset_event.wait()

    async def DoMove(self, action):
        cmd_id = self._next_id()
        action['cmd_id'] = cmd_id
        
        self.move_events[cmd_id] = asyncio.Event()
        self.scene_data_events[cmd_id] = asyncio.Event()
        await self.rpc_client.Publish('move_action/goal', 'niryo_moveit/MoveActionGoal', action)
        
        # Wait for command completion & newest scene data including command.
        await asyncio.wait_for(self.move_events[cmd_id].wait(), 0.2)
        await asyncio.wait_for(self.scene_data_events[cmd_id].wait(), 0.2)

        # Cleanup.
        del self.move_events[cmd_id]
        del self.scene_data_events[cmd_id]

    async def GetSceneData(self):
        if not self.latest_scene_data:
            await self.have_scene_data.wait()
        return self.latest_scene_data


