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

    async def Plan(self, plan_request):
        req = ros_service_pb2.ServiceRequest(
            service_name='pose_planner',
            service_type='niryo_moveit/PosePlanner',
            request=json.dumps(plan_request))
        res = await self.stub.CallService(req)
        return json.loads(res.response)

class RobotApi:

    def __init__(self, addr='ros-server:50051'):
        self.loop = asyncio.get_event_loop()
        self.rpc_client = RpcClient(addr)
        self.pp = pprint.PrettyPrinter(indent=4)
        self.reset_event = asyncio.Event()
        self.apply_force_event = asyncio.Event()
        self.apply_force_events = {}
        self.has_reached_goal = False
        self.move_events = {}
        self.scene_data_events = {}
        self.have_scene_data = asyncio.Event()
        self.have_car_scene_data = asyncio.Event()
        self.next_id = 0
        self.latest_scene_data = None
        self.latest_car_scene_data = None
        self.latest_overhead_camera_frame = None
        self.have_overhead_camera_frame = asyncio.Event()

        # Timeout counters. Incremented in each `except asyncio.TimeoutError`
        # branch below. Exposed via get_timeout_counts() so the trainer
        # in robotaxi.py can aggregate across all ParallelPyEnvironment
        # workers and write the totals as tf.summary scalars under the
        # `timeouts/` namespace in TensorBoard. The four buckets map to
        # the four wait_for() sites in this file:
        #   - reset:       DoReset, 4s wait on reset_event
        #   - apply_force: DoApplyForce, first wait on apply_force_event
        #   - scene_data:  DoApplyForce, second wait on scene_data_events[cmd_id]
        #   - move:        DoMove, either of the two wait_for()s in its
        #                  shared try/except (rare in current training)
        self.reset_timeouts = 0
        self.apply_force_timeouts = 0
        self.scene_data_timeouts = 0
        self.move_timeouts = 0

    def get_timeout_counts(self):
        """Snapshot the running tally of asyncio.TimeoutError occurrences.

        Returns a plain dict so it serialises cleanly across
        ParallelPyEnvironment's RPC layer when the trainer pulls
        per-actor counts via env.call('get_timeout_counts').
        Counts are cumulative since this RobotApi was constructed
        (one instance per env, so one per actor in multi-env mode).
        """
        return {
            'reset_timeouts': self.reset_timeouts,
            'apply_force_timeouts': self.apply_force_timeouts,
            'scene_data_timeouts': self.scene_data_timeouts,
            'move_timeouts': self.move_timeouts,
        }

    def _next_id(self):
        ret = self.next_id
        self.next_id += 1
        return ret

    async def Initialize(self):
        # Set up subscribers
        self.loop.create_task(self.rpc_client.Subscribe('scene_data', 'niryo_moveit/SceneData', self._on_scene_data))
        self.loop.create_task(self.rpc_client.Subscribe('car_scene_data', 'niryo_moveit/CarSceneData', self._on_car_scene_data))
        self.loop.create_task(self.rpc_client.Subscribe('sim_status', 'niryo_moveit/SimStatus', self._on_sim_status))
        self.loop.create_task(self.rpc_client.Subscribe('move_action/result', 'niryo_moveit/MoveActionResult', self._on_move_action_result))
        self.loop.create_task(self.rpc_client.Subscribe('camera/overhead', 'niryo_moveit/Camera', self._on_overhead_camera_frame))

    def _on_sim_status(self, sim_status):
        #print("sim_status: " + str(sim_status))
        if (sim_status['status'] == 1):
            self.reset_event.set()
        if (sim_status['status'] == 2):
             self.apply_force_event.set()

    def _on_scene_data(self, scene_data):
        #print("in on_scene_data")
        self.latest_scene_data = scene_data
        self.have_scene_data.set()
        # Check if there are command waiting on this scene data
        if scene_data['last_executed_cmd_id'] in self.scene_data_events:
            self.scene_data_events[scene_data['last_executed_cmd_id']].set()
    
    def _on_car_scene_data(self, car_scene_data):
        #print("in on_car_scene_data")
        self.latest_car_scene_data = car_scene_data
        self.have_car_scene_data.set()
        #print(f"car_secene_data: {car_scene_data}")
        # Check if there are command waiting on this scene data
        if car_scene_data["car"]['has_reached_goal']:
            self.has_reached_goal = True
            print("has_reached_goal: " + str(car_scene_data["car"]['has_reached_goal']))
        #print("car_scene_data['last_executed_cmd_id']: " + str(car_scene_data['last_executed_cmd_id']))
        #print("self.scene_data_events: " + str(self.scene_data_events))
        if car_scene_data['last_executed_cmd_id'] in self.scene_data_events:
            self.scene_data_events[car_scene_data['last_executed_cmd_id']].set()
            #print("in _on_car_scene_data for " + str(car_scene_data['last_executed_cmd_id']))

    def _on_overhead_camera_frame(self, frame):
        self.latest_overhead_camera_frame = frame
        self.have_overhead_camera_frame.set()


    def _on_move_action_result(self, result):
        if result['cmd_id'] in self.move_events:
            self.move_events[result['cmd_id']].set()
    
    
    async def _do_sim_command(self, command):
        #print(command)
        await self.rpc_client.Publish('sim_command', 'niryo_moveit/SimCommand', command)

    def DoResetBlocking(self, num_obstacles=20):
        asyncio.run_coroutine_threadsafe(self.DoReset(num_obstacles), self.loop).result()
    
    def DoApplyForceBlocking(self, acceleration=100.0, steering_angle=30.0, num_obstacles=20):
        result = asyncio.run_coroutine_threadsafe(
            self.DoApplyForce(acceleration, steering_angle, num_obstacles),
            self.loop
        ).result()
        # print("++++++++++++++++++++++")
        # print(result)
        # print("++++++++++++++++++++++")
        return result

    def DoMoveBlocking(self, action):
        return asyncio.run_coroutine_threadsafe(self.DoMove(action), self.loop).result()
        
    def GetSceneDataBlocking(self):
        return asyncio.run_coroutine_threadsafe(self.GetSceneData(), self.loop).result()
    
    def GetCarSceneDataBlocking(self):
        return asyncio.run_coroutine_threadsafe(self.GetCarSceneData(), self.loop).result()
    
    
    async def DoReset(self, num_obstacles=20):
        #print(673236)
        self.reset_event.clear()
        force_angle = {
            'acceleration': 0.0,
            'steering_angle': 0.0,
            'num_obstacles': num_obstacles
        }
        await self._do_sim_command( { 'cmd' : 0 , 'ApplyForce': force_angle} )
        try:
            await asyncio.wait_for(self.reset_event.wait(), 4)
        except asyncio.TimeoutError:
            self.reset_timeouts += 1
            print('timed out waiting for reset. Ignoring')
    
    async def DoApplyForce(self, acceleration=100.0, steering_angle=30.0, num_obstacles=20):
        # print("DoApplyForce: " + str(num_obstacles))
        cmd_id = self._next_id()
        force_angle = {
            'acceleration': acceleration,
            'steering_angle': steering_angle,
            'cmd_id': cmd_id,
            'num_obstacles': num_obstacles
        }
        self.apply_force_event.clear()
        self.scene_data_events[cmd_id] = asyncio.Event()
        await self._do_sim_command( { 'cmd' : 1, 'ApplyForce': force_angle } )
        # Timeout bumped 3s -> 8s. The apply_force_event wait is load-
        # bearing back-pressure (gates each DoApplyForce on Unity
        # reaching the "force applied" state of its loop), so we keep
        # it - but at 4-actor GPU contention the global sim_status==2
        # ack frequently arrives 3-7s after publish, generating thousands
        # of spurious "Apply force timed out waiting. Ignoring" prints
        # per run while the next wait (scene_data_events[cmd_id], 5s)
        # still completes successfully. Bumping to 8s captures those
        # late acks within the wait window so the print only fires on
        # genuinely stalled steps.
        try:
            await asyncio.wait_for(self.apply_force_event.wait(), 8)
        except asyncio.TimeoutError:
            self.apply_force_timeouts += 1
            print('Apply force timed out waiting. Ignoring')

        try:
            await asyncio.wait_for(self.scene_data_events[cmd_id].wait(), 5)
            #print("after wait")
        except asyncio.TimeoutError:
            self.scene_data_timeouts += 1
            print('Scene data events timed out waiting. Ignoring')
        
        del self.scene_data_events[cmd_id]
        return self.latest_car_scene_data

    async def DoMove(self, action, timeout=0.2):
        cmd_id = self._next_id()
        action['cmd_id'] = cmd_id
        
        self.move_events[cmd_id] = asyncio.Event()
        self.scene_data_events[cmd_id] = asyncio.Event()
        await self.rpc_client.Publish('move_action/goal', 'niryo_moveit/MoveActionGoal', action)
        
        # Wait for command completion & newest scene data including command.
        try:
            await asyncio.wait_for(self.move_events[cmd_id].wait(), timeout)
            await asyncio.wait_for(self.scene_data_events[cmd_id].wait(), timeout)
            
        except asyncio.TimeoutError:
            self.move_timeouts += 1
            print('timed out waiting for move. Ignoring')


        # Cleanup.
        del self.move_events[cmd_id]
        del self.scene_data_events[cmd_id]
    
    # async def DoMove(self, action, timeout=0.2):
    #     cmd_id = self._next_id()
    #     action['cmd_id'] = cmd_id
    #     print("in DoMove")
    #     print(action)
    #     await self._do_sim_command( { 'cmd' : 1 , 'ApplyForce': action.apply_force} )
    #     try:
    #         await asyncio.wait_for(self.apply_force_event.wait(), 2)
    #         print("we did it")
    #         print(self.latest_car_scene_data)
    #     except asyncio.TimeoutError:
    #         print('timed out waiting for applyforce. Ignoring')

    async def DoTrajectory(self, trajectory):
        action = {'cmd': {
            'cmd_type': 1,
            'trajectory': trajectory
        }}
        await self.DoMove(action, 10)

    async def DoOpenGripper(self):
        action = {'cmd': {
            'cmd_type': 2
        }}
        await self.DoMove(action, 10)

    async def DoCloseGripper(self):
        action = {'cmd': {
            'cmd_type': 3
        }}
        await self.DoMove(action, 10)

    async def GetPlan(self, pose):
        scene_data = await self.GetSceneData()
        plan_request = {
            'joint_00': scene_data['joint_00'],
            'joint_01': scene_data['joint_01'],
            'joint_02': scene_data['joint_02'],
            'joint_03': scene_data['joint_03'],
            'joint_04': scene_data['joint_04'],
            'joint_05': scene_data['joint_05'],
            'pose': pose 
        }
        plan = await self.rpc_client.Plan(plan_request)
        return plan

    async def GetSceneData(self):
        if not self.latest_scene_data:
            await self.have_scene_data.wait()
        return self.latest_scene_data
    
    async def GetCarSceneData(self):
        if not self.latest_car_scene_data:
            await self.have_car_scene_data.wait()
        return self.latest_car_scene_data

    async def GetOverheadCameraFrame(self):
        if not self.latest_overhead_camera_frame:
            await self.have_overhead_camera_frame.wait()
            self.have_overhead_camera_frame.clear()
        res = self.latest_overhead_camera_frame
        return res
