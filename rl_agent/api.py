import json
import pprint
import asyncio
import time
import grpc
from grpc import aio

from virtual_endpoint.proto import ros_service_pb2_grpc
from virtual_endpoint.proto import ros_service_pb2

class RpcClient:
    # Client-side per-RPC deadlines. Without these the gRPC stub awaits
    # the response indefinitely - which historically deadlocked the
    # trainer after BC pretraining: the long no-traffic window let the
    # channel go half-open in Docker's NAT/conntrack table, and the
    # next Publish blocked forever on a TCP write to a dead pipe with
    # no asyncio.TimeoutError ever firing (because the existing
    # wait_for() guards in api.py are AROUND the Publish, not on it).
    #
    # PUBLISH_TIMEOUT_S: the Publish RPC just ships a small JSON blob
    # to ros-server's gRPC endpoint on the local Docker network -
    # normal latency is sub-millisecond, so 3s is generous and any
    # excess means the channel is wedged.
    #
    # CALL_SERVICE_TIMEOUT_S: services like niryo_moveit/PosePlanner
    # do real work (motion planning) and can legitimately take
    # multiple seconds, so the deadline is correspondingly larger.
    PUBLISH_TIMEOUT_S = 3.0
    CALL_SERVICE_TIMEOUT_S = 10.0

    def __init__(self, addr):
        # gRPC keepalive: ping the peer every 600s (10 min) when the
        # channel is idle, expect an ack within 30s. Without these
        # options the client-side HTTP/2 channel can go half-open
        # during a long no-traffic window (e.g., the 5000-step BC
        # pretraining before initial_collect_actor.run()) - and once
        # that happens, the next Publish silently blocks forever on a
        # TCP write to a dead pipe. Keepalive forces the channel to
        # detect the dead peer within ~10-11 min and transparently
        # reconnect before the next RPC.
        #
        # WHY 600s AND NOT SOMETHING TIGHTER (e.g., 60s)?
        # ----------------------------------------------
        # The interval MUST cooperate with the SERVER's
        # `grpc.http2.min_ping_interval_without_data_ms`. gRPC's
        # DEFAULT server min is 300_000 (5 min) and the server counts
        # any earlier ping as a "strike"; after `max_ping_strikes`
        # (default 2) the server kills the channel with GOAWAY
        # ENHANCE_YOUR_CALM / "too many pings". We hit this on
        # 2026-05-25 with the previous 10_000ms value - every actor
        # Subscribe stream died after ~30s.
        #
        # We DO ship a server-side fix (see
        # docker/ros_server/ROS/src/virtual_endpoint/src/virtual_endpoint/virtual.py
        # which sets server min=10s + max_strikes=0), but that
        # requires a ros-server IMAGE REBUILD to take effect since
        # virtual.py is COPYed into the image at build time, not
        # bind-mounted. Until the operator rebuilds, the server is
        # still at the gRPC default of 5 min.
        #
        # 600s is safely above the default 5 min minimum (so works
        # with stock ros-server) AND works with our tuned server
        # (which tolerates any rate). When the rebuild lands, this
        # value can be tightened to 60s for faster dead-channel
        # detection - but doing so is optional.
        #
        # permit_without_calls=1: send keepalive pings even when no
        # RPCs are in flight (default would only ping with active
        # streams). Required for our case because BC pretraining has
        # no in-flight RPCs.
        #
        # max_pings_without_data=0: disable gRPC's default cap of 2
        # consecutive pings without data. With permit_without_calls=1
        # we WILL send pings without data; without this override the
        # peer would close the channel after 2 pings, defeating the
        # purpose.
        options = [
            ('grpc.keepalive_time_ms', 600000),
            ('grpc.keepalive_timeout_ms', 30000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),
        ]
        self.channel = aio.insecure_channel(addr, options=options)
        self.stub = ros_service_pb2_grpc.RosNodeStub(self.channel)

    async def Subscribe(self, topic, msg_type, on_message):
        # Subscribe is a server-streaming RPC that runs for the
        # lifetime of the trainer (the iterator yields messages
        # forever). NO timeout - that would just kill the stream
        # prematurely.
        call = self.stub.Subscribe(ros_service_pb2.SubscribeRequest(topic=topic, msg_type=msg_type))
        async for topic_message in call:
            on_message(json.loads(topic_message.data))

    async def Publish(self, topic, msg_type, data, timeout=None):
        """Publish to a ROS topic with a client-side deadline.

        timeout=None uses PUBLISH_TIMEOUT_S. Pass a float to override
        for a specific call (e.g., known-large message). Raises
        grpc.aio.AioRpcError with code DEADLINE_EXCEEDED on timeout;
        callers in RobotApi catch this and convert to a counter +
        log + continue (mirrors the asyncio.TimeoutError sites in
        DoReset/DoApplyForce/DoMove).
        """
        req = ros_service_pb2.PublishRequest(topic=topic, msg_type=msg_type, data=json.dumps(data))
        await self.stub.Publish(
            req, timeout=self.PUBLISH_TIMEOUT_S if timeout is None else timeout)

    async def Plan(self, plan_request, timeout=None):
        """Call the niryo_moveit/PosePlanner service.

        timeout=None uses CALL_SERVICE_TIMEOUT_S; the planner is real
        compute and can take seconds, so the default is larger than
        Publish's. Override for very-large planning problems.
        """
        req = ros_service_pb2.ServiceRequest(
            service_name='pose_planner',
            service_type='niryo_moveit/PosePlanner',
            request=json.dumps(plan_request))
        res = await self.stub.CallService(
            req, timeout=self.CALL_SERVICE_TIMEOUT_S if timeout is None else timeout)
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
        # `timeouts/` namespace in TensorBoard. The five buckets map to
        # the five timeout-handling sites in this file:
        #   - reset:       DoReset, 4s wait on reset_event
        #   - apply_force: DoApplyForce, first wait on apply_force_event
        #   - scene_data:  DoApplyForce, second wait on scene_data_events[cmd_id]
        #   - move:        DoMove, either of the two wait_for()s in its
        #                  shared try/except (rare in current training)
        #   - publish:     _do_sim_command / DoMove's grpc Publish RPC
        #                  itself hits DEADLINE_EXCEEDED. This counter
        #                  ticking - especially after a long no-traffic
        #                  window like BC pretraining - means the
        #                  HTTP/2 channel to ros-server went stale and
        #                  the keepalive params on the channel are
        #                  either disabled or insufficient.
        self.reset_timeouts = 0
        self.apply_force_timeouts = 0
        self.scene_data_timeouts = 0
        self.move_timeouts = 0
        self.publish_timeouts = 0

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
            'publish_timeouts': self.publish_timeouts,
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
        # Catch the gRPC deadline from rpc_client.Publish here rather
        # than letting it propagate out through DoReset / DoApplyForce
        # and up into ParallelPyEnvironment's _worker. If we let it
        # propagate the actor subprocess would crash, take its
        # multiprocessing pipe with it, and the main trainer would
        # hang in recv() exactly the way today's bug manifests. By
        # catching here we keep the same "log + counter + continue"
        # contract as the asyncio.wait_for() blocks below: the next
        # wait_for(reset_event / apply_force_event / scene_data_events)
        # will still time out cleanly if the publish never actually
        # reached Unity, and a transient channel hiccup that resolved
        # by the time the next RPC fires costs us one warning line.
        try:
            await self.rpc_client.Publish(
                'sim_command', 'niryo_moveit/SimCommand', command)
        except aio.AioRpcError as e:
            self.publish_timeouts += 1
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print(
                    f'sim_command publish DEADLINE_EXCEEDED after '
                    f'{RpcClient.PUBLISH_TIMEOUT_S}s; channel may be '
                    f'stale (gRPC keepalive should reconnect shortly).',
                    flush=True)
            else:
                # UNAVAILABLE on a freshly-killed peer, INTERNAL on a
                # protocol error, etc. We treat all RPC failures the
                # same way here - count + log + return. Recovery is
                # transparent because gRPC reconnects automatically;
                # the next call has a fresh channel.
                print(
                    f'sim_command publish RPC error {e.code()}: '
                    f'{e.details()}',
                    flush=True)

    def DoResetBlocking(self, num_obstacles=20, corner_radius=10.0, curvature_difficulty=0.0):
        asyncio.run_coroutine_threadsafe(
            self.DoReset(num_obstacles, corner_radius, curvature_difficulty),
            self.loop).result()
    
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
    
    
    async def DoReset(self, num_obstacles=20, corner_radius=10.0, curvature_difficulty=0.0):
        #print(673236)
        self.reset_event.clear()
        # corner_radius / curvature_difficulty drive the procedural
        # TrackGenerator on the Unity side (regenerated on RESTART). The ROS
        # ApplyForce.msg declares both as float64, and the JSON->ROS converter
        # runs with check_types=True, so they MUST be floats here (an int
        # would raise TypeError in message_converter).
        force_angle = {
            'acceleration': 0.0,
            'steering_angle': 0.0,
            'num_obstacles': num_obstacles,
            'corner_radius': float(corner_radius),
            'curvature_difficulty': float(curvature_difficulty),
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
        # Same DEADLINE_EXCEEDED handling as _do_sim_command above; see
        # that method for the full rationale. Catching here keeps DoMove
        # from raising into _worker (which would crash the actor
        # subprocess) on a stuck channel; the next wait_for() in the
        # try block below times out cleanly so the trainer recovers.
        try:
            await self.rpc_client.Publish(
                'move_action/goal', 'niryo_moveit/MoveActionGoal', action)
        except aio.AioRpcError as e:
            self.publish_timeouts += 1
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print(
                    f'move_action/goal publish DEADLINE_EXCEEDED after '
                    f'{RpcClient.PUBLISH_TIMEOUT_S}s; channel may be stale.',
                    flush=True)
            else:
                print(
                    f'move_action/goal publish RPC error {e.code()}: '
                    f'{e.details()}',
                    flush=True)

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
