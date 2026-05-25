import threading
import time
import grpc
import asyncio

from virtual_endpoint.proto import ros_service_pb2_grpc

from .server import RpcServer

# gRPC server keepalive tolerance. Must be co-tuned with the trainer-
# side options in rl_agent/api.py::RpcClient.__init__; keep this
# server min STRICTLY less than the client's keepalive_time_ms or the
# server will GOAWAY the channel with ENHANCE_YOUR_CALM (we observed
# this on 2026-05-25 when the client pinged every 10s while this
# server defaulted to a 5-minute minimum, causing every actor's
# Subscribe stream to die after ~30s).
#
# Settings rationale:
#   * min_ping_interval_without_data_ms = 10_000: accept client pings
#     as fast as every 10s. Our client pings at 60s so this leaves
#     a comfortable 6x margin.
#   * max_ping_strikes = 0: don't count strikes at all; once we trust
#     the interval, we trust it. Belt-and-suspenders for clients with
#     slight clock jitter.
#   * keepalive_permit_without_calls = 1: also let the SERVER ping
#     the client when streams are idle but the channel is alive (used
#     for long-running Subscribe streams that don't send data for
#     extended periods).
_GRPC_SERVER_OPTIONS = [
    ('grpc.http2.min_ping_interval_without_data_ms', 10_000),
    ('grpc.http2.max_ping_strikes', 0),
    ('grpc.keepalive_permit_without_calls', 1),
]


class VirtualNode:

    async def _start_rpc_server_and_wait(self):
        server = grpc.aio.server(options=_GRPC_SERVER_OPTIONS)
        ros_service_pb2_grpc.add_RosNodeServicer_to_server(RpcServer(), server)
        # TODO: dynamic port selection.
        server.add_insecure_port('[::]:50051')
        await server.start()
        await server.wait_for_termination()

    async def main(self):
        await self._start_rpc_server_and_wait()