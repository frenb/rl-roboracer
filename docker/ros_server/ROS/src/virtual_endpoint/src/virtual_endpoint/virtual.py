import threading
import time
import grpc
import asyncio

from virtual_endpoint.proto import ros_service_pb2_grpc

from .server import RpcServer

class VirtualNode:

    async def _start_rpc_server_and_wait(self):
        server = grpc.aio.server()
        ros_service_pb2_grpc.add_RosNodeServicer_to_server(RpcServer(), server)
        # TODO: dynamic port selection.
        server.add_insecure_port('[::]:50051')
        await server.start()
        await server.wait_for_termination()

    async def main(self):
        await self._start_rpc_server_and_wait()