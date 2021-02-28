#! /usr/bin/python
# Local server that serves log file contents to clients.
# TODO: In production, we would not use this and instead configure a docker logging driver.
# to send to syslog server or similar.

import sys
import os
import asyncio
import socket

async def handle_client(client, log_path):
    loop = asyncio.get_running_loop()
    proc = await asyncio.create_subprocess_shell(
        'tail -n 1000 -F ' + log_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE) 
    
    async for line in proc.stdout:
        await loop.sock_sendall(client, line)
    

async def run_server(server, log_path):
    loop = asyncio.get_running_loop()
    while True:
        client, _ = await loop.sock_accept(server)
        loop.create_task(handle_client(client, log_path))


if __name__ == '__main__':
    port = int(sys.argv[1])
    log_path = sys.argv[2]
    addr = ('', port)
    
    if socket.has_dualstack_ipv6():
        server = socket.create_server(addr, family=socket.AF_INET6, dualstack_ipv6=True)
    else:
        server = socket.create_server(addr)
    
    server.listen()
    server.setblocking(False)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_server(server, log_path))

    