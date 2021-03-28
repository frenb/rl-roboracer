#! /usr/bin/python
# Local server that can run the code in the /python_ws/src directory on behalf of the client.

import sys
import os
import asyncio
import socket
import json
import signal
import traceback

def kill(proc):
    try:
        if proc:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        traceback.print_exc()

def check_task_exception(task):
    try:
        if task and task.exception():
            raise task.exception()
    except asyncio.InvalidStateError:
        pass


async def send_output(proc, client):
    async for line in proc.stdout:
        await loop.sock_sendall(client, line)

async def handle_client(client, python_ws):
    loop = asyncio.get_running_loop()
    proc = None
    send_task = None
    try:
        while True:
            check_task_exception(send_task)
            data = await loop.sock_recv(client, 10000)
            command = json.loads(data.decode('utf-8'))
            print('client command: ' + json.dumps(command))
            if command['cmd'] == 'run':
                kill(proc)
                src = command['src']
                shell_str = '/bin/bash -c "export PYTHONUNBUFFERED=1 && cd %s && source ./venv/bin/activate && cd src && python %s 2>&1"' % (python_ws, src)
                proc = await asyncio.create_subprocess_shell(
                    shell_str,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    preexec_fn=os.setsid)

                send_task = loop.create_task(send_output(proc, client))
            elif command['cmd'] == 'stop':
                kill(proc)
                proc = None
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        traceback.print_exc()

    finally:
        kill(proc)

    

async def run_server(server, python_ws):
    loop = asyncio.get_running_loop()
    running_task = None
    while True:
        client, _ = await loop.sock_accept(server)
        # Only one instance at a time.
        if running_task:
            running_task.cancel()
        running_task = loop.create_task(handle_client(client, python_ws))


if __name__ == '__main__':
    port = int(sys.argv[1])
    python_ws = sys.argv[2]
    addr = ('', port)
    
    if socket.has_dualstack_ipv6():
        server = socket.create_server(addr, family=socket.AF_INET6, dualstack_ipv6=True)
    else:
        server = socket.create_server(addr)
    
    server.listen()
    server.setblocking(False)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_server(server, python_ws))

    