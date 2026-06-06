"""Debug tool: subscribe to the `policy_rollouts` topic and print what the
trainer is publishing for the trajectory-rollout visualization.

Run inside the sim-controller container (it has the gRPC stubs + network
access to the ros-servers):

    docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
        sh -c "cd /python_ws/src && python check_rollouts.py"

Requires the trainer to be running WITH ROLLOUT_VIZ_ENABLED=1 and a job
actively training (rollouts only publish from the collect/eval loop). Listens
for ~30s, prints each message's size + a JSON preview, then exits.

Optional arg: ros-server address (default ros-server-0:50051, where actor 0
publishes).
"""
import asyncio
import sys

from api import RpcClient

ADDR = sys.argv[1] if len(sys.argv) > 1 else "ros-server-0:50051"
LISTEN_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30


async def main():
    client = RpcClient(ADDR)
    count = [0]

    def on_msg(msg):
        count[0] += 1
        # std_msgs/String arrives as {"data": "<rollout json string>"}.
        data = msg.get("data") if isinstance(msg, dict) else msg
        s = str(data)
        preview = s[:220].replace("\n", " ")
        print(f"[{count[0]}] policy_rollouts: {len(s)} chars | {preview}", flush=True)

    print(f"Subscribing to 'policy_rollouts' on {ADDR} for {LISTEN_SECONDS}s ...",
          flush=True)
    try:
        await asyncio.wait_for(
            client.Subscribe("policy_rollouts", "std_msgs/String", on_msg),
            timeout=LISTEN_SECONDS)
    except asyncio.TimeoutError:
        pass
    print(f"Done. Received {count[0]} message(s). "
          f"{'OK - data is flowing.' if count[0] else 'NONE - check ROLLOUT_VIZ_ENABLED + that a job is training.'}",
          flush=True)


asyncio.run(main())
