"""Debug tool: subscribe to the `policy_rollouts` topic and print what the
trainer is publishing for the trajectory-rollout visualization.

Run inside the sim-controller container (it has the gRPC stubs + network
access to the ros-servers):

    docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
        sh -c "cd /python_ws/src && python check_rollouts.py"

Requires the trainer to be running WITH ROLLOUT_VIZ_ENABLED=1 and a job
actively training (rollouts only publish from the collect/eval loop). Listens
for ~30s, prints each message's size + a JSON preview, then exits.

Optional args: ros-server address (default ros-server-0:50051, where actor 0
publishes), listen seconds, and topic. The topic argument also makes this the
step-8 check for the fly-brain overlay:

    python check_rollouts.py ros-server-0:50051 30 fly_brain_activity
"""
import asyncio
import sys
import time

from api import RpcClient

ADDR = sys.argv[1] if len(sys.argv) > 1 else "ros-server-0:50051"
LISTEN_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TOPIC = sys.argv[3] if len(sys.argv) > 3 else "policy_rollouts"


async def main():
    client = RpcClient(ADDR)
    count = [0]
    first = [None]

    def on_msg(msg):
        count[0] += 1
        if first[0] is None:
            first[0] = time.time()
        # std_msgs/String arrives as {"data": "<json string>"}.
        data = msg.get("data") if isinstance(msg, dict) else msg
        s = str(data)
        preview = s[:220].replace("\n", " ")
        print(f"[{count[0]}] {TOPIC}: {len(s)} chars | {preview}", flush=True)

    print(f"Subscribing to '{TOPIC}' on {ADDR} for {LISTEN_SECONDS}s ...",
          flush=True)
    try:
        await asyncio.wait_for(
            client.Subscribe(TOPIC, "std_msgs/String", on_msg),
            timeout=LISTEN_SECONDS)
    except asyncio.TimeoutError:
        pass
    # Rate matters as much as arrival for the activity stream, which is
    # supposed to hold 20 Hz.
    rate = ""
    if count[0] > 1 and first[0]:
        span = time.time() - first[0]
        if span > 0:
            rate = f" ~{(count[0] - 1) / span:.1f} Hz."
    print(f"Done. Received {count[0]} message(s).{rate} "
          f"{'OK - data is flowing.' if count[0] else 'NONE - check the publisher is enabled, a job is running, and the topic is a RosSubscriber in unity_node.py.'}",
          flush=True)


asyncio.run(main())
