"""Environment factory for robotaxi training.

Each call to :func:`make_env` constructs a brand-new :class:`RobotaxiEnv`
backed by its own :class:`RobotApi` and its own asyncio event loop running
in a daemon thread. That means the factory is safe to call from any
process: when ``tf_agents.environments.parallel_py_environment`` forks N
worker processes for parallel collection, each subprocess gets its own
self-contained ROS connection and loop without touching state from
sibling subprocesses or the parent.

Why a daemon thread per env (rather than `asyncio.run`):

  - ``RobotApi.Initialize`` schedules five long-lived ROS subscriptions
    via ``loop.create_task`` and returns. Those tasks need a continually
    running loop to actually receive messages.
  - The synchronous wrappers on ``RobotApi`` (``DoApplyForceBlocking`` etc.)
    use ``asyncio.run_coroutine_threadsafe(coro, self.loop).result()``,
    which requires ``self.loop`` to be running in another thread.
  - A daemon ``loop.run_forever()`` thread satisfies both. The thread
    dies with the process, no manual cleanup needed.
"""

import asyncio
import threading

from api import RobotApi
from environments import RobotaxiEnv


def _start_api(addr):
    """Start an asyncio loop in a daemon thread, construct a RobotApi bound
    to that loop, await its Initialize, and return ``(api, loop, thread)``.

    ``RobotApi.__init__`` calls ``asyncio.get_event_loop()`` and creates
    several ``asyncio.Event`` instances; both bind to whichever loop is
    current at construction time. We construct it from inside a coroutine
    submitted to the target loop so those bindings are correct.
    """
    loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_runner, name=f"RobotApi-loop[{addr}]", daemon=True)
    thread.start()

    async def _build():
        api = RobotApi(addr=addr)
        await api.Initialize()
        return api

    api = asyncio.run_coroutine_threadsafe(_build(), loop).result(timeout=30)
    return api, loop, thread


def make_env(grpc_addr='ros-server-0:50051', course_type='donut'):
    """Construct a :class:`RobotaxiEnv` connected to the given gRPC endpoint.

    Args:
      grpc_addr: ``host:port`` of a ros-server's gRPC virtual_endpoint.
        For multi-actor training, pass ``ros-server-0:50051``,
        ``ros-server-1:50051``, etc. For single-actor training, the default
        ``ros-server-0:50051`` matches the network alias declared on the
        base ros-server service in ``docker-compose.yml`` (so single-env
        works without the ``compose/scale.yml`` overlay).
      course_type: ``'donut'`` (default) or ``'simple'``; selected when
        building the inner course.

    Returns:
      A :class:`RobotaxiEnv` instance with an attached, running
      :class:`RobotApi`. Backing thread + loop references are stashed on
      the env (``_api``, ``_api_loop``, ``_api_thread``) so they outlive
      function scope and stay alive for the env's lifetime.
    """
    api, loop, thread = _start_api(grpc_addr)
    env = RobotaxiEnv(api, course_type=course_type)
    env._api = api
    env._api_loop = loop
    env._api_thread = thread
    return env
