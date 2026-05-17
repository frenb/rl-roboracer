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

# Install uvloop as the asyncio event-loop policy BEFORE the first
# `asyncio.new_event_loop()` call below in `_start_loop_thread`.
# Mirrors the install block at the top of robotaxi.py - we need it
# here too because ParallelPyEnvironment spawns subprocess workers
# that re-import this module fresh; the parent's loop policy doesn't
# propagate, so each subprocess must call uvloop.install() itself
# before its first loop is created.
#
# See robotaxi.py for the full rationale (Cython libuv-backed loop,
# 2-4x speedup on the DoApplyForce hot path, etc.). Linux/macOS only;
# try/except so a missing uvloop falls back gracefully to the stdlib
# loop instead of crashing module import.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import asyncio
import sys
import threading

# Verify uvloop actually took over the asyncio event-loop policy in
# this subprocess. Each ParallelPyEnvironment worker re-imports this
# module fresh, so this line fires once per worker - with the
# _PrefixedStream wrapper installed by _install_actor_prefix below,
# the messages render as e.g. "[actor-0] event loop policy: ...".
#
# Inspects the policy class rather than calling get_event_loop()
# because modern uvloop's get_event_loop() raises RuntimeError when
# there is no current loop in the thread. Reading the policy is
# safe at any time and answers the same question.
#
# Remove once the swap is confirmed sticking.
_policy_t = type(asyncio.get_event_loop_policy())
print(f"event loop policy: {_policy_t.__module__}.{_policy_t.__name__}", flush=True)
del _policy_t

from api import RobotApi
from environments import RobotaxiEnv


class _PrefixedStream:
    """File-like wrapper that prepends a fixed tag to every output line.

    Multi-actor training spawns one ParallelPyEnvironment worker per
    actor; the workers inherit the parent's stdout/stderr (Linux + spawn
    multiprocessing), so all per-actor prints land in
    sim-controller's ``robotaxi.out`` interleaved without any
    per-source label. Wrapping each worker's streams with this class
    gives every emitted line a stable ``[actor-N] `` prefix so the
    dashboard log view (and the raw file) become readable.

    Implementation notes:

    - Buffers partial lines so the prefix lands once per logical line,
      regardless of whether ``print`` writes the trailing newline as
      one call or two.
    - ``__getattr__`` falls through to the underlying stream so callers
      that introspect ``isatty()``, ``fileno()``, ``encoding``, etc.
      see the same answers they would on the unwrapped stream.
    - ``flush()`` emits any partial line (with the prefix) before
      flushing the underlying stream, so a programmatic Ctrl+C or
      crash doesn't leave a half-line of un-prefixed output.
    """

    def __init__(self, underlying, prefix):
        self._underlying = underlying
        self._prefix = prefix
        self._partial = ""

    def write(self, s):
        if not s:
            return
        # splitlines(keepends=True) preserves trailing newlines so we
        # can tell which fragments are complete lines (emit) vs partial
        # tails (buffer for next write).
        lines = (self._partial + s).splitlines(keepends=True)
        self._partial = ""
        out_parts = []
        for line in lines:
            if line.endswith(("\n", "\r")):
                out_parts.append(self._prefix)
                out_parts.append(line)
            else:
                self._partial = line
        if out_parts:
            self._underlying.write("".join(out_parts))

    def flush(self):
        if self._partial:
            self._underlying.write(self._prefix + self._partial)
            self._partial = ""
        self._underlying.flush()

    def __getattr__(self, name):
        return getattr(self._underlying, name)


def _install_actor_prefix(actor_index):
    """Wrap this process's stdout/stderr with an ``[actor-N] `` prefix.

    Idempotent: re-installing in a worker that's already wrapped is a
    no-op (the second factory call would otherwise nest a prefix
    inside the existing wrapped stream).
    """
    prefix = f"[actor-{actor_index}] "
    if not isinstance(sys.stdout, _PrefixedStream):
        sys.stdout = _PrefixedStream(sys.stdout, prefix)
    if not isinstance(sys.stderr, _PrefixedStream):
        sys.stderr = _PrefixedStream(sys.stderr, prefix)


def _silence_grpc_blockingio_errors(loop):
    """Suppress the noisy asyncio log spam from grpc.aio's poller EAGAIN.

    ``grpcio``'s ``PollerCompletionQueue._handle_events`` uses a
    non-blocking file descriptor for its internal completion queue.
    Under high load (and especially with N parallel-env workers each
    running their own gRPC client) the poller occasionally returns
    ``EAGAIN`` (errno 11) -> raises ``BlockingIOError`` inside an
    asyncio callback. The library handles this transparently by
    retrying on the next loop iteration, but asyncio's default
    exception handler logs a full traceback every time, producing
    pages of:

        BlockingIOError: [Errno 11] Resource temporarily unavailable
        ERROR:asyncio:Exception in callback PollerCompletionQueue._handle_events(...)

    that drown out the actual training output in robotaxi.out. We
    install a filter that drops just this exception class and lets
    everything else fall through to the default handler. Genuine
    BlockingIOError from elsewhere would also be silenced, but at the
    asyncio level any such error is by definition a "would block"
    that asyncio is supposed to handle by yielding - so silencing the
    log is the right behavior, not just a cosmetic bandage.
    """
    default_handler = loop.get_exception_handler()

    def _filtered(loop, context):
        exc = context.get('exception')
        if isinstance(exc, BlockingIOError):
            return
        if default_handler is not None:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_filtered)


def _start_api(addr):
    """Start an asyncio loop in a daemon thread, construct a RobotApi bound
    to that loop, await its Initialize, and return ``(api, loop, thread)``.

    ``RobotApi.__init__`` calls ``asyncio.get_event_loop()`` and creates
    several ``asyncio.Event`` instances; both bind to whichever loop is
    current at construction time. We construct it from inside a coroutine
    submitted to the target loop so those bindings are correct.
    """
    loop = asyncio.new_event_loop()
    _silence_grpc_blockingio_errors(loop)

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


def make_env(grpc_addr='ros-server-0:50051', course_type='donut',
             actor_index=None):
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
      actor_index: when provided, the worker's stdout/stderr are wrapped
        to prefix every line with ``[actor-N] ``. Single-actor callers
        leave this ``None`` so the existing un-decorated logs are
        preserved; multi-actor callers pass the per-factory index so
        ``robotaxi.out`` can be read back un-multiplexed even though
        all N workers share the parent's stdout.

    Returns:
      A :class:`RobotaxiEnv` instance with an attached, running
      :class:`RobotApi`. Backing thread + loop references are stashed on
      the env (``_api``, ``_api_loop``, ``_api_thread``) so they outlive
      function scope and stay alive for the env's lifetime.
    """
    if actor_index is not None:
        _install_actor_prefix(actor_index)
    api, loop, thread = _start_api(grpc_addr)
    env = RobotaxiEnv(api, course_type=course_type)
    env._api = api
    env._api_loop = loop
    env._api_thread = thread
    return env
