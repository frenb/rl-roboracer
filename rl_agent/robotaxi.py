# Install uvloop as the asyncio event-loop policy BEFORE any module
# in this process creates an asyncio loop. uvloop is a Cython wrapper
# around libuv that replaces asyncio's pure-Python loop with a C
# implementation; benchmarks typically show 2-4x throughput on small-
# message gRPC + event.wait() patterns, which is exactly the
# DoApplyForce/RobotApi hot path here.
#
# Why at the very top:
#   - tf_agents (imported below) eventually touches asyncio.
#   - grpc.aio (imported transitively by api.py / RobotApi) creates
#     its own loops.
#   - asyncio.get_event_loop() before install() locks in the stdlib
#     loop; uvloop.install() won't retroactively swap it.
#
# Linux/macOS only - no Windows native build. sim-controller is a
# Linux container so this is safe in production. try/except makes it
# a graceful no-op if uvloop is ever absent (Dockerfile transition,
# pip pin rolling back, etc.) so we degrade to the stdlib loop rather
# than crash on import.
#
# Important: this must also be present at the top of envs.py because
# ParallelPyEnvironment spawns subprocess workers that re-import the
# env factory module fresh - each subprocess needs its own
# uvloop.install() before its own RobotApi creates a loop. Keeping
# the two install blocks in sync is intentional.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

# Verify uvloop actually took over the asyncio event-loop policy.
# Inspects the policy class rather than calling get_event_loop()
# because modern uvloop's get_event_loop() raises RuntimeError when
# there is no current loop in the thread, whereas the policy is
# safe to read at any time.
#
# Prints:
#   "event loop policy: uvloop.EventLoopPolicy"      when uvloop is active
#   "event loop policy: asyncio.unix_events._UnixDefaultEventLoopPolicy"
#                                                    when uvloop didn't install
#
# Remove this line once you're confident the swap is sticking
# across restarts.
import asyncio
_policy_t = type(asyncio.get_event_loop_policy())
print(f"event loop policy: {_policy_t.__module__}.{_policy_t.__name__}", flush=True)
del _policy_t

import random
import os
import math
import shutil
import tempfile
import time
import threading
import json
import datetime

# tf-agents 0.11.0rc0 ships an OpenAIGymStateSaver (registered globally
# in tf_agents.system.system_multiprocessing._STATE_SAVERS at import
# time) that runs inside every ParallelPyEnvironment worker and does
#     if not isinstance(state, gym.envs.registration.EnvRegistry): ...
# That class existed in gym <= 0.23. gym 0.26 (what this container has)
# made the registry a plain dict and removed EnvRegistry entirely, so
# every worker dies with AttributeError before our env factories run.
# Aliasing EnvRegistry to the registry's actual type makes the
# isinstance check pass without affecting any real gym usage. Done at
# module level so the spawn-reimported children pick it up too.
import gym.envs.registration as _gym_reg
if not hasattr(_gym_reg, 'EnvRegistry'):
    _gym_reg.EnvRegistry = type(_gym_reg.registry)
del _gym_reg

import numpy as np
from scipy.interpolate import interp1d
from numpy import interp
import tensorflow as tf

# ---- GPU memory sharing with the Unity clients --------------------------
# The Unity gym clients run on the same GPU as this trainer (they need an
# active renderer for physics + raycasting; see RosBootstrap.cs). By
# default TensorFlow GREEDILY pre-allocates almost all GPU memory on first
# use, which leaves the Unity clients unable to create their D3D graphics
# device — they wedge at "GfxDevice: creating device client" with no
# window if the trainer starts before them. Enabling memory growth makes
# TF allocate only what it actually needs (far less than the full GPU for
# the SAC nets here), so the Unity clients always have headroom regardless
# of which side starts first.
#
# Runs at module import so every process that imports robotaxi.py — the
# main trainer AND each ParallelPyEnvironment subprocess — applies it
# before any GPU op initializes the device. set_memory_growth must be
# called before the GPU is touched; doing it here guarantees that.
#
# Optionally cap TF to a hard ceiling (MB) via ROBOTAXI_GPU_MEMORY_LIMIT_MB
# to reserve a fixed slice for the Unity clients. Unset = memory growth.
def _configure_gpu_memory():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return
        limit_mb = os.environ.get("ROBOTAXI_GPU_MEMORY_LIMIT_MB", "").strip()
        if limit_mb:
            mb = int(limit_mb)
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=mb)])
            print(f"[gpu] capped TF to {mb} MB/GPU "
                  f"(reserving the rest for Unity clients)", flush=True)
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[gpu] enabled memory growth on {len(gpus)} GPU(s) "
                  f"(TF takes only what it needs; leaves room for Unity)",
                  flush=True)
    except Exception as _e:  # noqa: BLE001
        # Never let GPU config block the trainer from starting; a failure
        # here just falls back to TF's default greedy allocation.
        print(f"[gpu] memory config skipped: {_e}", flush=True)


_configure_gpu_memory()

from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.environments import py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from pymongo import MongoClient

from environments.courses import donut_course,simple_course
from envs import make_env
from replay import make_local_replay
import collect_training_data
from rollout_viz import get_viz
import logging
import signal
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _ts():
    """Millisecond-precision local wall-clock timestamp for log/print lines.

    Added 2026-07-19 so diagnostic prints (steer-diag, crash-pos, stage
    transitions, etc.) can be correlated after the fact against screen
    recordings of the car driving - see scripts/analyze_video.py. NOTE: this
    is the sim-controller CONTAINER's local clock, not necessarily the
    recording machine's (e.g. the container runs UTC while the Windows host
    recording the video is UTC-7) - analyze_video.py auto-detects and
    corrects for that offset, so don't assume this timestamp is directly
    comparable to a wall-clock time you read off the host. Plain
    HH:MM:SS.mmm (no date) since correlation windows are always a few
    minutes within a single session; kept out of the standard `logging`
    formatter (which only prefixes logger.info/etc, not the many raw
    print() diagnostics here) so this is a single opt-in call site per
    print rather than a global format change.
    """
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


# Trainer-wide state visible to the SIGTERM emergency-pause handler.
# Populated by main() right after the Learner is constructed; reset
# back to {} when main() exits cleanly. The handler reads this to
# know which job/Learner/train_step to flush on a Docker/Windows-
# initiated shutdown.
#
# Module-level instead of a closure because signal handlers
# installed via signal.signal() outlive the main() call frame; we
# want each successive job pickup to install its own state without
# leaking the previous job's references.
_emergency_state = {}


def _emergency_pause_handler(signum, frame):
    """Best-effort graceful pause on SIGTERM (Docker / WSL shutdown).

    Triggered when the container is asked to stop, e.g. by:
      * `docker compose stop` / `docker compose down`,
      * Docker Desktop shutting down on Windows Update / host reboot,
      * `docker stop <container>`.

    Docker sends SIGTERM with a default 10s grace period before
    SIGKILL. tf-agents' Checkpointer.save() is ~100ms of local-disk
    I/O so we have ample headroom in the typical case.

    Two writes happen in the handler:
      1. Learner checkpoint to /tmp/active/<id>/learner/train/checkpoints/.
         The next trainer pickup of this job will auto-restore it via
         the Learner constructor (see _detect_resume_for_train_job's
         crash-recovery branch).
      2. Mongo: status=PAUSED + paused_at_step=<current train_step>.
         This is the EXPLICIT-resume signal the dashboard uses.
         Without it, the recovery would still work via crash-recovery
         detection - this is just belt-and-suspenders so the operator
         sees the job clearly as PAUSED in the UI rather than as a
         stale IN_PROGRESS that "happens to resume".

    Sys.exit(0) at the end so we cooperate with Docker's shutdown
    sequence; tf-agents' multiprocess actor children will be cleaned
    up by Python's atexit handlers + the container teardown.

    Exceptions inside the handler are swallowed (we're already on a
    shutdown path; raising would just turn a graceful pause into a
    SIGABRT and lose the checkpoint anyway).
    """
    state = _emergency_state
    if not state:
        # No active job - either the trainer hasn't reached the Learner
        # construction step yet (still in env setup / BC pretrain), or
        # we're between jobs. Just exit; nothing to save.
        print(
            f"SIGTERM ({signum}): no active job state - exiting cleanly.",
            flush=True)
        sys.exit(0)
    job_id = state.get("job_id")
    agent_learner = state.get("learner")
    train_step = state.get("train_step")
    try:
        step_val = int(train_step.numpy()) if train_step is not None else -1
        if agent_learner is not None and step_val >= 0:
            try:
                agent_learner._checkpointer.save(step_val)
                print(
                    f"SIGTERM ({signum}): saved emergency Learner "
                    f"checkpoint at train_step={step_val} for job {job_id}.",
                    flush=True)
            except Exception as _e:  # noqa: BLE001
                print(
                    f"SIGTERM ({signum}): Learner checkpoint save FAILED: "
                    f"{_e}. Falling back to the most recent automatic "
                    f"checkpoint (checkpoint_interval=100); the crash-"
                    f"recovery detector in do_job will still resume.",
                    flush=True)
        if job_id is not None:
            try:
                update_job(job_id, step_val, "paused_at_step")
                update_job(job_id, "PAUSED", "status")
                print(
                    f"SIGTERM ({signum}): marked job {job_id} PAUSED "
                    f"(paused_at_step={step_val}). Resume from the "
                    f"dashboard once the stack is back up.",
                    flush=True)
            except Exception as _e:  # noqa: BLE001
                print(
                    f"SIGTERM ({signum}): Mongo pause write FAILED: {_e}. "
                    f"Job will appear IN_PROGRESS but crash-recovery "
                    f"detection in do_job will still resume on next "
                    f"trainer pickup.",
                    flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"SIGTERM ({signum}): emergency handler error: {_e}",
              flush=True)
    sys.exit(0)


# Install the SIGTERM handler eagerly at module import. SIGINT is
# left at its default (Python raises KeyboardInterrupt) because the
# interactive Ctrl-C case is "operator wants out now"; for that we
# rely on the periodic Learner checkpoints (checkpoint_interval=100)
# + the crash-recovery detector instead of trying to do extra work
# during the shutdown.
try:
    signal.signal(signal.SIGTERM, _emergency_pause_handler)
except (ValueError, OSError) as _e:  # noqa: BLE001
    # Some environments (Windows + non-main thread, embedded
    # interpreters) refuse signal.signal. We don't want that to be
    # fatal - the trainer still works, it just won't catch SIGTERM
    # gracefully. The periodic checkpoint is the real safety net.
    print(f"robotaxi: could not install SIGTERM handler: {_e}", flush=True)

client = MongoClient('mongo', 
    username='root',
    password='example')
# db = client.local
#set database_name variable to environment variable DATABASE_NAME
database_name = os.environ['DATABASE_NAME']
db = client.robotaxi


# ---- Git provenance ------------------------------------------------ *
#
# Read the trainer's git_sha + git_branch ONCE at module import time
# and cache. Stamped onto every model document (via add_model) and
# every job document (via do_job at pickup) so the future multiverse
# / research-agent tooling can answer "which code version produced
# this result?" without timestamp inference.
#
# We parse .git/HEAD directly (rather than shelling out to `git`)
# because the sim-controller image isn't guaranteed to have the git
# binary, and the parse is trivial. /git_meta is the read-only bind
# mount of the host's ./.git declared in docker-compose.yml.
#
# Failures are silently swallowed -> _GIT_SHA / _GIT_BRANCH stay None.
# Records inserted in that state are flagged "no git provenance" by
# the dashboard's Models tab; not a fatal condition.
def _read_git_provenance(git_dir="/git_meta"):
    try:
        head_path = os.path.join(git_dir, "HEAD")
        with open(head_path, "r") as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[len("ref: "):]
            branch = ref.rsplit("/", 1)[-1]
            ref_path = os.path.join(git_dir, ref)
            try:
                with open(ref_path, "r") as f:
                    sha = f.read().strip()
            except FileNotFoundError:
                # Branch ref might live in packed-refs instead of a
                # loose file (common on freshly-cloned repos). Walk
                # packed-refs for the matching line.
                sha = None
                packed = os.path.join(git_dir, "packed-refs")
                try:
                    with open(packed, "r") as f:
                        for line in f:
                            if line.startswith("#") or "^" in line:
                                continue
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2 and parts[1] == ref:
                                sha = parts[0]
                                break
                except FileNotFoundError:
                    pass
            return sha, branch
        # Detached HEAD - raw SHA.
        return head, None
    except Exception as e:
        print(f"_read_git_provenance: failed ({e}); records will carry no git_sha/branch.",
              flush=True)
        return None, None

_GIT_SHA, _GIT_BRANCH = _read_git_provenance()
print(f"trainer git provenance: sha={_GIT_SHA} branch={_GIT_BRANCH}", flush=True)

def get_save_dir_root(policy):
    policy_type = get_policy_type_name(policy)
    saved_models_dir = os.getenv('SAVED_MODELS_DIR')
    robot_type = os.getenv('ROBOT_TYPE')
    return os.path.join(saved_models_dir,robot_type,policy_type)

def get_policy_type_name(policy):
    if (isinstance(policy, str)):
        policy_type = policy
        debug_print(policy_type)
    else:
        policy_type = type(policy).__name__
        # AWAC is a SacAgent subclass; its saved policy is a plain tf-agents
        # policy evaluated as a SacAgent (job model_type='SacAgent'), and the
        # /saved_models/<robot>/SacAgent/ tree is the one that exists. Map the
        # subclass name back so model saving + the Models-tab/EVAL dispatch
        # all resolve to the same SacAgent directory.
        if policy_type == "AwacSacAgent":
            policy_type = "SacAgent"
    return policy_type

def get_next_model_version(policy):
    path=get_save_dir_root(policy)
    file_list = os.listdir(path)
    sorted_file_list=sorted(file_list,key=str,reverse=True)
    num_dirs = len(sorted_file_list)
    next_model_version=str(num_dirs)
    debug_print(file_list)
    debug_print(sorted_file_list)
    debug_print(next_model_version)
    return path, next_model_version

def get_save_dir_name(policy):
    path, next_dir_name=get_next_model_version(policy)
    return os.path.join(path,next_dir_name)

def get_latest_save_dir_name(policy):
    path=get_save_dir_root(policy)
    file_list = os.listdir(path)
    sorted_file_list=sorted(file_list,reverse=True)
    return os.path.join(path,sorted_file_list[0])

def get_save_dir_by_version(policy, version):
    path=get_save_dir_root(policy)
    file_list = os.listdir(path)
    sorted_file_list=sorted(file_list,reverse=True)
    return os.path.join(path, version)

def print_replay_buffer_size(reverb_replay, table_name, replay_buffer_capacity):
    # Query the Reverb server for the current stats
    server_info = reverb_replay.py_client.server_info()
    # Extract the current size of your specific table
    current_size = server_info[table_name].current_size
    print(f"Current Replay Buffer length: {current_size} / {replay_buffer_capacity}")


def read_timeout_counts(env):
    """Aggregate per-actor RobotApi timeout counters across all sub-envs.

    Each ParallelPyEnvironment worker holds its own RobotApi instance
    with its own counters (incremented in api.py's
    `except asyncio.TimeoutError` branches whenever a Unity round-trip
    misses its deadline). We sum across actors so the TensorBoard
    scalars reflect the *total* count of dropped waits the stack has
    experienced - useful for spotting whether a particular eval
    interval saw a spike vs the steady-state rate.

    Single env: just one RobotApi's counts (no aggregation).
    Parallel envs: dispatch to each underlying ProcessPyEnvironment and
    sum.

    The returned dict has the five counter keys
    ('reset_timeouts', 'apply_force_timeouts', 'scene_data_timeouts',
    'move_timeouts', 'publish_timeouts'); the trainer prefixes them
    with 'timeouts/' when writing to tf.summary so they group together
    in TensorBoard's UI.

    publish_timeouts in particular catches the case where the gRPC
    channel to ros-server goes stale (historically after a long no-
    traffic window like BC pretraining) and a Publish RPC hits its
    client-side DEADLINE_EXCEEDED rather than completing. Before that
    counter existed the failure mode was a silent deadlock; now it
    surfaces as a tick on the TensorBoard scalar.
    """
    from tf_agents.environments import parallel_py_environment
    keys = ['reset_timeouts', 'apply_force_timeouts',
            'scene_data_timeouts', 'move_timeouts', 'publish_timeouts']
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [proc_env.call('get_timeout_counts')
                    for proc_env in env._envs]
        per_actor = [promise() for promise in promises]
        return {k: sum(d.get(k, 0) for d in per_actor) for k in keys}
    return env.get_timeout_counts()


def read_course_raw_counters(env):
    """Read the inner course's raw counters from a (possibly batched) env.

    Single env: just delegate to env.get_course_raw_counters().
    Parallel envs: dispatch to each underlying ProcessPyEnvironment and
    SUM across actors. Summing (rather than averaging) is correct
    because raw counters are accumulating tallies (total speed across
    all steps, total episode count, etc.); the consumer in run_policy
    snapshots this sum before and after each trial and divides
    appropriate-numerator-deltas by appropriate-denominator-deltas to
    get the per-trial average across all actors' contribution.

    Returns a dict with the same keys as ``BaseCourse.RAW_COUNTER_KEYS``.
    Missing keys default to 0 so a course that doesn't track a
    particular counter cleanly contributes nothing rather than
    breaking the aggregation.
    """
    from tf_agents.environments import parallel_py_environment
    keys = ('steps_total', 'num_episodes_total', 'speeds_total',
            'goals_per_episode_total', 'steering_angle_ratio_total')
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [proc_env.call('get_course_raw_counters')
                    for proc_env in env._envs]
        per_actor = [promise() for promise in promises]
        return {k: float(sum(d.get(k, 0) for d in per_actor)) for k in keys}
    raw = env.get_course_raw_counters()
    return {k: float(raw.get(k, 0)) for k in keys}


def read_course_metrics(env):
    """Read the inner course metric snapshot from a (possibly batched) env.

    Single env: just delegate to env.get_course_metrics().
    Parallel envs: dispatch to each underlying ProcessPyEnvironment and
    aggregate across actors - max() for max_* keys, mean() for the rest.
    Returns a dict with the same keys as a single env, so the
    TensorBoard scalar names are unchanged regardless of N.

    The same caveat as configure_env applies: ParallelPyEnvironment in
    tf-agents 0.11 has no public .call() proxy of its own, so we go
    through the per-subprocess wrappers in env._envs and use the
    fire-all-then-wait-all promise pattern that the library's own
    seed() helper uses.
    """
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [proc_env.call('get_course_metrics')
                    for proc_env in env._envs]
        per_actor = [promise() for promise in promises]
        out = {}
        for k in per_actor[0]:
            vals = [d[k] for d in per_actor]
            out[k] = max(vals) if k.startswith('max_') else sum(vals) / len(vals)
        return out
    return env.get_course_metrics()


def read_course_recent_goals(env, n):
    """Return the most-recent ``n`` per-episode goal counts from a course.

    Single env: delegate to env.get_recent_goals_per_episode(n).
    Parallel envs: concatenate each worker's recent-n (rarely needed - the
    curriculum gate calls this on the single-gym eval env). Returns a list of
    floats; empty if the course/env doesn't expose the accessor.
    """
    from tf_agents.environments import parallel_py_environment
    try:
        if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
            promises = [proc_env.call('get_recent_goals_per_episode', n)
                        for proc_env in env._envs]
            out = []
            for promise in promises:
                out.extend(promise() or [])
            return out
        return list(env.get_recent_goals_per_episode(n) or [])
    except Exception as e:  # noqa: BLE001
        print(f"read_course_recent_goals failed: {e}", flush=True)
        return []


def build_train_env(num_envs, course_type='donut'):
    """Construct the training env (single or parallel) for main()."""
    if num_envs <= 1:
        return make_env('ros-server-0:50051', course_type=course_type)

    from tf_agents.environments import parallel_py_environment
    # Quiesce the rollout-viz background sampler around the multiprocessing
    # spawn below. A concurrent TF/GPU inference in that thread races the
    # ParallelPyEnvironment fork and can segfault the whole process (observed
    # 2026-06-10, mid-`sample_rollouts_multi` while spawning a new job's env).
    # pause_viz() blocks until any in-flight inference tick finishes; the
    # finally re-enables it once all workers are started. No-op when viz is
    # disabled or not yet constructed.
    from rollout_viz import pause_viz, resume_viz
    # actor_index=i wraps each worker's stdout/stderr with [actor-N] so
    # robotaxi.out (and the dashboard log view) become legible when
    # multiple workers are emitting interleaved per-step prints.
    pause_viz()
    try:
        return parallel_py_environment.ParallelPyEnvironment(
            [(lambda i=i: make_env(f'ros-server-{i}:50051',
                                   course_type=course_type,
                                   actor_index=i))
             for i in range(num_envs)])
    finally:
        resume_viz()


def configure_env(env, job_id="", pass_through_actions=False,
                  corner_radius=10.0, curvature_difficulty=0.0,
                  chicanes_north=0, chicanes_east=0, chicanes_south=0,
                  chicanes_west=0, env_discount=None):
    """Apply per-job config to a single env or all parallel subprocess envs.

    tf-agents 0.11 doesn't put a public `call()` proxy on
    ParallelPyEnvironment itself; the dispatch lives on each underlying
    ProcessPyEnvironment in env._envs. ProcessPyEnvironment.call() is
    asynchronous and returns a no-arg promise-callable; following the
    same fire-all-then-wait-all pattern that ParallelPyEnvironment.seed
    uses internally lets all four subprocess configure() calls run
    concurrently instead of serially.

    ``corner_radius`` / ``curvature_difficulty`` are the procedural-track
    curriculum knobs; they're constant across a job, so every actor gets the
    same values and the shared Unity scene regenerates one consistent track.
    ``chicanes_north/east/south/west`` (2026-07-18) are the per-edge absolute
    chicane counts that actually drive chicane placement (curvature_difficulty
    is kept for logging/back-compat only - see TrackGenerator.cs).
    """
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [
            proc_env.call('configure', job_id, pass_through_actions,
                          corner_radius, curvature_difficulty,
                          chicanes_north, chicanes_east, chicanes_south,
                          chicanes_west, env_discount)
            for proc_env in env._envs
        ]
        for promise in promises:
            promise()
    else:
        env.configure(job_id, pass_through_actions,
                      corner_radius, curvature_difficulty,
                      chicanes_north, chicanes_east, chicanes_south,
                      chicanes_west, env_discount)


def set_immediate_reset_on_failure(env, enabled):
    """Enable/disable immediate Unity reset on episode failure.

    During TRAINING, immediate reset is desirable so the car doesn't sit
    dead for 10+ seconds while the learner processes. During EVAL, immediate
    reset is counterproductive because it blocks the fast eval stepping.

    Dispatches to all subprocess envs in a ParallelPyEnvironment.
    """
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [
            proc_env.call('set_immediate_reset_on_failure', enabled)
            for proc_env in env._envs
        ]
        for promise in promises:
            try:
                promise()
            except Exception:
                pass  # Env might not have the method (backward compat)
    else:
        if hasattr(env, 'set_immediate_reset_on_failure'):
            env.set_immediate_reset_on_failure(enabled)
        elif hasattr(env, '_immediate_reset_on_failure'):
            env._immediate_reset_on_failure = enabled


def install_reward_design_on_env(env, name, code):
    """Install a reward design on the env (single- or multi-env aware).

    Mirrors configure_env's dispatch pattern: in multi-env training the
    course instances live inside subprocess envs (ParallelPyEnvironment),
    so we have to ask each ProcessPyEnvironment to compile and install
    the design inside its own process via the .call() proxy.

    The user code is a string (not function objects), so it serialises
    cleanly across the process boundary - each subprocess does its own
    compile + monkey-patch via RobotaxiEnv.install_reward_design.

    Returns:
      A dict ``{installed: [fn_names], penalty_reward: float, actors: int}``
      summarising what was patched. For multi-env training we report the
      union of installed function names across actors and the actor count;
      the dict shape matches the single-env case so callers can log
      uniformly. None is never returned on a successful install - any
      failure raises RewardDesignError.

    Raises:
      rl_agent.reward_designs.RewardDesignError if the design fails to
      compile/load in any sub-env. Caller in do_job catches this and
      surfaces as job.eval_error.
    """
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [
            proc_env.call('install_reward_design', name, code)
            for proc_env in env._envs
        ]
        # Wait for all installs so any RewardDesignError propagates
        # before training starts (rather than at the first env step
        # of some actor). We capture each actor's returned dict so the
        # caller (main()) can log the actual list of patched function
        # names rather than just "installed=None" (which was a logging
        # bug that hid whether the install actually took effect).
        per_actor = []
        for promise in promises:
            per_actor.append(promise())
        installed = sorted({
            fn
            for result in per_actor
            if isinstance(result, dict)
            for fn in (result.get("installed") or [])
        })
        penalty_reward = next(
            (result.get("penalty_reward") for result in per_actor
             if isinstance(result, dict) and result.get("penalty_reward") is not None),
            None)
        return {
            "installed": installed,
            "penalty_reward": penalty_reward,
            "actors": len(per_actor),
        }
    else:
        result = env.install_reward_design(name, code)
        # Single-env path returns the raw dict from install_on_course
        # ({installed, penalty_reward}); wrap with actors=1 to match
        # the multi-env shape so the caller can render both uniformly.
        if isinstance(result, dict):
            return {
                "installed": list(result.get("installed") or []),
                "penalty_reward": result.get("penalty_reward"),
                "actors": 1,
            }
        # Defensive fallback in case a future env subclass returns
        # something else - we still know the install didn't raise, so
        # we report an unknown-names success rather than masking the
        # outcome as None.
        return {"installed": [], "penalty_reward": None, "actors": 1}


def _measure_demo_batch_fraction(demo_rb, online_rb, ratio, batch_size,
                                 sequence_length=2, n_batches=300):
    """Empirically measure the realized demo fraction of the mixed sampler.

    Mirrors replay.make_local_replay's
    ``tf.data.Dataset.sample_from_datasets([demo, online], weights=[r, 1-r])``
    construction, but tags each source so we can count how many sampled rows
    actually originate in the protected demo table vs the online RL table.

    Note on granularity: both source datasets already yield *batches* of
    ``batch_size`` rows, so ``sample_from_datasets`` interleaves at batch
    granularity - each drawn batch is wholly demo OR wholly online. The
    per-batch composition is therefore all-or-nothing; it's the AGGREGATE
    row fraction over many batches that converges to ``ratio``. This probe
    reports that aggregate.

    Pure measurement: builds its own throwaway datasets off the same two
    Reverb tables (a second set of readers), so it does NOT perturb the
    learner's training stream. Returns (realized_fraction, demo_rows,
    total_rows).
    """
    import tensorflow as tf
    demo_ds = demo_rb.as_dataset(
        sample_batch_size=batch_size, num_steps=sequence_length)
    online_ds = online_rb.as_dataset(
        sample_batch_size=batch_size, num_steps=sequence_length)
    # Replace each batch with a same-shape source flag (0=demo, 1=online) so
    # sample_from_datasets carries only the tag, not the (large) trajectory.
    demo_flags = demo_ds.map(lambda *a: tf.zeros([batch_size], tf.int32))
    online_flags = online_ds.map(lambda *a: tf.ones([batch_size], tf.int32))
    mixed = tf.data.Dataset.sample_from_datasets(
        [demo_flags, online_flags],
        weights=[ratio, 1.0 - ratio],
        stop_on_empty_dataset=False)
    demo_rows = 0
    total_rows = 0
    for flags in mixed.take(n_batches):
        f = flags.numpy()
        demo_rows += int((f == 0).sum())
        total_rows += int(f.size)
    realized = (demo_rows / total_rows) if total_rows else float('nan')
    return realized, demo_rows, total_rows


def _append_eval_curve(job_id, step, avg_return, avg_ep_len):
    """Best-effort append of one in-training eval point to a per-job CSV.

    Gives a fast, file-based record of the (step, AverageReturn,
    AverageEpisodeLength) learning curve so offline stall/recovery analysis
    (e.g. "did this seed ever recover above its BC start?") is a trivial CSV
    read instead of a slow unindexed db.logs scan.

    Written under the job's /tmp/active/<id>/ dir, which move_all_jobs_data
    archives to /tmp/jobsdata/<id>/ at job end - so the curve persists with
    the job's other artifacts. Pure logging: never raises, so it can't
    disturb the eval/training loop.
    """
    if not job_id:
        return
    try:
        d = os.path.join("/tmp/active", str(job_id))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "eval_curve.csv")
        write_header = not os.path.exists(path)
        with open(path, "a") as f:
            if write_header:
                f.write("step,avg_return,avg_ep_len\n")
            r = "" if avg_return is None else f"{float(avg_return):.6f}"
            e = "" if avg_ep_len is None else f"{float(avg_ep_len):.4f}"
            f.write(f"{int(step)},{r},{e}\n")
    except Exception:  # noqa: BLE001 - logging must never disturb eval
        pass


class CurriculumScheduler:
    """Performance-gated track-difficulty scheduler for curriculum training.

    Advances through a sequence of (corner_radius, curvature_difficulty) stages
    based on the agent's avg_goals_per_episode at each eval cycle.

    Each stage dict:
        corner_radius (float):       track turn radius in metres
        curvature_difficulty (float): DEPRECATED chicane density [0, 1],
                                      default 0.0 - logging/back-compat only,
                                      no longer drives chicane count (see
                                      chicanes_north/east/south/west below).
        chicanes_north/east/south/west (int): absolute chicane count on
                                      each edge for this stage, default 0.
        advance_goals (float):       goals/ep threshold to advance
        consecutive (int):           consecutive evals above threshold, default 1

    The last stage has no advance_goals (terminal; training continues on it).

    Usage:
        sched = CurriculumScheduler(stages, env)
        # at each eval cycle:
        advanced = sched.update(avg_goals_per_episode, train_step)
    """

    def __init__(self, stages, env, eval_env=None, start_stage=0):
        if not stages:
            raise ValueError("CurriculumScheduler requires at least one stage")
        self.stages = stages
        self.env = env
        # Dedicated single-gym eval env (multi-env runs). When present and
        # distinct from the collect env, each stage change must reconfigure it
        # too so eval measures performance on the CURRENT stage geometry rather
        # than whatever stage the eval env was built at. None (or the same
        # object as env) in single-env runs where collect and eval share one env.
        self.eval_env = eval_env if (eval_env is not None
                                     and eval_env is not env) else None
        # Starting stage index (curriculum_start_stage). Default 0 = normal
        # bottom-up curriculum. Set > 0 to pin the run higher up the ladder,
        # e.g. start_stage=len(stages)-1 trains directly on the final/hardest
        # geometry (terminal stage: no advance_goals, so update() is a no-op
        # and the run stays there). Clamped into range so an out-of-bounds
        # config can never IndexError.
        try:
            start_stage = int(start_stage)
        except (TypeError, ValueError):
            start_stage = 0
        self.stage_idx = max(0, min(start_stage, len(stages) - 1))
        self._consecutive_above = 0
        self._apply_stage(train_step=0)

    def _apply_stage(self, train_step):
        s = self.stages[self.stage_idx]
        cr = float(s.get('corner_radius', 10.0))
        cd = float(s.get('curvature_difficulty', 0.0))
        ch_n = int(s.get('chicanes_north', 0))
        ch_e = int(s.get('chicanes_east', 0))
        ch_s = int(s.get('chicanes_south', 0))
        ch_w = int(s.get('chicanes_west', 0))
        configure_env(self.env, corner_radius=cr, curvature_difficulty=cd,
                      chicanes_north=ch_n, chicanes_east=ch_e,
                      chicanes_south=ch_s, chicanes_west=ch_w)
        if self.eval_env is not None:
            configure_env(self.eval_env, corner_radius=cr,
                          curvature_difficulty=cd,
                          chicanes_north=ch_n, chicanes_east=ch_e,
                          chicanes_south=ch_s, chicanes_west=ch_w)
        print(
            f"[curriculum] stage {self.stage_idx}/{len(self.stages)-1}: "
            f"corner_radius={cr}, curvature_difficulty={cd}, "
            f"chicanes(N/E/S/W)={ch_n}/{ch_e}/{ch_s}/{ch_w} "
            f"(train_step={train_step})",
            flush=True)
        try:
            tf.summary.scalar('curriculum/stage', data=self.stage_idx,
                              step=train_step)
            tf.summary.scalar('curriculum/corner_radius', data=cr,
                              step=train_step)
            tf.summary.scalar('curriculum/curvature_difficulty', data=cd,
                              step=train_step)
            tf.summary.scalar('curriculum/chicanes_total',
                              data=ch_n + ch_e + ch_s + ch_w,
                              step=train_step)
        except Exception:  # noqa: BLE001
            pass

    def update(self, avg_goals_per_episode, train_step):
        """Call at each eval cycle. Returns True if stage advanced."""
        s = self.stages[self.stage_idx]
        if 'advance_goals' not in s:
            return False  # terminal stage — nothing to check
        threshold = float(s['advance_goals'])
        required = int(s.get('consecutive', 1))
        if avg_goals_per_episode >= threshold:
            self._consecutive_above += 1
            print(
                f"[curriculum] stage {self.stage_idx}: "
                f"goals/ep={avg_goals_per_episode:.2f} >= {threshold} "
                f"({self._consecutive_above}/{required} consecutive)",
                flush=True)
        else:
            if self._consecutive_above > 0:
                print(
                    f"[curriculum] stage {self.stage_idx}: "
                    f"goals/ep={avg_goals_per_episode:.2f} < {threshold}, "
                    f"resetting consecutive counter",
                    flush=True)
            self._consecutive_above = 0
        if self._consecutive_above >= required:
            self.stage_idx = min(self.stage_idx + 1, len(self.stages) - 1)
            self._consecutive_above = 0
            print(
                f"[curriculum] ADVANCING to stage {self.stage_idx}",
                flush=True)
            self._apply_stage(train_step)
            return True
        return False


def main(
    job_id="",
    num_envs=1,
    checkpoint_restore=False, 
    version=None,
    num_iterations_val=50000,
    pass_through_actions=False,
    initial_collect_steps_val=500,
    collect_steps_per_iteration_val=1,
    replay_buffer_capacity_val=75000,#5000,
    # BC-pretrain actor_net on the loaded expert demonstrations before
    # SAC takes over. SAC's actor loss only sees buffer actions
    # indirectly via the critic, so without this pre-step the actor
    # never directly imitates the expert and the historical
    # "starts-strong-then-drifts" curve flattens to mediocre. The
    # pre-existing call in collect_training_data (commented out when
    # this repo was created in commit dfef1f8) used 1000 steps; we
    # default higher because at batch_size=256 against ~50k expert
    # items each "epoch" is ~195 batches, so 1000 steps is only ~5
    # epochs - typically not enough to fully BC-fit a 512x512 actor.
    # Set to 0 to skip and run pure SAC.
    bc_pretrain_steps_val=5000,
    batch_size_val=256,
    critic_learning_rate_val=3e-5,
    actor_learning_rate_val=3e-5,
    alpha_learning_rate_val=3e-5,
    target_update_tau_val=0.005,
    target_update_period_val=1,
    gamma_val=0.99,
    reward_scale_factor_val=1.0,
    # ---- BC/RL replay-buffer composition ----------------------------
    # Controls the demo-prefill + demo-protection feature for BC+RL
    # research. All three default to back-compat values:
    #
    #   demo_prefill_count_val=50000 matches the trainer's previous
    #     hardcoded prefill cap (the 50k items_added break in the
    #     for-loop below).
    #   demo_min_keep_val=0 -> single-table mode: demos go into the
    #     ONLINE buffer alongside RL data and FIFO-displace over time,
    #     identical to the trainer's pre-Tier2 behavior.
    #   demo_sample_ratio_val=0.0 -> irrelevant in single-table mode.
    #
    # Set demo_min_keep_val > 0 from an experiment_design to switch on
    # the protected two-table mode. See replay.py::make_local_replay
    # for the algorithmic specifics + DQfD/DDPGfD paper references.
    demo_prefill_count_val=50000,
    demo_min_keep_val=0,
    demo_sample_ratio_val=0.0,
    # List of /tfrecords/job_<id> directories to load expert demonstrations
    # from, all concatenated into ONE combined expert dataset before prefill
    # + BC pretrain (see the expert_tfrecord_load phase below). None (the
    # default) preserves the original single-hardcoded-job behavior. Added
    # 2026-07-20 so a TRAIN job can combine multiple DEMO collection runs
    # (e.g. the long-standing default expert set alongside a fresh,
    # verified-clean post-bugfix collection) rather than being limited to
    # exactly one job's data.
    demo_record_dirs_val=None,
    # Optional list of EXACT per-source target step counts, parallel to
    # demo_record_dirs_val (same order/length). Overrides the default
    # behavior of using every source's full row count proportionally
    # (see the _source_ids comment in expert_tfrecord_load below) -
    # instead each source is resampled to exactly its target count, with
    # replacement (oversampling) if the source has fewer real rows than
    # the target. Added 2026-07-21 to let e.g. a small 33,512-step
    # post-fix demo collection be balanced 50/50 against a much larger
    # 500,001-step default set rather than being drowned out
    # proportionally. None (default) preserves the original
    # proportional-to-availability behavior.
    demo_source_counts_val=None,
    # ---- AWAC (advantage-weighted BC regularization) ----------------
    # Off by default (awac_lambda_val=0.0 -> plain SAC, bit-identical).
    # When >0 AND demo_min_keep_val>0, the actor loss gains an advantage-
    # weighted BC term sampled from the protected demo table each step, so
    # the demos shape the POLICY directly (not just the critic). See
    # awac_sac_agent.AwacSacAgent. Lets the actor inherit the expert's
    # survival without the one-shot BC-pretrain degradation, while the
    # advantage weighting avoids copying the expert's slowness.
    awac_lambda_val=0.0,
    awac_beta_val=1.0,
    awac_weight_clip_val=20.0,
    awac_lambda_decay_steps_val=0,
    actor_fc_layer_params_x=512,
    actor_fc_layer_params_y=512,
    critic_joint_fc_layer_params_x=512,
    critic_joint_fc_layer_params_y=512,
    log_interval_val=5000,
    num_eval_episodes_val=10,
    eval_interval_val=5000,
    # Target fraction of training-loop WALL-CLOCK spent in eval (the rest is
    # collect/training). 0.25 => ~75% train / 25% eval, so the clients spend
    # most of their time in the collect phase where the rollout-viz fans
    # render. Set <=0 or >=1 to fall back to the old step%eval_interval gate.
    eval_time_fraction_val=0.25,
    # Fixed TRAINING wall-clock (seconds) between evals. When > 0 this takes
    # precedence over eval_time_fraction: an eval fires after every
    # eval_train_interval_sec of training time, independent of how long each
    # eval takes. Unlike the time-fraction budget (which scales the train gap
    # with eval duration), this stays a fixed cadence even as long-episode
    # evals lengthen - "eval every N minutes". 0 disables (use the fraction /
    # step gates). Overridden at runtime by the EVAL_TRAIN_INTERVAL_SEC env var.
    eval_train_interval_sec_val=0,
    policy_save_interval_val=50,
    model_type="SacAgent",
    # ---- Reward-design plumbing -------------------------------------
    # When the user submits a TRAIN job with a reward_design_id field,
    # do_job fetches the matching reward_designs document from MongoDB
    # and passes the relevant pieces through here. ``reward_design`` is
    # the whole document (dict) so we can stamp id+version+code onto
    # every saved-model record via add_model() below; ``None`` means
    # "use the course's default reward formulas" (existing behavior).
    reward_design=None,
    # ---- Resume plumbing --------------------------------------------
    # do_job sets this to True when it detects an existing Learner
    # checkpoint dir for the job, meaning the job was previously paused
    # via the dashboard (status PAUSE_REQUESTED -> PAUSED -> resume by
    # setting NOT_STARTED). On resume:
    #   * skip BC pretrain (actor weights are restored from the
    #     checkpoint, no point re-training the actor on demos and
    #     overwriting that work),
    #   * still run demo prefill + initial_collect_actor since the
    #     Reverb buffer wasn't persisted (cheap, ~5-10s),
    #   * the Learner's tf.train.Checkpoint auto-restores actor +
    #     critic + target_critic + optimizers + train_step on
    #     construction, so the training loop picks up from the
    #     saved train_step without any extra wiring here.
    # False on fresh starts (existing behaviour bit-identical).
    is_resume_val=False,
    # ---- Experiment-design plumbing ---------------------------------
    # Parallel to reward_design above: the experiment_designs document
    # the trainer-loop hyperparameters came from. do_job applies the
    # doc's fields onto our kwargs BEFORE the call here (via
    # experiment_designs.apply_to_main_kwargs), so by the time main()
    # is invoked, num_iterations_val / actor_learning_rate_val / etc.
    # already carry the design's values. We still pass the doc through
    # so add_model can stamp experiment_design_id + name + version on
    # every checkpoint, giving each model record full provenance over
    # which (reward_design, experiment_design) pair produced it. None
    # means "trainer defaults" - the same effect as the canonical
    # "Default" seeded design.
    experiment_design=None,
    # ---- Track / curriculum plumbing (STAGED) -----------------------
    # corner_radius_val + curvature_difficulty_val come from the
    # experiment_designs SCHEMA's "Track / environment (curriculum)"
    # section and are overlaid here by apply_to_main_kwargs just like
    # any other design field. They are deliberately INERT today: the
    # course is hardcoded to "donut" (see build_train_env below) and
    # there is no Python->Unity channel for track geometry yet, so we
    # only record + log them. The follow-up that wires the ROS
    # ApplyForce message + SimController->TrackGenerator bridge will
    # thread these into the env reset. Accepting the kwargs now keeps
    # designs/jobs that carry these fields from raising a TypeError
    # when main() is invoked.
    curriculum_stages_val=None,
    # Curriculum starting stage index (0-based). 0 = normal bottom-up
    # curriculum. Set to len(curriculum_stages)-1 to start (and stay) on the
    # final/hardest geometry - useful to fine-tune a warm-started policy
    # directly on the terminal stage without re-climbing the ladder. Clamped
    # into [0, n_stages-1] by CurriculumScheduler.
    curriculum_start_stage_val=0,
    corner_radius_val=10.0,
    curvature_difficulty_val=0.0,
    # Per-edge absolute chicane counts (2026-07-18), applied when there is
    # no curriculum_stages_val (a fixed-geometry job) - the curriculum path
    # below reads these same fields per-stage from curriculum_stages_val
    # instead. curvature_difficulty_val above is retained for logging/
    # back-compat only; these are what actually drive chicane placement.
    chicanes_north_val=0,
    chicanes_east_val=0,
    chicanes_south_val=0,
    chicanes_west_val=0,
    # Per-step env discount returned by the course on non-terminal steps;
    # compounds with gamma_val (effective discount = gamma * env_discount).
    # Default 0.90 preserves legacy behavior; set 1.0 (via an experiment
    # design) to let gamma alone govern the horizon - matters for speed/lap
    # objectives where the ~9-step legacy horizon is too myopic.
    env_discount_val=0.90,
    # ---- Seed plumbing ----------------------------------------------
    # Optional RNG seed used for the bit-identical validation
    # procedure (see rl_agent/reward_designs.py docs). When set, we
    # seed every RNG we know about (Python's random, numpy, TF) so
    # paired single-env training runs are reproducible. The env's
    # subprocess-level seed gets propagated via the standard
    # tf-agents ParallelPyEnvironment seeding path when num_envs > 1.
    seed=None):

    #tempdir = tempfile.gettempdir()
    tempdir = "/tmp/active/"
    env_name = "NiryoPoleCart-v0" # @param {type:"string"}
    #tf.debugging.experimental.enable_dump_debug_info(tempdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    num_iterations=num_iterations_val # @param {type:"integer"}
    initial_collect_steps = initial_collect_steps_val # @param {type:"integer"}
    collect_steps_per_iteration = collect_steps_per_iteration_val # @param {type:"integer"}
    replay_buffer_capacity = replay_buffer_capacity_val # @param {type:"integer"}
    batch_size = batch_size_val # @param {type:"integer"}
    critic_learning_rate = critic_learning_rate_val # @param {type:"number"}
    actor_learning_rate = actor_learning_rate_val # @param {type:"number"}
    alpha_learning_rate = alpha_learning_rate_val # @param {type:"number"}
    target_update_tau = target_update_tau_val # @param {type:"number"}
    target_update_period = target_update_period_val # @param {type:"number"}
    gamma = gamma_val # @param {type:"number"}
    reward_scale_factor = reward_scale_factor_val # @param {type:"number"}
    # BC/RL replay composition (Tier 2). Defaults preserve the trainer's
    # pre-Tier2 single-table behavior; see the kwarg comments in main()'s
    # signature for the full rationale.
    demo_prefill_count = int(demo_prefill_count_val) if demo_prefill_count_val is not None else 50000
    demo_min_keep = int(demo_min_keep_val) if demo_min_keep_val is not None else 0
    demo_sample_ratio = float(demo_sample_ratio_val) if demo_sample_ratio_val is not None else 0.0
    actor_fc_layer_params = (actor_fc_layer_params_x, actor_fc_layer_params_y)
    critic_joint_fc_layer_params = (critic_joint_fc_layer_params_x, critic_joint_fc_layer_params_y)
    log_interval = log_interval_val # @param {type:"integer"}
    num_eval_episodes = num_eval_episodes_val # @param {type:"integer"}
    eval_interval = eval_interval_val # @param {type:"integer"}
    policy_save_interval = policy_save_interval_val # @param {type:"integer"}
    # Track / curriculum knobs are applied live: configure_env() below stamps
    # them onto every env, and the course forwards them to Unity's
    # TrackGenerator on each episode reset. Log the effective values for
    # provenance (a non-default value here = a non-default procedural track).
    print(
        f"[track-curriculum] corner_radius={corner_radius_val}, "
        f"curvature_difficulty={curvature_difficulty_val}, "
        f"chicanes(N/E/S/W)={chicanes_north_val}/{chicanes_east_val}/"
        f"{chicanes_south_val}/{chicanes_west_val} "
        f"(applied on each Unity reset by TrackGenerator).",
        flush=True)
    # ---- Startup phase timing (measurement only) --------------------
    # Lightweight wall-clock instrumentation for the cold-start /
    # restart path. _phase(label) closes the currently-open phase
    # (attributing all wall-time since the previous _phase call to it)
    # and opens a new one; _dump_startup_timings() prints the
    # accumulated breakdown right before the training loop begins.
    # Pure measurement - no control-flow or data changes, so training
    # behavior is bit-identical with or without these calls.
    _startup_timings = []
    _phase_state = {"label": None, "t0": time.time()}

    def _phase(label):
        now = time.time()
        if _phase_state["label"] is not None:
            _startup_timings.append(
                (_phase_state["label"], now - _phase_state["t0"]))
        _phase_state["label"] = label
        _phase_state["t0"] = now

    def _dump_startup_timings():
        # Close the final open phase, then print a padded table.
        _phase(None)
        if not _startup_timings:
            return
        total = sum(d for _, d in _startup_timings)
        width = max(len(name) for name, _ in _startup_timings)
        print("\n[startup-timing] cold-start phase breakdown "
              "(restart -> first TRAIN iteration):", flush=True)
        for name, dur in _startup_timings:
            pct = (100.0 * dur / total) if total > 0 else 0.0
            print(f"[startup-timing]   {name.ljust(width)}  "
                  f"{dur:8.2f}s  ({pct:5.1f}%)", flush=True)
        print(f"[startup-timing]   {'TOTAL'.ljust(width)}  "
              f"{total:8.2f}s  (100.0%)\n", flush=True)

    _phase("env_spawn (Unity handshake)")
    # Environment. Single-env runs use one env for both collect and eval.
    env = build_train_env(num_envs, course_type="donut")
    # Dedicated single-gym EVAL env (option-a, 2026-07-25). In multi-env runs
    # the collect env is a ParallelPyEnvironment that steps ALL gyms in
    # lockstep; reusing it for eval meant the eval phase hammered every client
    # at max rate (no learner delay between steps) through that lockstep
    # barrier, so any single client's hiccup stalled them all -> the pauses and
    # crashes that biased eval metrics (spuriously short episodes). Eval is a
    # measurement, not a throughput task, so we run it on ONE client
    # (ros-server-0) while the collect ParallelPyEnvironment keeps BOTH gyms for
    # throughput. Two RobotApi connections to ros-server-0 (this eval env +
    # collect worker-0) coexist fine: ROS pub/sub allows multiple
    # subscribers/publishers and collection is PAUSED during eval, so only one
    # side drives Unity at a time. Caveat: right after an eval, collect
    # worker-0's course tracking (target goal / steps_since_last_goal) is briefly
    # out of sync with the car's eval-moved position; it self-corrects on that
    # gym's next episode reset (~1 short episode). Single-env runs are unchanged.
    if num_envs > 1:
        eval_env = make_env('ros-server-0:50051', course_type="donut")
    else:
        eval_env = env
    # Bookkeeping that used to live on env.course; tracking it in main() lets
    # us share the same code path for single- vs multi-env training (in
    # multi-env mode the per-subprocess course state isn't reachable from
    # main).
    avg_return_arr = []
    max_avg_return = 0.0
    # Resume-aware seeding of max_avg_return. The is_new_max_avg gate
    # below only saves a new Model record when the eval's
    # AverageReturn beats this high-water mark. Initializing to 0.0
    # on every main() entry would mean that the FIRST eval after a
    # pause/resume (or after a crash recovery) always saves a model
    # record, even if its avg_return is lower than the historical
    # best from a previous resume slice. The result: dozens of
    # "false best" model rows per job (we saw 20 for job
    # 6a13c4946a957f1c5552cd47 on 2026-05-25, only one of which was
    # actually globally best).
    #
    # Fix: when this is a resume, query Mongo for the existing
    # models linked to this job_id and seed max_avg_return to the
    # actual historical best. New saves below will then only fire
    # on a true new global best, regardless of how many resume
    # slices the run was broken into.
    if is_resume_val and job_id is not None:
        try:
            historical_max = 0.0
            for prev_model in db.models.find(
                    {"job_id": str(job_id)},
                    {"avg_return": 1, "_id": 0}):
                v = prev_model.get("avg_return")
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v > historical_max:
                    historical_max = v
            if historical_max > 0.0:
                max_avg_return = historical_max
                print(
                    f"main: RESUME - seeding max_avg_return from "
                    f"existing models for job {job_id}: "
                    f"max_avg_return={max_avg_return:.4f}. Subsequent "
                    f"model checkpoints will only be saved on a TRUE "
                    f"new global best (not a per-slice local best).",
                    flush=True)
            else:
                print(
                    f"main: RESUME - no historical models found for "
                    f"job {job_id}; max_avg_return stays at 0.0.",
                    flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"main: RESUME - failed to seed max_avg_return: {_e}",
                  flush=True)
    print(f"Job arguments = num_envs: {num_envs}, num_iterations: {num_iterations}, nn_size_x: {actor_fc_layer_params_x}, nn_size_x: {actor_fc_layer_params_y}")
    learner_dir = os.path.join(tempdir, str(job_id),"learner")
    saved_model_dir = os.path.join(learner_dir, learner.POLICY_SAVED_MODEL_DIR)
    log_dir = os.path.join(tempdir, str(job_id),"metrics")
    train_dir=os.path.join(tempdir, str(job_id),"train")
    eval_dir=os.path.join(tempdir, str(job_id),"eval")
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()
    _phase("strategy+config+reward")
    # Strategy
    use_gpu = True #@param {type:"boolean"}
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    # Existing code unconditionally overwrote pass_through_actions to False
    # immediately after assigning the requested value, so the requested
    # value never took effect. Preserve that here by passing False directly.
    # Resolve initial track geometry. If curriculum_stages_val is provided,
    # use the first stage's geometry; otherwise use the fixed design values.
    _curriculum_stages = None
    # Starting stage index (curriculum_start_stage). Resolved once here so the
    # PRE-LOOP env geometry below and the CurriculumScheduler both key off the
    # same stage. Clamped after the stage list is known.
    _curriculum_start = 0
    if curriculum_stages_val:
        try:
            import json as _json
            if isinstance(curriculum_stages_val, str):
                _curriculum_stages = _json.loads(curriculum_stages_val)
            else:
                _curriculum_stages = list(curriculum_stages_val)
            try:
                _curriculum_start = int(curriculum_start_stage_val)
            except (TypeError, ValueError):
                _curriculum_start = 0
            _curriculum_start = max(
                0, min(_curriculum_start, len(_curriculum_stages) - 1))
            _init_cr = float(_curriculum_stages[_curriculum_start].get('corner_radius', corner_radius_val))
            _init_cd = float(_curriculum_stages[_curriculum_start].get('curvature_difficulty', curvature_difficulty_val))
            _init_ch_n = int(_curriculum_stages[_curriculum_start].get('chicanes_north', chicanes_north_val))
            _init_ch_e = int(_curriculum_stages[_curriculum_start].get('chicanes_east', chicanes_east_val))
            _init_ch_s = int(_curriculum_stages[_curriculum_start].get('chicanes_south', chicanes_south_val))
            _init_ch_w = int(_curriculum_stages[_curriculum_start].get('chicanes_west', chicanes_west_val))
            print(f"[curriculum] enabled: {len(_curriculum_stages)} stages, "
                  f"starting at stage {_curriculum_start}/"
                  f"{len(_curriculum_stages) - 1}: corner_radius={_init_cr}, "
                  f"curvature_difficulty={_init_cd}, "
                  f"chicanes(N/E/S/W)={_init_ch_n}/{_init_ch_e}/{_init_ch_s}/{_init_ch_w}",
                  flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[curriculum] failed to parse curriculum_stages_val: {_e}; "
                  f"falling back to fixed track geometry.", flush=True)
            _curriculum_stages = None
            _curriculum_start = 0
    if _curriculum_stages:
        _init_cr = float(_curriculum_stages[_curriculum_start].get('corner_radius', corner_radius_val))
        _init_cd = float(_curriculum_stages[_curriculum_start].get('curvature_difficulty', curvature_difficulty_val))
        _init_ch_n = int(_curriculum_stages[_curriculum_start].get('chicanes_north', chicanes_north_val))
        _init_ch_e = int(_curriculum_stages[_curriculum_start].get('chicanes_east', chicanes_east_val))
        _init_ch_s = int(_curriculum_stages[_curriculum_start].get('chicanes_south', chicanes_south_val))
        _init_ch_w = int(_curriculum_stages[_curriculum_start].get('chicanes_west', chicanes_west_val))
    else:
        _init_cr = corner_radius_val
        _init_cd = curvature_difficulty_val
        _init_ch_n = chicanes_north_val
        _init_ch_e = chicanes_east_val
        _init_ch_s = chicanes_south_val
        _init_ch_w = chicanes_west_val
    configure_env(env, job_id=job_id, pass_through_actions=False,
                  corner_radius=_init_cr,
                  curvature_difficulty=_init_cd,
                  chicanes_north=_init_ch_n, chicanes_east=_init_ch_e,
                  chicanes_south=_init_ch_s, chicanes_west=_init_ch_w,
                  env_discount=env_discount_val)
    # Mirror the same job config onto the dedicated eval env (multi-env runs).
    # The curriculum scheduler keeps them in sync on later stage changes; this
    # sets the initial stage-0 geometry / discount before the first eval.
    if eval_env is not env:
        configure_env(eval_env, job_id=job_id, pass_through_actions=False,
                      corner_radius=_init_cr,
                      curvature_difficulty=_init_cd,
                      chicanes_north=_init_ch_n, chicanes_east=_init_ch_e,
                      chicanes_south=_init_ch_s, chicanes_west=_init_ch_w,
                      env_discount=env_discount_val)
    print(f"pass_through_actions: False")
    print(f"env_discount={env_discount_val} (effective gamma*env_discount "
          f"= {gamma_val * env_discount_val:.4f})", flush=True)

    # Seed every RNG we know about so bit-identical comparison runs are
    # reproducible when the caller supplies a fixed `seed`. Only runs
    # for single-env training; multi-env (--num-envs > 1) has its own
    # subprocess-level seed dispatch which we don't try to coordinate
    # here. Pass `seed=None` (the default) to keep the historical
    # nondeterministic behavior.
    if seed is not None:
        try:
            import random
            random.seed(int(seed))
            np.random.seed(int(seed))
            tf.random.set_seed(int(seed))
            print(f"main: seeded RNGs with seed={seed}", flush=True)
        except Exception as e:
            print(f"main: failed to seed RNGs ({e}); continuing without seed",
                  flush=True)

    # Install the reward design on the env's course (single-env via
    # direct call; multi-env dispatches through ParallelPyEnvironment).
    # If this fails we let the exception propagate up to do_job which
    # records it as job.eval_error and marks the job DONE - no point
    # training a model whose reward formula didn't even load.
    if reward_design and reward_design.get("code"):
        rd_name = reward_design.get("name", "<unnamed>")
        try:
            result = install_reward_design_on_env(
                env, rd_name, reward_design["code"])
            # Surface the FULL install result so operators can confirm
            # which scalar reward methods got monkey-patched (and which
            # were left as course defaults). Empty list = nothing got
            # patched, which technically counts as success here only
            # because load_reward_design would have raised if it could
            # find no functions at all - but we make it visible so the
            # user can tell at a glance.
            installed = (result.get("installed") if isinstance(result, dict) else []) or []
            actors = result.get("actors") if isinstance(result, dict) else 1
            penalty = result.get("penalty_reward") if isinstance(result, dict) else None
            if installed:
                print(
                    f"main: installed reward design name={rd_name!r} "
                    f"version={reward_design.get('version')} "
                    f"installed={installed} "
                    f"actors={actors} "
                    f"penalty_reward={penalty}",
                    flush=True)
            else:
                # Loud warning so a no-op install (nothing was patched
                # despite a design being selected) doesn't pass silently.
                # This can happen if a future change loosens
                # load_reward_design to tolerate signature mismatches
                # by skipping them - operators should still see it here.
                print(
                    f"main: WARNING reward design name={rd_name!r} "
                    f"version={reward_design.get('version')} loaded but "
                    f"no functions were patched (installed=[], actors={actors}). "
                    f"The course is still using its default reward formulas. "
                    f"Check that your design defines reward_standard / "
                    f"reward_success / reward_failure with the expected signatures.",
                    flush=True)
        except Exception as e:
            print(f"main: reward design {rd_name!r} failed to install: {e}",
                  flush=True)
            raise
    # Critic network.
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(env))

    # Publish the env's current specs to MongoDB so the dashboard's
    # Models-tab "Compat" column can mark rows that no longer match
    # the live env. Done once per TRAIN job start (upsert; the doc is
    # keyed by robot_type so multiple training runs against the same
    # robot are idempotent).
    publish_env_spec(env)

    _phase("network_build (actor+critic)")
    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
    
    # Actor network.
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

    _phase("expert_tfrecord_load (500k records)")
    # demo_record_dirs_val lets a job combine MULTIPLE DEMO collections into
    # one expert dataset (see the kwarg's docstring above); None falls back
    # to the single long-standing default job's directory, unchanged from
    # the original hardcoded behavior.
    record_dirs = demo_record_dirs_val or ['/tfrecords/job_64168c1b58d4d8ccdb76e721']
    # 1. Load each directory's demos separately, then concatenate into ONE
    # Trajectory object. Only `observation`/`action` actually vary by
    # source - `step_type`/`next_step_type`/`reward`/`discount` are the
    # same synthetic length-1 constants read_files_from_directory always
    # produces (see collect_training_data.convert_tfrecord_to_trajectory),
    # so it's safe to keep just one copy of those and stretch later via
    # match_length below.
    # Acceleration-floor for expert demos: drop every transition whose
    # commanded acceleration (action[:, 0] - the force channel, see
    # collect_expert_demos' `np.array([action_apply_force,
    # action_steering_angle])`) is below the current action_spec minimum.
    # Added 2026-07-22 alongside pulling the acceleration floor from -1.0
    # to -0.01: the TrackGen demo collection (job 6a5ec0d1...) was recorded
    # while min_force was -1.0, so it contains hard active-braking
    # transitions down to -1.0 that are now OUTSIDE the policy's action
    # range - feeding them into prefill/BC would train the actor toward
    # targets it can no longer emit.
    #
    # Raised -0.01 -> 0.05 (2026-07-24) to track the action_spec floor moving
    # to a positive 0.05 (see DonutCourse.action_spec). This now also drops
    # every near-idle transition (0 <= force < 0.05) from ANY reused legacy
    # set - including the old w-course [0.001, 0.2] data whose median force
    # was ~0.012, i.e. most of it - which is intentional: those idle rows are
    # exactly the crawl prior we're eliminating. Freshly-collected demos use
    # min_force=0.2 (> 0.05) so they lose 0 rows. The per-source "dropped N"
    # log below makes which source was affected explicit. Keep in sync with
    # DonutCourse.action_spec's minimum[0].
    _DEMO_MIN_ACCEL = 0.05
    _per_dir_trajectories = []
    _per_dir_counts = []
    for _record_dir in record_dirs:
        _traj = collect_training_data.read_files_from_directory(_record_dir)
        _n_loaded = int(tf.shape(_traj.observation)[0])
        # .numpy() so the boolean mask below works whether
        # read_files_from_directory handed back tf.Tensors or numpy arrays
        # (same reason the resampling block downstream does this).
        _obs_np = (_traj.observation.numpy()
                   if hasattr(_traj.observation, "numpy")
                   else _traj.observation)
        _act_np = (_traj.action.numpy()
                   if hasattr(_traj.action, "numpy")
                   else _traj.action)
        _keep_mask = _act_np[:, 0] >= _DEMO_MIN_ACCEL
        _n_steps = int(_keep_mask.sum())
        _n_dropped = _n_loaded - _n_steps
        if _n_dropped > 0:
            # Only reconstruct (and only touch observation/action) when we
            # actually drop rows. step_type/next_step_type/reward/discount
            # are the synthetic length-constants read_files_from_directory
            # produces and are stretched to length downstream via
            # match_length - NOT per-row arrays - so they're passed through
            # unchanged, exactly like the resampling block below does.
            _traj = trajectory.Trajectory(
                step_type=_traj.step_type,
                observation=_obs_np[_keep_mask],
                action=_act_np[_keep_mask],
                policy_info=(),
                next_step_type=_traj.next_step_type,
                reward=_traj.reward,
                discount=_traj.discount,
            )
        print(f"Loaded {_n_steps} expert steps from {_record_dir} "
              f"(dropped {_n_dropped} of {_n_loaded} with acceleration < "
              f"{_DEMO_MIN_ACCEL})", flush=True)
        _per_dir_trajectories.append(_traj)
        _per_dir_counts.append(_n_steps)

    # Optional balanced/oversampled per-source resampling (2026-07-21): see
    # demo_source_counts_val's docstring above. Runs BEFORE concatenation so
    # every source is independently resampled to its own EXACT target count
    # (fixed seed per-source index for reproducibility) - sources with fewer
    # real rows than their target are oversampled WITH replacement, sources
    # with more are subsampled WITHOUT replacement. Everything downstream
    # (concatenation, _source_ids, the full shuffle, the prefill loop's
    # demo_prefill_count cap) is unchanged and just sees the resampled
    # per-dir trajectories/counts as if that's how many rows always existed.
    if demo_source_counts_val:
        if len(demo_source_counts_val) != len(record_dirs):
            raise ValueError(
                f"demo_source_counts_val has {len(demo_source_counts_val)} "
                f"entries but there are {len(record_dirs)} record_dirs: "
                f"{record_dirs}")
        _resample_rng = np.random.RandomState(42)
        for _i, _target_n in enumerate(demo_source_counts_val):
            _avail_n = _per_dir_counts[_i]
            _target_n = int(_target_n)
            _with_replacement = _target_n > _avail_n
            if _with_replacement:
                print(
                    f"Oversampling {record_dirs[_i]}: {_avail_n} real steps "
                    f"-> {_target_n} target (WITH replacement, "
                    f"{_target_n / _avail_n:.2f}x)", flush=True)
            else:
                print(
                    f"Subsampling {record_dirs[_i]}: {_avail_n} real steps "
                    f"-> {_target_n} target (without replacement)",
                    flush=True)
            _idx = _resample_rng.choice(
                _avail_n, size=_target_n, replace=_with_replacement)
            _src_traj = _per_dir_trajectories[_i]
            # trajectory.first() (inside read_files_from_directory) converts
            # the numpy observation/action arrays into tf.Tensors, which do
            # NOT support numpy-style fancy indexing with an arbitrary int
            # array via `[...]` (only slices/scalars) - must go through
            # .numpy() first (no-op if already a numpy array) so the
            # fancy-index gather below works regardless of which type
            # read_files_from_directory happened to hand back.
            _obs_np = (_src_traj.observation.numpy()
                       if hasattr(_src_traj.observation, "numpy")
                       else _src_traj.observation)
            _act_np = (_src_traj.action.numpy()
                       if hasattr(_src_traj.action, "numpy")
                       else _src_traj.action)
            _per_dir_trajectories[_i] = trajectory.Trajectory(
                step_type=_src_traj.step_type,
                observation=_obs_np[_idx],
                action=_act_np[_idx],
                policy_info=(),
                next_step_type=_src_traj.next_step_type,
                reward=_src_traj.reward,
                discount=_src_traj.discount,
            )
            _per_dir_counts[_i] = _target_n

    expert_trajectories = trajectory.Trajectory(
        step_type=_per_dir_trajectories[0].step_type,
        observation=np.concatenate(
            [t.observation for t in _per_dir_trajectories], axis=0),
        action=np.concatenate(
            [t.action for t in _per_dir_trajectories], axis=0),
        policy_info=(),
        next_step_type=_per_dir_trajectories[0].next_step_type,
        reward=_per_dir_trajectories[0].reward,
        discount=_per_dir_trajectories[0].discount,
    )
    print(f"Combined {len(record_dirs)} demo source(s) into "
          f"{expert_trajectories.observation.shape[0]} total expert steps: "
          f"{record_dirs}", flush=True)
    print(f"Loaded trajectories shape: {expert_trajectories.step_type.shape}")
    # Per-row source attribution (2026-07-21): remembers which record_dir
    # each concatenated row came from, so downstream consumers - chiefly the
    # demo_prefill loop below - can report a verifiable per-source
    # breakdown of what actually ended up in Reverb, not just what was
    # loaded into memory. THE BUG THIS FIXES: demo_prefill_count (e.g.
    # 50000) is almost always far smaller than the first source's row
    # count (e.g. 500001), and trajectory_dataset was previously consumed
    # in raw concatenation order with NO shuffle - so with 2+ sources
    # concatenated one-after-another, the prefill loop's `break` after
    # demo_prefill_count items would exhaust its budget entirely within
    # source #1 and NEVER reach source #2's rows at all. The combined
    # "loaded" log above was accurate, but the ACTUAL Reverb demo table
    # silently contained 100% source #1 / 0% every other source. Shuffling
    # below (with source ids carried alongside) fixes this by making the
    # first demo_prefill_count items an unbiased random draw across ALL
    # sources instead of a prefix of source #1.
    _source_ids = np.concatenate([
        np.full(_n, _i, dtype=np.int32)
        for _i, _n in enumerate(_per_dir_counts)])

    # 2. Find our target length (e.g., 500001) based on the observation tensor
    num_steps = tf.shape(expert_trajectories.observation)[0]
    # Helper function to stretch length-1 tensors to match num_steps
    def match_length(tensor, target_length):
        if tensor.shape[0] == 1 and target_length > 1:
            return tf.repeat(tensor, target_length, axis=0)
        return tensor

    # 3. Create a new, shape-aligned Trajectory object
    aligned_trajectories = trajectory.Trajectory(
        step_type=match_length(expert_trajectories.step_type, num_steps),
        observation=expert_trajectories.observation,
        action=expert_trajectories.action,
        policy_info=(), # Keep empty
        next_step_type=match_length(expert_trajectories.next_step_type, num_steps),
        reward=match_length(expert_trajectories.reward, num_steps),
        discount=match_length(expert_trajectories.discount, num_steps)
    )

    # 4. Now slice the perfectly aligned trajectories!
    trajectory_dataset = tf.data.Dataset.from_tensor_slices(aligned_trajectories)

    # Zip in the per-row source id and shuffle BOTH together (same
    # permutation applied to each, since zip keeps them aligned element-
    # for-element) - see the _source_ids comment above for why this must
    # happen before anything takes only a prefix of this dataset.
    # buffer_size = the full row count so this is a true full-dataset
    # shuffle, not a sliding-window approximation - the underlying arrays
    # are already fully materialized in memory (they came from
    # np.concatenate above), so this costs no extra I/O, just an
    # index permutation. Fixed seed => reproducible prefill/BC-pretrain
    # composition across runs of the same job for easier debugging.
    # reshuffle_each_iteration=False so multiple full passes (e.g. BC
    # pretrain's many epochs over this same dataset) see a STABLE order
    # rather than re-shuffling every epoch, matching this dataset's
    # original (unshuffled-but-fixed-order) semantics as closely as
    # possible while still fixing the source-starvation bug.
    _dataset_with_source = tf.data.Dataset.zip((
        trajectory_dataset, tf.data.Dataset.from_tensor_slices(_source_ids)
    )).shuffle(buffer_size=int(num_steps), seed=42, reshuffle_each_iteration=False)
    # Downstream consumers that only want the Trajectory (BC pretrain, the
    # AWAC demo iterator, etc.) get a plain projection that preserves the
    # SAME shuffled order - so every consumer of `trajectory_dataset` from
    # this point on is implicitly drawing from the same shuffled pool the
    # prefill loop below verifies against.
    trajectory_dataset = _dataset_with_source.map(lambda traj, _sid: traj)

    # 5. Add a batch dimension of 1 for Reverb
    #batched_parsed_dataset = trajectory_dataset.batch(1)
 

    # parsed_dataset = collect_training_data.get_parsed_dataset(file)
    
    # collect_training_data.train_agent_sampling(
    #     actor_net,
    #     record_dir, 
    #     training_steps=1000,
    #     sampling_fraction=0.1,
    #     parsed_dataset=parsed_dataset)


    _phase("agent_build (SAC)")
    # Agent.
    with strategy.scope():
        train_step = train_utils.create_train_step()

        # AWAC is opt-in via experiment-design config (awac_lambda_val>0) and
        # requires the protected demo table (demo_min_keep>0) to sample expert
        # transitions for the advantage-weighted BC term. Off => plain SacAgent
        # (bit-identical to before). The demo iterator is attached AFTER
        # make_local_replay builds the demo table (it needs collect_data_spec).
        _awac_on = (awac_lambda_val is not None and float(awac_lambda_val) > 0.0
                    and demo_min_keep > 0)
        _agent_kwargs = dict(
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step,
            debug_summaries=True,
            summarize_grads_and_vars=True,
        )
        if _awac_on:
            from awac_sac_agent import AwacSacAgent
            tf_agent = AwacSacAgent(
                time_step_spec,
                action_spec,
                awac_lambda=float(awac_lambda_val),
                awac_beta=float(awac_beta_val),
                awac_weight_clip=float(awac_weight_clip_val),
                awac_lambda_decay_steps=int(awac_lambda_decay_steps_val),
                awac_lambda_min=0.0,
                demo_iter=None,
                **_agent_kwargs)
            print(f"main: AWAC ENABLED (lambda={float(awac_lambda_val)}, "
                  f"beta={float(awac_beta_val)}, "
                  f"weight_clip={float(awac_weight_clip_val)}, "
                  f"decay_steps={int(awac_lambda_decay_steps_val)})", flush=True)
        else:
            tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                **_agent_kwargs)

        tf_agent.initialize()
    _phase("reverb_setup")
    # Replay Buffer.
    # collect_observer is fan-out-aware: in multi-env mode it splits the
    # batched Trajectory produced by ParallelPyEnvironment into N
    # per-env writes. expert_observer is always plain-unbatched and is
    # what the offline expert-demo loop below feeds directly. In
    # single-env mode the two are the same instance.
    # Reverb checkpointing directory: a fixed path inside the Learner dir
    # so pause/resume can save and restore the replay table contents
    # without searching for a random temp path. The directory is created
    # by make_local_replay when checkpointing_dir is set.
    _reverb_ckpt_dir = os.path.join(learner_dir, "reverb_checkpoint")
    # Periodic (not just on-pause) Reverb online-table checkpoint cadence.
    # Added 2026-07-22: previously the ONLY time the online replay buffer
    # was snapshotted was a graceful dashboard pause (see the
    # _lifecycle == 'pause' branch below); any ungraceful crash (a worker-
    # subprocess segfault, OOM-kill, host reboot) lost the entire online
    # buffer even though the Learner's own checkpoint_interval=100 kept
    # the network weights/train_step safe - discovered when job
    # 6a601d44a8350bdc89d19a60 crashed at train_step=177057 and resumed
    # with weights intact but buffer_size back near 0. 1000 matches
    # StepPerSecondLogTrigger's cadence and is cheap relative to a
    # 250k-step job; REVERB_PERIODIC_CHECKPOINT_KEEP bounds disk usage
    # since Reverb's checkpointer never prunes on its own (see
    # _prune_reverb_checkpoints).
    REVERB_PERIODIC_CHECKPOINT_INTERVAL = 1000
    REVERB_PERIODIC_CHECKPOINT_KEEP = 2
    (reverb_server, reverb_replay, dataset,
     rb_observer, expert_observer,
     demo_replay, demo_observer) = make_local_replay(
        tf_agent.collect_data_spec,
        capacity=replay_buffer_capacity,
        sample_batch_size=batch_size,
        sequence_length=2,
        stride_length=1,
        num_envs=num_envs,
        # Two-table demo-protected mode kicks in when demo_min_keep > 0.
        # Otherwise demo_capacity=0 keeps the original single-table
        # behavior bit-identical (demo prefill into the online table).
        demo_capacity=demo_min_keep,
        demo_sample_ratio=demo_sample_ratio,
        checkpointing_dir=_reverb_ckpt_dir)
    table_name = 'uniform_table'
    experience_dataset_fn = lambda: dataset
    # AWAC: attach a demo-only iterator so the agent can sample expert
    # transitions for its advantage-weighted BC term. Independent of the
    # mixed training `dataset` above (which stochastically blends demo/online
    # at demo_sample_ratio); AWAC needs a clean expert-only stream. Attached
    # here (after the demo table exists) and before the Learner first traces
    # actor_loss. No-op unless AWAC is on and the demo table was built.
    if _awac_on and demo_replay is not None:
        _demo_only_ds = demo_replay.as_dataset(
            sample_batch_size=batch_size, num_steps=2).prefetch(5)
        tf_agent.set_demo_iter(iter(_demo_only_ds))
        print("main: AWAC demo iterator attached (expert-only stream).",
              flush=True)
    # Log the resolved buffer composition so robotaxi.out makes the
    # active mode explicit even when the experiment_design overlay
    # changed defaults silently.
    if demo_min_keep > 0:
        print(
            f"main: replay buffer = TWO TABLES "
            f"(online cap={replay_buffer_capacity}, "
            f"demo cap={demo_min_keep}, "
            f"sample_ratio_from_demo={demo_sample_ratio:.3f}, "
            f"prefill_count={demo_prefill_count})", flush=True)
    else:
        print(
            f"main: replay buffer = SINGLE TABLE "
            f"(cap={replay_buffer_capacity}, demo prefill={demo_prefill_count}, "
            f"demos will FIFO-evict as RL data accumulates)", flush=True)
    
    # Policies
    # EVAL uses a GREEDY wrapper (2026-07-25). tf-agents' SacAgent sets both
    # `policy` and `collect_policy` to the SAME stochastic ActorPolicy, so
    # without this wrap eval would sample from the tanh-Gaussian with full
    # exploration noise - one unlucky sampled steering action can crash the car
    # mid-corner, so AverageReturn / episode-length carried the policy's
    # exploration VARIANCE rather than its learned competence (the spuriously
    # short eval episodes). GreedyPolicy takes the distribution's MODE
    # (tanh(mean)) so eval measures the deterministic learned behavior - lower
    # variance, more meaningful best-model selection. Collection still uses the
    # stochastic collect_policy below (exploration is required there).
    tf_eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)
    

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
        env.time_step_spec(), env.action_spec())
    
    # Actors. rb_observer is constructed by make_local_replay() above.
    # The expert demonstrations were saved as single-actor trajectories
    # and must go through the always-unbatched expert_observer; in
    # multi-env mode rb_observer is a fan-out that would slice into
    # leaves expecting a leading parallel-env batch dim.
    # Pick which observer the prefill loop feeds:
    #   * Two-table mode (demo_min_keep > 0): demos go into the
    #     dedicated demo table via demo_observer; the online table
    #     stays empty so the first RL collect_actor.run() finds it
    #     ready (Reverb's MinSize(1) rate-limiter is satisfied by the
    #     initial collect actor, not by prefill).
    #   * Single-table mode (demo_min_keep == 0): demos go into the
    #     ONLINE table via expert_observer - identical to the trainer's
    #     pre-Tier2 path.
    # The prefill cap is the experiment-design-controlled
    # demo_prefill_count (previously hardcoded 50000).
    if demo_observer is not None:
        prefill_target = demo_observer
        prefill_label = "DEMO table (protected)"
    else:
        prefill_target = expert_observer
        prefill_label = "ONLINE table (single-table mode)"
    # Resume optimisation. The demo Reverb tables are populated for
    # two reasons:
    #   * to feed BC pretrain (`bc_pretrain_actor_net`), and
    #   * to feed SAC's training dataset when demo_sample_ratio > 0
    #     (the DDPGfD-style demo over-sampling mode added in Tier 2).
    #
    # On RESUME, BC pretrain is skipped (the saved actor checkpoint
    # already contains its output). If demo_sample_ratio is also 0
    # (the default for the seeded canonical experiment design),
    # demos are never sampled during SAC either - so the prefill is
    # pure wasted disk I/O, typically 5-30s depending on demo size.
    # Skip it in that case so resume comes up quickly.
    #
    # The opposite case (resume + demo_sample_ratio > 0) DOES need
    # the prefill to repopulate the demo table that wasn't serialised
    # at pause (Option A: buffer state is not preserved across the
    # pause/resume boundary). Without it, SAC's mixed sampler would
    # underflow at every gradient step.
    _phase("demo_prefill (-> Reverb)")
    skip_demo_prefill = (
        is_resume_val
        and demo_prefill_count > 0
        and demo_sample_ratio <= 0.0
    )
    if skip_demo_prefill:
        print(
            f"RESUME: skipping demo prefill into {prefill_label}. "
            f"BC pretrain is skipped + demo_sample_ratio={demo_sample_ratio} "
            f"means demos are unused during SAC training; saves the "
            f"trajectory-load wall time.",
            flush=True)
        items_added = 0
    else:
        print(f"Loading expert demonstrations into Reverb -> {prefill_label} (cap={demo_prefill_count})...")
        items_added = 0
        # Tallies which record_dir (source) each PREFILLED row actually
        # came from - the verifiable, ground-truth answer to "does the
        # replay buffer really contain both sources", as opposed to just
        # trusting the "Combined N demo source(s)..." load-time log above
        # (which only proves the sources were concatenated in memory, not
        # that any given one survived the demo_prefill_count cutoff - see
        # the _source_ids comment above for the bug this closed).
        _prefill_source_counts = {i: 0 for i in range(len(record_dirs))}
        if demo_prefill_count > 0:
            for unbatched_traj, _source_id in _dataset_with_source:
                prefill_target(unbatched_traj)
                items_added += 1
                _prefill_source_counts[int(_source_id.numpy())] += 1
                if items_added >= demo_prefill_count:
                    break
                if items_added % 10000 == 0:
                    print(f"Batch trajectory {items_added} added")

        print(f"Successfully loaded {items_added} expert steps into Reverb ({prefill_label}).")
        if items_added > 0:
            _breakdown = ", ".join(
                f"{record_dirs[_i]}={_prefill_source_counts[_i]} "
                f"({100.0 * _prefill_source_counts[_i] / items_added:.1f}%)"
                for _i in range(len(record_dirs)))
            print(f"Prefill source breakdown ({prefill_label}): {_breakdown}",
                  flush=True)

    print_replay_buffer_size(reverb_replay,table_name,replay_buffer_capacity)
    # If we're in two-table mode, also log the demo table's size so
    # robotaxi.out shows the actual loaded count for both tables.
    if demo_replay is not None:
        try:
            demo_size = demo_replay.py_client.server_info()[
                'demo_table'].current_size
            print(f"Demo table: {demo_size}/{demo_min_keep} samples loaded.",
                  flush=True)
        except Exception as e:
            print(f"Could not read demo table size: {e}", flush=True)

    # BC pretrain actor_net on the expert demos before SAC takes over.
    # See collect_training_data.bc_pretrain_actor_net for the full
    # rationale; in short, pure SAC's actor loss does not directly
    # consume buffer actions, so without this step expert demos only
    # influence the actor indirectly via the critic and the policy
    # never imitates the expert before on-policy data dilutes the
    # buffer. tf_agent.actor_network is the same Python object as
    # actor_net, so weight updates here are visible to SAC.
    #
    # SKIPPED on resume: the actor weights already came back from the
    # Learner checkpoint constructed above (tf-agents' Learner auto-
    # restores from learner_dir/train/checkpoint on init). BC-
    # pretraining on top would overwrite that with another round of
    # imitation, wasting the saved policy progress.
    _phase("bc_pretrain")
    if bc_pretrain_steps_val > 0 and not is_resume_val:
        collect_training_data.bc_pretrain_actor_net(
            actor_net=actor_net,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            strategy=strategy,
            trajectory_dataset=trajectory_dataset,
            training_steps=bc_pretrain_steps_val,
            batch_size=batch_size)
    elif is_resume_val:
        # NOTE: we deliberately don't print int(train_step.numpy()) here:
        # the Learner that restores train_step from the checkpoint isn't
        # constructed until ~30 lines below this point. Reading the
        # counter here would always say 0 (its initial value), which
        # gave the misleading impression on 2026-05-25 that the restore
        # itself had failed. The restored value is printed after the
        # Learner constructor instead - search for "RESUME - preserving
        # restored train_step" below.
        print(
            f"main: RESUME - skipping BC pretrain (Learner checkpoint "
            f"will be auto-restored when the Learner is constructed "
            f"shortly).",
            flush=True)

    # ---- Reverb table restore (resume path) --------------------------------
    # On a clean pause the trainer calls reverb_server.localhost_client()
    # .checkpoint() which writes the full online table to _reverb_ckpt_dir.
    #
    # IMPORTANT (fixed 2026-07-22): the actual restore-from-disk already
    # happened (or didn't) automatically back at make_local_replay()'s
    # ``reverb.Server(tables, checkpointer=DefaultCheckpointer(path=
    # _reverb_ckpt_dir))`` call, several dozen lines above this point.
    # reverb's DefaultCheckpointer loads the latest valid checkpoint found
    # under its ``path`` at SERVER CONSTRUCTION time - there is no separate
    # "restore into an already-running server" step, and none is possible:
    # reverb.Client's actual methods are just checkpoint/insert/
    # mutate_priorities/reset/sample/server_address/server_info/
    # trajectory_writer/writer - `checkpoint()` is write-only, nothing
    # named set_checkpoint or similar exists to trigger a read. (Verified
    # empirically: constructing two Reverb servers back-to-back against the
    # same checkpoint dir, the second one's table already contains the
    # first's data immediately after construction, no extra call needed.)
    #
    # The previous version of this block called
    # ``reverb_server.localhost_client().set_checkpoint(_latest)``, which
    # always raised ``AttributeError: 'Client' object has no attribute
    # 'set_checkpoint'`` - meaning every resume, graceful pause or crash,
    # silently landed in the except branch and fell back to
    # initial_collect, since the day this restore feature was first added.
    # The block below is therefore pure DETECTION + LOGGING of whatever the
    # server already did at construction, using current_size as the only
    # reliable signal, rather than an action that triggers restoration.
    _reverb_restored = False
    if is_resume_val:
        try:
            _ckpt_entries = [
                f for f in os.listdir(_reverb_ckpt_dir)
                if os.path.isdir(os.path.join(_reverb_ckpt_dir, f))
            ] if os.path.isdir(_reverb_ckpt_dir) else []
            if _ckpt_entries:
                _restored_size = reverb_replay.py_client.server_info()[
                    'uniform_table'].current_size
                if _restored_size > 0:
                    print(
                        f"main: RESUME - Reverb online table auto-restored "
                        f"at server construction from "
                        f"{sorted(_ckpt_entries)[-1]} "
                        f"({_restored_size} items).",
                        flush=True)
                    _reverb_restored = True
                else:
                    # Checkpoint entries exist on disk but current_size is 0
                    # - the auto-restore-at-construction above didn't pick
                    # them up (e.g. a table config/schema mismatch between
                    # write time and this run). Report this explicitly
                    # rather than claiming success with an empty buffer.
                    print(
                        f"main: RESUME - Reverb checkpoint entries exist in "
                        f"{_reverb_ckpt_dir} but the online table is empty "
                        f"after server construction (auto-restore didn't "
                        f"take); falling back to initial_collect.",
                        flush=True)
            else:
                print(
                    "main: RESUME - no Reverb checkpoint found in "
                    f"{_reverb_ckpt_dir}; will run initial_collect.",
                    flush=True)
        except Exception as _e:  # noqa: BLE001
            print(
                f"main: RESUME - Reverb restore check failed ({_e}); "
                "falling back to initial_collect.",
                flush=True)

    _phase("initial_collect")
    if _reverb_restored:
        # Buffer already warm from restored checkpoint — skip initial_collect.
        # The restored online table is large enough that Reverb's MinSize(1)
        # rate-limiter is satisfied and SAC sampling starts immediately.
        print(
            "main: Skipping initial_collect — Reverb online table already "
            f"warm from restored checkpoint.",
            flush=True)
    else:
        initial_collect_actor = actor.Actor(
            env,
            random_policy,
            train_step,
            steps_per_run=initial_collect_steps,
            observers=[rb_observer])
        print("initial_collect_actor.run() :)")
        initial_collect_actor.run()
        print("Initial collection done")

    env_step_metric = py_metrics.EnvironmentSteps()
    print("number of steps: " + str(env_step_metric.result()))
    collect_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=train_dir,
        observers=[rb_observer, env_step_metric])
    
    _phase("learner+eval_actor build")
    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=eval_dir)
    
    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    # checkpoint_interval bounds the worst-case progress lost to a
    # hard kill (Docker stop without grace, host OOM, Windows-Update
    # reboot, etc.). tf-agents' Learner DEFAULTS this to 100_000 train
    # steps - way bigger than our typical 5_000-iter job, so the
    # automatic Checkpointer trigger never fires and a crash mid-run
    # wipes the entire run's progress. We pull it down to 100 so a
    # crash costs at most ~100 train_steps (a percent or two of
    # progress) instead of the whole run.
    #
    # max_checkpoints_to_keep is left at the default 3, which bounds
    # disk usage to 3 * ~12 MB = ~36 MB per job. The checkpoint write
    # is ~100ms so even at interval=100 it's <1% of training wall time.
    #
    # The crash-recovery detection in do_job() (see _detect_resume_
    # for_train_job below) reads these auto-saves; without them that
    # detection has nothing to restore.
    agent_learner = learner.Learner(
        learner_dir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        checkpoint_interval=100)

    # Wire the Learner + train_step + job_id into the module-level
    # SIGTERM handler so a Docker / WSL shutdown can flush an
    # emergency checkpoint + mark the job PAUSED. See
    # _emergency_pause_handler above for the full handshake. Cleared
    # at end of main() via the try/finally so the next job's pickup
    # registers fresh state without a stale reference to the
    # previous job's Learner.
    _emergency_state["learner"] = agent_learner
    _emergency_state["train_step"] = train_step
    _emergency_state["job_id"] = job_id
    
    def get_eval_metrics():
        # Pull the NumberOfEpisodes counter out of the eval_actor's
        # metrics list so we can measure episodes-completed-during-this-
        # call. The counter is cumulative across the whole training run
        # (eval_actor never resets between get_eval_metrics() calls), so
        # we snapshot it before/after the run() and subtract. This is
        # especially useful with a ParallelPyEnvironment(num_envs=N)
        # because each subprocess env's LAST step type counts as one
        # episode toward the run's episodes_per_run budget; the delta
        # tells you whether the eval actually finished its budget or
        # bailed for some reason (e.g., a wedged env step).
        pre_episodes_metric = next(
            (m for m in eval_actor.metrics if m.name == 'NumberOfEpisodes'),
            None)
        pre_episodes = (int(pre_episodes_metric.result())
                        if pre_episodes_metric is not None else None)
        eval_step = int(train_step.numpy())
        eval_start = time.time()
        print(f"EVAL begin: train_step={eval_step} "
              f"target_episodes={num_eval_episodes} "
              f"cumulative_episodes_so_far="
              f"{pre_episodes if pre_episodes is not None else '?'}",
              flush=True)

        # Keep the rollout viz live during the (blocking) eval run. In-training
        # eval now steps the DEDICATED single-gym eval env (ros-server-0), not
        # the N-actor collect env, so only that one client is active during
        # eval - but the viz context is otherwise only refreshed in the TRAIN
        # loop, leaving a frozen collect fan during eval. This helper thread
        # refreshes the viz context (greedy eval_policy + the eval env's CACHED
        # current_time_step) every ~50ms so the eval client renders its live
        # fan. It only READS the cached time_step (no subprocess call), so it
        # doesn't race the eval stepping; update_context is a no-op when
        # ROLLOUT_VIZ is disabled.
        _eval_viz_stop = threading.Event()
        # Eval runs on a single client regardless of collect num_envs.
        _eval_num_envs = 1 if eval_env is not env else num_envs

        def _feed_eval_viz():
            while not _eval_viz_stop.is_set():
                try:
                    get_viz().update_context(eval_policy, eval_env, eval_step,
                                             mode="eval")
                    # Keep the HUD on EVAL even when the fan is disabled.
                    get_viz().publish_mode("eval", _eval_num_envs, eval_step)
                except Exception:  # noqa: BLE001 - viz must never break eval
                    pass
                _eval_viz_stop.wait(0.05)

        _eval_viz_thread = threading.Thread(
            target=_feed_eval_viz, name="eval-viz-feed", daemon=True)
        _eval_viz_thread.start()
        # Disable immediate reset during eval - eval steps quickly and the
        # blocking reset just slows it down. The normal reset path is fast
        # enough when there's no learner delay between steps. Toggle it on the
        # eval env (the single-gym eval client in multi-env runs).
        set_immediate_reset_on_failure(eval_env, False)
        try:
            eval_actor.run()
        finally:
            set_immediate_reset_on_failure(eval_env, True)
            _eval_viz_stop.set()
            _eval_viz_thread.join(timeout=1.0)

        eval_elapsed = time.time() - eval_start
        post_episodes = (int(pre_episodes_metric.result())
                         if pre_episodes_metric is not None else None)
        episodes_completed = (post_episodes - pre_episodes
                              if (pre_episodes is not None
                                  and post_episodes is not None)
                              else None)

        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
            print("metric.name:" + str(metric.name))
            print("metric.result():" + str(metric.result()))

        # Single-line structured summary, easy to grep for in
        # robotaxi.out (search 'EVAL end:' to step through eval points
        # in time-order and read off training progress without scrolling
        # past per-step ACTION traces).
        avg_return = results.get('AverageReturn')
        avg_ep_len = results.get('AverageEpisodeLength')
        avg_return_str = f"{avg_return:.4f}" if avg_return is not None else "N/A"
        avg_ep_len_str = f"{avg_ep_len:.2f}" if avg_ep_len is not None else "N/A"
        episodes_completed_str = (str(episodes_completed)
                                  if episodes_completed is not None else "?")
        post_episodes_str = (str(post_episodes)
                             if post_episodes is not None else "?")
        print(f"EVAL end:   train_step={eval_step} "
              f"episodes_completed={episodes_completed_str} "
              f"elapsed_sec={eval_elapsed:.2f} "
              f"AverageReturn={avg_return_str} "
              f"AverageEpisodeLength={avg_ep_len_str} "
              f"cumulative_episodes={post_episodes_str}",
              flush=True)
        _append_eval_curve(job_id, eval_step, avg_return, avg_ep_len)
        return results

    _phase("first_eval")
    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        eval_results_blob= {}
        for name, result in metrics.items():
            eval_results_blob[str(name)] = float(result)
        eval_results_blob["step"] = int(step)
        eval_results_blob["type"] = "step update"
        # Use the local job_id arg of main() rather than env.job_id. On
        # multi-env training the parent is a ParallelPyEnvironment and
        # the actual job_id-bearing RobotApi lives on each subprocess
        # env (set via env._envs[i].call('configure', job_id, ...) in
        # configure_env above), so a parent-level env.job_id read raises
        # AttributeError. The same value is in scope as a closure.
        eval_results_blob["job_id"] = job_id
        log_blob(eval_results_blob)
        print('step = {0}: {1}'.format(step, eval_results), flush=True)

    def bc_agent_training(training_steps=100):
        """In-loop BC top-up for the actor network.

        Periodically re-fits ``actor_net`` to the expert demonstration
        distribution during the SAC training loop. This anchors the
        policy to expert behavior as the replay buffer dilutes with
        on-policy data and the FIFO remover starts evicting expert
        demos, recovering the historical "starts-strong-stays-strong"
        curve that was lost when the in-loop BC top-up was disabled in
        commit dfef1f8 (see also the bc_pretrain_steps_val rationale at
        the top of main()).

        Currently NOT wired into the training loop. Re-enable by adding
        a call inside the ``if eval_interval and step % eval_interval
        == 0:`` branch below, e.g.:

            if eval_interval and step % eval_interval == 0:
                bc_agent_training()  # in-loop BC top-up
                metrics = get_eval_metrics()
                ...

        Reuses ``collect_training_data.bc_pretrain_actor_net`` (the same
        machinery as the pre-loop BC pretraining) but with a smaller
        default ``training_steps`` because this is meant to be a
        periodic anchoring nudge rather than a full BC fit. The
        ``actor_net`` reference is shared with ``tf_agent.actor_network``,
        so weight updates here are immediately visible to the SAC
        agent on its next gradient step.

        Args:
          training_steps: number of supervised BC gradient updates per
            top-up call. Defaults to 100 - small enough to not stall
            the SAC loop on every eval interval, large enough to apply
            a meaningful expert-anchor gradient. Adjust based on how
            often this is called and how rapidly the policy drifts.
        """
        print(f"bc agent training started ({training_steps} steps)",
              flush=True)
        collect_training_data.bc_pretrain_actor_net(
            actor_net=actor_net,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            strategy=strategy,
            trajectory_dataset=trajectory_dataset,
            training_steps=training_steps,
            batch_size=batch_size)
        print("bc agent training done", flush=True)

    log_eval_metrics(0, metrics)

    # Reset the train step.
    #
    # Pre-resume rationale: after BC pretrain, the agent's
    # train_step counter may have been incremented by the inner BC
    # optimizer loop; we zero it out so training metrics in
    # TensorBoard start cleanly from 0 (= "first SAC training step")
    # rather than from "first SAC step + N_BC_steps".
    #
    # SKIPPED on resume: when we're resuming a paused job, the
    # Learner's auto-restore (inside its constructor a few dozen
    # lines above) just loaded the saved train_step value back into
    # this counter. Resetting to 0 here would silently throw away
    # that restore - the training loop's first per-iteration write
    # of `percent_complete = step / num_iterations * 100` would then
    # report 0%, undoing both the saved progress and the
    # percent_complete-preservation logic in do_job(). (We observed
    # this on 2026-05-25: pause at step 800 -> resume -> progress
    # bar pops back to 0 because the Learner restored to 800, then
    # this line wiped it back to 0.)
    if not is_resume_val:
        tf_agent.train_step_counter.assign(0)
    else:
        print(
            f"main: RESUME - preserving restored train_step="
            f"{int(train_step.numpy())} (skipping post-BC reset).",
            flush=True)

    _phase("pre_train_eval")
    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]
    curr_iteration=0
    print("Num iterations: " + str(num_iterations), flush=True)
    print("Eval interval: " + str(eval_interval), flush=True)
    print("Log interval: " + str(log_interval), flush=True)
    min_write_step = 0

    # Time-budgeted eval cadence. Instead of evaluating every eval_interval
    # steps (which made eval ~90% of wall-clock when each eval runs many
    # episodes), keep eval to ~eval_time_fraction of the loop's wall-clock so
    # the clients spend the majority of time in the collect phase (where the
    # rollout-viz fans render). After an eval of duration D we train for
    # (1-frac)/frac * D before the next eval; the FIRST eval is bootstrapped
    # off eval_interval so we can measure an initial D. Falls back to the old
    # step%eval_interval gate when eval_time_fraction is outside (0, 1).
    eval_time_fraction = eval_time_fraction_val
    # Optional env override for quick tuning without touching the job config,
    # e.g. EVAL_TIME_FRACTION=0.2 for 80/20. Out-of-range disables budgeting.
    try:
        _env_frac = os.environ.get("EVAL_TIME_FRACTION")
        if _env_frac is not None:
            eval_time_fraction = float(_env_frac)
    except (TypeError, ValueError):
        pass
    _eval_time_budgeted = (0.0 < eval_time_fraction < 1.0)
    _eval_train_ratio = ((1.0 - eval_time_fraction) / eval_time_fraction
                         if _eval_time_budgeted else 0.0)
    train_time_since_eval = 0.0
    last_eval_duration = None
    # Fixed training-time eval interval (seconds). When > 0 it takes precedence
    # over the time-fraction budget: eval fires after this many seconds of
    # TRAINING wall-clock (train_time_since_eval), giving a constant "eval every
    # N minutes" cadence regardless of eval duration. Env override for quick
    # tuning without touching the job/experiment-design config.
    eval_train_interval_sec = eval_train_interval_sec_val
    try:
        _env_eti = os.environ.get("EVAL_TRAIN_INTERVAL_SEC")
        if _env_eti is not None:
            eval_train_interval_sec = float(_env_eti)
    except (TypeError, ValueError):
        pass
    _eval_fixed_interval = (eval_train_interval_sec is not None
                            and eval_train_interval_sec > 0)
    if _eval_fixed_interval:
        print(f"Eval cadence: fixed {eval_train_interval_sec:.0f}s "
              f"(~{eval_train_interval_sec / 60.0:.1f} min) of TRAINING "
              f"wall-clock between evals (overrides time-fraction budget)",
              flush=True)
    elif _eval_time_budgeted:
        print(f"Eval time budget: ~{eval_time_fraction * 100:.0f}% eval / "
              f"{(1 - eval_time_fraction) * 100:.0f}% train (train "
              f"{_eval_train_ratio:.1f}x each eval's wall-clock)", flush=True)

    # Bound the loop by REMAINING work, not a flat num_iterations count.
    # `train_step` is the global step counter; on a fresh start it's 0
    # (range = num_iterations, unchanged), but on a RESUME / crash-recovery
    # it's been restored to where the prior run left off. Looping a flat
    # num_iterations from there would train far past the target and leave
    # percent_complete (= step / num_iterations) stuck above 100% with the
    # job perpetually IN_PROGRESS. Training only the remaining iterations
    # makes a resumed job stop at the target, hit 100%, exit this loop,
    # and let do_job's trailer flip it to DONE. If it's already complete
    # on resume, remaining is 0 and we skip straight to the DONE path.
    start_step = int(train_step.numpy())
    remaining_iterations = max(0, num_iterations - start_step)
    if is_resume_val or start_step > 0:
        print(f"main: training {remaining_iterations} more iteration(s) "
              f"(start_step={start_step}, target={num_iterations}).",
              flush=True)

    # Curriculum scheduler (None when curriculum_stages_val is unset).
    # eval_env is passed so stage changes also reconfigure the dedicated
    # single-gym eval env (multi-env runs); it's a no-op when eval_env is env.
    _curriculum = (
        CurriculumScheduler(_curriculum_stages, env, eval_env=eval_env,
                            start_stage=_curriculum_start)
        if _curriculum_stages else None
    )

    # Optional policy trajectory-rollout visualization (off unless
    # ROLLOUT_VIZ_ENABLED). A shared background thread samples action sequences
    # from the live policy at ~20Hz and publishes them to Unity for the
    # candidate-path fan; the loop just feeds it the latest policy+obs below.
    # See rl_agent/rollout_viz.py + docs/trajectory-rollout-viz.md.
    rollout_viz = get_viz()

    _dump_startup_timings()

    # ---- Demo/online batch-composition probe (two-table mode only) ------
    # One-shot empirical check that the mixed sampler actually draws
    # ~demo_sample_ratio of its rows from the protected demo table. Logs a
    # single [demo-ratio] line; pure measurement, never aborts training.
    if (demo_min_keep > 0 and demo_replay is not None
            and 0.0 < demo_sample_ratio < 1.0):
        try:
            _realized, _dr, _tr = _measure_demo_batch_fraction(
                demo_replay, reverb_replay, demo_sample_ratio,
                batch_size, sequence_length=2, n_batches=300)
            print(
                f"[demo-ratio] configured={demo_sample_ratio:.3f} "
                f"realized={_realized:.3f} "
                f"({_dr}/{_tr} sampled rows from the demo table over 300 "
                f"batches; demo_table protected at {demo_min_keep} samples)",
                flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[demo-ratio] measurement failed (non-fatal): {_e}",
                  flush=True)

    for _ in range(remaining_iterations):
        # Cooperative lifecycle check. The dashboard's Jobs-tab Pause
        # button writes status=PAUSE_REQUESTED; Set-to-done / Delete
        # write DONE / NOT_STARTED / etc. Without this poll the
        # trainer would keep training against a now-stale job. One
        # sub-millisecond Mongo find_one per iteration vs ~250ms env
        # step, so the check is cheap and the UX is crisp - the
        # operator's click becomes effective within a single iteration.
        #
        # Ctrl-C / SIGTERM is handled by a separate KeyboardInterrupt
        # catch in run_jobs_loop, so this branch is specifically for
        # dashboard-initiated lifecycle changes.
        _lifecycle = _get_job_lifecycle_state(job_id)
        if _lifecycle == 'pause':
            # Save the Learner checkpoint (actor + critic + target_critic
            # + optimizers + train_step counter) so a later resume can
            # pick up from this exact training state. tf-agents'
            # Learner constructs a Checkpointer at init that auto-
            # restores from the same dir on the next instantiation,
            # so we just call .save(...) here and the resume side is
            # transparent.
            #
            # Saves both the Learner checkpoint (actor + critic + target_critic
            # + optimizers + train_step counter) AND, on success, a Reverb
            # online-table snapshot so resume can restore the buffer without
            # re-running initial_collect (eliminating the demo-heavy warmup
            # bias that previously caused a distributional shift for ~5k steps).
            try:
                _ckpt_step = int(train_step.numpy())
                agent_learner._checkpointer.save(_ckpt_step)
            except Exception as _e:  # noqa: BLE001
                pass  # handled below
            else:
                # Learner checkpoint succeeded — also snapshot the Reverb
                # online table so resume can restore it without re-running
                # initial_collect and eliminates the demo-heavy warmup bias.
                # Non-fatal: a Reverb checkpoint failure is logged but doesn't
                # block the pause (the Learner checkpoint is already safe).
                try:
                    _rv_path = reverb_server.localhost_client().checkpoint()
                    print(
                        f"main: Reverb online table checkpointed to {_rv_path}",
                        flush=True)
                except Exception as _rv_e:  # noqa: BLE001
                    print(
                        f"main: Reverb checkpoint failed (non-fatal): {_rv_e}",
                        flush=True)
                _e = None  # checkpoint succeeded; clear sentinel
            if _e is not None:
                # If the checkpoint write fails (disk full, weird
                # tf-agents version mismatch on _checkpointer access)
                # we still want to record the pause attempt rather than
                # silently keep training. Surface the error on the job
                # so the operator sees it, then mark PAUSED anyway -
                # the job won't actually resume from a checkpoint but
                # at least it's stopped.
                print(
                    f"main: pause checkpoint save FAILED for job {job_id}: {_e}. "
                    f"Marking PAUSED anyway; resume will start from scratch.",
                    flush=True)
                try:
                    update_job(job_id,
                               f"pause checkpoint save failed: {_e}"[:4000],
                               "eval_error")
                except Exception:
                    pass
            print(
                f"main: job {job_id} pause requested; saved Learner "
                f"checkpoint at train_step={int(train_step.numpy())}, "
                f"iter={curr_iteration + 1}/{num_iterations}. Marking "
                f"status=PAUSED; resume by setting status=NOT_STARTED.",
                flush=True)
            try:
                update_job(job_id, int(train_step.numpy()), "paused_at_step")
                update_job(job_id, "PAUSED", "status")
            except Exception as _e:  # noqa: BLE001
                print(f"main: pause status write failed: {_e}", flush=True)
        if _lifecycle == 'cancel':
            print(
                f"main: job {job_id} status changed externally; "
                f"breaking out of training loop at iter "
                f"{curr_iteration + 1}/{num_iterations} "
                f"(train_step={int(train_step.numpy())}). "
                f"do_job will skip the trailing status update so the "
                f"externally-set status is preserved.",
                flush=True)
            break

        def diagnostic_check():
            print("\n--- DIAGNOSTIC CHECK ---")
            # 1. Check if the environment is giving rewards
            test_time_step = env.reset()
            print(f"Initial Step Type: {test_time_step.step_type}")
            print(f"Initial Reward: {test_time_step.reward}")
            # 2. Check what the Actor Network is outputting
            # We pass the observation through the policy to see if it's producing NaNs
            test_action_step = tf_eval_policy.action(test_time_step)
            print(f"Agent Action Output: {test_action_step.action.numpy()}")
            # 3. Step the environment with that action
            next_test_time_step = env.step(test_action_step.action)
            print(f"Next Step Type: {next_test_time_step.step_type}")
            print(f"Next Reward: {next_test_time_step.reward}")
            print("------------------------\n")
        
        # Training.
        #
        # One outer-loop iteration = one collect step (collect_actor.run
        # advances the env by collect_steps_per_iteration steps per env
        # and writes the produced trajectory to the Reverb table) + one
        # SAC gradient update (agent_learner.run(iterations=1) samples a
        # batch_size=256 batch from the table and applies one critic
        # + actor + alpha optimizer step).
        #
        # Logged with TRAIN begin / TRAIN end lines mirroring the
        # EVAL begin / EVAL end lines emitted by get_eval_metrics()
        # above, so robotaxi.out reads as a clean alternating sequence
        # of TRAIN / EVAL events. Grep 'TRAIN end:' for a per-iter
        # timing+loss+buffer trace, or 'EVAL end:' for the periodic
        # policy-quality snapshots.
        train_step_before = int(train_step.numpy())
        train_iter_start = time.time()
        print(f"TRAIN begin: iter={curr_iteration + 1}/{num_iterations} "
              f"train_step={train_step_before}", flush=True)

        collect_start = time.time()
        collect_actor.run()
        collect_elapsed = time.time() - collect_start

        learner_start = time.time()
        loss_info = agent_learner.run(iterations=1)
        learner_elapsed = time.time() - learner_start

        train_iter_elapsed = time.time() - train_iter_start
        step = agent_learner.train_step_numpy
        # Accrue training wall-clock toward the eval time budget (Section: the
        # time-budgeted eval gate below decides when enough train time has
        # passed to justify the next eval).
        train_time_since_eval += train_iter_elapsed

        # Publish a rollout-viz fan for the live policy (throttled; no-op
        # unless ROLLOUT_VIZ_ENABLED). collect_policy is the stochastic
        # SAC policy, so its sampled action sequences show real spread.
        rollout_viz.maybe_publish(collect_policy, step, env)
        # Keep the Unity HUD's TRAINING indicator live regardless of the fan:
        # publish_mode is independent of ROLLOUT_VIZ_ENABLED and does no TF
        # inference, so the HUD shows "TRAINING" even when the fan is off.
        # Also forward the current curriculum stage (None for non-curriculum
        # jobs) so the HUD can show "stage: X/N".
        rollout_viz.publish_mode(
            "train", num_envs, step,
            stage=_curriculum.stage_idx if _curriculum is not None else None,
            num_stages=len(_curriculum.stages) if _curriculum is not None else None)

        # Buffer-size readout via Reverb's server_info gRPC (same query
        # print_replay_buffer_size used to do). One round-trip per
        # iteration is negligible overhead next to the collect step.
        buffer_size = (reverb_replay.py_client.server_info()
                       [table_name].current_size)

        loss_value = (float(loss_info.loss.numpy())
                      if hasattr(loss_info.loss, 'numpy')
                      else float(loss_info.loss))

        print(f"TRAIN end:   iter={curr_iteration + 1}/{num_iterations} "
              f"train_step={step} "
              f"elapsed_sec={train_iter_elapsed:.2f} "
              f"collect_sec={collect_elapsed:.2f} "
              f"learner_sec={learner_elapsed:.2f} "
              f"loss={loss_value:.4f} "
              f"buffer_size={buffer_size}/{replay_buffer_capacity}",
              flush=True)

        # Periodic Reverb online-table checkpoint - see
        # REVERB_PERIODIC_CHECKPOINT_INTERVAL's docstring above for why
        # this exists alongside (not instead of) the on-pause checkpoint
        # in the _lifecycle == 'pause' branch. Non-fatal on failure: a
        # missed periodic snapshot just means a future resume falls back
        # to initial_collect, exactly like before this was added.
        if step > 0 and step % REVERB_PERIODIC_CHECKPOINT_INTERVAL == 0:
            try:
                _rv_path = reverb_server.localhost_client().checkpoint()
                print(
                    f"main: periodic Reverb online-table checkpoint "
                    f"written to {_rv_path} (train_step={step})",
                    flush=True)
                _prune_reverb_checkpoints(
                    _reverb_ckpt_dir, keep=REVERB_PERIODIC_CHECKPOINT_KEEP)
            except Exception as _rv_periodic_e:  # noqa: BLE001
                print(
                    f"main: periodic Reverb checkpoint failed "
                    f"(non-fatal): {_rv_periodic_e}", flush=True)

        # Time-budgeted eval gate (see pre-loop setup): once an initial eval
        # duration is known, eval only after train_time_since_eval reaches
        # _eval_train_ratio x that duration (keeps eval ~eval_time_fraction of
        # wall-clock). Before the first eval, and when budgeting is disabled,
        # fall back to the classic step%eval_interval cadence.
        _do_eval = False
        if _eval_fixed_interval:
            # Fixed training-time cadence (2026-07-26): eval after every
            # eval_train_interval_sec of TRAINING wall-clock, regardless of eval
            # duration or step count. train_time_since_eval is reset to 0 after
            # each eval (below), so this fires cleanly every interval and also
            # avoids the resume-time eval burst the step%eval_interval bootstrap
            # caused (eval_interval is small = 10 just to bootstrap the budget).
            _do_eval = (train_time_since_eval >= eval_train_interval_sec)
        elif eval_interval:
            if _eval_time_budgeted and last_eval_duration is not None:
                _do_eval = (train_time_since_eval
                            >= _eval_train_ratio * last_eval_duration)
            else:
                _do_eval = (step % eval_interval == 0)
        if _do_eval:
            # In-training eval is bracketed by EVAL CYCLE begin / end
            # markers so robotaxi.out reads as a clean nested sequence:
            #
            #   TRAIN begin / TRAIN end                <- training iter
            #   EVAL CYCLE begin                       <- entering eval phase
            #     EVAL begin / EVAL end                <- eval_actor.run()
            #     step = N: AverageReturn = ...        <- log_eval_metrics
            #   EVAL CYCLE end                         <- summary + saved flag
            #   TRAIN begin / TRAIN end                <- back to training
            #
            # Grep 'EVAL CYCLE end:' for a flat per-eval timeline of
            # current/max return, is_new_max, and whether a checkpoint
            # was written.
            eval_cycle_start = time.time()
            percent_complete = step / num_iterations
            print(f"EVAL CYCLE begin: train_step={step} "
                  f"iter={curr_iteration + 1}/{num_iterations} "
                  f"percent={percent_complete * 100:.1f}%",
                  flush=True)

            update_job(job_id, percent_complete * 100, "percent_complete")
            update_job(job_id, int(step), "training_steps")

            # bc_agent_training() is intentionally not called here; the
            # function lives at main()-scope above (next to
            # get_eval_metrics) so it can be re-enabled in this branch
            # later as the in-loop BC top-up - see Option 2 in the
            # earlier expert-demo-routing investigation. Pure SAC
            # without periodic BC anchoring will start strong (BC-
            # pretrained actor) and drift as the buffer dilutes.
            # Snapshot the EVAL env's cumulative goal/episode counters BEFORE
            # the eval so the curriculum gate can isolate THIS eval cycle's
            # episodes (goals_per_episode_total / num_episodes_total accumulate
            # over the course's lifetime; the post-eval delta gives exactly the
            # episodes of the most recently completed eval). Only the eval env
            # drives Unity between these two reads (collection is paused during
            # eval), so the delta is eval-only even in single-env runs.
            try:
                _eval_ctr_before = read_course_raw_counters(eval_env)
            except Exception:  # noqa: BLE001
                _eval_ctr_before = {}
            metrics = get_eval_metrics()
            course_metrics = read_course_metrics(env)
            for name, value in course_metrics.items():
                tf.summary.scalar(name, data=value, step=step)
            # Also snapshot the EVAL env's course metrics (dedicated single-gym
            # eval env in multi-env runs; the shared env in single-env runs).
            # These reflect the GREEDY eval rollouts, not the exploration-noisy
            # collect rollouts. Logged under an 'eval/' namespace so TB shows the
            # eval-rollout goals/ep (rolling last-30, cumulative, etc.) for
            # reference; the curriculum GATE below uses the per-eval delta.
            eval_course_metrics = read_course_metrics(eval_env)
            for name, value in eval_course_metrics.items():
                tf.summary.scalar('eval/' + name, data=value, step=step)
            # Per-eval-cycle goals: episode count + per-episode goal counts
            # banked by the EVAL env during THIS eval only. num_episodes_total
            # delta tells us how many episodes this eval ran; the tail of the
            # eval env's goals_per_episode_arr gives their individual goal counts.
            try:
                _eval_ctr_after = read_course_raw_counters(eval_env)
            except Exception:  # noqa: BLE001
                _eval_ctr_after = {}
            _eval_goals_delta = (
                _eval_ctr_after.get('goals_per_episode_total', 0.0)
                - _eval_ctr_before.get('goals_per_episode_total', 0.0))
            _eval_eps_delta = int(
                _eval_ctr_after.get('num_episodes_total', 0.0)
                - _eval_ctr_before.get('num_episodes_total', 0.0))
            # Per-episode goal counts for THIS eval's episodes (last N entries).
            _eval_ep_goals = read_course_recent_goals(
                eval_env,
                _eval_eps_delta if _eval_eps_delta > 0 else num_eval_episodes)
            # Gate metric: mean of the TOP-K episodes by goal count within this
            # eval (2026-07-27). Averaging only the best K episodes measures the
            # policy's demonstrated best-case competence and is robust to a few
            # unlucky early crashes dragging a full-episode mean down. K is
            # capped at the number of episodes actually available.
            _EVAL_TOPK = 3
            _eval_topk_goals = sorted(_eval_ep_goals, reverse=True)[:_EVAL_TOPK]
            _eval_topk_metric = (
                sum(_eval_topk_goals) / len(_eval_topk_goals)
                if _eval_topk_goals else 0.0)
            # Full-eval mean kept for logging / reference (not the gate).
            _eval_mean_metric = (
                _eval_goals_delta / _eval_eps_delta if _eval_eps_delta > 0
                else 0.0)
            tf.summary.scalar('eval/goals_per_episode_this_eval',
                              data=_eval_mean_metric, step=step)
            tf.summary.scalar('eval/goals_per_episode_top3_this_eval',
                              data=_eval_topk_metric, step=step)
            print(f"EVAL goals/ep (this eval only): "
                  f"top{_EVAL_TOPK}-mean={_eval_topk_metric:.2f} "
                  f"(top={[round(g, 1) for g in _eval_topk_goals]}), "
                  f"full-mean={_eval_mean_metric:.2f} over "
                  f"{_eval_eps_delta} episodes "
                  f"all={[round(g, 1) for g in _eval_ep_goals]}", flush=True)
            # Curriculum update: gate advancement on the TOP-K-by-goal-count mean
            # of ONLY the most recently completed eval (2026-07-27). Rationale:
            #   * EVAL (greedy, single clean gym) signal reflects learned
            #     competence, not the collect env's exploration variance.
            #   * ONLY the latest eval's episodes -> the gate tracks the CURRENT
            #     policy, not a rolling window lagged by earlier weaker evals.
            #   * TOP-K (best 3) rewards demonstrated best-case driving and is
            #     robust to a few early crashes; per-eval variance is absorbed by
            #     the stage 'consecutive' requirement (bar cleared N evals in a
            #     row).
            if _curriculum is not None:
                _curriculum.update(_eval_topk_metric, train_step=step)
            # Cumulative asyncio.TimeoutError counts per category,
            # summed across actors. Each value is monotonically non-
            # decreasing across the training run, so the TensorBoard
            # plot reads as a cumulative curve - a flat slope means
            # no new timeouts since the last eval, a steep slope means
            # a burst. The 'timeouts/' namespace groups the four
            # counters together in TB's left-rail filter.
            timeout_counts = read_timeout_counts(env)
            for name, value in timeout_counts.items():
                tf.summary.scalar('timeouts/' + name, data=value, step=step)
            log_eval_metrics(step, metrics)
            current_avg_return = metrics["AverageReturn"]
            avg_return_arr.append(current_avg_return)
            avg_return_arr = avg_return_arr[-100:]
            max_avg_return = max(current_avg_return, max_avg_return)
            returns.append([current_avg_return])
            is_new_max_avg = (current_avg_return + 1e-5) > max_avg_return

            saved_checkpoint = False
            if step >= min_write_step and is_new_max_avg: # step % write_policy_interval == 0
                # Export the GREEDY policy (tf_eval_policy = GreedyPolicy(agent
                # .policy)) so the deployed best model takes the distribution's
                # MODE (tanh(mean)) instead of sampling. A deployed SAC actor
                # should drive deterministically; saving the raw stochastic
                # policy would ship the exploration noise into production. This
                # matches the greedy eval used to SELECT this best model, so the
                # exported artifact behaves like what was measured.
                tf_policy_saver = policy_saver.PolicySaver(tf_eval_policy)
                save_dir_name=get_save_dir_name(tf_agent)+ "_step_" + str(step)
                tf_policy_saver.save(save_dir_name)
                robot_type = os.getenv('ROBOT_TYPE')
                model_type=get_policy_type_name(tf_agent)
                training_iterations=num_iterations
                # Stamp the current env's specs onto the model record
                # so the Models-tab "Compat" column can flag rows that
                # were trained against a different observation/action
                # shape than the live env. observation_spec and
                # action_spec are the unbatched per-step specs already
                # in scope from spec_utils.get_tensor_specs(env) above.
                add_model(
                    save_dir_name,
                    robot_type,
                    model_type,
                    training_iterations,
                    avg_return=metrics["AverageReturn"],
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    # Capture which reward design produced this
                    # checkpoint so the Models tab can label it and
                    # the Analysis tab can group/compare by design.
                    # None when training used the course default.
                    reward_design=reward_design,
                    # Capture which experiment design (training-loop
                    # hyperparameters) produced this checkpoint. None
                    # means the trainer's hardcoded defaults were used.
                    experiment_design=experiment_design,
                    # Stamp the training job id so the Models tab can
                    # link each row back to its TensorBoard run for
                    # the multi-model TB comparison flow.
                    job_id=job_id)
                saved_checkpoint = True

            eval_cycle_elapsed = time.time() - eval_cycle_start
            # Feed the eval duration back into the time budget and reset the
            # train-time accumulator, so the next eval fires only after
            # ~_eval_train_ratio x this eval's wall-clock of training.
            last_eval_duration = eval_cycle_elapsed
            train_time_since_eval = 0.0
            print(f"EVAL CYCLE end:   train_step={step} "
                  f"iter={curr_iteration + 1}/{num_iterations} "
                  f"elapsed_sec={eval_cycle_elapsed:.2f} "
                  f"current_avg_return={current_avg_return:.4f} "
                  f"max_avg_return={max_avg_return:.4f} "
                  f"is_new_max={is_new_max_avg} "
                  f"returns_history_len={len(returns)} "
                  f"saved_checkpoint={saved_checkpoint}",
                  flush=True)
        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
        curr_iteration=curr_iteration+1
    print("Training completed")
    # If we trained all the way to the target step (normal completion or
    # an already-complete resume), stamp percent_complete=100 so the Jobs
    # tab matches the imminent DONE flip. A pause/cancel breaks out with
    # train_step < num_iterations, so this won't fire on those paths.
    if int(train_step.numpy()) >= num_iterations:
        try:
            update_job(job_id, 100.0, "percent_complete")
        except Exception as _e:  # noqa: BLE001
            print(f"main: final percent_complete=100 write failed: {_e}", flush=True)
    rb_observer.close()
    # In multi-env mode rb_observer is the fan-out wrapper; expert_observer
    # is a separate writer into the same table that we have to close on
    # its own. In single-env mode they're the same instance and the
    # second close() is a harmless no-op on an already-closed writer.
    if expert_observer is not rb_observer:
        expert_observer.close()
    reverb_server.stop()
    # tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
    # save_dir_name=get_save_dir_name(tf_agent)
    # tf_policy_saver.save(save_dir_name)
    # robot_type = os.getenv('ROBOT_TYPE')
    # model_type=get_policy_type_name(tf_agent)
    # training_iterations=num_iterations
    # add_model(save_dir_name, robot_type, model_type, training_iterations)


# Internal control-flow exception for cooperative cancellation inside
# run_policy. Raised by the per-chunk _is_job_cancelled() check inside
# run_one_trial; caught by the outer trial loop so we can return the
# partial results collected so far without inventing a special return-
# tuple shape. Subclasses Exception (not BaseException) so a stray
# try/except in calling code still catches it - but the only code
# expected to catch this name is run_policy itself.
class _EvalCancelled(Exception):
    pass


def _build_eval_result(returns, episode_lengths, avg_speeds,
                       avg_goals_per_episode, avg_steering_angle_ratios,
                       partial=False, episode_counts=None):
    """Package run_policy's per-trial metric arrays into the dict
    save_results_to_db consumes.

    Centralised here so the two return paths in run_policy (clean
    completion + cancellation) construct the dict identically.

    ``partial=True`` is set on the cancellation path; consumers that
    care about distinguishing complete-eval from cancelled-eval can
    branch on it (currently save_results_to_db treats both the
    same - records whatever data we have).

    ``episode_counts``: per-trial count of episodes (= crashes, since
    every donut-course episode ends on a collision). Stored alongside
    the ratio metrics so the Leaderboard tab can show raw crash counts
    and compute steps-per-goal without needing to re-derive them.
    """
    return {
        "returns": returns,
        "episode_lengths": episode_lengths,
        "avg_speeds": avg_speeds,
        "avg_goals_per_episode": avg_goals_per_episode,
        "avg_steering_angle_ratios": avg_steering_angle_ratios,
        "episode_counts": episode_counts or [],
        "partial": partial,
    }


def run_policy(saved_policy, tf_env, job_id="",
                                  max_episodes=3,
                                  num_eval_episodes=5,
                                  log_interval=10,
                                  max_steps_per_episode=0,
                                  max_goals_per_episode=100):
    """Run an EVAL job for ``saved_policy`` and report the metrics.

    Args:
      saved_policy: a ``PyPolicy`` to evaluate (a saved SAC actor or a
        ``RandomPyPolicy`` baseline; whatever the EVAL dispatch passed in).
      tf_env: a single (non-batched) ``PyEnvironment`` connected to
        ros-server-0. Wrapped here in a ``BatchedPyEnvironment(batch=1)``
        so it satisfies ``actor.Actor``'s batched-env contract.
      job_id: MongoDB job ObjectId; used for the per-job TensorBoard
        eval-log directory ``/tmp/active/<job_id>_eval/``.
      max_episodes: number of outer *trials*; each trial runs
        ``num_eval_episodes`` episodes and produces one
        ``(AverageReturn, AverageEpisodeLength)`` sample point. Default
        3, so a stock EVAL job evaluates 3 x 5 = 15 episodes total and
        the returned ``returns`` list has 3 entries (one mean-of-N per
        trial). Useful for measuring run-to-run variance of a stochastic
        policy without manually re-submitting jobs. The parameter name
        is historical; semantically it's the *trial count*, not an
        episode count.
      num_eval_episodes: episodes within one trial. Used both as the
        target episode count per trial (we keep stepping the env in
        ``log_interval``-sized chunks until this many episodes have
        completed) and as ``actor.eval_metrics``'s buffer size (each
        metric averages over the last N episodes). Default 5.
      log_interval: env-step granularity for in-trial progress logging.
        Drives ``actor.Actor.steps_per_run``, so each call to
        ``eval_actor.run()`` advances exactly ``log_interval`` env
        steps before printing one progress line; the trial loop keeps
        calling ``.run()`` until ``num_eval_episodes`` have completed.
        Default 10.
      max_steps_per_episode: maximum env-steps allowed in a single
        episode before it is forcibly truncated. Set to 0 or None to
        disable (unlimited). Default 0 (disabled; use max_goals_per_episode
        instead which is more policy-invariant).
      max_goals_per_episode: maximum goals the car may reach in a single
        episode before it is forcibly truncated. Cleaner than
        max_steps_per_episode because it is invariant to driving speed —
        a fast policy hits the cap in fewer steps than a slow one, but
        both produce the same goal count per episode. When the cap is
        hit the episode is counted as done and the env is reset.
        Set to 0 or None to disable. Default 100.
    """
    print("run policy")
    tempdir = "/tmp/active/"
    train_step = train_utils.create_train_step()
    eval_dir=os.path.join( tempdir,
        "eval" if str(job_id) == "" else str(job_id) + "_eval")
    batch_tf_env = batched_py_environment.BatchedPyEnvironment((tf_env,))
    debug_print("after batch_tf_env")
    time_step = batch_tf_env.reset()
    debug_print("after reset")

    # Per-trial eval metrics + a fresh NumberOfEpisodes counter to know
    # when each trial's episode budget has been spent. The counter
    # piggybacks on actor.Actor's metrics= argument so step_type=LAST
    # transitions auto-increment it; we read .result() between
    # ``eval_actor.run()`` calls and bail when it reaches
    # num_eval_episodes.
    eval_metrics_list = list(actor.eval_metrics(num_eval_episodes))
    episodes_metric = py_metrics.NumberOfEpisodes()
    debug_print("after eval metrics")

    # steps_per_run=log_interval (instead of episodes_per_run) so each
    # .run() call advances a fixed step budget regardless of where the
    # env is in the current episode. That gives us a stable cadence
    # for the in-trial progress prints (one line per log_interval
    # steps), matching main()'s inline-eval logging style. Without
    # this, .run() would block for the full trial before printing
    # anything.
    eval_actor = actor.Actor(
        batch_tf_env,
        saved_policy,
        train_step,
        steps_per_run=log_interval,
        metrics=eval_metrics_list + [episodes_metric],
        summary_dir=eval_dir)
    debug_print("after eval actor")
    print(eval_actor.metrics)

    # Total episodes across the full EVAL run = trials x episodes/trial.
    # Used by both the per-chunk progress update below and the
    # cancellation-check log line.
    total_episodes_target = max(1, max_episodes * num_eval_episodes)

    def _update_eval_progress(trial_idx, episodes_done_in_trial):
        """Stamp percent_complete on the job document.

        Mirrors the per-eval-cycle update in main()'s TRAIN loop so the
        Jobs tab's progress bar updates live for EVAL jobs too (previously
        EVAL jobs sat at 0% the entire run since nobody wrote
        percent_complete during the eval loop).

        Computation: each trial contributes a flat
        ``num_eval_episodes`` slice of the total budget; within the
        active trial we grow linearly by ``episodes_done_in_trial``.
        Capped at 100 because the LAST-step transition can push
        ``episodes_metric`` one episode past the target on the final
        chunk (we'd otherwise show 101%, which the dashboard's bar
        renderer clamps anyway but reads oddly in the Mongo doc).

        Best-effort: a Mongo blip during the eval shouldn't take the
        whole job with it. We log and continue.
        """
        if not job_id:
            return
        completed = (trial_idx - 1) * num_eval_episodes + episodes_done_in_trial
        pct = max(0.0, min(100.0, 100.0 * completed / total_episodes_target))
        try:
            update_job(job_id, pct, "percent_complete")
        except Exception as e:  # noqa: BLE001
            print(f"_update_eval_progress: Mongo update failed: {e}",
                  flush=True)

    # Optional rollout-viz for eval. NOTE: saved_policy is typically the
    # GREEDY saved policy, so its sampled "fan" collapses to (near-)identical
    # lines - the plumbing works but the spread is degenerate. Off unless
    # ROLLOUT_VIZ_ENABLED. Same shared background sampler as the train loop
    # (get_viz singleton); we just feed it the eval policy + eval-env obs so the
    # fan keeps updating at 20Hz through the long eval windows too.
    eval_rollout_viz = get_viz()

    def run_one_trial(trial_idx):
        """Step the env in log_interval-sized chunks until
        num_eval_episodes episodes have completed. Returns the final
        per-trial metric dict + the per-trial COURSE counter delta.

        The course counter delta is computed by snapshotting
        ``read_course_raw_counters(tf_env)`` BEFORE the trial starts
        and again at the end. The Analysis tab uses these per-trial
        deltas to compute reward-design-invariant metrics (goals per
        episode, average speed, steering-angle ratio) that can be
        meaningfully compared across models trained with different
        reward designs - which avg_return cannot.
        """
        # Reset metrics at the start of each trial so AverageReturn /
        # AverageEpisodeLength reflect only this trial's episodes; the
        # outer ``returns`` list collects per-trial means independently.
        for m in eval_metrics_list:
            m.reset()
        episodes_metric.reset()

        # Snapshot the underlying course counters BEFORE the trial.
        # The course's metric machinery accumulates raw sums over the
        # env's lifetime; subtracting the pre-trial snapshot from the
        # post-trial snapshot gives us the per-trial averages without
        # needing to mutate course state mid-run. read_course_raw_counters
        # transparently handles single-env vs ParallelPyEnvironment.
        try:
            counters_before = read_course_raw_counters(tf_env)
        except Exception as e:  # noqa: BLE001
            # Defensive: if the course doesn't track raw counters
            # (e.g., a future course without RAW_COUNTER_KEYS) we
            # gracefully degrade to zero deltas, which the consumer
            # treats as "metric unavailable".
            print(f"run_one_trial: counters_before read failed: {e}",
                  flush=True)
            counters_before = {}

        step = 0
        _steps_this_episode = 0   # steps since last episode boundary
        _goals_this_episode = 0   # goals reached since last episode boundary
        _truncated_episodes = 0   # episodes ended by cap, not crash
        _ep_step_cap  = int(max_steps_per_episode)  if max_steps_per_episode  else 0
        _ep_goal_cap  = int(max_goals_per_episode)  if max_goals_per_episode  else 0
        # Snapshot of the course goals counter at the last episode boundary.
        _goals_counter_at_boundary = 0
        try:
            _goals_counter_at_boundary = float(
                read_course_raw_counters(tf_env).get('goals_per_episode_total', 0))
        except Exception:  # noqa: BLE001
            pass
        while (int(episodes_metric.result()) + _truncated_episodes) < num_eval_episodes:
            # Cooperative cancellation, same pattern as main()'s TRAIN
            # loop. Lets the dashboard's "Set to done" button stop a
            # multi-trial EVAL run within one env-chunk (~2-5s) rather
            # than waiting potentially minutes for the full trial loop
            # to drain.
            if _is_job_cancelled(job_id):
                print(
                    f"run_one_trial: job {job_id} status changed externally; "
                    f"breaking out of trial {trial_idx} at "
                    f"episode {int(episodes_metric.result())}/"
                    f"{num_eval_episodes}.",
                    flush=True)
                # Surface to the outer loop via a sentinel exception so
                # the rest of run_policy can stop without inventing a
                # return-tuple-shape change just for this case.
                raise _EvalCancelled()

            _ep_before = int(episodes_metric.result())
            eval_actor.run()
            step += log_interval
            _steps_this_episode += log_interval

            # Read current goals counter to compute goals this episode.
            _goals_now = 0
            try:
                _goals_now = float(
                    read_course_raw_counters(tf_env).get('goals_per_episode_total', 0))
            except Exception:  # noqa: BLE001
                pass
            _goals_this_episode = _goals_now - _goals_counter_at_boundary

            # If a real episode ended (crash), reset per-episode counters.
            if int(episodes_metric.result()) > _ep_before:
                _steps_this_episode = 0
                _goals_this_episode = 0
                _goals_counter_at_boundary = _goals_now

            # Episode truncation by goal cap (preferred) or step cap.
            # tf-agents' episodes_metric only increments on a terminal
            # time-step (crash); we bypass it via _truncated_episodes
            # so strong (non-crashing) policies finish in bounded time
            # while accumulating accurate course counter deltas.
            elif (_ep_goal_cap > 0 and _goals_this_episode >= _ep_goal_cap) or \
                 (_ep_step_cap > 0 and _steps_this_episode >= _ep_step_cap):
                cap_kind = f"{int(_goals_this_episode)} goals" \
                    if _ep_goal_cap > 0 and _goals_this_episode >= _ep_goal_cap \
                    else f"{_steps_this_episode} steps"
                print(
                    f"run_one_trial: episode truncated at {cap_kind} "
                    f"(goal_cap={_ep_goal_cap}, step_cap={_ep_step_cap}); "
                    f"counting as done and resetting env.",
                    flush=True)
                try:
                    tf_env.reset()
                except Exception:  # noqa: BLE001
                    pass
                _truncated_episodes += 1
                _steps_this_episode = 0
                _goals_this_episode = 0
                _goals_counter_at_boundary = _goals_now
            # Rollout-viz fan (throttled; no-op unless ROLLOUT_VIZ_ENABLED).
            # ts from the batched eval env, publish via the RobotaxiEnv.
            eval_rollout_viz.maybe_publish(saved_policy, step, batch_tf_env, tf_env)
            # Keep the Unity HUD on EVAL even when the fan is disabled (single
            # eval env -> one client).
            eval_rollout_viz.publish_mode("eval", 1, step)
            episodes_done = int(episodes_metric.result())
            partial = ', '.join(
                '{} = {:.4f}'.format(m.name, float(m.result()))
                for m in eval_metrics_list)
            print('trial {0}: step = {1}: {2}/{3} episodes, {4}'.format(
                trial_idx, step, episodes_done, num_eval_episodes, partial),
                flush=True)
            # Stamp the latest percent_complete on the job after each
            # progress print. Cadence matches the print's (one
            # log_interval ≈ 2-5s under normal load), which is dense
            # enough for a smooth Jobs-tab bar without churning Mongo.
            _update_eval_progress(trial_idx, min(episodes_done, num_eval_episodes))

        # Post-trial counter snapshot. Subtract from pre-trial to get
        # this trial's contribution. Course counter deltas can be 0
        # when the course doesn't track that key (SimpleCourse) or
        # when no time elapsed in that dimension (e.g., episodes that
        # were too short for the speed accumulator); the consumer
        # handles those as missing.
        try:
            counters_after = read_course_raw_counters(tf_env)
        except Exception as e:  # noqa: BLE001
            print(f"run_one_trial: counters_after read failed: {e}",
                  flush=True)
            counters_after = {}

        counter_delta = {
            k: float(counters_after.get(k, 0) - counters_before.get(k, 0))
            for k in (counters_after.keys() | counters_before.keys())
        }

        tf_metrics_result = {m.name: m.result() for m in eval_metrics_list}
        return tf_metrics_result, counter_delta

    def log_eval_metrics(trial_idx, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('trial {0} final: {1}'.format(trial_idx, eval_results), flush=True)

    # Stamp 0% at the start so the Jobs tab transitions from "no bar"
    # to "0% bar present" immediately, signalling that the eval kicked
    # off. Without this the bar stays at whatever ghost value Mongo
    # had (typically null, rendered as no bar) until the first chunk
    # completes - which can be 10+ seconds on a slow first-episode
    # reset.
    _update_eval_progress(trial_idx=1, episodes_done_in_trial=0)

    # Per-trial arrays. ``returns`` is the legacy (reward-shaped)
    # AverageReturn array; ``episode_lengths`` / ``avg_speeds`` /
    # ``avg_goals_per_episode`` / ``avg_steering_angle_ratios`` are
    # the reward-DESIGN-INVARIANT metrics the Analysis tab uses for
    # apples-to-apples comparison across models trained with
    # different reward shapings. Each array has exactly one entry per
    # completed trial. Parallel indexing - returns[i] and
    # avg_speeds[i] both refer to trial i+1.
    returns = []
    episode_lengths = []
    avg_speeds = []
    avg_goals_per_episode_arr = []
    avg_steering_angle_ratios = []
    episode_counts = []   # raw crash count per trial (num_episodes_total delta)

    def _safe_div(num, den):
        """Treat 0/0 (no episodes / no steps in this trial) as None.

        Returning None instead of 0 communicates "metric unavailable
        for this trial" (caller filters None out before computing
        stats) rather than "metric was zero" (which would skew means
        toward 0).
        """
        if den == 0:
            return None
        return float(num) / float(den)

    curr_trial=0
    # Disable immediate reset for standalone EVAL jobs - same rationale as
    # in-training eval: eval steps quickly without learner delay.
    set_immediate_reset_on_failure(tf_env, False)
    try:
        while curr_trial < max_episodes:
            debug_print("in loop")
            metrics, counter_delta = run_one_trial(curr_trial + 1)
            log_eval_metrics(curr_trial + 1, metrics)
            # avg_return = the tf-agents AverageReturn metric for this
            # trial. Legacy field name kept for back-compat.
            avg_return = metrics["AverageReturn"]
            returns.append(float(avg_return))
            # AverageEpisodeLength comes free from tf-agents'
            # eval_metrics list. Previously we computed and threw it
            # away; now it's persisted as a reward-invariant signal
            # of "how long the agent survives" (longer = better for
            # navigation tasks).
            ep_len = metrics.get("AverageEpisodeLength")
            episode_lengths.append(
                float(ep_len) if ep_len is not None else None)
            # Course-level unbiased metrics from the counter delta.
            # avg_speed = total speed / total steps for this trial.
            # avg_goals_per_episode = total goals / total episodes.
            # avg_steering_angle_ratio = total ratio / total steps.
            steps_delta = counter_delta.get('steps_total', 0)
            episodes_delta = counter_delta.get('num_episodes_total', 0)
            speeds_delta = counter_delta.get('speeds_total', 0)
            goals_delta = counter_delta.get('goals_per_episode_total', 0)
            steering_delta = counter_delta.get('steering_angle_ratio_total', 0)
            avg_speeds.append(_safe_div(speeds_delta, steps_delta))
            avg_goals_per_episode_arr.append(
                _safe_div(goals_delta, episodes_delta))
            avg_steering_angle_ratios.append(
                _safe_div(steering_delta, steps_delta))
            episode_counts.append(int(episodes_delta))
            print(
                f"trial {curr_trial + 1} unbiased: "
                f"episode_length={episode_lengths[-1]} "
                f"avg_speed={avg_speeds[-1]} "
                f"avg_goals_per_episode={avg_goals_per_episode_arr[-1]} "
                f"avg_steering_angle_ratio={avg_steering_angle_ratios[-1]}",
                flush=True)
            curr_trial=curr_trial+1
            # Snap percent to the trial-boundary exactly. The per-chunk
            # update inside run_one_trial may have lagged a beat (e.g.,
            # if the final chunk happened to land right on the episode
            # count without an extra progress print). Stamping the
            # boundary here keeps the Jobs tab from showing e.g. 87%
            # while the eval is actually fully done with trial 3 of 5.
            _update_eval_progress(
                trial_idx=curr_trial + 1, episodes_done_in_trial=0)
    except _EvalCancelled:
        # Cooperative cancellation. We've already logged the break;
        # return whatever metrics we collected so far so the calling
        # code in do_job can still record partial results (or, more
        # commonly, the cancellation also marked the job DONE/FAILED
        # and the caller skips writing).
        return _build_eval_result(
            returns, episode_lengths, avg_speeds,
            avg_goals_per_episode_arr, avg_steering_angle_ratios,
            partial=True, episode_counts=episode_counts)
    finally:
        # Re-enable immediate reset for any subsequent training on this env.
        set_immediate_reset_on_failure(tf_env, True)

    # Final snap to 100% on a clean completion so the Jobs tab bar
    # reaches full width even if the last per-chunk update landed
    # at 99.something% due to integer rounding.
    if job_id:
        try:
            update_job(job_id, 100.0, "percent_complete")
        except Exception as e:  # noqa: BLE001
            print(f"run_policy: final percent_complete update failed: {e}",
                  flush=True)
    return _build_eval_result(
        returns, episode_lengths, avg_speeds,
        avg_goals_per_episode_arr, avg_steering_angle_ratios,
        partial=False, episode_counts=episode_counts)

def get_saved_model(policy_type, version=None, path_arg=None):
    if path_arg is not None:
        path=path_arg
    elif version is None:
        path=get_latest_save_dir_name(policy_type)
    else:
        path=get_save_dir_by_version(policy_type, version)
    debug_print(path)
    saved_policy = tf.saved_model.load(path)
    return saved_policy, path

def load_saved_model(policy_type, version=None, path=None, job_id="",
                     num_trials=None, num_eval_episodes=None,
                     max_steps_per_episode=None,
                     max_goals_per_episode=None,
                     corner_radius=10.0, curvature_difficulty=0.0,
                     chicanes_north=0, chicanes_east=0,
                     chicanes_south=0, chicanes_west=0):
    """Build an env, load the policy at ``path`` (or by version), run an
    EVAL through it, persist the resulting per-trial means to MongoDB.

    Args:
      policy_type: 'SacAgent' / 'GreedyPolicy' / ... - dispatches the
        loader.
      version: integer version under the saved_models dir tree. Mutually
        exclusive with ``path``; one of the two must be supplied.
      path: explicit path to a saved model dir.
      job_id: MongoDB ObjectId for this EVAL job. Threaded into env for
        course-metric tracking and into ``run_policy``'s TensorBoard
        log dir.
      num_trials: trial count to forward to ``run_policy`` (mapped to
        its ``max_episodes`` arg, the historical misnomer for trial
        count - see run_policy's docstring). ``None`` keeps the
        default. The dashboard's Models-tab Eval modal sets this.
      num_eval_episodes: episodes-per-trial to forward to
        ``run_policy``. ``None`` keeps the default. Currently not
        surfaced in the dashboard; reserved for a future refinement
        of the Eval modal.
      corner_radius: procedural-track corner tightness forwarded to
        Unity's TrackGenerator on every episode reset (matches the
        same-named TRAIN kwarg so eval geometry equals training geometry).
      curvature_difficulty: DEPRECATED, logging/back-compat only - see
        chicanes_north/east/south/west below.
      chicanes_north/east/south/west: per-edge absolute chicane counts
        forwarded to Unity's TrackGenerator on every episode reset (matches
        the same-named TRAIN kwargs so eval geometry equals training
        geometry).
    """
    env = make_env('ros-server-0:50051')
    env.job_id = job_id
    # Mirror main()'s configure_env call so the track geometry at eval
    # time matches what was used during training.  Without this, every
    # DoReset defaults to curvature_difficulty=0.0 regardless of what
    # the job doc (or the Unity Inspector) specifies.
    configure_env(env, job_id=job_id, pass_through_actions=False,
                  corner_radius=corner_radius,
                  curvature_difficulty=curvature_difficulty,
                  chicanes_north=chicanes_north, chicanes_east=chicanes_east,
                  chicanes_south=chicanes_south, chicanes_west=chicanes_west)
    # Publish the current env's specs to MongoDB so the Models-tab
    # "Compat" column has up-to-date data for every robot_type the
    # sim-controller has touched. Cheap (one Mongo upsert), runs
    # even if the spec check below later rejects this particular
    # model. See publish_env_spec for the doc layout.
    publish_env_spec(env)

    if path is not None:
        saved_policy, path = get_saved_model(policy_type, path_arg=path)
    else:
        saved_policy, path = get_saved_model(policy_type, version)

    # Two-stage spec safety:
    #
    #   1. **Best-effort pre-flight** - extract specs from the loaded
    #      SavedModel's traced action() signature and compare against
    #      the env. If extraction returns None (e.g. tf_agents version
    #      doesn't expose the spec the way we read it, or the file
    #      isn't a PolicySaver-format SavedModel), we silently skip
    #      the pre-flight rather than refuse the eval. A missing spec
    #      is NOT a mismatch - we'd rather try the eval and convert a
    #      runtime crash than block a perfectly working model.
    #
    #   2. **Runtime catch** - any ValueError surfacing from inside
    #      run_policy with the "matching concrete function" wording
    #      that tf's restored_function_body uses to signal an input
    #      spec mismatch gets re-raised as EvalSpecMismatchError. This
    #      is the actual safety net that prevents sim-controller from
    #      crashing on a stale-model EVAL; do_job catches the
    #      EvalSpecMismatchError and surfaces it as job.eval_error.
    #
    # The previous implementation relied solely on a strict pre-flight
    # against `saved_policy.time_step_spec`, which doesn't exist on
    # the _UserObject that tf.saved_model.load() returns. That broke
    # every EVAL ("'_UserObject' object has no attribute
    # 'time_step_spec'"). This rewrite restores correctness while
    # keeping the crash-prevention benefit via the runtime catch.
    env_obs_d = _spec_to_dict(env.observation_spec())
    env_act_d = _spec_to_dict(env.action_spec())
    policy_obs, policy_act = _extract_savedmodel_specs(saved_policy)
    policy_obs_d = _spec_to_dict(policy_obs)
    policy_act_d = _spec_to_dict(policy_act)

    # Only enforce when we have BOTH sides. None on the model side =
    # extraction failed = skip this dimension; let the runtime catch
    # handle any real mismatch.
    if policy_obs_d is not None and not _specs_compatible(env_obs_d, policy_obs_d):
        raise EvalSpecMismatchError(
            f"Observation spec mismatch: env produces {env_obs_d} but "
            f"SavedModel at {path} expects {policy_obs_d}. The model was "
            "likely trained against a different observation set. Either "
            "retrain against the current env, or revert the env's "
            "observation builder to match the model's training-time spec.")
    if policy_act_d is not None and not _specs_compatible(env_act_d, policy_act_d):
        raise EvalSpecMismatchError(
            f"Action spec mismatch: env expects actions {env_act_d} but "
            f"SavedModel at {path} produces {policy_act_d}. The model "
            "was likely trained against a different action_spec.")
    if policy_obs_d is None and policy_act_d is None:
        print(
            f"load_saved_model: SavedModel specs at {path} were not "
            "extractable; skipping pre-flight check and relying on the "
            "runtime safety net.", flush=True)

    # Build run_policy kwargs from the optional overrides so we only
    # mention args the caller actually chose; everything unset falls
    # through to run_policy's own defaults. Without this, passing
    # num_trials=None would override max_episodes with None and break.
    run_kwargs = {}
    if num_trials is not None:
        run_kwargs["max_episodes"] = int(num_trials)
    if num_eval_episodes is not None:
        run_kwargs["num_eval_episodes"] = int(num_eval_episodes)
    if max_steps_per_episode is not None:
        run_kwargs["max_steps_per_episode"] = int(max_steps_per_episode)
    if max_goals_per_episode is not None:
        run_kwargs["max_goals_per_episode"] = int(max_goals_per_episode)

    # Runtime safety net (stage 2 above). tf's
    # restored_function_body raises ValueError with the wording
    # "Could not find matching concrete function to call loaded from
    # the SavedModel" when the input shape doesn't fit any traced
    # signature - this is the exact symptom that originally killed
    # sim-controller. Catch the narrow message pattern and convert;
    # let unrelated ValueErrors propagate so they still surface in
    # full detail in the logs.
    try:
        results = run_policy(saved_policy, env, job_id=job_id, **run_kwargs)
    except ValueError as e:
        msg = str(e)
        if ('matching concrete function' in msg
                or 'concrete_function' in msg
                or 'TensorSpec' in msg and 'expects' in msg):
            raise EvalSpecMismatchError(
                f"Runtime spec mismatch: SavedModel at {path} rejected "
                f"the env's time_step / action_step. Env produces "
                f"{env_obs_d} / actions {env_act_d}. Original tf error: "
                f"{msg}")
        raise

    save_results_to_db(path, results)

# ---------------------------------------------------------------- *
# Spec compatibility plumbing
#
# Background: a SavedModel-loaded policy is a TF concrete function
# specialized to a specific observation/action TensorSpec. Calling it
# with the wrong shape (e.g. (1, 32) when it expects (None, 10))
# raises a ValueError from deep inside function_deserialization.py
# and propagates all the way out of eval_actor.run() into
# run_jobs_loop, which kills the sim-controller process.
#
# We make three changes here to turn that catastrophic failure into
# something the dashboard can surface and the user can recover from:
#
#   1. Every model record gets observation_spec / action_spec stamped
#      onto it at training time (via add_model).
#   2. Every env-creating path publishes the env's current specs into
#      a MongoDB env_specs collection (one document per robot_type,
#      upserted). The dashboard reads this to flag incompatible models.
#   3. load_saved_model does a pre-flight comparison of the loaded
#      policy's specs vs. the current env's specs. On mismatch it
#      raises EvalSpecMismatchError, which do_job catches and surfaces
#      as eval_error on the job document - the job is marked DONE and
#      sim-controller keeps running.
#
# Comparison is shape + dtype only; we deliberately ignore bound
# values because SavedModel restoration doesn't validate them either,
# so requiring them to match would produce false-positive
# incompatibilities (e.g. action_spec.minimum changed but the model
# still works).
# ---------------------------------------------------------------- *

class EvalSpecMismatchError(ValueError):
    """Raised at EVAL pre-flight when the loaded SavedModel's specs
    don't match the current env. Caught by do_job and surfaced as an
    eval_error string on the job document rather than crashing the
    whole job loop."""

def _spec_to_dict(spec):
    """Serialise a tf_agents / numpy spec object to a small, JSON-
    friendly dict capturing the dimensions that determine SavedModel
    inference compatibility.

    Works with both TF backend (TensorSpec with TensorShape) and NP
    backend (BoundedArraySpec with plain tuple). A leading batch-like
    dim (None or -1) is stripped so specs from a SavedModel (which
    expose shape (None, K)) compare cleanly to specs from a per-
    timestep env method (which expose shape (K,)).
    """
    if spec is None:
        return None
    shape_attr = getattr(spec, 'shape', None)
    if shape_attr is None:
        return {"shape": None, "dtype": str(getattr(spec, 'dtype', 'unknown'))}
    try:
        shape_list = (
            list(shape_attr.as_list())
            if hasattr(shape_attr, 'as_list')
            else list(shape_attr))
    except Exception:
        shape_list = []
    # Strip a single leading batch dim if it looks like one (None or
    # -1). We do NOT strip a leading 1, because a legitimate 1-element
    # observation/action axis is a real (1,) shape that should be
    # preserved for comparison.
    if shape_list and (shape_list[0] is None or shape_list[0] == -1):
        shape_list = shape_list[1:]
    return {
        "shape": [int(d) if (d is not None and d != -1) else None
                  for d in shape_list],
        "dtype": str(getattr(spec, 'dtype', 'unknown')),
    }

def _normalize_dtype_str(s):
    """Map the various dtype string representations we encounter to
    a canonical form. The same logical dtype shows up as:
      "<dtype: 'float32'>"     (tf.float32 stringified)
      "tf.float32"             (tf.dtypes.DType repr)
      "float32"                (numpy / canonical)
    All three should compare equal.
    """
    if not s:
        return ''
    s = str(s).strip().lower()
    if "'" in s:
        # "<dtype: 'float32'>" -> "float32"
        try:
            return s.split("'")[1]
        except IndexError:
            return s
    return s.replace('tf.', '').replace('numpy.', '').replace('np.', '')

def _specs_compatible(a, b):
    """True iff two _spec_to_dict() outputs describe specs that the
    SavedModel inference path will accept. None entries in the shape
    list (after batch-strip) act as wildcards on either side.

    A missing spec on either side is treated as "unknown" and returns
    False - the conservative choice for the dashboard's "Compat"
    column, since we'd rather warn the user on a legacy model than
    silently green-light it.
    """
    if a is None or b is None:
        return False
    a_shape = a.get('shape') or []
    b_shape = b.get('shape') or []
    if len(a_shape) != len(b_shape):
        return False
    for da, db in zip(a_shape, b_shape):
        if da is None or db is None:
            continue  # wildcard match
        if da != db:
            return False
    a_dt = _normalize_dtype_str(a.get('dtype', ''))
    b_dt = _normalize_dtype_str(b.get('dtype', ''))
    # Empty dtype on either side -> can't reject just on that.
    if a_dt and b_dt and a_dt != b_dt:
        return False
    return True

def publish_env_spec(env, robot_type=None):
    """Write the current env's specs into MongoDB so the dashboard can
    flag model-vs-env compatibility per row in the Models tab.

    One document per robot_type, upserted on each call - so when the
    env shape changes (e.g. someone added a sensor), the document
    gets overwritten and the Models-tab flags update on the next
    poll. Idempotent and cheap (one Mongo write per training/eval
    job start); failure is logged but never propagates because env-
    spec publication is a UX nicety, not a correctness requirement.
    """
    robot_type = robot_type or os.getenv('ROBOT_TYPE') or 'unknown'
    try:
        obs_d = _spec_to_dict(env.observation_spec())
        act_d = _spec_to_dict(env.action_spec())
    except Exception as e:
        print(f"publish_env_spec: failed to extract specs: {e}", flush=True)
        return
    try:
        ts = time.time()
        db.env_specs.update_one(
            {"robot_type": robot_type},
            {"$set": {
                "robot_type": robot_type,
                "observation_spec": obs_d,
                "action_spec": act_d,
                "updated_at": datetime.datetime.fromtimestamp(ts, None),
            }},
            upsert=True)
        print(
            f"publish_env_spec: robot_type={robot_type} "
            f"obs={obs_d} act={act_d}", flush=True)
    except Exception as e:
        print(f"publish_env_spec: mongo write failed: {e}", flush=True)


def _extract_savedmodel_specs(saved_policy):
    """Best-effort spec extraction from a tf.saved_model.load'd policy.

    Returns:
      (observation_spec, action_spec) where either or both may be
      None if the respective spec couldn't be extracted. **Never
      raises**; the caller uses None as a signal to skip the
      corresponding compatibility check.

    Background:
      tf_agents' PolicySaver writes a SavedModel whose top-level
      object exposes ``action(time_step, policy_state)`` as a tf
      ConcreteFunction. The function's ``structured_input_signature``
      carries the TimeStep namedtuple of TensorSpecs, and
      ``structured_outputs`` carries the PolicyStep namedtuple whose
      ``.action`` field is the action TensorSpec. We pull from those
      attributes rather than ``saved_policy.time_step_spec``, which
      exists on a live-in-memory PySavedModelPolicy wrapper but NOT
      on the raw ``_UserObject`` that ``tf.saved_model.load()``
      returns. The original pre-flight check used the live-wrapper
      attribute and broke every EVAL job ("'_UserObject' object has
      no attribute 'time_step_spec'") - the runtime-error fallback
      in load_saved_model is the actual safety net that prevents
      the sim-controller crash.

    Compatible with tf_agents 0.10 / 0.11 SavedModels (which is what
    this codebase produces). If a future tf_agents version reshapes
    the saved-policy surface, this just returns (None, None) and we
    degrade to "no pre-flight" - never to a crash and never to a
    bogus EvalSpecMismatchError.
    """
    obs_spec = None
    act_spec = None
    try:
        action_fn = getattr(saved_policy, 'action', None)
        if action_fn is None:
            return obs_spec, act_spec
        # structured_input_signature is a 2-tuple (args, kwargs)
        # where `args` is the positional-arg signature - typically
        # (TimeStep,) for tf_agents-saved policies. The TimeStep is
        # itself a namedtuple of TensorSpecs; we want .observation.
        sig = getattr(action_fn, 'structured_input_signature', None)
        if sig and len(sig) > 0 and len(sig[0]) > 0:
            ts = sig[0][0]
            obs_spec = getattr(ts, 'observation', None)
        # structured_outputs is the PolicyStep namedtuple of
        # TensorSpecs returned by action(). Its .action field is the
        # action's TensorSpec.
        out_sig = getattr(action_fn, 'structured_outputs', None)
        if out_sig is not None:
            act_spec = getattr(out_sig, 'action', None)
    except Exception:
        # Quietly degrade. The caller treats None as "spec unknown,
        # skip pre-flight" rather than as a mismatch.
        pass
    return obs_spec, act_spec


def add_model(path, robot_type, model_type, training_iterations, avg_return=None,
              observation_spec=None, action_spec=None, reward_design=None,
              job_id=None, experiment_design=None, is_global_best=True):
    """Persist a new saved-model record to MongoDB.

    Extended with ``observation_spec`` / ``action_spec`` kwargs so the
    Models tab's "Compat" column can compare a model's training-time
    expectations against the current env. Both default to None (legacy
    callers + RandomPyPolicy that doesn't actually have a learned
    spec); the dashboard renders missing specs as "unknown" compat
    rather than failing.

    Also extended with ``reward_design``: the reward_designs document
    that was active during this model's training, if any. We stamp the
    design's id, version, name, AND raw code onto the model record so
    historical models remain reproducible even if the design is later
    edited or archived. Models trained without a reward design store
    None in all those fields and render as "— default —" on the
    Models tab's Reward column.

    And extended with ``job_id``: the ObjectId of the TRAIN job that
    produced this model. Stored as a string so it's JSON-friendly for
    the dashboard's /get_models endpoint. The dashboard's
    Models -> Analysis -> TensorBoard comparison flow uses this to
    look up the model's TB run directory (``<job_id>/learner/...``
    under /tmp/active or /tmp/jobsdata) and build a regex filter for
    TB's run-name search box. Legacy models predating this field have
    job_id=None and won't be filterable in TB; the new flow degrades
    gracefully by just leaving those out of the filter pattern.
    """
    # "_id": ObjectID(),
    # model_type: 'SacAgent',
    # training_iterations: 50000,
    # location: '/saved_models/niryo/SacAgent/8',
    # notes: 'this is a dummy field',
    # robot_type: 'niryo'
    ts = time.time()
    iso_date = datetime.datetime.fromtimestamp(ts, None)
    # Pull the reward-design metadata into top-level fields on the
    # model record. We deliberately store the code STRING, not a
    # reference - if the user later edits or deletes the original
    # design, this record still tells us exactly what reward shaped
    # the model.
    if reward_design:
        rd_id = reward_design.get("_id")
        rd_version = reward_design.get("version")
        rd_name = reward_design.get("name")
        rd_code = reward_design.get("code")
    else:
        rd_id = rd_version = rd_name = rd_code = None
    # Experiment-design provenance. Unlike reward designs we don't
    # store the full fields dict on the model (it's redundant with
    # the experiment_designs collection, and re-storing every knob on
    # every checkpoint bloats /get_models). Storing _id + name +
    # version is enough to look up the exact design doc the trainer
    # used, even if the design has since been edited (each save bumps
    # version, so version=N identifies a frozen field-set).
    if experiment_design:
        ed_id = experiment_design.get("_id")
        ed_version = experiment_design.get("version")
        ed_name = experiment_design.get("name")
    else:
        ed_id = ed_version = ed_name = None

    # Maintain the "exactly one model per job_id has is_global_best=True"
    # invariant. When the caller is saving a new global best (the
    # default + only case the trainer currently emits), demote every
    # prior is_global_best=True record for the same job to False before
    # inserting this one. Models predating this field have it absent;
    # they stay absent (the dashboard treats absent as "unknown"). A
    # one-shot backfill script can mark historical records all at once
    # if the operator wants the field populated retroactively.
    if is_global_best and job_id is not None:
        try:
            res = db.models.update_many(
                {"job_id": str(job_id), "is_global_best": True},
                {"$set": {"is_global_best": False}})
            if res.modified_count:
                print(
                    f"add_model: demoted {res.modified_count} prior "
                    f"is_global_best=True record(s) for job {job_id}.",
                    flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"add_model: failed to demote previous best: {_e}",
                  flush=True)

    db.models.insert_one(
        {
            "create_date": iso_date,
            "location": path,
            "robot_type": robot_type,
            "model_type": model_type,
            "training_iterations": training_iterations,
            "notes": "NA",
            "avg_return": float(avg_return) if avg_return is not None else None,
            # Spec metadata for compatibility checking at EVAL time and
            # on the Models tab. _spec_to_dict handles None cleanly.
            "observation_spec": _spec_to_dict(observation_spec),
            "action_spec": _spec_to_dict(action_spec),
            # Reward-design provenance. All None when training used the
            # course's default reward formulas (no design selected).
            "reward_design_id": rd_id,
            "reward_design_version": rd_version,
            "reward_design_name": rd_name,
            "reward_design_code": rd_code,
            # Experiment-design provenance. _id + name + version is
            # enough to look up the frozen field-set on the
            # experiment_designs collection; we deliberately don't
            # store the field dict here to keep /get_models slim.
            "experiment_design_id": str(ed_id) if ed_id is not None else None,
            "experiment_design_version": ed_version,
            "experiment_design_name": ed_name,
            # Training-job id as a string so the dashboard can map this
            # model back to its TensorBoard run directory for the
            # multi-model comparison flow. None for legacy callers; the
            # dashboard treats those as "not TB-filterable".
            "job_id": str(job_id) if job_id is not None else None,
            # Trainer git provenance (see _read_git_provenance above).
            # Lets the future multiverse tooling identify which code
            # version produced this checkpoint. Both fields may be
            # None on a misconfigured container (missing /git_meta
            # mount) - dashboard renders that as "no git provenance".
            "git_sha": _GIT_SHA,
            "git_branch": _GIT_BRANCH,
            # Whether this record was the global-best avg_return for
            # its job AT TIME OF INSERT. With the resume-aware
            # max_avg_return seeding in main(), every saved model IS
            # a new global best - so this is True by default. The
            # invariant "at most one model per job has
            # is_global_best=True" is maintained by the demote query
            # above. Field is absent on records pre-2026-05-25
            # (run backfill_global_best.py to populate them).
            "is_global_best": bool(is_global_best),
        })

def save_results_to_db(path, results):
    """Persist an EVAL run's per-trial samples to db.leaderboard_scores.

    ``results`` accepts either:
      * a legacy list (just AverageReturn per trial), for back-compat
        with any direct callers that haven't migrated, OR
      * a dict produced by run_policy / _build_eval_result with the
        full set of per-trial arrays (AverageReturn + episode length
        + course-level unbiased metrics: speed, goals_per_episode,
        steering_angle_ratio).

    The leaderboard_scores schema gains four new parallel arrays
    alongside the existing ``scores``:

      episode_lengths           - tf-agents AverageEpisodeLength per trial
      avg_speeds                - course speeds_total / steps_total per trial
      avg_goals_per_episode     - course goals / episodes per trial
      avg_steering_angle_ratios - course steering ratio sum / steps per trial

    All four are reward-design INVARIANT, so the Analysis tab can use
    them to compare models trained with different reward shapings.
    A ``None`` entry in any of these arrays means "metric unavailable
    for this trial" (e.g., the course doesn't track that counter, or
    the trial completed zero episodes); the Analysis tab filters those
    out before computing means / CIs.

    The aggregate fields (mean_score, median_score, min_score,
    max_score) are still computed off ``scores`` so existing
    consumers (Leaderboard tab, /get_leaderboard_scores) keep working.
    """
    # Back-compat: accept the legacy list-of-floats shape too.
    if isinstance(results, dict):
        returns = results.get("returns") or []
        episode_lengths = results.get("episode_lengths") or []
        avg_speeds = results.get("avg_speeds") or []
        avg_goals = results.get("avg_goals_per_episode") or []
        avg_steering = results.get("avg_steering_angle_ratios") or []
        episode_counts = results.get("episode_counts") or []
    else:
        returns = list(results or [])
        episode_lengths = []
        avg_speeds = []
        avg_goals = []
        avg_steering = []
        episode_counts = []

    if not returns:
        print("Warning: No results to save. Skipping DB insert.", flush=True)
        return
    print("Has results to save.", flush=True)
    saved_model_object = db.models.find_one({"location": path})
    if saved_model_object is None:
        print(f"Warning: no model found at location {path}; skipping leaderboard_scores insert.",
              flush=True)
        return
    ts = time.time()
    iso_date = datetime.datetime.fromtimestamp(ts, None)
    # Aggregate stats are computed off ``returns`` (the legacy
    # AverageReturn array) for back-compat with the Leaderboard tab.
    # The reward-invariant per-trial arrays are stored as-is for the
    # Analysis tab to do its own aggregation.
    db.leaderboard_scores.insert_one(
        {
            "create_date": iso_date,
            "path": saved_model_object["location"],
            "mean_score": float(np.mean(returns)),
            "robot_type": saved_model_object["robot_type"],
            "model_type": saved_model_object["model_type"],
            "model_id": saved_model_object["_id"],
            "median_score": float(np.median(returns)),
            "min_score": float(np.min(returns)),
            "max_score": float(np.max(returns)),
            "scores": np.asarray(returns).tolist(),
            # Reward-design-invariant per-trial arrays for the Analysis
            # tab. Each is parallel-indexed with ``scores``. Entries
            # can be None (= "metric unavailable for this trial").
            "episode_lengths": list(episode_lengths),
            "avg_speeds": list(avg_speeds),
            "avg_goals_per_episode": list(avg_goals),
            "avg_steering_angle_ratios": list(avg_steering),
            # Raw per-trial crash count (num_episodes_total delta). One
            # episode = one crash in the donut course. Stored so the
            # Leaderboard tab can show crash frequency and steps-per-goal
            # without re-deriving them from the ratio fields.
            "episode_counts": list(episode_counts),
        })

def log_reward(job_id, type, score, diff=None, extra_data=None, step_costs=[], position_history=[],stat_array=[]):
    dilimeter = ","
    step_costs_valid = None if len(step_costs) == 0 else dilimeter.join([str(i) for i in step_costs])
    position_history_valid = None if len(position_history) == 0 else dilimeter.join([str(i) for i in position_history])
    stat_array_valid = None if len(stat_array) == 0 else dilimeter.join([str(i) for i in stat_array])
    new_log = {
        "job_id": job_id,
        "type": type,
        "score": score,
        "diff": diff,
        "extra_data": extra_data,
        "step_costs": step_costs_valid,
        "position_history": position_history_valid,
        "stat_array": stat_array_valid
    }
    db.logs.insert_one(new_log)
def log_blob(blob):
    db.logs.insert_one(blob)

def _is_queue_paused():
    """Global queue pause switch (separate from per-job pause).

    Reads a singleton document in the `queue_control` collection that the
    dashboard's Pause/Resume-queue button toggles. When paused, run_jobs_loop
    stops picking up NEW jobs (job statuses are left untouched), so the
    operator can halt the queue, test something, then resume right where the
    queue left off. Fails open (returns False) on any error so a Mongo hiccup
    can't wedge the trainer into a permanent idle.
    """
    try:
        doc = db.queue_control.find_one({"_id": "singleton"})
        return bool(doc and doc.get("paused"))
    except Exception as e:  # noqa: BLE001
        print(f"_is_queue_paused check failed (treating as not paused): {e}",
              flush=True)
        return False


def get_jobs():
    debug_print("in get_jobs")
    is_not_started = {"status": "NOT_STARTED"}
    is_in_progress = {"status": "IN_PROGRESS"}
    # Drain the queue in a predictable FIFO order by creation time so the
    # "next" job picked up after one pauses/finishes is the oldest queued
    # one, not an arbitrary Mongo natural-order pick. create_date is
    # stamped at job creation; jobs missing it (legacy) sort first under
    # ascending order, which is fine.
    jobs = db.jobs.find({"$or":[is_not_started, is_in_progress]}).sort("create_date", 1)
    debug_print(jobs)
    return jobs

def _signal_gym_switch(job):
    """Tell the Unity supervisors which gym binary this job wants.

    POSTs the job's gym file_path to the dashboard's /set_desired_gym
    endpoint (broadcast to all actor indices via index='*'). Each
    RunClientWrapper.ps1 supervisor on the Windows host polls
    /get_desired_gym?index=<N> and hot-swaps the Unity binary when the
    requested source build differs from what it's running.

    Fires for ANY job type that carries gym_file_path. No-op (and never
    raises) when the job has no gym or the dashboard is unreachable, so a
    missing/locked dashboard never blocks a job from running.
    """
    raw_path = job.get("gym_file_path") or ""
    # The Gyms-tab form may have stored a path the user pasted WITH
    # surrounding quotes (e.g. copied from Explorer's "Copy as path",
    # which wraps the value in double quotes). Strip a single matching
    # pair of leading/trailing single or double quotes so the path the
    # supervisor receives is a clean filesystem path.
    gym_file_path = raw_path.strip()
    if len(gym_file_path) >= 2 and gym_file_path[0] == gym_file_path[-1] and gym_file_path[0] in ("'", '"'):
        gym_file_path = gym_file_path[1:-1].strip()

    gym_name = job.get("gym_name") or job.get("gym_id") or "(none)"
    print(f"do_job: gym={gym_name!r} file_path={gym_file_path!r}", flush=True)

    if not gym_file_path:
        return

    try:
        import urllib.request, json as _json
        payload = _json.dumps({
            "gym_id":    str(job.get("gym_id", "")),
            "gym_name":  str(job.get("gym_name", "")),
            "file_path": gym_file_path,
        }).encode()
        req = urllib.request.Request(
            "http://dashboard/set_desired_gym",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(
                f"do_job: gym switch signalled to dashboard "
                f"({resp.status}): {gym_file_path!r}",
                flush=True,
            )
    except Exception as e:
        print(f"do_job: gym switch signal failed (non-fatal): {e}", flush=True)


def do_job(job, num_envs=1):
    print(job["job_type"])
    # Decide resume-vs-fresh BEFORE flipping status to IN_PROGRESS, so
    # _detect_resume_for_train_job can still see the operator's
    # original pickup-time status (= the discriminator between
    # NOT_STARTED + paused_at_step = explicit resume, and IN_PROGRESS
    # = crash recovery). Once we write IN_PROGRESS to Mongo, that
    # signal is destroyed.
    _is_train_resume = (
        job.get("job_type") == "TRAIN"
        and _detect_resume_for_train_job(job))

    update_job(job["_id"], "IN_PROGRESS")

    # Clear any stale `ended_at` from a previous run of this job (a
    # re-queue, resume, or crash-recovery). Without this, the dashboard's
    # duration cell sees a leftover ended_at and renders the FROZEN old
    # duration (ended - started) instead of ticking live; and if started_at
    # was refreshed to be newer than the stale ended_at, it clamps to 0 -
    # the "zero / wrong duration" symptom. Nulling it here means a running
    # job has started_at + null ended_at, so the UI shows live elapsed,
    # and the end-of-job trailer stamps a fresh ended_at on completion.
    update_job(job["_id"], None, "ended_at")

    # Conditional percent_complete reset.
    #   * Fresh pickup: zero out so a re-queued job doesn't carry stale
    #     progress from a previous failed run.
    #   * Resume (either explicit Pause/Resume cycle OR crash recovery
    #     where the trainer just came back up against an IN_PROGRESS
    #     job with a Learner checkpoint on disk): preserve the prior
    #     percent so the Jobs-tab progress bar picks up where the
    #     operator left it. The training loop's first iteration will
    #     refresh it from the restored train_step.
    if not _is_train_resume:
        update_job(job["_id"], 0, "percent_complete")
    else:
        print(
            f"do_job: preserving percent_complete on resume of "
            f"{job['_id']} (paused_at_step={job.get('paused_at_step')}, "
            f"pickup_status={job.get('status')!r}); first training-loop "
            f"update will refresh it.",
            flush=True)
    # Stamp trainer git provenance on the job at pickup so a job
    # re-picked-up after a code edit (the user marks it NOT_STARTED
    # again, or the trainer crash-restarts an IN_PROGRESS job) carries
    # the SHA of whichever trainer process is actually running it now,
    # not the SHA from when the job was originally queued.
    update_job(job["_id"], _GIT_SHA, "trainer_git_sha")
    update_job(job["_id"], _GIT_BRANCH, "trainer_git_branch")
    # Wall-clock duration tracking. We stamp `started_at` here (right
    # after the IN_PROGRESS transition) and `ended_at` immediately
    # before the DONE transition at the bottom of this function. Stored
    # as a timezone-aware UTC datetime so pymongo writes a BSON Date
    # and Express's res.json serialises it as ISO-8601 with a 'Z'
    # suffix. The Jobs grid in dashboard/jobs.html uses these to
    # render a Duration column that ticks live for IN_PROGRESS jobs.
    #
    # We stamp started_at = NOW on every FRESH pickup, OVERWRITING any
    # pre-existing value. This is deliberate: some jobs arrive with a
    # queue-time started_at already set (e.g. older MadScientist
    # orchestrator builds stamped it at creation, and you can see it on
    # never-run NOT_STARTED jobs whose started_at == create_date). If we
    # merely preserved that, Duration would measure time-since-QUEUED -
    # a job that waited days in the queue then trained briefly would show
    # a multi-day duration. Overwriting on fresh pickup makes Duration
    # reflect actual run time.
    #
    # On RESUME (pause/resume or crash-recovery) we KEEP the original
    # started_at so the duration spans the whole training (including the
    # paused gap). A future "active training time" metric could subtract
    # the PAUSED windows, but for now total wall-clock-since-first-run is
    # the natural meaning for a resumed job.
    if not _is_train_resume:
        update_job(job["_id"], datetime.datetime.now(datetime.timezone.utc), "started_at")
    #Move all data for jobs with _id = job["_id"] from /tmp to /jobsdata
    job_type = job["job_type"]

    # Gym hot-swap signal — fires for EVERY job type that carries a gym so
    # the Unity supervisors switch to the requested binary before the job
    # runs. Must run BEFORE the job_type branches below (TRAIN starts the
    # env immediately, EVAL builds its env right away too), so the
    # supervisor has the desired-gym path queued when the env first resets.
    _signal_gym_switch(job)

    if job_type == "DEMO":
        # Robust against every shape the New-job form (or a hand-written
        # queue script) can send: field omitted entirely (KeyError on plain
        # dict indexing), empty string, explicit 0/None - all fall back to
        # the same defaults do_job has always advertised for DEMO jobs.
        def _demo_int_field(name, default):
            v = job.get(name)
            if v is None or v == "":
                return default
            try:
                v = int(v)
            except (TypeError, ValueError):
                return default
            return v if v > 0 else default
        num_iterations = _demo_int_field("num_iterations", 50000)
        demo_training_steps = _demo_int_field("training_steps", 20000)
        # Optional job-level override to skip straight to a later curriculum
        # stage instead of always starting collection at stage 0 - added
        # 2026-07-19 to let a cloned validation job re-test just the stage
        # that was crashing (e.g. stage 1, the first one with a chicane)
        # without re-collecting the easy stage(s) before it every time.
        # Default 0 preserves the original "always start at stage 0" behavior.
        demo_start_stage = _demo_int_field("demo_start_stage", 0)
        # Number of Unity gym instances to collect from IN PARALLEL (added
        # 2026-07-21). Each gym runs its own independent
        # collect_expert_demos() loop (own env, own course/goal state, own
        # sub-batch tfrecord files - see gym_index param) on a background
        # thread; do_job blocks until every gym finishes the current
        # stage/geometry before moving on. Threads (not processes) are fine
        # here: almost all of collect_expert_demos' per-step time is spent
        # blocked on gRPC calls to Unity, which release the GIL, so N
        # threads collecting concurrently scale close to linearly despite
        # the GIL. Default 1 preserves the original single-gym behavior for
        # every job that doesn't opt in.
        demo_num_gyms = max(1, _demo_int_field("demo_num_gyms", 1))
        env = make_env('ros-server-0:50051')
        _demo_envs = [env] + [
            make_env(f'ros-server-{_i}:50051', actor_index=_i)
            for _i in range(1, demo_num_gyms)]
        if demo_num_gyms > 1:
            print(f"do_job: DEMO collecting across {demo_num_gyms} parallel "
                  f"Unity gyms (ros-server-0..{demo_num_gyms - 1})",
                  flush=True)

        def _collect_on_all_gyms(num_episodes, batch_number, stage=None,
                                  num_stages=None, goal_budget=None):
            """Run collect_expert_demos on every gym env concurrently.

            Each gym independently targets the FULL goal_budget/num_episodes
            (not a split share), so demo_num_gyms>1 multiplies total demo
            volume collected per stage in roughly the SAME wall-clock time
            as a single gym, rather than trading volume for speed.
            """
            if len(_demo_envs) == 1:
                collect_expert_demos(
                    _demo_envs[0], num_episodes, job["_id"],
                    batch_number=batch_number, stage=stage,
                    num_stages=num_stages, goal_budget=goal_budget,
                    gym_index=0)
                return
            _threads = [
                threading.Thread(
                    target=collect_expert_demos,
                    args=(_gym_env, num_episodes, job["_id"]),
                    kwargs=dict(
                        batch_number=batch_number, stage=stage,
                        num_stages=num_stages, goal_budget=goal_budget,
                        gym_index=_gi),
                    name=f"demo-gym-{_gi}",
                    daemon=True)
                for _gi, _gym_env in enumerate(_demo_envs)]
            for _t in _threads:
                _t.start()
            for _t in _threads:
                _t.join()

        # Optional curriculum-spread collection: if the job references an
        # experiment_design with curriculum_stages (the same field TRAIN jobs
        # use), split num_iterations evenly across every stage and reconfigure
        # the env's corner_radius/curvature_difficulty between chunks, instead
        # of collecting entirely on RobotaxiEnv's fixed default geometry. This
        # gives the demo buffer real examples at every difficulty level a
        # curriculum TRAIN job will traverse, rather than only the easiest one.
        _demo_curriculum_stages = None
        _demo_ed_id_raw = job.get("experiment_design_id")
        if _demo_ed_id_raw:
            from bson import ObjectId
            _demo_ed_doc = None
            try:
                _demo_ed_doc = db.experiment_designs.find_one(
                    {"_id": ObjectId(str(_demo_ed_id_raw))})
            except Exception:
                _demo_ed_doc = None
            if _demo_ed_doc is None:
                _demo_ed_doc = db.experiment_designs.find_one(
                    {"_id": str(_demo_ed_id_raw)})
            if _demo_ed_doc and _demo_ed_doc.get("curriculum_stages"):
                _demo_curriculum_stages = _demo_ed_doc["curriculum_stages"]

        if _demo_curriculum_stages:
            n_stages = len(_demo_curriculum_stages)
            _start_stage = min(max(demo_start_stage, 0), n_stages - 1)
            _stages_to_run = n_stages - _start_stage
            # Budget is GOALS REACHED per stage, not raw episode count
            # (changed 2026-07-20 - see collect_expert_demos' goal_budget
            # docstring). A fixed episode count made per-stage wall-clock
            # time wildly unpredictable once episode length became
            # policy-dependent: a well-driving car can now survive
            # thousands of steps/episode instead of crashing quickly, so
            # "12,500 episodes" that used to take a few hours could take
            # months at the new, much healthier episode-survival rate.
            # Goals reached is a direct measure of how much useful demo
            # data was actually collected, independent of how long each
            # episode happens to last.
            #
            # Fixed per-stage goal target (2026-07-20, same day): was
            # `num_iterations // _stages_to_run` (e.g. 12,500/stage from a
            # 50,000 total) - at the observed ~800-850 goals/hour collection
            # rate that meant ~14.5 hours per stage, far longer than
            # intended. Replaced with a flat GOALS_PER_STAGE constant so
            # stage transitions happen at a predictable, short cadence
            # (~100 goals / ~800/hr =~ 7-8 minutes/stage) regardless of
            # num_iterations, which no longer has any effect on DEMO
            # curriculum timing.
            GOALS_PER_STAGE = 100
            # Episode-count safety cap so a policy that can't reach goals
            # at all (or only very rarely) can't collect forever - see
            # collect_expert_demos' goal_budget docstring. Generous: at
            # even a very poor ~50 steps/episode this still allows 5M+
            # steps before the cap alone would stop a stage.
            _episode_safety_cap = 100000
            print(
                f"do_job: DEMO collecting across {n_stages} curriculum "
                f"stages from experiment design {_demo_ed_id_raw!r} "
                f"({GOALS_PER_STAGE} goals/stage, "
                f"starting at stage {_start_stage}/{n_stages - 1})",
                flush=True)
            for _stage_idx, _stage in enumerate(_demo_curriculum_stages):
                if _stage_idx < _start_stage:
                    continue
                _cr = float(_stage.get("corner_radius", 10.0))
                _cd = float(_stage.get("curvature_difficulty", 0.0))
                _ch_n = int(_stage.get("chicanes_north", 0))
                _ch_e = int(_stage.get("chicanes_east", 0))
                _ch_s = int(_stage.get("chicanes_south", 0))
                _ch_w = int(_stage.get("chicanes_west", 0))
                _n_goals = GOALS_PER_STAGE
                for _gym_env in _demo_envs:
                    configure_env(_gym_env, corner_radius=_cr, curvature_difficulty=_cd,
                                 chicanes_north=_ch_n, chicanes_east=_ch_e,
                                 chicanes_south=_ch_s, chicanes_west=_ch_w)
                print(
                    f"[demo-curriculum] t={_ts()} stage {_stage_idx}/{n_stages - 1}: "
                    f"corner_radius={_cr}, curvature_difficulty={_cd}, "
                    f"chicanes(N/E/S/W)={_ch_n}/{_ch_e}/{_ch_s}/{_ch_w}, "
                    f"goal_budget={_n_goals}/gym x {demo_num_gyms} gym(s)", flush=True)
                _collect_on_all_gyms(
                    _episode_safety_cap, batch_number=_stage_idx,
                    stage=_stage_idx, num_stages=n_stages, goal_budget=_n_goals)
        else:
            # No curriculum on the referenced experiment design (or none
            # referenced at all) - fall back to a single fixed geometry from
            # the job's own corner_radius_val/curvature_difficulty_val
            # fields (the New-job form's "Corner radius"/"Curvature
            # difficulty" inputs), defaulting to RobotaxiEnv's own defaults
            # (10.0 / 0.0) if the job doesn't set them either.
            _demo_cr = float(job.get("corner_radius_val") or 10.0)
            _demo_cd = float(job.get("curvature_difficulty_val") or 0.0)
            _demo_ch_n = int(job.get("chicanes_north_val") or 0)
            _demo_ch_e = int(job.get("chicanes_east_val") or 0)
            _demo_ch_s = int(job.get("chicanes_south_val") or 0)
            _demo_ch_w = int(job.get("chicanes_west_val") or 0)
            for _gym_env in _demo_envs:
                configure_env(_gym_env, corner_radius=_demo_cr, curvature_difficulty=_demo_cd,
                             chicanes_north=_demo_ch_n, chicanes_east=_demo_ch_e,
                             chicanes_south=_demo_ch_s, chicanes_west=_demo_ch_w)
            print(
                f"do_job: DEMO collecting on fixed geometry "
                f"corner_radius={_demo_cr}, curvature_difficulty={_demo_cd}, "
                f"chicanes(N/E/S/W)={_demo_ch_n}/{_demo_ch_e}/{_demo_ch_s}/{_demo_ch_w} "
                f"({num_iterations} episodes/gym x {demo_num_gyms} gym(s))", flush=True)
            _collect_on_all_gyms(num_iterations, batch_number=0)

        root_dir = "/tfrecords/job_" + str(job["_id"])
        collect_training_data.train_agent(root_dir, demo_training_steps)
        print("after collect_expert_demos")
    elif job_type == "BC_TRAINING_ONLY":
        root_dir = "/tfrecords/job_" + str(job["demo_job_id"])
        collect_training_data.train_agent(root_dir, int(job["training_steps"]))
        print("after collect_expert_demos")
    elif job_type == "TRAIN":
        # Resume detection MUST run before move_all_jobs_data, because
        # move_all_jobs_data's trailing self-cleanup (move_data) would
        # archive THIS job's eval/metrics/train/learner subdirs out of
        # /tmp/active/<id>/ otherwise - taking the Learner checkpoint
        # we're about to read with them.
        #
        # The authoritative resume signal is the job document's
        # paused_at_step (written by main()'s pause break path). If
        # it's set, we know the trainer previously saved a Learner
        # checkpoint for this job; _restore_paused_active_dir handles
        # putting it back where main() expects it, regardless of
        # whether the archive sits at /tmp/jobsdata/<id>/ (singly
        # nested - pause-and-resume cycle alone) or
        # /tmp/jobsdata/<id>/<id>/ (doubly nested - a different job
        # ran between pause and resume).
        # Resume gating. _is_train_resume was computed up at the top
        # of do_job (before status was flipped to IN_PROGRESS) and
        # covers BOTH explicit pause/resume AND crash recovery.
        is_resume = False
        if _is_train_resume:
            is_resume = _restore_paused_active_dir(str(job["_id"]))
            if not is_resume and job.get("paused_at_step") is not None:
                # Explicit-resume signal with no restoreable data on
                # disk. Confusing for the UI (the job will appear
                # "resumable" but each Resume click will silently fall
                # through to a fresh start). Clear paused_at_step so
                # future Restart operations behave as expected.
                print(
                    f"do_job: job {job['_id']} has paused_at_step="
                    f"{job.get('paused_at_step')} but no Learner "
                    f"checkpoint could be restored. Treating as fresh "
                    f"start; BC pretrain will run and the previous "
                    f"actor weights are lost.",
                    flush=True)
                try:
                    db.jobs.update_one(
                        {"_id": job["_id"]},
                        {"$unset": {"paused_at_step": ""}})
                except Exception as _e:  # noqa: BLE001
                    print(f"do_job: failed to clear stale paused_at_step: {_e}",
                          flush=True)
            elif is_resume and job.get("paused_at_step") is not None:
                # Consume the explicit-resume signal. Crash recovery
                # path doesn't have paused_at_step set, so nothing to
                # $unset there.
                try:
                    db.jobs.update_one(
                        {"_id": job["_id"]},
                        {"$unset": {"paused_at_step": ""}})
                except Exception as _e:  # noqa: BLE001
                    print(f"do_job: failed to clear paused_at_step on resume: {_e}",
                          flush=True)
        # Skip the self-cleanup tail of move_all_jobs_data on resume
        # so the data we just restored isn't immediately archived back
        # out. The outer-loop pass that archives OTHER jobs' dirs out
        # of /tmp/active/ still runs - we always want to clean those.
        move_all_jobs_data(job["_id"], skip_current_cleanup=is_resume)
        # Bound /tmp/jobsdata growth: keep only the JOBSDATA_MAX_ARCHIVES
        # most-recent job buckets (default 100). Protects the current job.
        prune_jobsdata(current_id=job["_id"])
        # Job fields can arrive from Mongo as either None (never set)
        # or the empty string '' (cleared via dashboard form). Both
        # mean "fall back to default". Casting '' to int blows up
        # with ValueError, so route every nullable-numeric field
        # through this helper.
        def _int_or(val, default):
            if val is None or val == "":
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        num_iterations = _int_or(job.get("num_iterations"), 50000)
        pass_through_actions = (
            job["pass_through_actions"]
            if job.get("pass_through_actions") not in (None, "")
            else False)
        _nn_size_x = _int_or(job.get("nn_size_x"), 512)
        _nn_size_y = _int_or(job.get("nn_size_y"), 512)
        actor_fc_layer_params_x = _nn_size_x
        actor_fc_layer_params_y = _nn_size_y
        critic_joint_fc_layer_params_x = _nn_size_x
        critic_joint_fc_layer_params_y = _nn_size_y
        # Resolve the optional reward design and seed from the job
        # document. reward_design_id can be either an ObjectID (sent
        # by the future dashboard Reward-design tab) or the special
        # string id used for the seeded passthrough design - look up
        # by _id first, fall back to a stable string id match. None
        # means "use course defaults" and we pass None down to main()
        # which short-circuits the install. RewardDesignError raised
        # inside main() is allowed to propagate; the standard
        # eval_error capture at the bottom of this branch catches it.
        reward_design_doc = None
        rd_id_raw = job.get("reward_design_id")
        if rd_id_raw:
            from bson import ObjectId
            try:
                # Try ObjectId first - that's the common case for
                # designs created via the dashboard.
                reward_design_doc = db.reward_designs.find_one(
                    {"_id": ObjectId(str(rd_id_raw))})
            except Exception:
                reward_design_doc = None
            if reward_design_doc is None:
                # Fall back to string match for the seeded passthrough
                # design (whose _id is the string PASSTHROUGH_DESIGN_ID).
                reward_design_doc = db.reward_designs.find_one(
                    {"_id": str(rd_id_raw)})
            if reward_design_doc and reward_design_doc.get("archived"):
                # Archived designs still work for backward-compat with
                # in-flight jobs, but log the fact so a user wondering
                # "why is my training still using that design?" can
                # find the answer in robotaxi.out.
                print(
                    f"do_job: reward design {rd_id_raw} is archived; "
                    "using it for this job anyway. Unarchive on the "
                    "Reward design tab if you want it visible in the "
                    "new-job dropdown.", flush=True)
            if reward_design_doc is None:
                print(
                    f"do_job: reward_design_id {rd_id_raw!r} not found in "
                    "reward_designs collection; falling back to course "
                    "defaults.", flush=True)
        seed = job.get("seed")
        try:
            seed = int(seed) if seed not in (None, "") else None
        except (TypeError, ValueError):
            seed = None

        # Resolve the optional experiment_design_id off the job doc.
        # Same pattern as reward_design_id resolution above:
        #   - try ObjectId match first (dashboard-created designs),
        #   - fall back to string match (seeded canonical "Default"
        #     which uses experiment_designs.DEFAULT_DESIGN_ID).
        # Missing / unknown / archived all fall through to "trainer
        # defaults" without aborting the job.
        experiment_design_doc = None
        ed_id_raw = job.get("experiment_design_id")
        if ed_id_raw:
            from bson import ObjectId
            try:
                experiment_design_doc = db.experiment_designs.find_one(
                    {"_id": ObjectId(str(ed_id_raw))})
            except Exception:
                experiment_design_doc = None
            if experiment_design_doc is None:
                experiment_design_doc = db.experiment_designs.find_one(
                    {"_id": str(ed_id_raw)})
            if experiment_design_doc and experiment_design_doc.get("archived"):
                print(
                    f"do_job: experiment design {ed_id_raw} is archived; "
                    "using it for this job anyway. Unarchive on the "
                    "Experiment design tab if you want it visible in the "
                    "new-job dropdown.", flush=True)
            if experiment_design_doc is None:
                print(
                    f"do_job: experiment_design_id {ed_id_raw!r} not found in "
                    "experiment_designs collection; falling back to trainer "
                    "hardcoded defaults.", flush=True)

        print(job)
        print(f"in do_job pass_through_actions: {pass_through_actions}")
        print(
            f"in do_job reward_design={reward_design_doc.get('name') if reward_design_doc else None!r} "
            f"version={reward_design_doc.get('version') if reward_design_doc else None} "
            f"experiment_design={experiment_design_doc.get('name') if experiment_design_doc else None!r} "
            f"version={experiment_design_doc.get('version') if experiment_design_doc else None} "
            f"seed={seed}")
        # Wrap main() in a RewardDesignError catch so a broken
        # user-supplied reward design surfaces as job.eval_error (like
        # the EvalSpecMismatchError flow below for EVAL) and the
        # trainer can pick up the next queued job. Anything else
        # propagates so real crashes still surface in full detail.
        #
        # NOTE on the import path: robotaxi.py runs with CWD set to
        # /python_ws/src/ inside the sim-controller container (that's
        # the bind-mount of host's ./rl_agent), so reward_designs.py
        # is a SIBLING module, not under a `rl_agent` package. The
        # bare-name `from reward_designs import ...` matches the
        # existing convention `import collect_training_data` at the
        # top of this file; `from rl_agent.reward_designs import ...`
        # raises ModuleNotFoundError because Python doesn't see the
        # `rl_agent` package from inside it.
        from reward_designs import RewardDesignError
        # Build the base kwargs from legacy job-doc fields, then let
        # the experiment_design (if any) overlay its values on top.
        # Order matters: anything explicitly stamped on the job
        # document via the dashboard New-job form (num_iterations,
        # nn_size_x/y) wins for legacy compatibility; the
        # experiment_design fills in everything ELSE (learning rates,
        # gamma, replay capacity, etc.) that the legacy form never
        # exposed.
        #
        # To honour that precedence, do_job's legacy values go into
        # base_kwargs BEFORE the overlay; apply_to_main_kwargs writes
        # over ANY key the design specifies. If users want the
        # design's num_iterations to win, just leave job.num_iterations
        # blank / submit a fresh job that doesn't override it.
        # NOTE: is_resume was decided above (before move_all_jobs_data)
        # so we could pass skip_current_cleanup=is_resume into that call.
        # We just thread the same flag into main() here.

        def _job_float(key, default):
            """Read a float from the job doc, falling back to default."""
            v = job.get(key)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _job_int(key, default):
            """Read an int from the job doc, falling back to default."""
            v = job.get(key)
            if v is None or v == "":
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        # demo_job_ids (optional, job-doc field): extra DEMO-job ids whose
        # /tfrecords/job_<id> directories get ADDED to the long-standing
        # default expert-demo job, all concatenated into one combined
        # expert dataset for prefill + BC pretrain (see main()'s
        # demo_record_dirs_val). Additive rather than a replacement so the
        # default set is never silently dropped just because a job wants to
        # supplement it with a fresh collection.
        _DEFAULT_DEMO_JOB_ID = "64168c1b58d4d8ccdb76e721"
        _extra_demo_job_ids = job.get("demo_job_ids") or []
        if isinstance(_extra_demo_job_ids, str):
            _extra_demo_job_ids = [
                s.strip() for s in _extra_demo_job_ids.split(",") if s.strip()]
        _demo_job_ids = [_DEFAULT_DEMO_JOB_ID] + [
            j for j in _extra_demo_job_ids if j and j != _DEFAULT_DEMO_JOB_ID]
        demo_record_dirs_val = [
            f"/tfrecords/job_{_jid}" for _jid in _demo_job_ids]
        if len(_demo_job_ids) > 1:
            print(f"do_job: TRAIN combining {len(_demo_job_ids)} expert-demo "
                  f"sources: {_demo_job_ids}", flush=True)

        # demo_source_counts (optional, job-doc field): exact per-source
        # target step counts, parallel to _demo_job_ids/demo_record_dirs_val
        # above (same order - index 0 is always the default job). See
        # main()'s demo_source_counts_val docstring. None/absent preserves
        # the default proportional-to-availability behavior.
        _demo_source_counts = job.get("demo_source_counts") or None
        if _demo_source_counts is not None:
            if len(_demo_source_counts) != len(demo_record_dirs_val):
                print(
                    f"do_job: WARNING demo_source_counts has "
                    f"{len(_demo_source_counts)} entries but there are "
                    f"{len(demo_record_dirs_val)} demo sources "
                    f"{demo_record_dirs_val}; ignoring demo_source_counts.",
                    flush=True)
                _demo_source_counts = None
            else:
                print(f"do_job: TRAIN using explicit per-source demo "
                      f"counts {_demo_source_counts} for sources "
                      f"{_demo_job_ids}", flush=True)

        base_kwargs = dict(
            demo_record_dirs_val=demo_record_dirs_val,
            demo_source_counts_val=_demo_source_counts,
            job_id=job["_id"],
            num_envs=num_envs,
            num_iterations_val=num_iterations,
            pass_through_actions=pass_through_actions,
            actor_fc_layer_params_x=actor_fc_layer_params_x,
            actor_fc_layer_params_y=actor_fc_layer_params_y,
            critic_joint_fc_layer_params_x=critic_joint_fc_layer_params_x,
            critic_joint_fc_layer_params_y=critic_joint_fc_layer_params_y,
            eval_interval_val=10,
            reward_design=reward_design_doc,
            experiment_design=experiment_design_doc,
            seed=seed,
            is_resume_val=is_resume,
            # Track-geometry curriculum knobs read directly from the job doc
            # so the user can set them in the New-job form without needing an
            # experiment design. The experiment_design overlay (apply_to_main_kwargs
            # below) still wins if the design also specifies these fields, which
            # preserves the "design is authoritative" precedence everywhere else.
            corner_radius_val=_job_float("corner_radius_val", 10.0),
            curvature_difficulty_val=_job_float("curvature_difficulty_val", 0.0),
            chicanes_north_val=_job_int("chicanes_north_val", 0),
            chicanes_east_val=_job_int("chicanes_east_val", 0),
            chicanes_south_val=_job_int("chicanes_south_val", 0),
            chicanes_west_val=_job_int("chicanes_west_val", 0),
        )
        if is_resume:
            print(
                f"do_job: RESUMING job {job['_id']} - found existing "
                f"Learner checkpoint. Skipping BC pretrain; SAC will "
                f"continue from the saved train_step.",
                flush=True)
        # Bare-name sibling import; see reward_designs import note above.
        from experiment_designs import apply_to_main_kwargs as _apply_ed
        main_kwargs = _apply_ed(experiment_design_doc, base_kwargs)
        try:
            main(**main_kwargs)
        except RewardDesignError as e:
            err_msg = str(e)
            print(
                f"TRAIN reward design error for job {job['_id']}: {err_msg}",
                flush=True)
            update_job(job["_id"], err_msg, "eval_error")
    elif job_type == "EVAL":
        # Dispatch on model_type (a constrained dropdown in the
        # dashboard's job form: SacAgent / GreedyPolicy /
        # RandomPyPolicy) rather than substring-matching the
        # free-text ``location`` field.
        #
        # ``location`` is no longer captured by the create-job form -
        # the supported way to evaluate an existing saved snapshot is
        # to open the dashboard's Models tab, select the row, and
        # click Add Jobs. That flow POSTs an EVAL job whose
        # ``location`` is read from the model's MongoDB record. Form-
        # created EVAL jobs (RandomPyPolicy baseline) reach this
        # branch with no ``location`` field; ``.get(...)`` defaults
        # safely. Saved-model EVAL jobs from the form would have an
        # empty ``location`` and crash inside tf.saved_model.load - by
        # design, since that workflow is now Models-tab-only.
        model_type=job["model_type"]
        location=job.get("location", "")
        debug_print(location)
        # Pull the optional per-job overrides off the document. The
        # Models-tab "Eval selected" modal sets ``num_trials``; legacy
        # rows and the jobs.html "+ New job" form omit it and we fall
        # back to run_policy's defaults (max_episodes=3,
        # num_eval_episodes=5) by passing None through. ``int(...)`` is
        # defensive against the field being stored as a stringified
        # number by some older codepath; falsy/empty values short-
        # circuit to None.
        def _opt_int(v):
            if v is None or v == "":
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        def _opt_float(v, default):
            if v is None or v == "":
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
        num_trials = _opt_int(job.get("num_trials"))
        num_eval_episodes = _opt_int(job.get("num_eval_episodes"))
        max_steps_per_episode = _opt_int(job.get("max_steps_per_episode"))
        max_goals_per_episode = _opt_int(job.get("max_goals_per_episode"))
        # Mirror the TRAIN path: forward the same track-curriculum knobs
        # that main() reads from experiment_designs so the eval track
        # matches the training track geometry.  Defaults match Unity's
        # TrackGenerator inspector defaults so omitting them from the
        # job doc is safe.
        eval_corner_radius = _opt_float(job.get("corner_radius_val"), 10.0)
        eval_curvature_difficulty = _opt_float(job.get("curvature_difficulty_val"), 0.0)
        eval_chicanes_north = _opt_int(job.get("chicanes_north_val")) or 0
        eval_chicanes_east = _opt_int(job.get("chicanes_east_val")) or 0
        eval_chicanes_south = _opt_int(job.get("chicanes_south_val")) or 0
        eval_chicanes_west = _opt_int(job.get("chicanes_west_val")) or 0
        # Wrap the EVAL dispatch in an EvalSpecMismatchError catch so a
        # model with stale observation/action specs against the current
        # env reports cleanly instead of tearing down the whole
        # sim-controller process (see EvalSpecMismatchError docstring
        # for the full motivation). The catch only handles spec
        # mismatches; any OTHER exception (env crash, OOM, etc.) is
        # still allowed to propagate and surface as a real crash, since
        # those usually indicate a deeper problem the user should see.
        try:
            if model_type == "RandomPyPolicy":
                run_randompolicy(
                    job_id=job["_id"],
                    num_trials=num_trials,
                    num_eval_episodes=num_eval_episodes,
                    corner_radius=eval_corner_radius,
                    curvature_difficulty=eval_curvature_difficulty,
                    chicanes_north=eval_chicanes_north,
                    chicanes_east=eval_chicanes_east,
                    chicanes_south=eval_chicanes_south,
                    chicanes_west=eval_chicanes_west)
            else:
                load_saved_model(
                    model_type, path=location, job_id=job["_id"],
                    num_trials=num_trials,
                    num_eval_episodes=num_eval_episodes,
                    max_steps_per_episode=max_steps_per_episode,
                    max_goals_per_episode=max_goals_per_episode,
                    corner_radius=eval_corner_radius,
                    curvature_difficulty=eval_curvature_difficulty,
                    chicanes_north=eval_chicanes_north,
                    chicanes_east=eval_chicanes_east,
                    chicanes_south=eval_chicanes_south,
                    chicanes_west=eval_chicanes_west)
        except EvalSpecMismatchError as e:
            # Record the precise reason on the job doc so the dashboard
            # (Jobs tab and, eventually, the Models tab Compat column)
            # can surface it. Then fall through to the bottom of do_job
            # which stamps ended_at + status=DONE; the next queued job
            # picks up normally on the next get_jobs() iteration.
            err_msg = str(e)
            print(
                f"EVAL spec mismatch for job {job['_id']}: {err_msg}",
                flush=True)
            update_job(job["_id"], err_msg, "eval_error")
        debug_print(model_type)
        debug_print(location)
    else:
        return
    # End-of-job trailer.
    #
    # We unconditionally stamp ended_at so the Jobs tab's Duration
    # column can show the run length.
    #
    # The status update is *conditional*: we only set status=DONE
    # if the job is still IN_PROGRESS. This preserves any externally-
    # written status (e.g., the dashboard's "Set to done" button
    # producing a deliberate DONE while we were mid-training, or a
    # future "Cancel" / "Mark failed" button). Without this guard the
    # trainer would overwrite that user intent with another DONE
    # (benign today, but it would clobber non-DONE statuses we add
    # later).
    try:
        current = db.jobs.find_one({"_id": job["_id"]}, {"status": 1})
    except Exception:  # noqa: BLE001
        current = None
    current_status = (current or {}).get("status")
    # PAUSED is intentionally NOT terminal - the job is expected to
    # resume later and we don't want a misleading duration in the Jobs
    # tab or an ended_at timestamp that suggests the run is over. Skip
    # both the ended_at stamp and the DONE flip in that case.
    if current_status == "PAUSED":
        print(
            f"do_job: job {job['_id']} is PAUSED; skipping trailing "
            f"ended_at + status writes so a future resume can reuse the "
            f"original started_at and the duration UI keeps making sense.",
            flush=True)
    else:
        update_job(job["_id"], datetime.datetime.now(datetime.timezone.utc), "ended_at")
        if current_status == "IN_PROGRESS":
            update_job(job["_id"], "DONE")
        else:
            print(
                f"do_job: skipping trailing status=DONE for {job['_id']} "
                f"(status is already {current_status!r}; preserving external "
                f"intent).",
                flush=True)

def _learner_checkpoint_dir_exists(root):
    """True iff ``root`` contains the tf-agents Learner Checkpointer dir.

    tf-agents' ``common.Checkpointer`` writes to
    ``<learner_root>/train/checkpoints/`` (NOTE: the subdir is
    ``checkpoints`` PLURAL, with TF checkpoint v2 files like
    ``ckpt-N.index`` and ``ckpt-N.data-*`` inside). We treat the
    existence of that directory as the canonical "Learner has been
    checkpointed at least once" signal. The directory existence alone
    isn't quite enough - tf-agents creates it during Learner __init__
    even before any save - so we also peek inside for at least one
    ``ckpt-*`` file. Cheap and unambiguous.
    """
    ckpt_dir = os.path.join(root, "learner", "train", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    try:
        for name in os.listdir(ckpt_dir):
            if name.startswith("ckpt-") or name == "checkpoint":
                return True
    except OSError:
        return False
    return False


def _has_learner_checkpoint_for_job(job_id_str):
    """True if a Learner checkpoint exists ANYWHERE for this job_id.

    Checks the three locations where the trainer might have left a
    Learner checkpoint:
      * /tmp/active/<id>/learner/train/checkpoints/   (live - most
                                                       common case
                                                       for an
                                                       in-place
                                                       crash recovery
                                                       where no
                                                       other job has
                                                       run since the
                                                       crash)
      * /tmp/jobsdata/<id>/learner/train/checkpoints/  (singly-nested
                                                       archive, from
                                                       move_data's
                                                       self-cleanup
                                                       tail)
      * /tmp/jobsdata/<id>/<id>/learner/train/checkpoints/ (doubly-
                                                            nested
                                                            archive,
                                                            from
                                                            move_all_jobs_data's
                                                            outer loop
                                                            archiving
                                                            a stale
                                                            entry)

    Used by _detect_resume_for_train_job to decide if a job that's
    IN_PROGRESS at trainer startup (= the trainer was hard-killed
    mid-run; the dashboard never had a chance to set
    PAUSE_REQUESTED) has restoreable state on disk.
    """
    if not job_id_str:
        return False
    candidates = [
        os.path.join("/tmp/active", job_id_str),
        os.path.join("/tmp/jobsdata", job_id_str),
        os.path.join("/tmp/jobsdata", job_id_str, job_id_str),
    ]
    return any(_learner_checkpoint_dir_exists(c) for c in candidates)


def _detect_resume_for_train_job(job):
    """Decide whether to take the resume code path for a TRAIN job pickup.

    Returns True if EITHER:

      1. EXPLICIT RESUME: job has paused_at_step set in Mongo. Written
         by main()'s pause-break path when the operator clicks Pause
         in the dashboard; survives the trainer Ctrl-C and the
         resume click that flips status back to NOT_STARTED.

      2. CRASH RECOVERY: the job was IN_PROGRESS at trainer pickup
         (job["status"] is the snapshot from get_jobs() = whatever
         Mongo had BEFORE do_job's IN_PROGRESS write) AND a Learner
         checkpoint exists on disk for it. This is the
         spontaneous-Windows-Update / OOM / SIGKILL case: the
         trainer was killed mid-run, never wrote paused_at_step,
         but the Learner's auto-checkpoint trigger may have left
         restoreable state behind. Without this branch, the next
         pickup would archive the partial active dir + re-run BC
         from scratch + lose all training progress.

    A fresh NOT_STARTED job with no paused_at_step returns False
    (= treat as a brand-new run, BC pretrain + Learner from random
    init).
    """
    if job.get("paused_at_step") is not None:
        return True
    if job.get("status") == "IN_PROGRESS":
        job_id_str = str(job.get("_id"))
        if _has_learner_checkpoint_for_job(job_id_str):
            print(
                f"_detect_resume_for_train_job: CRASH RECOVERY for "
                f"{job_id_str} - job was IN_PROGRESS at trainer pickup "
                f"AND a Learner checkpoint exists on disk. Treating as "
                f"resume; BC pretrain will be skipped and the saved "
                f"train_step + actor/critic weights will be restored.",
                flush=True)
            return True
    return False


def _restore_paused_active_dir(job_id_str):
    """Move a paused job's archived training data back into /tmp/active/.

    Called from do_job's TRAIN branch when the job document carries
    ``paused_at_step`` (the trainer-side signal that this job was
    previously paused via the dashboard's Pause button and is now
    being picked up for resume).

    Two archive layouts to handle:

      * Singly-nested - ``/tmp/jobsdata/<id>/{eval,learner,metrics,train}/``.
        Produced by ``move_all_jobs_data``'s tail-call to ``move_data``
        when THIS job's previous attempt put its subdirs into
        /tmp/active/<id>/* and a later pickup wanted to "clean up
        leftovers". This is the case for pause-and-immediately-resume:
        the trainer's pause break exits main(), do_job's trailer
        leaves the active dir alone (PAUSED skip), then on the next
        pickup move_all_jobs_data archives the active subdirs
        SINGLY-nested under /tmp/jobsdata/<id>/. (This is the case
        the operator hit on 2026-05-24.)

      * Doubly-nested - ``/tmp/jobsdata/<id>/<id>/...``. Produced by
        ``move_all_jobs_data``'s outer loop when a DIFFERENT job
        ran between pause and resume - that job's
        move_all_jobs_data sweeps /tmp/active/<paused_id>/ out as
        a non-current entry into /tmp/jobsdata/<paused_id>/<paused_id>/.

    The function:
      1. Returns True with no work if the Learner checkpoint is
         already in /tmp/active/<id>/learner/train/checkpoints/
         (could happen if the user clicks Resume so fast that no
         move_all_jobs_data has run between the two pickups - rare
         but possible).
      2. Otherwise tries inner-then-outer archive locations. The
         FIRST one that contains a Learner checkpoint wins; we
         move its contents into /tmp/active/<id>/, replacing any
         conflicts (e.g., a freshly-created empty metrics/ from a
         pre-restore attempt).
      3. Returns True if a restore actually happened, False if no
         archive was found.

    /tmp is bind-mounted so moves are cheap intra-filesystem renames.
    """
    if not job_id_str:
        return False
    active_dest = os.path.join("/tmp/active", job_id_str)
    # Case 1: already live. Happens when the user clicks Pause then
    # Resume before any other job pickup intervenes (so the archive
    # cycle never fired).
    if _learner_checkpoint_dir_exists(active_dest):
        print(
            f"_restore_paused_active_dir: /tmp/active/{job_id_str}/ "
            f"already holds a Learner checkpoint; no restore needed.",
            flush=True)
        return True

    archived_outer = os.path.join("/tmp/jobsdata", job_id_str)
    archived_inner = os.path.join(archived_outer, job_id_str)

    # Pick which archive layout actually has the checkpoint. Doubly-
    # nested only happens after a third-party job intervened; check
    # both. Order is "doubly first" so we don't accidentally pick the
    # outer wrapper that contains the inner dir (the outer DOES exist
    # in the doubly-nested case, but holds only the inner subdir, not
    # the real Learner data).
    candidates = []
    if _learner_checkpoint_dir_exists(archived_inner):
        candidates.append(archived_inner)
    if _learner_checkpoint_dir_exists(archived_outer):
        candidates.append(archived_outer)
    if not candidates:
        print(
            f"_restore_paused_active_dir: no Learner checkpoint found "
            f"for {job_id_str} (looked in {archived_inner}/ and "
            f"{archived_outer}/). Treating as fresh start; BC pretrain "
            f"will run.",
            flush=True)
        return False

    src = candidates[0]
    try:
        os.makedirs(active_dest, exist_ok=True)
        # Merge src into active_dest: for each top-level entry in src,
        # replace the matching entry in dest. We can't simply
        # shutil.move(src, active_dest) because shutil.move puts src
        # INSIDE dest when dest already exists as a directory (which it
        # often does - either because a fresh metrics/ got created
        # earlier in this do_job pickup, or because case-1 left an
        # empty active dir).
        moved = []
        for entry in os.listdir(src):
            src_entry = os.path.join(src, entry)
            dst_entry = os.path.join(active_dest, entry)
            if os.path.exists(dst_entry):
                if os.path.isdir(dst_entry) and not os.path.islink(dst_entry):
                    shutil.rmtree(dst_entry)
                else:
                    os.remove(dst_entry)
            shutil.move(src_entry, dst_entry)
            moved.append(entry)

        # Clean up the (now empty) archive wrapper(s).
        try:
            if os.path.isdir(src) and not os.listdir(src):
                os.rmdir(src)
        except OSError:
            pass
        # If we restored from the inner dir of a doubly-nested layout,
        # also try to remove the outer wrapper if it's empty.
        if src == archived_inner:
            try:
                if os.path.isdir(archived_outer) and not os.listdir(archived_outer):
                    os.rmdir(archived_outer)
            except OSError:
                pass

        print(
            f"_restore_paused_active_dir: restored {len(moved)} entries "
            f"{moved} from {src}/ -> {active_dest}/ for resume.",
            flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(
            f"_restore_paused_active_dir: failed to restore "
            f"{src} -> {active_dest}: {e}. Falling back to fresh start.",
            flush=True)
        return False


# Back-compat name for the older detector. New code should call
# _restore_paused_active_dir directly with the paused_at_step signal
# from the job doc instead of doing pure-filesystem inspection.
def _detect_and_restore_resume_state(job_id_str):
    return _restore_paused_active_dir(job_id_str)


def _prune_reverb_checkpoints(reverb_ckpt_dir, keep):
    """Delete all but the ``keep`` newest entries under reverb_ckpt_dir.

    Reverb's DefaultCheckpointer writes each ``.checkpoint()`` call to a
    NEW timestamped subdirectory under reverb_ckpt_dir and never prunes
    old ones itself. Added 2026-07-22 alongside periodic (not just
    on-pause) Reverb checkpointing in the training loop - without this,
    a 250k-iteration job checkpointing every
    REVERB_PERIODIC_CHECKPOINT_INTERVAL steps would accumulate ~250
    full online-table snapshots (each up to replay_buffer_capacity
    items) over its lifetime. Only the newest is ever read on resume
    (see _restore_paused_active_dir's ``sorted(_ckpt_entries)[-1]``), so
    older ones are pure waste once a newer one lands successfully.

    Best-effort: failures here must never break training, since the
    checkpoint write itself already succeeded by the time this runs.
    """
    try:
        entries = sorted(
            f for f in os.listdir(reverb_ckpt_dir)
            if os.path.isdir(os.path.join(reverb_ckpt_dir, f)))
        for stale in entries[:-keep] if keep > 0 else entries:
            try:
                shutil.rmtree(os.path.join(reverb_ckpt_dir, stale))
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass


def move_all_jobs_data(id, skip_current_cleanup=False):
    """Archive every /tmp/active/ entry that isn't the current job's.

    Background: TensorBoard runs with ``--logdir /tmp/active`` (see
    sim-controller's docker-compose command), so anything sitting in
    /tmp/active/ shows up as a separate Run in the TensorBoard UI. The
    intent of this function is to keep that view to exactly one run at
    a time - the job currently training. Old jobs' summaries are moved
    out to /tmp/jobsdata/ where the dashboard service can browse them.

    The previous implementation enumerated stale jobs by MongoDB query
    (``get_job_ids`` with ``.limit(10)``) and only archived directories
    matching the canonical ``<job_id>`` naming convention. That left two
    classes of cruft visible in TensorBoard forever:

      * Older legacy ``<id>_eval`` / ``<id>_<suffix>`` flat directories
        produced by an earlier version of the training loop never got
        recognized.
      * Anything beyond the 10 most-recent other jobs (in MongoDB
        order) was never enumerated, so accumulated tail cruft stayed.

    This rewrite walks /tmp/active/ on the filesystem instead, which is
    the source of truth TensorBoard actually scans. Everything that
    isn't the current job_id (under either ``<id>`` or ``<id>_<suffix>``
    naming) gets moved out, and any leftover subdirs of the current job
    from a previous failed attempt of the same id get cleaned out so
    this run starts with empty train/eval/metrics/learner.
    """
    print(f"archiving prior /tmp/active entries (keeping {id})")
    active_root = "/tmp/active"
    if not os.path.isdir(active_root):
        return

    str_id = str(id)
    for entry in os.listdir(active_root):
        # Keep anything belonging to the current job. ``<id>`` is the
        # canonical layout used by main(); ``<id>_<suffix>`` is also
        # accepted because some legacy code paths produced flat dirs
        # like ``<id>_eval`` for the same job.
        if entry == str_id or entry.startswith(str_id + "_"):
            continue
        src = os.path.join(active_root, entry)
        if not os.path.exists(src):
            continue

        # Group all dir variants of the same underlying job id under
        # one /tmp/jobsdata/<id>/... archive, so the dashboard sees a
        # single bucket per job rather than fragmenting into
        # /tmp/jobsdata/<id>/ and /tmp/jobsdata/<id>_eval/.
        base_id = entry.split("_", 1)[0] if "_" in entry else entry
        archive_root = os.path.join("/tmp/jobsdata", base_id)
        dst = os.path.join(archive_root, entry)
        os.makedirs(archive_root, exist_ok=True)

        if os.path.isdir(dst):
            print(f"  archive dst {dst} already exists; replacing")
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        print(f"  archived {src} -> {dst}")

    # Cleanup of any leftover subdirs of THIS job from a previous
    # failed attempt with the same job_id. Without this, a partial
    # /tmp/active/<id>/train from a crash would be merged with the new
    # run's summaries and TensorBoard would show two overlapping
    # learning curves under the same Run name.
    #
    # SKIPPED on resume: when a paused job is being picked back up,
    # _restore_paused_active_dir() has just populated /tmp/active/<id>/
    # with the saved Learner checkpoint + summary dirs. Running this
    # cleanup would immediately archive the restored data right back
    # out, defeating the resume. The caller (do_job) sets
    # skip_current_cleanup=True in that case.
    if skip_current_cleanup:
        print(
            f"move_all_jobs_data: skipping self-cleanup of "
            f"/tmp/active/{id}/* (resume path; restored data must "
            f"survive into main()).",
            flush=True)
    else:
        move_data(id, folders=["eval", "metrics", "train", "learner"])


def prune_jobsdata(keep=None, current_id=None):
    """Cap /tmp/jobsdata to the `keep` most-recently-modified job buckets.

    /tmp/jobsdata accumulates one bucket per archived job (moved there by
    move_all_jobs_data). Left unbounded it grows without limit (it reached
    ~27 GB / 228 buckets in practice), which also makes the rmtree-on-
    replace and any filesystem scans over the bind mount slow.

    We rank top-level buckets by directory mtime (newest first), keep the
    `keep` newest, and rmtree the rest. The current job's bucket is always
    protected regardless of rank so an in-flight archive is never deleted.

    `keep` defaults to the JOBSDATA_MAX_ARCHIVES env var (fallback 100).
    A value <= 0 disables pruning entirely.

    Pruned buckets can no longer be opened in TensorBoard's "Compare in
    Analysis" view (that symlinks /tmp/jobsdata/<id> into the compare
    bucket). Saved models live under a separate mount (/saved_models) and
    are unaffected — only old jobs' TB scalar history is dropped.
    """
    if keep is None:
        try:
            keep = int(os.environ.get("JOBSDATA_MAX_ARCHIVES", "100"))
        except (TypeError, ValueError):
            keep = 100
    if keep <= 0:
        print("prune_jobsdata: disabled (keep <= 0)", flush=True)
        return

    jobsdata_root = "/tmp/jobsdata"
    if not os.path.isdir(jobsdata_root):
        return

    current_str = str(current_id) if current_id is not None else None

    # Gather (path, mtime) for every top-level bucket.
    buckets = []
    for entry in os.listdir(jobsdata_root):
        path = os.path.join(jobsdata_root, entry)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        buckets.append((entry, path, mtime))

    if len(buckets) <= keep:
        print(
            f"prune_jobsdata: {len(buckets)} bucket(s) <= keep={keep}; "
            "nothing to prune.", flush=True)
        return

    # Newest first; everything past `keep` is a deletion candidate. The
    # current job's bucket is force-protected even if it somehow sorts old.
    buckets.sort(key=lambda b: b[2], reverse=True)
    survivors = buckets[:keep]
    candidates = buckets[keep:]

    deleted = 0
    freed_bytes = 0
    for entry, path, _ in candidates:
        if current_str is not None and entry == current_str:
            continue  # never delete the in-flight job's archive
        try:
            for dirpath, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        freed_bytes += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            shutil.rmtree(path)
            deleted += 1
        except Exception as e:  # noqa: BLE001
            print(f"prune_jobsdata: failed to remove {path}: {e}", flush=True)

    print(
        f"prune_jobsdata: kept {len(survivors)} newest, deleted {deleted} "
        f"bucket(s), freed ~{freed_bytes / (1024 * 1024):.1f} MB.",
        flush=True)


def move_data(job_id, folders=[""]):
    #Move all data for jobs with _id = job["_id"] from /tmp to /jobsdata
    # shutil.move is non-idempotent: if dst already exists as a
    # directory, it puts src INSIDE dst (so the next move with the same
    # arguments fails with "Destination already exists"). That's exactly
    # what bites us when a previous run of the same job crashed mid-way
    # through and we re-pick the same MongoDB IN_PROGRESS job. Clean up
    # any pre-existing dst first so the move is a true overwrite-with-
    # latest, not an accidental nested-recursive archive.
    for folder in folders:
        if folder == "":
            src = os.path.join("/tmp/active/", str(job_id))
            dst = os.path.join("/tmp/jobsdata/", str(job_id))
        else:
            src = os.path.join("/tmp/active/", str(job_id), folder)
            dst = os.path.join("/tmp/jobsdata/", str(job_id), folder)
        print(f"moving {src} to {dst}")
        if os.path.isdir(src):
            if os.path.isdir(dst):
                print(f"  dst {dst} already exists from a previous run; replacing")
                shutil.rmtree(dst)
            elif os.path.exists(dst):
                # rare: dst is a file or symlink, not a directory
                os.remove(dst)
            result = shutil.move(src, dst)
            print(f"moved {result}")

def update_job(id, value, field_name="status"):
    print(f"updating job {id} with {field_name} = {value}", flush=True)
    myquery = { "_id": id }
    newvalues = { "$set": { field_name: value } }
    db.jobs.update_one(myquery, newvalues)


def _get_job_lifecycle_state(job_id):
    """Has the dashboard externally changed this job's status?

    Polled by the training loop once per iteration. Returns one of:

      'cancel'  - status changed to anything-other-than IN_PROGRESS or
                  PAUSE_REQUESTED, OR job was hard-deleted from Mongo.
                  Trainer should break out of the training loop
                  IMMEDIATELY without saving a checkpoint. do_job's
                  trailer preserves whatever status the operator set.
      'pause'   - status changed to PAUSE_REQUESTED. Trainer should
                  save a Learner checkpoint, set status=PAUSED on the
                  job, then break. The job is resumable by setting
                  status back to NOT_STARTED later.
      'continue'- status is still IN_PROGRESS (or the job lookup hit
                  a transient error). Training should proceed.

    A missing job_id (single-env training without one queued via
    Mongo) is a safety-noop returning 'continue' so we never break a
    no-job-id run.

    A one-field find_one against the local Mongo container is sub-
    millisecond; comfortably cheaper than a 250ms env step. We
    project to just `status` so we never drag the whole job document
    across the wire.
    """
    if not job_id:
        return 'continue'
    try:
        doc = db.jobs.find_one({"_id": job_id}, {"status": 1})
    except Exception as e:  # noqa: BLE001
        # Mongo wobble. Don't kill training over a transient lookup
        # failure - next iteration's check picks up the lifecycle
        # change once Mongo recovers.
        print(f"_get_job_lifecycle_state: lookup failed for {job_id}: {e}",
              flush=True)
        return 'continue'
    if doc is None:
        # Job was hard-deleted from Mongo. Treat as a cancellation -
        # nothing for us to update at the end anyway.
        return 'cancel'
    status = doc.get("status")
    if status == 'PAUSE_REQUESTED':
        return 'pause'
    if status in (None, "", "IN_PROGRESS"):
        return 'continue'
    return 'cancel'


# Back-compat alias for any caller that still uses the previous bool
# API. New code should use _get_job_lifecycle_state directly so the
# distinction between 'cancel' and 'pause' is preserved.
def _is_job_cancelled(job_id):
    return _get_job_lifecycle_state(job_id) == 'cancel'

def run_randompolicy(job_id="", num_trials=None, num_eval_episodes=None,
                     corner_radius=10.0, curvature_difficulty=0.0,
                     chicanes_north=0, chicanes_east=0,
                     chicanes_south=0, chicanes_west=0):
    """Run a uniformly-random-action EVAL job as a baseline benchmark.

    Mirrors load_saved_model's pattern: builds a single env on
    ros-server-0, attaches the job_id for course metric tracking,
    constructs a RandomPyPolicy whose action distribution comes from
    the env's action_spec, and runs num_eval_episodes episodes through
    run_policy. The resulting AverageReturn /
    AverageEpisodeLength land in MongoDB via save_results_to_db so the
    dashboard leaderboard can compare a learned policy against the
    random baseline.

    Reachable from do_job's EVAL branch when ``job["location"]``
    contains ``"RandomPyPolicy"``. Multi-actor (--num-envs N) doesn't
    affect this path - same as load_saved_model, EVAL is hardcoded
    single-env on ros-server-0.

    ``num_trials`` / ``num_eval_episodes`` are the same optional
    overrides as load_saved_model - they map onto run_policy's
    ``max_episodes`` and ``num_eval_episodes`` args. None means
    "leave run_policy's default alone".

    ``corner_radius`` / ``curvature_difficulty`` mirror the TRAIN knobs
    so the random-policy baseline runs on the same track geometry as
    the trained model it is being compared against.

    Previously broke with `NameError: name 'env' is not defined`
    because it referenced a module-global env that was removed in the
    rl_agent factory refactor (commit fbe3bce). This fix builds its
    own env exactly like load_saved_model does.
    """
    debug_print("in random policy")
    env = make_env('ros-server-0:50051')
    env.job_id = job_id
    configure_env(env, job_id=job_id, pass_through_actions=False,
                  corner_radius=corner_radius,
                  curvature_difficulty=curvature_difficulty,
                  chicanes_north=chicanes_north, chicanes_east=chicanes_east,
                  chicanes_south=chicanes_south, chicanes_west=chicanes_west)
    # Same publication step as load_saved_model so the RandomPyPolicy
    # baseline path also keeps the dashboard's env_specs current.
    publish_env_spec(env)
    random_policy = random_py_policy.RandomPyPolicy(
        env.time_step_spec(), env.action_spec())
    run_kwargs = {}
    if num_trials is not None:
        run_kwargs["max_episodes"] = int(num_trials)
    if num_eval_episodes is not None:
        run_kwargs["num_eval_episodes"] = int(num_eval_episodes)
    results = run_policy(random_policy, env, job_id=job_id, **run_kwargs)
    debug_print(results)
    random_policy_path = get_latest_save_dir_name(random_policy)
    debug_print(random_policy_path)
    save_results_to_db(random_policy_path, results)

def create_traj(
    observation, 
    action,
    reward=tf.constant([1], dtype=tf.float32), 
    discount=tf.constant([0.99], dtype=tf.float32)):
    traj = trajectory.first(
        observation=observation, #1
        action=action, #2
        policy_info=(), #3
        reward=reward, #4
        discount=discount #5
    )
    return traj

def collect_expert_demos(environment, num_episodes, job_id=0, batch_number=0,
                         stage=None, num_stages=None, goal_budget=None,
                         flush_every_n_steps=5000, gym_index=0):
    """Collects expert demonstrations and returns a dataset of trajectories.

    ``batch_number`` picks the output filename prefix
    (``<batch_number zero-padded>_<sub-batch zero-padded>trajectories.tfrecord``
    - see ``flush_every_n_steps`` below). Callers that invoke this multiple
    times for the SAME job_id (e.g. once per curriculum stage, see the DEMO
    branch in do_job) MUST pass distinct batch_number values -
    collect_training_data.read_files_from_directory() reads every file in the
    job's /tfrecords directory, so distinct filenames make each stage's demos
    additive; reusing batch_number=0 would silently overwrite the prior
    stage's trajectories instead.

    ``stage`` / ``num_stages`` (0-indexed stage, total stage count) are
    forwarded to the Unity HUD's "stage: X/N" readout via
    rollout_viz.publish_mode (mode="demo") - see the DEMO branch in do_job,
    which passes the current curriculum stage when collecting across a
    curriculum-bearing experiment design. None for a non-curriculum
    collection (the fallback single-geometry path), in which case the HUD
    just shows "DEMO" with no stage line.

    ``goal_budget`` (2026-07-20): when set, this stage stops once the car
    has reached this many goals SINCE THIS CALL STARTED (checked only at
    episode boundaries - never mid-episode, matching the num_trajectories
    units-confusion fix from 2026-07-19), rather than after a fixed
    ``num_episodes``. Added because a fixed episode count made per-stage
    collection time wildly unpredictable once episode length itself became
    policy-dependent (a car that drives well now often survives 5,000+
    steps/episode instead of crashing in a few hundred) - goals reached is
    a much more direct proxy for "how much useful demonstration data did we
    actually collect" than either raw episode count or raw step count.
    ``num_episodes`` still applies as a hard safety cap alongside
    goal_budget (whichever bound is hit first stops the loop), so a policy
    that can't reach goals at all can't spin forever.

    ``flush_every_n_steps`` (2026-07-20): periodically writes the
    in-progress ``trajectories`` buffer to its own numbered tfrecord file
    and clears it, instead of only writing once at the very end of the
    entire call. Previously a single stage's ENTIRE collection (which can
    now run for many hours - see goal_budget above) lived only in this
    function's local `trajectories` list until the very last line, so
    killing/restarting the trainer at any point mid-stage (a wedge, a code
    deploy, anything) silently discarded ALL of that stage's data with
    nothing on disk to show for it - confirmed 2026-07-20 when a restart
    would have discarded ~10.5 hours / ~590k steps of stage-1 collection.
    Set to 0 to disable (write only once at the end, the old behavior).

    ``gym_index`` (2026-07-21): identifies which of N concurrently-running
    Unity gym instances this call is driving - see do_job's
    ``_collect_on_all_gyms`` helper, which spawns one collect_expert_demos()
    call per gym on its own thread when a DEMO job's ``demo_num_gyms`` > 1.
    Two roles: (1) every log line from this call is prefixed ``[gym-N]`` so
    interleaved output from concurrent gyms stays attributable, and (2) it's
    baked into the flushed tfrecord filenames so N concurrent calls sharing
    the same job_id/batch_number (writing to the same /tfrecords/job_<id>/
    directory at the same time) can never collide on a filename - without
    this, two gyms' flushes at the same batch_number/sub_batch would race
    and one would silently clobber the other's steps on disk.
    """
    # Prefix every print() in this function (including the nested
    # _flush_trajectories_to_disk below, which resolves `print` from this
    # enclosing scope) with [gym-N] - see gym_index docstring above. This
    # local reassignment shadows the builtin for the whole function body
    # regardless of where it's placed textually, since Python resolves
    # locals at compile time; placing it up front just keeps every call
    # site below unmodified individually. MUST fetch the real builtin via
    # the `builtins` module rather than the bare name `print` - the `def
    # print(...)` below makes `print` a local variable for the WHOLE
    # function scope the moment Python compiles it (regardless of where the
    # def appears), so `_builtin_print = print` would raise
    # UnboundLocalError (reading a local before its first assignment).
    import builtins as _builtins_module
    _builtin_print = _builtins_module.print
    def print(*args, **kwargs):  # noqa: A001 - intentional shadow, see above
        _builtin_print(f"[gym-{gym_index}]", *args, **kwargs)

    folder_name = "/tfrecords/job_"+str(job_id)
    create_folder(folder_name=folder_name)
    action_steering_angle = np.float32(0)
    action_apply_force = np.float32(1)
    trajectories = []
    num_trajectories=0
    crashes=0
    goal_cap_ends=0
    sub_batch=0

    def _flush_trajectories_to_disk():
        """Writes the current in-memory `trajectories` buffer to its own
        numbered file and clears it - see flush_every_n_steps docstring
        above. No-op on an empty buffer (nothing to write, e.g. called at
        the end of a stage that just flushed on its last step already)."""
        nonlocal trajectories, sub_batch
        if not trajectories:
            return
        path = (folder_name + "/" + zero_pad_integer(batch_number, 6) + "_g"
                + str(gym_index) + "_" + zero_pad_integer(sub_batch, 4)
                + "trajectories.tfrecord")
        print(f"t={_ts()} flushing {len(trajectories)} steps to {path}", flush=True)
        write_trajectories_to_file(trajectories, path)
        trajectories = []
        sub_batch += 1
    # NOTE: do NOT reassign `batch_number` here - it's a parameter (the
    # output-file index for multi-stage/curriculum demo collection). This used
    # to shadow the argument with a local 0, so every call silently wrote to
    # 000000trajectories.tfrecord regardless of what the caller passed.
    # base_force: 0.2 -> 0.8 (2026-07-19, target-speed increase to 5 m/s) ->
    # 0.25 (2026-07-19, same day): 0.8 turned out to be too aggressive once
    # MAX_SPEED/GOAL_BRAKE_ZONE_DIST below were also raised - video+log
    # cross-referencing (scripts/analyze_video.py) showed the car
    # overshooting MAX_SPEED by 15-20% (up to 5.8 m/s against a 5.0 m/s cap)
    # on long straights before the goal-proximity brake had enough distance
    # to react, causing corner crashes. Capped back down to 0.25 alongside
    # MAX_SPEED 5.0 -> 3.5 to reduce both how fast the car can ramp up AND
    # how much speed the braking logic ever has to shed before a corner.
    #
    # 0.25 -> 0.5 (2026-07-24): base_force is the UPPER bound the collector's
    # throttle ramps toward on straights (see the ACCEL_STEP branch below:
    # `action_apply_force = min(base_force, action_apply_force + ACCEL_STEP)`),
    # so it sets how hard/fast the demo car ever accelerates. With min_force
    # also raised to 0.2, the demo force now lives in [0.2, 0.5] instead of
    # the old crawl-prone [-0.01, 0.25] - a real forward-throttle band that
    # teaches driving speed. Still well under the action_spec max (1.0).
    base_force=0.5
    # min_force widened 0.001 -> -1.0 (2026-07-19, same day as the
    # action_spec range change above from [0.1,2.0] to [-1,1]): 0.001 was
    # effectively "coast at ~0 drive torque", the ONLY deceleration
    # available being passive rolling friction/drag - confirmed via
    # video+log cross-referencing (scripts/analyze_video.py) that this was
    # far too weak to shed overshoot speed (e.g. 5.6 m/s) before a corner,
    # producing "drives straight into the wall" crashes right after long
    # straights. -1.0 lets the decel branch below command genuine negative
    # motorTorque (active braking, see CarController.cs Accelerate()) once
    # coasting alone isn't cutting it fast enough.
    #
    # -1.0 -> -0.01 (2026-07-22): full -1.0 active braking was too strong,
    # so the decel floor is pulled back to a near-coast -0.01 (barely
    # negative torque) rather than hard reverse braking.
    #
    # -0.01 -> 0.2 (2026-07-24): demo-distribution analysis of the resulting
    # tfrecords showed the [-0.01, base_force] range let the collector sit at
    # near-zero throttle for the vast majority of steps (median force ~0.01-
    # 0.07, ~70% of W-course steps in the near-coast band), which AWAC's BC
    # term then faithfully reproduced as a ~0.5 m/s "crawl" the reward could
    # never overcome. Raising the FLOOR to 0.2 (with base_force=0.25) pins the
    # collector to a near-constant [0.2, 0.25] forward throttle - it can no
    # longer coast/brake below 0.2 - so a freshly-collected demo set teaches
    # actual driving speed instead of idling. NOTE: this only affects NEW demo
    # collection jobs; existing tfrecords are unchanged and must be re-
    # collected for this to take effect. 0.2 is >= the action_spec minimum
    # (0.05) so the _DEMO_MIN_ACCEL load filter drops 0 of these rows.
    min_force=0.2
    # Simple curvature-scaled heuristic (2026-07-18 rewrite, replacing the
    # previous corner_urgency/clearance-cone/curvature-asymmetry/force-cap
    # system - see git history for that version). That system's extra
    # anticipatory signals turned out not to prevent the car getting
    # physically wedged against track geometry (confirmed via per-step
    # logging: forward clearance reading a literal 0.000m while corner
    # urgency + force cap both maxed out, i.e. touching a wall, not just
    # cornering hard), so replacing it with two independent, easy-to-reason-
    # about proportional controllers instead:
    #
    #   * steering: a plain P-controller on the heading-error-to-goal signal
    #     (obs[0], normalized to [-1,1] = angle_deg/180), scaled by
    #     K_STEERING and clamped to [-1,1] so a clamped output is exactly a
    #     full-lock action of +-1.0. See the UNIT MISMATCH note below for why
    #     K_STEERING is 4.0 rather than 1.0.
    #
    #   * velocity: target speed backs off with the CURRENT steering
    #     command's own magnitude (the normalized action we just computed
    #     above, not a separate multi-ray urgency score) - straight-ahead
    #     cruises at MAX_SPEED, full-lock caps at MIN_SPEED. The falloff is
    #     QUADRATIC in (1 - steer_frac) rather than linear (2026-07-19, target
    #     speed raised 2.0 -> 5.0 m/s): the straight-vs-corner speed spread
    #     went from 1.2 m/s to 4.2 m/s, and a linear ramp only sheds
    #     meaningful speed once steer_frac is already large (i.e. already
    #     mid-turn) - too late at the new higher cruise speed. Quadratic
    #     falloff cuts speed much more aggressively as soon as ANY real
    #     steering is commanded, well before full lock, so "slow down as
    #     necessary to make the turn" starts happening early into the turn
    #     instead of only at its sharpest point.
    # UNIT MISMATCH FOUND (2026-07-19): obs[0] (raw_steer) and the
    # steering_angle action do NOT share a normalization, despite both living
    # in [-1,1] - obs[0] = GetAngleToGoal()/180 (SceneDataPublisher.cs) i.e.
    # normalized against a 180-degree span, while the action is applied as
    # `steerAngle = maxSteeringAngle * angle` with maxSteeringAngle=45
    # (CarController.cs) i.e. normalized against a 45-degree span. Those are
    # 4x apart (180/45), so the previous K_STEERING=1.0 only ever commanded
    # 25% of the wheel angle the heading error actually called for - the car
    # could only reach full lock if pointed nearly backwards from the goal
    # (~180 deg off), which never happens in practice. This silently capped
    # every turn to a shallow ~11 degrees of real wheel angle even when
    # heading error was 40-90+ degrees approaching a corner - confirmed via
    # the [steer-diag]/[crash-pos] logs (e.g. heading error -47 deg producing
    # only an actual -0.263 action = -11.8 deg of wheel angle at the moment
    # of a crash). K_STEERING=4.0 below cancels that mismatch so a real
    # heading error of +-45 degrees now correctly commands +-1.0 (full lock);
    # the clamp is reintroduced since heading error can exceed 45 degrees
    # (e.g. hairpins, right after a reset) and the resulting scaled value can
    # now exceed the [-1,1] action range.
    K_STEERING = 4.0           # proportional gain: heading-error -> steer command (compensates the 180 vs 45 degree normalization mismatch above)
    MAX_SPEED = 3.5            # m/s target when steering command ~ 0 (straight) - 2.0 -> 5.0 -> 3.5 (2026-07-19): 5.0 let the car overshoot up to 5.8 m/s on long straights (video+log cross-reference via scripts/analyze_video.py) faster than the goal-proximity brake below could react; 3.5 leaves less speed to shed per corner in the first place. Still not a hard cap - see base_force/ClampAccelerationForStandstill (CarController.cs) for the actual physical limits.
    MIN_SPEED = 0.8            # m/s target at full-lock steering command (unchanged - this is the tight-corner floor and was already working)
    ACCEL_STEP = 4e-3          # per-step force increase toward target speed (unchanged - braking authority matters more for safety than acceleration authority)
    DECEL_STEP = 2e-2          # per-step force decrease toward target speed - raised from 8e-3 (2026-07-19): more speed to shed per corner now (up to 4.2 m/s vs 1.2 m/s before), needs more braking authority to actually shed it in time
    # GOAL-PROXIMITY BRAKING (added 2026-07-19): the curvature-based target_speed
    # above only reacts to the CURRENT heading error, which is a purely reactive,
    # zero-lookahead signal - on a long straight it stays ~0 the whole way (there's
    # nothing to correct), so the car ramps all the way up to MAX_SPEED with no
    # warning that a turn is coming. The instant the car reaches the goal marking
    # the end of the straight, `next_goal` snaps to a sharp-turn waypoint, but by
    # then there's no distance left to both turn and shed speed - confirmed via
    # [crash-pos] logs showing repeat crashes at the SAME straight-to-corner
    # transition point (e.g. 9x at one location) at 3-5 m/s with near-zero steer,
    # some immediately after a logged "has_reached_goal: True". Since this is a
    # blind spot in principle (not just under-tuning), adding a second, independent
    # speed constraint based on dist_from_goal (already published per-step in
    # environment.data, see SceneDataPublisher.cs car.dist_from_goal - not part of
    # the 32-element observation vector) - the car eases toward MIN_SPEED as it
    # nears ANY goal, regardless of current heading error, since reaching a goal is
    # exactly when the required heading can suddenly change. Once past a goal with
    # dist_from_goal large again for the new next-goal, it re-accelerates - so on a
    # long straight with sparse goals this shows up as a brief slow-down "pulse" at
    # each goal crossing rather than one continuous cruise, trading a bit of
    # average speed for a real safety margin at the one place blind reactive
    # control can't see coming. Final target_speed is whichever constraint (this
    # one, or the existing curvature-based one) is more restrictive right now.
    GOAL_BRAKE_ZONE_DIST = 12.0  # meters - distance-to-goal at which proximity braking starts kicking in
    # SECOND units-confusion bug found alongside the reset_every_n_episodes
    # one below (2026-07-19): this loop used to ALSO break out of both the
    # inner while AND this outer for loop once `num_trajectories` (a per-STEP
    # counter - it's incremented once per environment._step() call below,
    # since one "trajectory" = one step's (obs,action,reward,discount) tuple
    # written to the tfrecord) exceeded `max_trajectories`, which was set to
    # `num_episodes` - a value the DEMO branch in do_job computes as an
    # EPISODE budget (e.g. num_iterations // num_curriculum_stages, commonly
    # in the thousands). Comparing a step count against an episode-sized
    # budget meant the entire function - and whatever episode happened to be
    # mid-flight at that instant - exited abruptly, with NO crash, after only
    # a handful of REAL episodes (a few hundred steps each) rather than the
    # `num_episodes` actually requested; the very next collect_expert_demos()
    # call (next curriculum stage, or a fresh call for the same stage) then
    # did its own environment.reset() - visible in Unity as a car reset with
    # no crash. This `for` loop already correctly bounds real episode count
    # (one full iteration = one environment.reset() + drive-until-episode-
    # ended), so the extra step-based break was pure liability. Removed.
    #
    # goal_budget support (2026-07-20, see docstring): `num_episodes` remains
    # a hard safety cap either way (rewritten as `while episode_idx <
    # num_episodes` below, equivalent to the old `for` when goal_budget is
    # None); the goal check only ever runs BETWEEN episodes, same principle
    # as removing the old mid-episode step-based break above - an episode
    # in progress always finishes naturally.
    _course = getattr(environment, 'course', None)
    _goals_at_call_start = getattr(_course, 'goals_per_episode_total', 0)
    episode_idx = 0
    while episode_idx < num_episodes:
        if goal_budget is not None:
            _goals_so_far = (
                getattr(_course, 'goals_per_episode_total', 0)
                - _goals_at_call_start)
            if _goals_so_far >= goal_budget:
                print(
                    f"t={_ts()} goal_budget {goal_budget} reached "
                    f"({_goals_so_far} goals since stage start) - "
                    f"advancing", flush=True)
                break
        episode_idx += 1
        environment.reset()
        action_apply_force = 0.2
        # Neutral start value; overwritten unconditionally by obs[0] the
        # instant the first step() call returns an observation below, so
        # this only matters for the very first action sent each episode
        # (before any observation exists).
        action_steering_angle = np.float32(0)
        while not environment._episode_ended:
            # NOTE (2026-07-19): this loop used to force an unconditional
            # environment.reset() every 1000 STEPS (a stale "reset_every_n_
            # episodes" counter that was actually checked against a
            # cumulative step count, not an episode count - see git history
            # of this function). That silently cut off any episode that
            # survived past 1000 steps and reset the car with no crash
            # logged, which is exactly the "car position resetting without
            # collisions" behavior reported once the steering-gain fix (see
            # K_STEERING above) started letting episodes actually survive
            # that long. Removed - episodes now end only via the natural
            # environment._episode_ended signal (crash, see donut_course.py
            # has_failed()).
            numpy_action = np.array([action_apply_force, action_steering_angle])
            action = tf.constant(numpy_action)
            next_time_step=environment._step(action)
            # data = environment._do_action(action)
            # print(f"data: {data}")
            obs = next_time_step.observation
            raw_steer = float(obs[0])
            speed = float(obs[1])

            # Steering: proportional controller on the heading-error signal.
            # Clamped to [-1,1] (see comment above) since K_STEERING=4.0
            # compensates the 180-vs-45-degree normalization gap between
            # obs[0] and the action, so heading errors beyond +-45 degrees
            # now legitimately overshoot the raw action range and must be
            # capped at full lock rather than sent through uncapped.
            action_steering_angle = max(-1.0, min(1.0, K_STEERING * raw_steer))

            # Velocity: curvature-scaled directly off THIS step's own
            # steering command magnitude - no separate clearance/asymmetry
            # cones (see comment block above). Quadratic (not linear) falloff
            # - see MAX_SPEED comment above for why.
            steer_frac = min(1.0, abs(float(action_steering_angle)))
            curvature_target_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (1.0 - steer_frac) ** 2

            # Goal-proximity braking - see GOAL_BRAKE_ZONE_DIST comment above.
            # Independent of steer_frac: eases toward MIN_SPEED as the car
            # nears ANY goal (a turn may follow), regardless of current
            # heading error. Falls back to "far away" (no constraint) if
            # dist_from_goal is ever unavailable, so a lookup failure can only
            # make this step behave like the braking zone didn't exist yet,
            # never force an unwanted stop.
            try:
                dist_from_goal = float(environment.data.get("car", {}).get("dist_from_goal"))
            except (TypeError, ValueError, AttributeError):
                dist_from_goal = GOAL_BRAKE_ZONE_DIST
            proximity_frac = 1.0 - min(1.0, max(0.0, dist_from_goal / GOAL_BRAKE_ZONE_DIST))
            proximity_target_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (1.0 - proximity_frac) ** 2

            target_speed = min(curvature_target_speed, proximity_target_speed)
            if speed < target_speed:
                action_apply_force = min(base_force, action_apply_force + ACCEL_STEP)
            else:
                action_apply_force = max(min_force, action_apply_force - DECEL_STEP)

            if num_trajectories % 15 == 0:
                # next_goal_name: pulled from environment.data (refreshed by
                # every _step() call - see RobotaxiEnv._step) rather than
                # `obs`, since the goal NAME isn't part of the 32-element
                # observation vector. NOTE: despite the Unity-side field/
                # method being called "current_goal"/GetCurrentGoalName(),
                # it actually holds goals[(goalIndex+1) % count].name - i.e.
                # the NEXT goal the car is steering toward right now (see
                # SceneDataPublisher.cs) - which is what obs[0] (below) is
                # the bearing to.
                try:
                    next_goal_name = environment.data.get("car", {}).get("current_goal")
                except Exception:  # noqa: BLE001 - diagnostic print must never break collection
                    next_goal_name = "?"
                print(
                    f"[steer-diag] t={_ts()} next_goal={next_goal_name} "
                    f"obs[0](raw_steer)={raw_steer:+.3f} "
                    f"action_steer(actual)={float(action_steering_angle):+.3f} "
                    f"action_force={float(action_apply_force):.4f} "
                    f"speed={speed:.3f} target_speed={target_speed:.3f} "
                    f"(curv={curvature_target_speed:.3f} prox={proximity_target_speed:.3f} "
                    f"dist_from_goal={dist_from_goal:.2f})",
                    flush=True)

            # Keep the Unity HUD's "DEMO" indicator + "stage: X/N" readout
            # live during heuristic demo collection - this is the only place
            # in this function's per-step loop, and publish_mode's own
            # internal ~1Hz throttle (see rollout_viz.py) makes calling it
            # every step cheap/safe. stage/num_stages are None for a
            # non-curriculum collection, so the HUD just shows "DEMO" with
            # no stage line in that case.
            try:
                get_viz().publish_mode(
                    "demo", 1, num_trajectories,
                    stage=stage, num_stages=num_stages)
            except Exception:  # noqa: BLE001 - HUD must never break collection
                pass

            traj = create_traj(
                next_time_step.observation,
                action,
                next_time_step.reward,
                next_time_step.discount)

            trajectories.append(traj)
            num_trajectories+=1
            
            if environment._episode_ended:
                # Distinguish a genuine crash-ending from the new
                # GOALS_PER_EPISODE_CAP success-ending (2026-07-20) - both
                # set environment._episode_ended, but only one is actually
                # a crash. `has_crashed` is the same field donut_course's
                # has_failed() reads, so it's authoritative here too.
                _is_crash = bool(environment.data.get("car", {}).get("has_crashed"))
                if _is_crash:
                    crashes=crashes+1
                else:
                    goal_cap_ends=goal_cap_ends+1
                # Log the actual world position at every episode end so we
                # can empirically check WHERE crashes cluster on the track,
                # instead of relying on subjective visual impression. Cheap:
                # environment.data is refreshed every _step() call regardless
                # (see RobotaxiEnv._step), this just reads it. Greppable via
                # "[crash-pos]" - x/z are TrackGenerator world coords (tile
                # grid * tileSize), so they can be checked against the 4
                # known corner positions for this loop
                # (loopWidthTiles x loopHeightTiles, tileSize=20).
                try:
                    _car_pos = environment.data.get("car", {})
                    # zone / easy-hard-straight crash tallies (2026-07-19):
                    # read straight off environment.course, which
                    # reward_failure() already incremented for THIS crash a
                    # moment ago (see donut_course.py _classify_car_zone) -
                    # no need to reclassify here, and DEMO jobs never hit
                    # the TRAIN loop's periodic TensorBoard course_metrics
                    # write (read_course_metrics), so this print is
                    # currently the only live visibility into these
                    # counters during DEMO collection.
                    _course = getattr(environment, 'course', None)
                    _tag = "[crash-pos]" if _is_crash else "[goal-cap-end]"
                    print(
                        f"{_tag} t={_ts()} x={_car_pos.get('location_x')} "
                        f"z={_car_pos.get('location_z')} "
                        f"speed={_car_pos.get('speed')} "
                        f"next_goal={_car_pos.get('current_goal')} "
                        f"steer={float(action_steering_angle):+.3f} "
                        f"step={num_trajectories} "
                        f"crashes(easy/hard/straight)="
                        f"{getattr(_course, 'crashes_easy_corner', '?')}/"
                        f"{getattr(_course, 'crashes_hard_corner', '?')}/"
                        f"{getattr(_course, 'crashes_straight', '?')} "
                        f"traversals(easy/hard)="
                        f"{getattr(_course, 'easy_corner_traversals', '?')}/"
                        f"{getattr(_course, 'hard_corner_traversals', '?')}",
                        flush=True)
                except Exception as _e:  # noqa: BLE001 - logging must never break collection
                    print(f"[crash-pos] logging failed (non-fatal): {_e}", flush=True)
            print(f"t={_ts()} num_trajectories(steps): {num_trajectories}, crashes: {crashes}, goal_cap_ends: {goal_cap_ends}")
        # Periodic crash-safe flush (see flush_every_n_steps docstring) -
        # checked at the episode boundary we're at right now (the inner
        # while loop above just exited via environment._episode_ended, or
        # this is the very start before any episode has run), never
        # mid-episode.
        if flush_every_n_steps and len(trajectories) >= flush_every_n_steps:
            _flush_trajectories_to_disk()
    _flush_trajectories_to_disk()  # final remainder, no-op if already empty

def zero_pad_integer(integer, length):
    """Pads an integer with zeros to a given length."""
    return str(integer).zfill(length)

def create_folder(folder_name):
    """Creates a folder if it doesn't already exist.

    exist_ok=True (rather than an exists-check-then-makedirs) because
    demo_num_gyms>1 (2026-07-21) has multiple collect_expert_demos() threads
    racing to create the SAME job folder at near-identical startup time -
    a naive check-then-create has a TOCTOU window where both threads pass
    the `not exists` check before either creates it, and the loser's
    makedirs() raises FileExistsError.
    """
    os.makedirs(folder_name, exist_ok=True)
 
def write_trajectories_to_file(trajectories, output_file):
    """Writes a list of trajectories to a TFRecord file."""
    writer = tf.io.TFRecordWriter(output_file)
    for traj in trajectories:
        print("writing trajectory")
        print(traj)
        observation = traj.observation
        action = traj.action
        reward = traj.reward
        discount = traj.discount
        print(f"reward: {reward}")
        print(f"discount: {discount}")
        feature_dict = {
            'action': tf.train.Feature(float_list=tf.train.FloatList(value=action.numpy().ravel())),
            'observation': tf.train.Feature(float_list=tf.train.FloatList(value=observation.numpy().ravel())),
            'reward': tf.train.Feature(float_list=tf.train.FloatList(value=reward.numpy().ravel())),
            'discount': tf.train.Feature(float_list=tf.train.FloatList(value=discount.numpy().ravel()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        writer.write(example.SerializeToString())
    writer.close()   

# def create_traj(
#     observation, action,
#     reward=tf.constant([1], dtype=tf.float32), 
#     discount=tf.constant([0.99], dtype=tf.float32)):
#     traj = trajectory.first(
#         observation=observation,
#         action=action,
#         policy_info=(),
#         reward=reward,
#         discount=discount)
#     return traj

def debug_print(text):
    debug_print_enabled = False
    if debug_print_enabled:
        print(text)

def _seed_canonical_reward_designs():
    """Idempotent upsert of built-in reward designs at sim-controller start.

    Currently seeds exactly one design: the canonical "Course default
    (passthrough)" which forwards every reward call straight back to
    the active course's default formula. Used as the validation
    baseline for the reward-design system (see
    rl_agent/reward_designs.py docstring) and as a starting template
    for new shaping experiments.

    The upsert is keyed by a stable string ``_id`` so this runs as a
    no-op on every container restart after the first. Bumps the
    ``version`` field whenever the canonical code changes so users
    can tell which version of the seed they have.

    Best-effort: failure to upsert is logged but doesn't block the
    trainer (the user can still create reward designs manually via
    the dashboard).
    """
    try:
        # Bare-name sibling import; see the comment in do_job's TRAIN
        # branch for why we don't use the `rl_agent.reward_designs`
        # package path.
        from reward_designs import (
            PASSTHROUGH_DESIGN_ID,
            PASSTHROUGH_DESIGN_NAME,
            PASSTHROUGH_DESIGN_CODE,
        )
        # Bump SEED_VERSION here when the canonical passthrough code
        # changes so the upsert refreshes the on-disk copy. Users who
        # manually edited the design will see their edits get replaced
        # only when this number bumps.
        SEED_VERSION = 1
        existing = db.reward_designs.find_one({"_id": PASSTHROUGH_DESIGN_ID})
        if existing and existing.get("version", 0) >= SEED_VERSION:
            return  # already current
        ts = time.time()
        iso_date = datetime.datetime.fromtimestamp(ts, None)
        db.reward_designs.update_one(
            {"_id": PASSTHROUGH_DESIGN_ID},
            {"$set": {
                "_id": PASSTHROUGH_DESIGN_ID,
                "name": PASSTHROUGH_DESIGN_NAME,
                "description": (
                    "No-op reward design. Forwards every reward to the active "
                    "course's default implementation. Use this as a sanity-"
                    "check baseline when validating the reward-design system "
                    "itself, and as a clean starting point for new shaping "
                    "experiments."),
                "code": PASSTHROUGH_DESIGN_CODE,
                "version": SEED_VERSION,
                "author": "system",
                "archived": False,
                "created_at": (existing or {}).get("created_at", iso_date),
                "updated_at": iso_date,
            }},
            upsert=True,
        )
        print(
            f"reward_designs: seeded canonical passthrough design "
            f"({PASSTHROUGH_DESIGN_ID}, v{SEED_VERSION})", flush=True)
    except Exception as e:
        print(f"reward_designs: seed upsert failed (continuing): {e}", flush=True)


def _seed_canonical_experiment_design():
    """Idempotent upsert of the canonical "Default" experiment design.

    Sibling of _seed_canonical_reward_designs above. The Default
    captures the trainer's current hardcoded hyperparameters as a
    Mongo document so:

      - the New-job form's Experiment design dropdown always has at
        least one option,
      - every other design can be diffed against a stable reference,
      - the future research-planning agent has a control config to
        compare new candidates against.

    Keyed by a stable string ``_id`` so re-runs are no-ops once
    seeded. Bumps the version field when SEED_VERSION below bumps,
    so a future change to the trainer's defaults can propagate
    through cleanly.

    Best-effort: failure to upsert is logged but doesn't block the
    trainer (operator can still create designs manually via the
    dashboard).
    """
    try:
        # Bare-name sibling import; see the comment in do_job's TRAIN
        # branch for why we don't use the `rl_agent.experiment_designs`
        # package path.
        from experiment_designs import (
            DEFAULT_DESIGN_ID,
            DEFAULT_DESIGN_NAME,
            DEFAULT_DESIGN_DESCRIPTION,
            default_design_fields,
        )
        # Bump this when SCHEMA defaults change in a meaningful way
        # (e.g., we tune the canonical actor_learning_rate down). The
        # upsert below uses the bump to refresh the on-disk copy;
        # otherwise it's a no-op so users' edits to Default are
        # preserved across restarts.
        SEED_VERSION = 1
        existing = db.experiment_designs.find_one({"_id": DEFAULT_DESIGN_ID})
        if existing and existing.get("version", 0) >= SEED_VERSION:
            return  # already current
        ts = time.time()
        iso_date = datetime.datetime.fromtimestamp(ts, None)
        fields = default_design_fields()
        db.experiment_designs.update_one(
            {"_id": DEFAULT_DESIGN_ID},
            {"$set": {
                "_id": DEFAULT_DESIGN_ID,
                "name": DEFAULT_DESIGN_NAME,
                "description": DEFAULT_DESIGN_DESCRIPTION,
                "version": SEED_VERSION,
                "author": "system",
                "archived": False,
                "created_at": (existing or {}).get("created_at", iso_date),
                "updated_at": iso_date,
                # Spread the canonical field-set so the doc directly
                # mirrors what the New-job form's dropdown sees.
                # Editing Default is allowed but discouraged - the
                # description string above tells users to clone for
                # variants instead.
                **fields,
            }},
            upsert=True,
        )
        print(
            f"experiment_designs: seeded canonical Default design "
            f"({DEFAULT_DESIGN_ID}, v{SEED_VERSION})", flush=True)
    except Exception as e:
        print(f"experiment_designs: seed upsert failed (continuing): {e}", flush=True)


def run_jobs_loop(num_envs=1):
    """Poll MongoDB for jobs and dispatch them indefinitely.

    Each individual job constructs its own env(s) via make_env() (and
    tears them down with the process when the job ends), so no
    module-global RobotApi or background thread is needed at startup.

    num_envs is process-wide configuration (passed in via --num-envs).
    Forwarded to do_job() so the TRAIN job_type can build a parallel
    env. DEMO and EVAL job types are unaffected.
    """
    # One-shot seed of the canonical reward + experiment designs so
    # the dashboard's New-job form always has at least one option in
    # each dropdown, and so the research-planning agent has stable
    # control references to compare against. Both upserts are
    # idempotent across container restarts.
    _seed_canonical_reward_designs()
    _seed_canonical_experiment_design()
    print(f"Polling for jobs (num_envs={num_envs})...")
    queue_paused_logged = False
    while True:
        # Global queue pause gate. When the dashboard's Pause-queue button
        # is on, we don't pick up new jobs - we just idle. Job statuses are
        # untouched, so resuming the queue continues exactly where it left
        # off. Logged only on transition so the idle loop isn't spammy.
        if _is_queue_paused():
            if not queue_paused_logged:
                print("Job queue PAUSED (global): not picking up new jobs "
                      "until resumed from the dashboard.", flush=True)
                queue_paused_logged = True
            time.sleep(5)
            continue
        if queue_paused_logged:
            print("Job queue RESUMED (global): picking up jobs.", flush=True)
            queue_paused_logged = False

        jobs = get_jobs()
        for j in jobs:
            # Re-check before each job so a pause issued mid-cycle stops
            # further pickups after the current job finishes (the running
            # job is never interrupted by the global pause - use per-job
            # Pause for that).
            if _is_queue_paused():
                print("Job queue paused mid-cycle; halting further pickups.",
                      flush=True)
                break
            print("doing job")
            # Outer safety net around do_job. Two graceful-error
            # codepaths inside do_job already (EvalSpecMismatchError,
            # RewardDesignError) write a clean ``eval_error`` and let
            # the job complete with status=DONE. THIS handler is for
            # everything else - import errors, KeyErrors, OOMs, env
            # connection failures, tf-agents internal crashes, etc.
            # Previously any such exception killed run_jobs_loop and
            # took the whole trainer with it; the container would
            # restart and immediately re-pick the same job, looping
            # forever. Now we capture the traceback, stamp it on the
            # job document as eval_error + status=FAILED, and continue
            # the poll loop so the queue keeps draining.
            #
            # Note we deliberately catch BaseException-minus-the-
            # exit-signals (so Ctrl+C / SIGTERM still propagate) by
            # excluding KeyboardInterrupt and SystemExit. That lets
            # ``docker compose stop`` shut down cleanly.
            try:
                do_job(j, num_envs=num_envs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # noqa: BLE001 - intentional broad catch
                import traceback as _tb
                tb_text = _tb.format_exc()
                # Print the FULL traceback to stdout for the operator
                # so debugging from container logs is still easy. The
                # job record only stores a trimmed first line + short
                # summary - Mongo isn't the right place for full
                # tracebacks, and the dashboard renders eval_error as
                # a single line in the Jobs table.
                print(
                    f"do_job uncaught exception for job {j.get('_id')}: "
                    f"{type(exc).__name__}: {exc}\n{tb_text}",
                    flush=True)
                # Trim the traceback to its last few frames so the
                # one-line dashboard rendering stays useful. Operators
                # who need the full thing have stdout/robotaxi.out.
                tb_lines = tb_text.strip().splitlines()
                tail = "\n".join(tb_lines[-6:]) if tb_lines else str(exc)
                err_summary = f"{type(exc).__name__}: {exc}\n...\n{tail}"
                try:
                    update_job(j["_id"], err_summary[:4000], "eval_error")
                    update_job(
                        j["_id"],
                        datetime.datetime.now(datetime.timezone.utc),
                        "ended_at")
                    update_job(j["_id"], "FAILED")
                except Exception as inner:  # noqa: BLE001
                    # If Mongo itself is the problem we can't record
                    # anything - just log and continue. The poll loop
                    # will keep retrying and the job stays NOT_STARTED
                    # so it gets another shot once Mongo recovers.
                    print(
                        f"do_job error-recording also failed for job "
                        f"{j.get('_id')}: {type(inner).__name__}: {inner}",
                        flush=True)
            finally:
                # Clear the per-job state visible to the SIGTERM
                # handler so a late shutdown signal between jobs
                # doesn't try to flush a stale Learner reference
                # from the previous run. Next job's pickup will
                # re-populate it inside main() after constructing
                # its own Learner.
                _emergency_state.clear()
        print("sleep")
        time.sleep(5)


# Fixed path for the trainer singleton lock. Held for the life of the main
# process; auto-released by the kernel if we crash, so a dead predecessor never
# wedges startup.
_TRAINER_LOCK_PATH = "/tmp/robotaxi_trainer.lock"
# Module-level ref keeps the locked fd (and thus the flock) alive for the whole
# process; letting it be garbage-collected would release the lock.
_trainer_lock_fh = None


def _reap_straggler_trainers():
    """SIGKILL orphaned robotaxi.py worker processes left by a dead main.

    Only called AFTER we win the singleton lock, so there is no live sibling
    main — any remaining robotaxi.py *python* process is an env worker orphaned
    by a previously crashed/killed main. Leaving them alive keeps them attached
    to the shared ros-server endpoints and re-introduces the cmd_id contention
    (scene-data timeouts / glitchy pauses) we are guarding against.

    Best-effort /proc scan (Linux container only). Skips self and skips the
    bash launcher (its argv0 is `bash`, not a python interpreter). Our own env
    workers don't exist yet at call time, so they are never targeted.
    """
    me = os.getpid()
    killed = []
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == me:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as c:
                    parts = c.read().split(b"\x00")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if not parts:
                continue
            argv0 = parts[0].decode("utf-8", "replace")
            cmd = b" ".join(parts).decode("utf-8", "replace")
            # Only kill actual python interpreters running robotaxi.py, never
            # the bash `-c ... robotaxi.py ...` launcher (argv0 == bash).
            if "robotaxi.py" in cmd and "python" in os.path.basename(argv0):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception as e:  # noqa: BLE001
        print(f"_reap_straggler_trainers: scan failed (non-fatal): {e}",
              flush=True)
    if killed:
        print(f"_reap_straggler_trainers: reaped {len(killed)} orphaned "
              f"robotaxi.py worker(s) from a previous run: {killed}",
              flush=True)


def _acquire_singleton_lock_or_exit():
    """Guarantee exactly ONE trainer main runs at a time.

    Running a second `python robotaxi.py` while one is already alive was the
    root cause of the "glitchy pause" incident: two mains publish sim_command
    to the same ros-server-0/1 endpoints and race each other's car_scene_data
    replies, so each blocks the full 5s scene-data timeout every few steps.

    We take a non-blocking advisory flock on a fixed path. If a live main holds
    it we REFUSE to start (never kill a healthy running trainer). If it is free
    we take it, record our pid, and reap any orphaned workers from a crashed
    predecessor. The lock releases automatically when this process exits.
    """
    global _trainer_lock_fh
    import fcntl  # Linux-only; entrypoint always runs in the sim-controller container.
    fh = open(_TRAINER_LOCK_PATH, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = ""
        try:
            with open(_TRAINER_LOCK_PATH) as r:
                holder = r.read().strip()
        except Exception:  # noqa: BLE001
            pass
        fh.close()
        print(
            "FATAL: another robotaxi.py trainer main is already running"
            + (f" (pid {holder})" if holder else "")
            + ". Refusing to start a second trainer: duplicate mains contend "
            "over the shared ros-server endpoints and cause scene-data "
            "timeouts / glitchy pauses. Stop the existing trainer first, e.g.:\n"
            "  pkill -9 -f robotaxi.py",
            flush=True)
        sys.exit(1)
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    _trainer_lock_fh = fh  # keep alive for the process lifetime
    print(f"trainer singleton lock acquired (pid {os.getpid()}, "
          f"{_TRAINER_LOCK_PATH}).", flush=True)
    _reap_straggler_trainers()


if __name__ == "__main__":
    import argparse
    import sys
    # Enforce single-trainer BEFORE spawning any env workers so a stray
    # duplicate command exits immediately instead of stacking up on the
    # shared ros-server endpoints. Env workers spawned later never run
    # __main__, so they don't touch this lock.
    _acquire_singleton_lock_or_exit()
    # tf-agents' ParallelPyEnvironment refuses to start unless the
    # multiprocessing 'spawn' context has been initialized at the program
    # entrypoint via system_multiprocessing.handle_main. The single-env
    # branch in build_train_env() doesn't trigger this requirement, so
    # the importable test from yesterday passed - but --num-envs > 1
    # crashes with "Unable to load multiprocessing context". Wrap the
    # job loop accordingly.
    from tf_agents.system import system_multiprocessing as multiprocessing

    p = argparse.ArgumentParser(description="Robotaxi RL training driver.")
    p.add_argument(
        '--num-envs', type=int, default=1,
        help="Number of parallel ros-server endpoints to collect from. "
             "Default 1 (single-actor). When >1, each TRAIN job uses "
             "ParallelPyEnvironment over ros-server-0..ros-server-(N-1).")
    # handle_main -> absl.app.run, which parses sys.argv and rejects
    # unknown flags. Consume our own flag with parse_known_args and
    # strip it so absl only sees what it understands.
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    def _absl_main(argv):
        del argv  # absl already parsed its own flags from sys.argv
        run_jobs_loop(num_envs=args.num_envs)

    multiprocessing.handle_main(_absl_main)
