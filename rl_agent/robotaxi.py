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
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

client = MongoClient('mongo', 
    username='root',
    password='example')
# db = client.local
#set database_name variable to environment variable DATABASE_NAME
database_name = os.environ['DATABASE_NAME']
db = client.robotaxi

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

    The returned dict has the four counter keys
    ('reset_timeouts', 'apply_force_timeouts', 'scene_data_timeouts',
    'move_timeouts'); the trainer prefixes them with 'timeouts/' when
    writing to tf.summary so they group together in TensorBoard's UI.
    """
    from tf_agents.environments import parallel_py_environment
    keys = ['reset_timeouts', 'apply_force_timeouts',
            'scene_data_timeouts', 'move_timeouts']
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [proc_env.call('get_timeout_counts')
                    for proc_env in env._envs]
        per_actor = [promise() for promise in promises]
        return {k: sum(d.get(k, 0) for d in per_actor) for k in keys}
    return env.get_timeout_counts()


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


def build_train_env(num_envs, course_type='donut'):
    """Construct the training env (single or parallel) for main()."""
    if num_envs <= 1:
        return make_env('ros-server-0:50051', course_type=course_type)

    from tf_agents.environments import parallel_py_environment
    # actor_index=i wraps each worker's stdout/stderr with [actor-N] so
    # robotaxi.out (and the dashboard log view) become legible when
    # multiple workers are emitting interleaved per-step prints.
    return parallel_py_environment.ParallelPyEnvironment(
        [(lambda i=i: make_env(f'ros-server-{i}:50051',
                               course_type=course_type,
                               actor_index=i))
         for i in range(num_envs)])


def configure_env(env, job_id="", pass_through_actions=False):
    """Apply per-job config to a single env or all parallel subprocess envs.

    tf-agents 0.11 doesn't put a public `call()` proxy on
    ParallelPyEnvironment itself; the dispatch lives on each underlying
    ProcessPyEnvironment in env._envs. ProcessPyEnvironment.call() is
    asynchronous and returns a no-arg promise-callable; following the
    same fire-all-then-wait-all pattern that ParallelPyEnvironment.seed
    uses internally lets all four subprocess configure() calls run
    concurrently instead of serially.
    """
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        promises = [
            proc_env.call('configure', job_id, pass_through_actions)
            for proc_env in env._envs
        ]
        for promise in promises:
            promise()
    else:
        env.configure(job_id, pass_through_actions)


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
    actor_fc_layer_params_x=512,
    actor_fc_layer_params_y=512,
    critic_joint_fc_layer_params_x=512,
    critic_joint_fc_layer_params_y=512,
    log_interval_val=5000,
    num_eval_episodes_val=10,
    eval_interval_val=5000,
    policy_save_interval_val=50,
    model_type="SacAgent"):

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
    actor_fc_layer_params = (actor_fc_layer_params_x, actor_fc_layer_params_y)
    critic_joint_fc_layer_params = (critic_joint_fc_layer_params_x, critic_joint_fc_layer_params_y)
    log_interval = log_interval_val # @param {type:"integer"}
    num_eval_episodes = num_eval_episodes_val # @param {type:"integer"}
    eval_interval = eval_interval_val # @param {type:"integer"}
    policy_save_interval = policy_save_interval_val # @param {type:"integer"}
    # Environment. Use same for eval and collection, though this does not seem standard?
    env = build_train_env(num_envs, course_type="donut")
    # Bookkeeping that used to live on env.course; tracking it in main() lets
    # us share the same code path for single- vs multi-env training (in
    # multi-env mode the per-subprocess course state isn't reachable from
    # main).
    avg_return_arr = []
    max_avg_return = 0.0
    print(f"Job arguments = num_envs: {num_envs}, num_iterations: {num_iterations}, nn_size_x: {actor_fc_layer_params_x}, nn_size_x: {actor_fc_layer_params_y}")
    learner_dir = os.path.join(tempdir, str(job_id),"learner")
    saved_model_dir = os.path.join(learner_dir, learner.POLICY_SAVED_MODEL_DIR)
    log_dir = os.path.join(tempdir, str(job_id),"metrics")
    train_dir=os.path.join(tempdir, str(job_id),"train")
    eval_dir=os.path.join(tempdir, str(job_id),"eval")
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()
    # Strategy
    use_gpu = True #@param {type:"boolean"}
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    # Existing code unconditionally overwrote pass_through_actions to False
    # immediately after assigning the requested value, so the requested
    # value never took effect. Preserve that here by passing False directly.
    configure_env(env, job_id=job_id, pass_through_actions=False)
    print(f"pass_through_actions: False")
    # Critic network.
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(env))

    # Publish the env's current specs to MongoDB so the dashboard's
    # Models-tab "Compat" column can mark rows that no longer match
    # the live env. Done once per TRAIN job start (upsert; the doc is
    # keyed by robot_type so multiple training runs against the same
    # robot are idempotent).
    publish_env_spec(env)

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

    record_dir = '/tfrecords/job_64168c1b58d4d8ccdb76e721'
    # 1. You already loaded the expert demos into memory as a Trajectory object
    # (Renaming the variable from 'files' to 'expert_trajectories' for clarity)
    expert_trajectories = collect_training_data.read_files_from_directory(record_dir)
    print(f"Loaded trajectories shape: {expert_trajectories.step_type.shape}")

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

    # 5. Add a batch dimension of 1 for Reverb
    #batched_parsed_dataset = trajectory_dataset.batch(1)
 

    # parsed_dataset = collect_training_data.get_parsed_dataset(file)
    
    # collect_training_data.train_agent_sampling(
    #     actor_net,
    #     record_dir, 
    #     training_steps=1000,
    #     sampling_fraction=0.1,
    #     parsed_dataset=parsed_dataset)


    # Agent.
    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
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
            debug_summaries = True,
            summarize_grads_and_vars = True
        )
        
        tf_agent.initialize()
    # Replay Buffer.
    # collect_observer is fan-out-aware: in multi-env mode it splits the
    # batched Trajectory produced by ParallelPyEnvironment into N
    # per-env writes. expert_observer is always plain-unbatched and is
    # what the offline expert-demo loop below feeds directly. In
    # single-env mode the two are the same instance.
    (reverb_server, reverb_replay, dataset,
     rb_observer, expert_observer) = make_local_replay(
        tf_agent.collect_data_spec,
        capacity=replay_buffer_capacity,
        sample_batch_size=batch_size,
        sequence_length=2,
        stride_length=1,
        num_envs=num_envs)
    table_name = 'uniform_table'
    experience_dataset_fn = lambda: dataset
    
    # Policies
    tf_eval_policy = tf_agent.policy
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
    print("Loading expert demonstrations into Reverb...")
    items_added = 0
    for unbatched_traj in trajectory_dataset:
        expert_observer(unbatched_traj)
        items_added += 1
        if items_added >=50000:
            break
        if items_added % 10000 == 0:
            print(f"Batch trajectory {items_added} added")
            
    print(f"Successfully loaded {items_added} expert steps into Reverb.")

    print_replay_buffer_size(reverb_replay,table_name,replay_buffer_capacity)

    # BC pretrain actor_net on the expert demos before SAC takes over.
    # See collect_training_data.bc_pretrain_actor_net for the full
    # rationale; in short, pure SAC's actor loss does not directly
    # consume buffer actions, so without this step expert demos only
    # influence the actor indirectly via the critic and the policy
    # never imitates the expert before on-policy data dilutes the
    # buffer. tf_agent.actor_network is the same Python object as
    # actor_net, so weight updates here are visible to SAC.
    if bc_pretrain_steps_val > 0:
        collect_training_data.bc_pretrain_actor_net(
            actor_net=actor_net,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            strategy=strategy,
            trajectory_dataset=trajectory_dataset,
            training_steps=bc_pretrain_steps_val,
            batch_size=batch_size)

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
    
    eval_actor = actor.Actor(
        env,
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

    agent_learner = learner.Learner(
        learner_dir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers)
    
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

        eval_actor.run()

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
        return results

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

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]
    curr_iteration=0
    print("Num iterations: " + str(num_iterations), flush=True)
    print("Eval interval: " + str(eval_interval), flush=True)
    print("Log interval: " + str(log_interval), flush=True)
    min_write_step = 0

    for _ in range(num_iterations):
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
        
        if eval_interval and step % eval_interval == 0:
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
            metrics = get_eval_metrics()
            course_metrics = read_course_metrics(env)
            for name, value in course_metrics.items():
                tf.summary.scalar(name, data=value, step=step)
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
                tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
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
                    action_spec=action_spec)
                saved_checkpoint = True

            eval_cycle_elapsed = time.time() - eval_cycle_start
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
 
def run_policy(saved_policy, tf_env, job_id="",
                                  max_episodes=3,
                                  num_eval_episodes=5,
                                  log_interval=10):
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

    def run_one_trial(trial_idx):
        """Step the env in log_interval-sized chunks until
        num_eval_episodes episodes have completed. Returns the final
        per-trial metric dict.
        """
        # Reset metrics at the start of each trial so AverageReturn /
        # AverageEpisodeLength reflect only this trial's episodes; the
        # outer ``returns`` list collects per-trial means independently.
        for m in eval_metrics_list:
            m.reset()
        episodes_metric.reset()

        step = 0
        while int(episodes_metric.result()) < num_eval_episodes:
            eval_actor.run()
            step += log_interval
            episodes_done = int(episodes_metric.result())
            partial = ', '.join(
                '{} = {:.4f}'.format(m.name, float(m.result()))
                for m in eval_metrics_list)
            print('trial {0}: step = {1}: {2}/{3} episodes, {4}'.format(
                trial_idx, step, episodes_done, num_eval_episodes, partial),
                flush=True)

        return {m.name: m.result() for m in eval_metrics_list}

    def log_eval_metrics(trial_idx, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('trial {0} final: {1}'.format(trial_idx, eval_results), flush=True)

    curr_trial=0
    returns=[]
    while curr_trial < max_episodes:
        debug_print("in loop")
        metrics = run_one_trial(curr_trial + 1)
        log_eval_metrics(curr_trial + 1, metrics)
        avg_return = metrics["AverageReturn"]
        returns.append(avg_return)
        curr_trial=curr_trial+1
    return returns

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
                     num_trials=None, num_eval_episodes=None):
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
    """
    env = make_env('ros-server-0:50051')
    env.job_id = job_id
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
              observation_spec=None, action_spec=None):
    """Persist a new saved-model record to MongoDB.

    Extended with ``observation_spec`` / ``action_spec`` kwargs so the
    Models tab's "Compat" column can compare a model's training-time
    expectations against the current env. Both default to None (legacy
    callers + RandomPyPolicy that doesn't actually have a learned
    spec); the dashboard renders missing specs as "unknown" compat
    rather than failing.
    """
    # "_id": ObjectID(),
    # model_type: 'SacAgent',
    # training_iterations: 50000,
    # location: '/saved_models/niryo/SacAgent/8',
    # notes: 'this is a dummy field',
    # robot_type: 'niryo'
    ts = time.time()
    iso_date = datetime.datetime.fromtimestamp(ts, None)
    db.models.insert_one(
        {
            "create_date": iso_date,
            "location": path,
            "robot_type": robot_type,
            "model_type": model_type,
            "training_iterations": training_iterations,
            "notes": "NA",
            "avg_return": float(avg_return) if avg_return is not None else None,
            # New: spec metadata for compatibility checking at EVAL
            # time and on the Models tab. _spec_to_dict handles None
            # cleanly.
            "observation_spec": _spec_to_dict(observation_spec),
            "action_spec": _spec_to_dict(action_spec),
        })

def save_results_to_db(path, results):
    if not results or len(results) == 0:
        print("Warning: No results to save. Skipping DB insert.")
        return
    else:
        print("Has results to save.")
    saved_model_object = db.models.find_one({"location": path})
    ts = time.time()
    iso_date = datetime.datetime.fromtimestamp(ts, None)
    db.leaderboard_scores.insert_one(
        {
            "create_date": iso_date,
            "path": saved_model_object["location"],
            "mean_score": float(np.mean(results)),
            "robot_type": saved_model_object["robot_type"],
            "model_type": saved_model_object["model_type"],
            "model_id": saved_model_object["_id"],
            "median_score": float(np.median(results)),
            "min_score": float(np.min(results)),
            "max_score": float(np.max(results)),
            "scores": np.asarray(results).tolist(),
            "model_id": saved_model_object["_id"]
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

def get_jobs():
    debug_print("in get_jobs")
    is_not_started = {"status": "NOT_STARTED"}
    is_in_progress = {"status": "IN_PROGRESS"}
    jobs = db.jobs.find({"$or":[is_not_started, is_in_progress]})
    debug_print(jobs)
    return jobs

def do_job(job, num_envs=1):
    print(job["job_type"])
    update_job(job["_id"], "IN_PROGRESS")
    update_job(job["_id"], 0, "percent_complete")
    # Wall-clock duration tracking. We stamp `started_at` here (right
    # after the IN_PROGRESS transition) and `ended_at` immediately
    # before the DONE transition at the bottom of this function. Stored
    # as a timezone-aware UTC datetime so pymongo writes a BSON Date
    # and Express's res.json serialises it as ISO-8601 with a 'Z'
    # suffix. The Jobs grid in dashboard/jobs.html uses these to
    # render a Duration column that ticks live for IN_PROGRESS jobs.
    # Jobs that crash mid-run leave `started_at` set but no `ended_at`
    # - the dashboard recognises that case and shows '—' for duration.
    update_job(job["_id"], datetime.datetime.now(datetime.timezone.utc), "started_at")
    #Move all data for jobs with _id = job["_id"] from /tmp to /jobsdata
    job_type = job["job_type"]
    
    if job_type == "DEMO":
        num_iterations=job["num_iterations"] if job["num_iterations"] != "" else 50000
        print(job)
        env = make_env('ros-server-0:50051')
        collect_expert_demos(env, num_iterations, job["_id"])
        root_dir = "/tfrecords/job_" + str(job["_id"])
        collect_training_data.train_agent(root_dir, int(job["training_steps"]))
        print("after collect_expert_demos")
    elif job_type == "BC_TRAINING_ONLY":
        root_dir = "/tfrecords/job_" + str(job["demo_job_id"])
        collect_training_data.train_agent(root_dir, int(job["training_steps"]))
        print("after collect_expert_demos")
    elif job_type == "TRAIN":
        move_all_jobs_data(job["_id"]) 
        num_iterations=job["num_iterations"] if job["num_iterations"] != "" else 50000
        pass_through_actions=job["pass_through_actions"] if job["pass_through_actions"] != "" else False,
        actor_fc_layer_params_x=512 if job.get("nn_size_x") == None else int(job.get("nn_size_x"))
        actor_fc_layer_params_y=512 if job.get("nn_size_y") == None else int(job.get("nn_size_y"))
        critic_joint_fc_layer_params_x=512 if job.get("nn_size_x") == None else job.get("nn_size_x")
        critic_joint_fc_layer_params_y=512 if job.get("nn_size_y") == None else job.get("nn_size_y")
        print(job)
        print(f"in do_job pass_through_actions: {pass_through_actions}")
        main(job_id=job["_id"],
            num_envs=num_envs,
            num_iterations_val=num_iterations,
            pass_through_actions=pass_through_actions,
            actor_fc_layer_params_x=actor_fc_layer_params_x,
            actor_fc_layer_params_y=actor_fc_layer_params_y,
            critic_joint_fc_layer_params_x=critic_joint_fc_layer_params_x,
            critic_joint_fc_layer_params_y=critic_joint_fc_layer_params_y,
            eval_interval_val=10)
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
        num_trials = _opt_int(job.get("num_trials"))
        num_eval_episodes = _opt_int(job.get("num_eval_episodes"))
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
                    num_eval_episodes=num_eval_episodes)
            else:
                load_saved_model(
                    model_type, path=location, job_id=job["_id"],
                    num_trials=num_trials,
                    num_eval_episodes=num_eval_episodes)
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
    # myquery = { "_id": job["_id"] }
    # newvalues = { "$set": { "status": "DONE" } }
    # print(f"updating job {job['_id']}")
    # db.jobs.update_one(myquery,newvalues)
    update_job(job["_id"], datetime.datetime.now(datetime.timezone.utc), "ended_at")
    update_job(job["_id"], "DONE")

def move_all_jobs_data(id):
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
    move_data(id, folders=["eval", "metrics", "train", "learner"])

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

def run_randompolicy(job_id="", num_trials=None, num_eval_episodes=None):
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

    Previously broke with `NameError: name 'env' is not defined`
    because it referenced a module-global env that was removed in the
    rl_agent factory refactor (commit fbe3bce). This fix builds its
    own env exactly like load_saved_model does.
    """
    debug_print("in random policy")
    env = make_env('ros-server-0:50051')
    env.job_id = job_id
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

def collect_expert_demos(environment, num_episodes, job_id=0):
    """Collects expert demonstrations and returns a dataset of trajectories."""
    folder_name = "/tfrecords/job_"+str(job_id)
    create_folder(folder_name=folder_name)
    action_steering_angle = np.float32(0)
    action_apply_force = np.float32(1)
    trajectories = []
    max_trajectories=num_episodes
    max_batch_size=1e3
    num_trajectories=0
    reset_every_n_episodes=1000
    crashes=0
    batch_number=0
    base_force=0.2
    min_force=0.001
    for _ in range(num_episodes):
        environment.reset()
        action_apply_force = 0.2
        while not environment._episode_ended:
            if num_trajectories % reset_every_n_episodes == 0:
                environment.reset()
                action_apply_force = base_force
            numpy_action = np.array([action_apply_force, action_steering_angle])
            action = tf.constant(numpy_action)
            next_time_step=environment._step(action)
            # data = environment._do_action(action)
            # print(f"data: {data}")
            action_steering_angle = next_time_step.observation[0]
            
            if next_time_step.observation[1] < 4:
                action_apply_force = action_apply_force + 1e-4
                action_apply_force = min(base_force, action_apply_force)
            else: 
                action_apply_force = max(min_force,action_apply_force - 1e-3)
            
            traj = create_traj(
                next_time_step.observation,
                action,
                next_time_step.reward,
                next_time_step.discount)

            trajectories.append(traj)
            num_trajectories+=1
            
            if environment._episode_ended:
                crashes=crashes+1
            print(f"num_trajectories: {num_trajectories} vs {max_batch_size}, crashes: {crashes}")
            # if len(trajectories) >= max_batch_size:
            #     path = folder_name + "/" + zero_pad_integer(batch_number,6) + "trajectories.tfrecord"
            #     print(f"writing to {path}")
            #     write_trajectories_to_file(trajectories, path)
            #     trajectories=[]
            #     batch_number+=1
            if num_trajectories > max_trajectories:
                break
        if num_trajectories > max_trajectories:
                break
    path = folder_name + "/" + zero_pad_integer(batch_number,6) + "trajectories.tfrecord"
    print(f"writing to {path}")
    write_trajectories_to_file(trajectories, path)
    # trajectories=[]
    # batch_number+=1

def zero_pad_integer(integer, length):
    """Pads an integer with zeros to a given length."""
    return str(integer).zfill(length)

def create_folder(folder_name):
    """Creates a folder if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)        
 
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

def run_jobs_loop(num_envs=1):
    """Poll MongoDB for jobs and dispatch them indefinitely.

    Each individual job constructs its own env(s) via make_env() (and
    tears them down with the process when the job ends), so no
    module-global RobotApi or background thread is needed at startup.

    num_envs is process-wide configuration (passed in via --num-envs).
    Forwarded to do_job() so the TRAIN job_type can build a parallel
    env. DEMO and EVAL job types are unaffected.
    """
    print(f"Polling for jobs (num_envs={num_envs})...")
    while True:
        jobs = get_jobs()
        for j in jobs:
            print("doing job")
            do_job(j, num_envs=num_envs)
        print("sleep")
        time.sleep(5)


if __name__ == "__main__":
    import argparse
    import sys
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
