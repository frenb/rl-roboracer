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
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
            print("metric.name:" + str(metric.name))
            print("metric.result():" + str(metric.result()))
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
        print("***************training agent**************", flush=True)
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)
        # Evaluating.
        step = agent_learner.train_step_numpy
        print("step: " + str(step), flush=True)
        print("eval_interval: " + str(eval_interval), flush=True)
        
        #********** PRINT REPLAY BUFFER SIZE **********#
        print_replay_buffer_size(
            reverb_replay,
            table_name,
            replay_buffer_capacity)
        
        if eval_interval and step % eval_interval == 0:
            print("***************evaluating agent**************", flush=True)
            print("step: " + str(step), flush=True)
            print("eval_interval: " + str(eval_interval))
            percent_complete = step/num_iterations
            update_job(job_id, percent_complete*100, "percent_complete")
            update_job(job_id, int(step), "training_steps")
            print(f"percent complete: {percent_complete}", flush=True)
            def bc_agent_training():
                print("bc agent training started", flush=True)
                collect_training_data.train_agent_sampling(
                    actor_net,
                    record_dir, 
                    training_steps=2,
                    sampling_fraction=0.002,
                    parsed_dataset=parsed_dataset)
                print("Evaluating agent")
            metrics = get_eval_metrics()
            course_metrics = read_course_metrics(env)
            for name, value in course_metrics.items():
                tf.summary.scalar(name, data=value, step=step)
            log_eval_metrics(step, metrics)
            current_avg_return = metrics["AverageReturn"]
            avg_return_arr.append(current_avg_return)
            avg_return_arr = avg_return_arr[-100:]
            max_avg_return = max(current_avg_return, max_avg_return)
            returns.append([current_avg_return])
            is_new_max_avg = (current_avg_return + 1e-5) > max_avg_return
            print("current step " + str(step))
            print("num returns " + str(len(returns)))
            print("current iteration " + str(curr_iteration))
            print(f"max return {max_avg_return}")
            print(f"current avg return {current_avg_return}")
            print(f"is new max average {is_new_max_avg}")
            
            if step >= min_write_step and is_new_max_avg: # step % write_policy_interval == 0
                tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
                save_dir_name=get_save_dir_name(tf_agent)+ "_step_" + str(step)
                tf_policy_saver.save(save_dir_name)
                robot_type = os.getenv('ROBOT_TYPE')
                model_type=get_policy_type_name(tf_agent)
                training_iterations=num_iterations
                add_model(
                    save_dir_name,
                    robot_type,
                    model_type,
                    training_iterations,
                    avg_return=metrics["AverageReturn"])
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
      max_episodes: number of outer passes; each pass runs
        ``num_eval_episodes`` episodes and produces one
        ``(AverageReturn, AverageEpisodeLength)`` sample point. Default
        3, so a stock EVAL job evaluates 3 x 5 = 15 episodes total and
        the returned ``returns`` list has 3 entries (one mean-of-N per
        pass). Useful for measuring run-to-run variance of a stochastic
        policy without manually re-submitting jobs.
      num_eval_episodes: episodes within one pass. Used both as the
        target episode count per pass (we keep stepping the env in
        ``log_interval``-sized chunks until this many episodes have
        completed) and as ``actor.eval_metrics``'s buffer size (each
        metric averages over the last N episodes). Default 5.
      log_interval: env-step granularity for in-pass progress logging.
        Drives ``actor.Actor.steps_per_run``, so each call to
        ``eval_actor.run()`` advances exactly ``log_interval`` env
        steps before printing one progress line; the pass loop keeps
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

    # Per-pass eval metrics + a fresh NumberOfEpisodes counter to know
    # when each pass's episode budget has been spent. The counter
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
    # for the in-pass progress prints (one line per log_interval
    # steps), matching main()'s inline-eval logging style. Without
    # this, .run() would block for the full pass before printing
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

    def run_one_pass(pass_idx):
        """Step the env in log_interval-sized chunks until
        num_eval_episodes episodes have completed. Returns the final
        per-pass metric dict.
        """
        # Reset metrics at the start of each pass so AverageReturn /
        # AverageEpisodeLength reflect only this pass's episodes; the
        # outer ``returns`` list collects per-pass means independently.
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
            print('pass {0}: step = {1}: {2}/{3} episodes, {4}'.format(
                pass_idx, step, episodes_done, num_eval_episodes, partial),
                flush=True)

        return {m.name: m.result() for m in eval_metrics_list}

    def log_eval_metrics(pass_idx, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('pass {0} final: {1}'.format(pass_idx, eval_results), flush=True)

    curr_pass=0
    returns=[]
    while curr_pass < max_episodes:
        debug_print("in loop")
        metrics = run_one_pass(curr_pass + 1)
        log_eval_metrics(curr_pass + 1, metrics)
        avg_return = metrics["AverageReturn"]
        returns.append(avg_return)
        curr_pass=curr_pass+1
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

def load_saved_model(policy_type, version=None, path=None, job_id=""):
    # Environment. Use same for eval and collection, though this does not seem standard?
    env = make_env('ros-server-0:50051')
    env.job_id = job_id
    if path is not None:
        saved_policy, path = get_saved_model(policy_type, path_arg=path)
    else:
        saved_policy, path = get_saved_model(policy_type, version)
    results = run_policy(saved_policy, env, job_id=job_id)
    save_results_to_db(path, results)

def add_model(path, robot_type, model_type, training_iterations, avg_return=None):
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
            "avg_return": float(avg_return)
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
        if model_type == "RandomPyPolicy":
            run_randompolicy(job_id=job["_id"])
        else:
            load_saved_model(model_type, path=location, job_id=job["_id"])
        debug_print(model_type)
        debug_print(location)
    else:
        return
    # myquery = { "_id": job["_id"] }
    # newvalues = { "$set": { "status": "DONE" } }
    # print(f"updating job {job['_id']}")
    # db.jobs.update_one(myquery,newvalues)
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

def run_randompolicy(job_id=""):
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

    Previously broke with `NameError: name 'env' is not defined`
    because it referenced a module-global env that was removed in the
    rl_agent factory refactor (commit fbe3bce). This fix builds its
    own env exactly like load_saved_model does.
    """
    debug_print("in random policy")
    env = make_env('ros-server-0:50051')
    env.job_id = job_id
    random_policy = random_py_policy.RandomPyPolicy(
        env.time_step_spec(), env.action_spec())
    results = run_policy(random_policy, env, job_id=job_id)
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
