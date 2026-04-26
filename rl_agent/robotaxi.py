import asyncio
import random
import os
import math
import shutil
import reverb
import tempfile
import threading
import time
import json
import datetime

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
from environments import RobotaxiEnv as pce
import collect_training_data
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from api import RobotApi

# Global reference to RobotApi
api = None
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

def main(
    job_id="",
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
    env = pce(api, course_type="donut")
    print(f"Job arguments = num_iterations: {num_iterations}, nn_size_x: {actor_fc_layer_params_x}, nn_size_x: {actor_fc_layer_params_y}")
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
    env.job_id=job_id
    env.pass_through_actions=bool(pass_through_actions)
    env.pass_through_actions = False
    print(f"pass_through_actions: {env.pass_through_actions}")
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
    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))
    
    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)

    dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
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
    
    # Actors
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver( #ReverbConcurrentAddBatchObserver
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)
    
    print("Loading expert demonstrations into Reverb...")
    items_added = 0
    for unbatched_traj in trajectory_dataset:
        rb_observer(unbatched_traj)
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
        eval_results_blob["job_id"] = env.job_id
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
            tf.summary.scalar('avg_goals_per_episode', data=env.course.avg_goals_per_episode, step=step)
            tf.summary.scalar('avg_goals_per_episode_last_30', data=env.course.avg_goals_per_episode_last_30, step=step)
            tf.summary.scalar('max_goals_per_episode', data=env.course.max_goals_per_episode, step=step)
            tf.summary.scalar('max_goals_per_episode_last_30', data=env.course.max_goals_per_episode_last_30, step=step)
            tf.summary.scalar('avg_steering_angle_ratio', data=env.course.avg_steering_angle_ratio, step=step)
            tf.summary.scalar('avg_steering_angle_ratio_last_30', data=env.course.avg_steering_angle_ratio_last_30, step=step)
            tf.summary.scalar('max_speed', data=env.course.max_speed, step=step)
            tf.summary.scalar('max_speed_last_30', data=env.course.max_speed_last_30, step=step)
            tf.summary.scalar('avg_speed', data=env.course.avg_speed, step=step)
            tf.summary.scalar('avg_speed_last_30', data=env.course.avg_speed_last_30, step=step)
            log_eval_metrics(step, metrics)
            current_avg_return = metrics["AverageReturn"]
            env.course.avg_return_arr.append(current_avg_return)
            env.course.max_avg_return = max(current_avg_return, env.course.max_avg_return)
            returns.append([current_avg_return])
            is_new_max_avg = (current_avg_return + 1e-5) > env.course.max_avg_return
            print("current step " + str(step))
            print("num returns " + str(len(returns)))
            print("current iteration " + str(curr_iteration))
            print(f"max return {env.course.max_avg_return}")
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
    reverb_server.stop()
    # tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
    # save_dir_name=get_save_dir_name(tf_agent)
    # tf_policy_saver.save(save_dir_name)
    # robot_type = os.getenv('ROBOT_TYPE')
    # model_type=get_policy_type_name(tf_agent)
    # training_iterations=num_iterations
    # add_model(save_dir_name, robot_type, model_type, training_iterations)
 
def run_episodes_and_create_video(saved_policy, tf_env, job_id=""):
    print("run episodes and create video")
    tempdir = "/tmp/active/"
    train_step = train_utils.create_train_step()
    log_interval = 10 # @param {type:"integer"}
    num_eval_episodes = 5 # @param {type:"integer"}
    max_episodes = 1
    eval_dir=os.path.join( tempdir,
        "eval" if str(job_id) == "" else str(job_id) + "_eval")
    batch_tf_env = batched_py_environment.BatchedPyEnvironment((tf_env,))
    debug_print("after batch_tf_env")
    time_step = batch_tf_env.reset()
    debug_print("after reset")
    actor.eval_metrics(num_eval_episodes)
    debug_print("after eval metrics")
    eval_actor = actor.Actor(
        batch_tf_env,
        saved_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=eval_dir)
    debug_print("after eval actor")
    print(eval_actor.metrics)
    
    def get_eval_metrics():
        debug_print("inside get_eval_metrics")
        eval_actor.run()
        debug_print("after eval_actor.run()")
        results = {}
        print(eval_actor.metrics)
        for metric in eval_actor.metrics:
            print("metric: " + str(metric))
            results[metric.name] = metric.result()
        return results
    
    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))
    curr_episode=0
    returns=[]
    while curr_episode < max_episodes:
        debug_print("in loop")
        metrics = get_eval_metrics()
        log_eval_metrics(0, metrics)
        avg_return = metrics["AverageReturn"]
        returns.append(avg_return)
        curr_episode=curr_episode+1
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
    env = pce(api)
    env.job_id = job_id
    if path is not None:
        saved_policy, path = get_saved_model(policy_type, path_arg=path)
    else:
        saved_policy, path = get_saved_model(policy_type, version)
    results = run_episodes_and_create_video(saved_policy, env, job_id=job_id)
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

async def robot_com():
    global api
    api_not_init = RobotApi()
    await api_not_init.Initialize()
    api = api_not_init
    while True:
        await asyncio.sleep(1)

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

def do_job(job):
    print(job["job_type"])
    update_job(job["_id"], "IN_PROGRESS")
    update_job(job["_id"], 0, "percent_complete")
    #Move all data for jobs with _id = job["_id"] from /tmp to /jobsdata
    job_type = job["job_type"]
    
    if job_type == "DEMO":
        num_iterations=job["num_iterations"] if job["num_iterations"] != "" else 50000
        print(job)
        env = pce(api)
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
            num_iterations_val=num_iterations,
            pass_through_actions=pass_through_actions, 
            actor_fc_layer_params_x=actor_fc_layer_params_x,
            actor_fc_layer_params_y=actor_fc_layer_params_y,
            critic_joint_fc_layer_params_x=critic_joint_fc_layer_params_x,
            critic_joint_fc_layer_params_y=critic_joint_fc_layer_params_y,
            eval_interval_val=10)
    elif job_type == "EVAL":
        model_type=job["model_type"]
        location=job["location"]
        debug_print(location)
        if "RandomPyPolicy" in location:
            run_randompolicy()
        else:
            load_saved_model(model_type, path=job["location"], job_id=job["_id"])
        debug_print(model_type)
        debug_print(job["location"])
    else:
        return
    # myquery = { "_id": job["_id"] }
    # newvalues = { "$set": { "status": "DONE" } }
    # print(f"updating job {job['_id']}")
    # db.jobs.update_one(myquery,newvalues)
    update_job(job["_id"], "DONE")

def move_all_jobs_data(id):
    print(f"moving all jobs data excluding job with {id}")
    # move eval, metrics, train
    move_data(id, folders=["eval", "metrics", "train"])
    jobs = get_job_ids(id)
    print(f"jobs: {jobs}")
    for job in jobs:
        move_data(job["_id"])

def get_job_ids(id):
    # get job ids that are not equal to id
    myquery = { "_id": { "$ne": id } }
    # sort by _id in descending order
    mysort = { "_id": -1 }
    # db.jobs.find using myquery and sorted by mysort
    results = db.jobs.find(myquery).sort("_id", -1).limit(10)
    return results

def move_data(job_id, folders=[""]):
    #Move all data for jobs with _id = job["_id"] from /tmp to /jobsdata
    for folder in folders:
        if folder == "":
            src = os.path.join("/tmp/active/", str(job_id))
            dst = os.path.join("/tmp/jobsdata/", str(job_id))
        else:
            src = os.path.join("/tmp/active/", str(job_id), folder)
            dst = os.path.join("/tmp/jobsdata/", str(job_id), folder)
        print(f"moving {src} to {dst}")
        #check if directory with name = job_id exists in /tmp
        if os.path.isdir(src):
            result = shutil.move(src, dst)
            print(f"moved {result}")

def update_job(id, value, field_name="status"):
    print(f"updating job {id} with {field_name} = {value}", flush=True)
    myquery = { "_id": id }
    newvalues = { "$set": { field_name: value } }
    db.jobs.update_one(myquery, newvalues)

def robot_com_main():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(robot_com())
    loop.close()

def run_randompolicy():
    debug_print("in random policy")
    random_policy = random_py_policy.RandomPyPolicy(
          env.time_step_spec(), env.action_spec())
    results=run_episodes_and_create_video(random_policy, env)
    debug_print(results)
    random_policy_path=get_latest_save_dir_name(random_policy)
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

def run_randompolicy_collect():
    debug_print(666)
    time_step = env._reset()
    debug_print(time_step)
    rewards = []
    steps = []
    number_of_episodes = 100
    episode_steps=0
    episode_reward=0
    current_step=0
    for _ in range(number_of_episodes):
        current_step=current_step+1
        reward_t=0
        steps_t=0
        env.reset()
        while True:
            action_apply_force = tf.random.uniform((1,),0,10,tf.dtypes.float32).numpy().tolist()[0]
            action_steering_angle = tf.random.uniform((1,),-45,45,tf.dtypes.float32).numpy().tolist()[0]
            numpy_action = np.array([action_apply_force, action_steering_angle])
            action = tf.constant(numpy_action)
            next_time_step=env.step(action)
            if next_time_step.is_last():
                break
            episode_steps += 1
            episode_reward += next_time_step.reward
        rewards.append(episode_reward)
        mean_reward = np.mean(rewards)
        print("mean reward: " + str(mean_reward))
        steps.append(episode_steps)
        mean_no_of_steps = np.mean(steps)
        print("mean number of steps: " + str(mean_no_of_steps))
        
    mean_reward = np.mean(rewards)
    mean_no_of_steps = np.mean(steps)
    print("mean reward: " + str(mean_reward))
    print("mean number of steps: " + str(mean_no_of_steps))

def debug_print(text):
    debug_print_enabled = False
    if debug_print_enabled:
        print(text)

if __name__ == "__main__":
    robot_com_thread = threading.Thread(target=robot_com_main)
    robot_com_thread.start()

    # Wait for api to be initialized.
    while not api:
        time.sleep(0.1)
    print("Robot API initialized")
    env = pce(api)
    print('action_spec:', env.action_spec())
    print('time_step_spec.observation:', env.time_step_spec().observation)
    print('time_step_spec.step_type:', env.time_step_spec().step_type)
    print('time_step_spec.discount:', env.time_step_spec().discount)
    print('time_step_spec.reward:', env.time_step_spec().reward)

    while True:
        jobs = get_jobs()
        for j in jobs:
            print("doing job")
            do_job(j)
        print("sleep")
        time.sleep(5)




