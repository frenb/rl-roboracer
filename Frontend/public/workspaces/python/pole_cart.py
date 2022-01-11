import asyncio
import random
import os
import reverb
import tempfile
import threading
import time
import json
import datetime

import numpy as np
import tensorflow as tf

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
from pymongo import MongoClient

from api import RobotApi

# Global reference to RobotApi
api = None
client = MongoClient('mongo', 
    username='root',
    password='example')
db = client.local
class PoleCartEnv(py_environment.PyEnvironment):
    def __init__(self, api):
        self._api = api
        # action_apply_force = tf.random.uniform((1,),0,10,tf.dtypes.float32).numpy().tolist()[0]
        # action_steering_angle = tf.random.uniform((1,),-45,45,tf.dtypes.float32).numpy().tolist()[0]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2, ), dtype=np.float32, 
                minimum=[0,-45], 
                maximum=[10,45], name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(10,), dtype=np.float32,
        #     minimum=[-3.06, -1.90, -1.4, -3.06, -1.75, -2.62, -30, -30, -1.5, -1.5],
        #     maximum=[3.06, 0.63, 1.57, 3.06, 1.92, 2.62, 30, 30, 1.5, 1.5],
        #     name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(9,), dtype=np.float32,
            minimum=[0, -25, -10000, -10000, -10000,-100,0,0,0],
            maximum=[1, 100, 10000, 10000, 10000,100,1000,1000,360],
            name='observation')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._state = self._get_empty_state()
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        #print(9999)
        self._episode_ended = False
        #print(222222)
        #print(7777777)
        self._api.DoResetBlocking()
        data = self._api.GetCarSceneDataBlocking()
        #self._state = self._get_empty_state()
        data_arr=self._scene_data_array(data)
        self._step_costs=[]
        #return ts.restart(np.array([data_arr], dtype=np.float32))
        return ts.restart(np.array(data_arr, dtype=np.float32))
        #return ts.restart(np.array(self._state, dtype=np.int32))
   
    def __has_failed(self, data_arr):
        #print("has_failed: " + str(data_arr))
        has_fallen = data_arr[3] < -0.5
        int_rotation_z = round(data_arr[8])
        has_flipped = int_rotation_z == 90 or int_rotation_z == 180 or int_rotation_z == 270
        #print("has_fallen: " + str(has_fallen))
        #print("has_flipped: " + str(has_flipped) + " " + str(int_rotation_z))
        return has_fallen or has_flipped
    
    def __has_succeeded(self, data_arr):
        #print("has_succeeded: " + str(data_arr))
        has_succeeded = data_arr[0] == 1
        #print("has_succeeded: " + str(has_succeeded))
        return has_succeeded
    
    def _apply_force(self):
        self._api.DoApplyForceBlocking()

    def _step(self, action):
        #print("in _step")
        time.sleep(2)
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        #print("before do action: " + str(self._state))
        result = self._do_action(action)
        debug_print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" + str(result))
        data = self._api.GetCarSceneDataBlocking()
        debug_print("data: " + str(data))
        data_arr = self._scene_data_array(data)
        self._step_costs.append(data_arr[6] + data_arr[7])
        #print("self._state")
        #print("in_step: " + str(self._state))
        if self.__has_failed(data_arr):
            debug_print("has failed")
            self._episode_ended = True
            #print('before termination')
            if False:
                term_time_step = ts.termination(np.array([data_arr], dtype=np.float32), -1)
            else:
                term_time_step = ts.termination(np.array(data_arr, dtype=np.float32), -1)
            #term_time_step = ts.termination(np.array(self._state, dtype=np.int32), -1)
            #print("is_last(): " + str(term_time_step.is_last()))
            return term_time_step
            #return ts.termination(np.array([self._state]), reward=-1)
        if self.__has_succeeded(data_arr):
            debug_print("has succeeded")
            self._episode_ended = True
            #print('before termination')
            reward = (1 / (1+np.sum(self._step_costs)))*100
            if False:
                term_time_step = ts.termination(np.array([data_arr], dtype=np.float32), reward)
            else:
                term_time_step = ts.termination(np.array(data_arr, dtype=np.float32), reward)
            #term_time_step = ts.termination(np.array(self._state, dtype=np.int32), reward)
            #print("is_last(): " + str(term_time_step.is_last()))
            return term_time_step
        else:
            debug_print("did not fail")
            self._episode_ended = False
            #ts.transition(np.array([self._state], dtype=np.int32), reward=0.05, discount=1.0)
            
            if False:
                return ts.transition(np.array([data_arr], dtype=np.float32), reward=1.0, discount=0.90)
            else:
                return ts.transition(np.array(data_arr, dtype=np.float32), reward=1.0, discount=0.90)
            #return ts.transition(np.array(self._state, dtype=np.int32), reward=1.0, discount=0.90)
    
    # def _do_action(self, action):
    #     # print('Action:')
    #     # print(action)
    #     # print(self._current_time_step.observation[0])
    #     positions = {
    #         'joint_00': (self._current_time_step.observation[0] + action[0]).item(),
    #         'joint_01': (self._current_time_step.observation[1] + action[1]).item(),
    #         'joint_02': (self._current_time_step.observation[2] + action[2]).item(),
    #         'joint_03': (self._current_time_step.observation[3]).item(),
    #         'joint_04': (self._current_time_step.observation[4]).item(),
    #         'joint_05': (self._current_time_step.observation[5]).item(),
    #     }
    #     cmd = {
    #         'cmd_type': 4,
    #         'positions': positions
    #     }
    #     self._api.DoMoveBlocking({'cmd': cmd})
    
    def _do_action(self, action):
        debug_print(action)
        debug_print(type(action).__name__)
        if type(action).__name__ == "ndarray":
            #action_arr = action[0].tolist()
            action_arr = action.tolist()
        else:
            action_arr = action.numpy().tolist()
        debug_print('Action: ' + str(action_arr))
        acceleration=action_arr[0]
        steering_angle=action_arr[1]
        self._api.DoApplyForceBlocking(
            acceleration,
            steering_angle)

    def _get_empty_state(self):
        emptyState = [0,0,0,0,0,0,0,0,0]
        return emptyState

    
    def _scene_data_array(self, scene_data):
        #print("in scene_data_array")
        #print(scene_data)
        # float64 speed
        # float64 location_x
        # float64 location_y
        # float64 location_z
        # float64 cost
        # float64 dist_from_traj
        # float64 dist_from_goal
        # bool has_not_crashed
        arr = [
            1 if scene_data["car"]['has_not_crashed'] else 0,
            scene_data["car"]['speed'],
            scene_data["car"]['location_x'],
            scene_data["car"]['location_y'],
            scene_data["car"]['location_z'],
            scene_data["car"]["cost"],
            scene_data["car"]["dist_from_traj"],
            scene_data["car"]["dist_from_goal"],
            scene_data["car"]["rotation_z"],
        ]
        # arr = [
        #     scene_data['joint_00'],
        #     scene_data['joint_01'],
        #     scene_data['joint_02'],
        #     scene_data['joint_03'],
        #     scene_data['joint_04'],
        #     scene_data['joint_05'],
        #     scene_data['pole_cart']['pole_hand_angle'],
        #     scene_data['pole_cart']['pole_hand_angle_b'],
        #     scene_data['pole_cart']['pole_angular_speed'],
        #     scene_data['pole_cart']['pole_angular_speed_b']
        # ]
        return np.array(arr, dtype=np.float32)

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
    sorted_file_list=sorted(file_list,key=int,reverse=True)
    next_model_version=str(int(sorted_file_list[0])+1)
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

def main(
    job_id="",
    checkpoint_restore=False, 
    version=None,
    num_iterations_val=50000,
    initial_collect_steps_val=500,
    collect_steps_per_iteration_val=1,
    replay_buffer_capacity_val=1000,
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
    log_interval_val=50,
    num_eval_episodes_val=20,
    eval_interval_val=50,
    policy_save_interval_val=50,
    model_type="SacAgent"):
    tempdir = tempfile.gettempdir()
    env_name = "NiryoPoleCart-v0" # @param {type:"string"}

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
    critic_joint_fc_layer_params = (critic_joint_fc_layer_params_x,critic_joint_fc_layer_params_y)
    log_interval = log_interval_val # @param {type:"integer"}
    num_eval_episodes = num_eval_episodes_val # @param {type:"integer"}
    eval_interval = eval_interval_val # @param {type:"integer"}
    policy_save_interval = policy_save_interval_val # @param {type:"integer"}
    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
    # Environment. Use same for eval and collection, though this does not seem standard?
    env = PoleCartEnv(api)
    train_dir=os.path.join(tempdir, learner.TRAIN_DIR if str(job_id) == "" else str(job_id) + "_" + learner.TRAIN_DIR)
    eval_dir=os.path.join(tempdir, "eval" if str(job_id) == "" else str(job_id) + "_eval")
    # Strategy
    use_gpu = False #@param {type:"boolean"}
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


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
            train_step_counter=train_step)
        
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
    
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)
    # Policies

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

    initial_collect_actor = actor.Actor(
        env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])
        
    if checkpoint_restore is True:
        checkpoint_dir="/tmp/policies/checkpoints"
        # eval_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        #     policy_dir, env.time_step_spec(), env.action_spec())
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=tf_agent.policy,
            replay_buffer=reverb_replay,
            global_step=global_step
        )
    else:
        debug_print("initial_collect_actor.run() :)")
        initial_collect_actor.run()
        debug_print("Initial collection done")

    # restore from checkpoint
    debug_print("+++++")
    debug_print(train_dir) #os.path.join(tempdir, learner.TRAIN_DIR))
    env_step_metric = py_metrics.EnvironmentSteps()
    debug_print("number of steps: " + str(py_metrics.EnvironmentSteps()))
    collect_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=train_dir, # os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])
    
    eval_actor = actor.Actor(
        env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=eval_dir) # os.path.join(tempdir, 'eval'),)
    
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
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers)
    
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        
        print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)


    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]
    curr_iteration=0
    for _ in range(num_iterations):
        # Training.
        
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)
        debug_print("loss_info: " + str(loss_info))
        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
        curr_iteration=curr_iteration+1
    rb_observer.close()
    reverb_server.stop()
    tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
    save_dir_name=get_save_dir_name(tf_agent)
    tf_policy_saver.save(save_dir_name)
    robot_type=robot_type = os.getenv('ROBOT_TYPE')
    model_type=get_policy_type_name(tf_agent)
    training_iterations=num_iterations
    add_model(save_dir_name, robot_type, model_type, training_iterations)
 
def run_episodes_and_create_video(saved_policy, tf_env, job_id=""):
    debug_print("run episodes and create video")
    train_step = train_utils.create_train_step()
    log_interval = 10 # @param {type:"integer"}
    num_eval_episodes = 5 # @param {type:"integer"}
    max_episodes = 10
    eval_dir=os.path.join(tempfile.gettempdir(), "eval" if str(job_id) == "" else str(job_id) + "_eval")
    batch_tf_env = batched_py_environment.BatchedPyEnvironment((tf_env,))
    debug_print("after batch_tf_env")
    time_step = batch_tf_env.reset()
    debug_print("after reset")
    actor.eval_metrics(num_eval_episodes)
    debug_print("after eval metrics")
    eval_actor = actor.Actor(
        batch_tf_env,
        #tf_env,
        saved_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=eval_dir) #os.path.join("/tmp", 'eval'),)
    debug_print("after eval actor")
    print(eval_actor.metrics)
    def get_eval_metrics():
        debug_print("inside get_eval_metrics")
        eval_actor.run()
        debug_print("after eval_actor.run()")
        results = {}
        debug_print(eval_actor.metrics)
        for metric in eval_actor.metrics:
            debug_print("metric: " + str(metric))
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
    env = PoleCartEnv(api)
    # Strategy
    use_gpu = False #@param {type:"boolean"}
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    if path is not None:
        saved_policy, path = get_saved_model(policy_type, path_arg=path)
    else:
        saved_policy, path = get_saved_model(policy_type, version)
    results = run_episodes_and_create_video(saved_policy, env, job_id=job_id)
    save_results_to_db(path, results)

def add_model(path, robot_type, model_type, training_iterations):
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
            "notes": "NA"
        })

def save_results_to_db(path, results):
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

def get_jobs():
    debug_print("in get_jobs")
    jobs = db.jobs.find({"status":"NOT_STARTED"})
    debug_print(jobs)
    # for x in jobs:
    #     print(x)
    return jobs

def do_job(job):
    if job["job_type"] == "TRAIN":
        num_iterations=job["num_iterations"]
        main(job_id=job["_id"], num_iterations_val=num_iterations)
    elif job["job_type"] == "EVAL":
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
    myquery = { "_id": job["_id"] }
    newvalues = { "$set": { "status": "DONE" } }
    db.jobs.update_one(myquery,newvalues)

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
    
def run_randompolicy_collect():
    debug_print(666)
    time_step = env._reset()
    debug_print(time_step)
    #print(1)
    rewards = []
    steps = []
    number_of_episodes = 100
    episode_steps=0
    episode_reward=0
    current_step=0
    for _ in range(number_of_episodes):
        #print(2)
        #print("current step: " + str(current_step))
        current_step=current_step+1
        reward_t=0
        steps_t=0
        env.reset()
        while True:
            action_apply_force = tf.random.uniform((1,),0,10,tf.dtypes.float32).numpy().tolist()[0]
            action_steering_angle = tf.random.uniform((1,),-45,45,tf.dtypes.float32).numpy().tolist()[0]
            #action = tf.TensorArray([action_apply_force,action_steering_angle])
            numpy_action = np.array([action_apply_force, action_steering_angle])
            action = tf.constant(numpy_action)
            #print("modified action: "+str(action))
            next_time_step=env.step(action)
            #env._do_action(action)
            #print(3)
            #print("is last: " + str(next_time_step.is_last()))
            #print(next_time_step)
            #print(next_time_step.reward)
            #print(env.current_time_step())
            #print("is last: " + str(next_time_step.is_last()))
            if next_time_step.is_last():
                #print("before break")
                break
            episode_steps += 1
            #print(4)
            #print(episode_steps)
            #ts.transition(self._state, reward=1.0, discount=0.90)
            episode_reward += next_time_step.reward
            #print(episode_reward)
            time.sleep(1)
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
    env = PoleCartEnv(api)
    print('action_spec:', env.action_spec())
    print('time_step_spec.observation:', env.time_step_spec().observation)
    print('time_step_spec.step_type:', env.time_step_spec().step_type)
    print('time_step_spec.discount:', env.time_step_spec().discount)
    print('time_step_spec.reward:', env.time_step_spec().reward)
    #tf_env = tf_py_environment.TFPyEnvironment(env)
    #main(num_iterations_val=100)
    #print(500)
    #utils.validate_py_environment(env, episodes=5)
    #run_randompolicy()
    #run_randompolicy_collect()
    # results = db.models.find_one({"model_type": "SacAgent"})
    #env._apply_force()
    # while True:
    #     print("before getSceneData")
    #     data = env._api.GetCarSceneDataBlocking()
    #     print(data)
    #     time.sleep(5)
    #env._reset()
    while True:
        jobs = get_jobs()
        for j in jobs:
            print("doing job")
            do_job(j)
        print("sleep")
        time.sleep(5)




