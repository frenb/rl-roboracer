import asyncio
import random
import os
import reverb
import tempfile
import threading
import time

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
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from api import RobotApi

# Global reference to RobotApi
api = None

class PoleCartEnv(py_environment.PyEnvironment):
    def __init__(self, api):
        self._api = api
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3, ), dtype=np.float32, minimum=-0.01, maximum=0.01, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(10,), dtype=np.float32,
            minimum=[-3.06, -1.90, -1.4, -3.06, -1.75, -2.62, -30, -30, -1.5, -1.5],
            maximum=[3.06, 0.63, 1.57, 3.06, 1.92, 2.62, 30, 30, 1.5, 1.5],
            name='observation')
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._api.DoResetBlocking()
        data = self._api.GetSceneDataBlocking()
        return ts.restart(self._scene_data_array(data))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._do_action(action)

        data = self._api.GetSceneDataBlocking()

        if not data['pole_cart']['upright']:
            self._episode_ended = True
            return ts.termination(self._scene_data_array(data), reward=0)
        else:
            return ts.transition(self._scene_data_array(data), reward=1.0, discount=0.95)
    
    def _do_action(self, action):
        # print('Action:')
        # print(action)
        # print(self._current_time_step.observation[0])
        positions = {
            'joint_00': (self._current_time_step.observation[0] + action[0]).item(),
            'joint_01': (self._current_time_step.observation[1] + action[1]).item(),
            'joint_02': (self._current_time_step.observation[2] + action[2]).item(),
            'joint_03': (self._current_time_step.observation[3]).item(),
            'joint_04': (self._current_time_step.observation[4]).item(),
            'joint_05': (self._current_time_step.observation[5]).item(),
        }
        cmd = {
            'cmd_type': 4,
            'positions': positions
        }
        self._api.DoMoveBlocking({'cmd': cmd})

    
    def _scene_data_array(self, scene_data):
        arr = [
            scene_data['joint_00'],
            scene_data['joint_01'],
            scene_data['joint_02'],
            scene_data['joint_03'],
            scene_data['joint_04'],
            scene_data['joint_05'],
            scene_data['pole_cart']['pole_hand_angle'],
            scene_data['pole_cart']['pole_hand_angle_b'],
            scene_data['pole_cart']['pole_angular_speed'],
            scene_data['pole_cart']['pole_angular_speed_b']
        ]
        return np.array(arr, dtype=np.float32)


def main():
    tempdir = tempfile.gettempdir()

    env_name = "NiryoPoleCart-v0" # @param {type:"string"}

    num_iterations = 100000 # @param {type:"integer"}
    initial_collect_steps = 1000 # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 1000 # @param {type:"integer"}
    batch_size = 256 # @param {type:"integer"}
    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}
    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)
    log_interval = 50 # @param {type:"integer"}
    num_eval_episodes = 20 # @param {type:"integer"}
    eval_interval = 100 # @param {type:"integer"}
    policy_save_interval = 100 # @param {type:"integer"}

    # Environment. Use same for eval and collection, though this does not seem standard?
    env = PoleCartEnv(api)

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
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
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
    
    print("initial_collect_actor.run() :)")
    initial_collect_actor.run()
    print("Initial collection done")


    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])

    
    eval_actor = actor.Actor(
        env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'eval'),)


    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
    print('saved_model_dir = ' + saved_model_dir)

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

    for _ in range(num_iterations):
        # Training.
        
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()


# def random_walk_blocking(api):
#     print("******" + " inside random walk blocking " + "******")
#     api.DoResetBlocking()
#     while True:
#         scene_data = api.GetSceneDataBlocking()
#         positions = {
#             'joint_00': scene_data['joint_00'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_01': scene_data['joint_01'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_02': scene_data['joint_02'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_03': scene_data['joint_03'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_04': scene_data['joint_04'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_05': scene_data['joint_05'] + random.uniform(-1, 1) * 3.14 / 360.0,
#         }
#         action = {
#             'cmd_type': 4,
#             'positions': positions
#         }
#         api.DoMoveBlocking({'cmd': action})

# async def random_walk(api):
#     print("******" + " inside random walk " + "******")
#     while True:
#         scene_data = await api.GetSceneData()
#         positions = {
#             'joint_00': scene_data['joint_00'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_01': scene_data['joint_01'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_02': scene_data['joint_02'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_03': scene_data['joint_03'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_04': scene_data['joint_04'] + random.uniform(-1, 1) * 3.14 / 360.0,
#             'joint_05': scene_data['joint_05'] + random.uniform(-1, 1) * 3.14 / 360.0,
#         }
#         action = {
#             'cmd_type': 4,
#             'positions': positions
#         }
#         await api.DoMove({'cmd': action})

async def robot_com():
    global api
    api_not_init = RobotApi()
    await api_not_init.Initialize()
    api = api_not_init
    while True:
        await asyncio.sleep(1)


def robot_com_main():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(robot_com())
    loop.close()


if __name__ == "__main__":
    robot_com_thread = threading.Thread(target=robot_com_main)
    robot_com_thread.start()

    # Wait for api to be initialized.
    while not api:
        time.sleep(0.1)
    print("Robot API initialized")

    main()




