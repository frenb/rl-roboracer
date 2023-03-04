import tensorflow as tf
import tf_agents as tf_agents
import numpy as np
import os
from pymongo import MongoClient
import time
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.environments import tf_py_environment
#from sklearn.model_selection import train_test_split
from tf_agents.environments import py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import sequential
from tf_agents.keras_layers import inner_reshape
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import spec_utils
from tf_agents.policies import policy_saver
from tf_agents.eval import metric_utils
import datetime

current_time = datetime.datetime.now()
#job_63f3d9e66882eb19364bbcb7 -- was breaking on the first step, but changing action spec fixed it
#job_63fdb96b9864cc9a57f65398 -- is breaking on the first step
#job_63fc2233eadebe470b7d6641 -- maybe works
#3/3 job_6401b355124fee671f8dffe3
#3/4 job_6402f67e124fee671f8dffe7
root_dir = "C:\\Users\\benja\\Documents\\robots\\LATEST\\tfrecords\\job_6402f67e124fee671f8dffe7"
# batch_size = 50000
training_steps = 5000
observation_size = 32 
action_size = 2

class robotaxi():
    def __init__(self):
        self._action_spec = tensor_spec.BoundedTensorSpec( #BoundedArraySpec(
            shape=(2, ), dtype=np.float32, 
            minimum=[0.0,-1], 
            maximum=[2, 1],
            name='action')
        # self._observation_spec = tensor_spec.BoundedTensorSpec( #BoundedArraySpec(
        #     shape=(32,), dtype=np.float32,
        #     minimum=0,
        #     maximum=1,
        #     name='observation')
        self._observation_spec = tensor_spec.BoundedTensorSpec(
            shape=(32,), dtype=np.float32,
            minimum=[
                -1, #scene_data["car"]["dist_from_traj"] angle to next goal
                -10, #scene_data["car"]["speed"] magnitude of car velocity
                -1, #scene_data["car"]["goal_2"] angle from velocity to car
                0, #scene_data["car"]["left"]
                0, #scene_data["car"]["forward_left"]
                0, #scene_data["car"]["forward_left_left"]
                0, #scene_data["car"]["n_27_50"]
                0, #scene_data["car"]["n_25_00"]
                0, #scene_data["car"]["n_22_50"]
                0, #scene_data["car"]["n_20_00"]
                0, #scene_data["car"]["n_17_50"]
                0, #scene_data["car"]["n_15_00"]
                0, #scene_data["car"]["n_12_50"]
                0, #scene_data["car"]["n_10_00
                0, #scene_data["car"]["n_07_50"],
                0, #scene_data["car"]["n_05_00"],
                0, #scene_data["car"]["n_02_50"]
                0, #scene_data["car"]["forward"], # float64 forward
                0, #scene_data["car"]["p_02_50"],# float64 p_02_50
                0, #scene_data["car"]["p_05_00"],# float64 p_05_00
                0, #scene_data["car"]["p_07_50"],# float64 p_07_50
                0, #scene_data["car"]["p_10_00"],# float64 p_10_00
                0, #scene_data["car"]["p_12_50"],# float64 p_12_50
                0, #scene_data["car"]["p_15_00"],# float64 p_15_00
                0, #scene_data["car"]["p_17_50"],# float64 p_17_50
                0, #scene_data["car"]["p_20_00"],# float64 p_20_00
                0, #scene_data["car"]["p_22_50"],# float64 p_22_50
                0, #scene_data["car"]["p_25_00"],# float64 p_25_00
                0, #scene_data["car"]["p_27_50"],# float64 p_27_50
                0, #scene_data["car"]["forward_right_right"],# float64 forward_right_right
                0, #scene_data["car"]["forward_right"],
                0, #scene_data["car"]["right"]
                ],
            maximum=[
                1, #scene_data["car"]["dist_from_traj"]
                10, #scene_data["car"]["speed"] magnitude of car velocity
                1, #scene_data["car"]["goal_2"] angle from velocity to car
                1000, #scene_data["car"]["left"]
                1000, #scene_data["car"]["forward_left"]
                1000, #scene_data["car"]["forward_left_left"]
                1000, #scene_data["car"]["n_27_50"]
                1000, #scene_data["car"]["n_25_00"]
                1000, #scene_data["car"]["n_22_50"]
                1000, #scene_data["car"]["n_20_00"]
                1000, #scene_data["car"]["n_17_50"]
                1000, #scene_data["car"]["n_15_00"]
                1000, #scene_data["car"]["n_12_50"]
                1000, #scene_data["car"]["n_10_00
                1000, #scene_data["car"]["n_07_50"],
                1000, #scene_data["car"]["n_05_00"],
                1000, #scene_data["car"]["n_02_50"]
                1000, #scene_data["car"]["forward"], # float64 forward
                1000, #scene_data["car"]["p_02_50"],# float64 p_02_50
                1000, #scene_data["car"]["p_05_00"],# float64 p_05_00
                1000, #scene_data["car"]["p_07_50"],# float64 p_07_50
                1000, #scene_data["car"]["p_10_00"],# float64 p_10_00
                1000, #scene_data["car"]["p_12_50"],# float64 p_12_50
                1000, #scene_data["car"]["p_15_00"],# float64 p_15_00
                1000, #scene_data["car"]["p_17_50"],# float64 p_17_50
                1000, #scene_data["car"]["p_20_00"],# float64 p_20_00
                1000, #scene_data["car"]["p_22_50"],# float64 p_22_50
                1000, #scene_data["car"]["p_25_00"],# float64 p_25_00
                1000, #scene_data["car"]["p_27_50"],# float64 p_27_50
                1000, #scene_data["car"]["forward_right_right"],# float64 forward_right_right
                1000, #scene_data["car"]["forward_right"],
                1000, #scene_data["car"]["right"]
                ],
            name='observation')
        self._reward_spec = tensor_spec.BoundedTensorSpec(
            shape=(1,), 
            dtype=np.float32,
            minimum=0,
            maximum=1, 
            name='reward')
        self._discount_spec = tensor_spec.BoundedTensorSpec(
            shape=(1,), 
            dtype=np.float32,
            minimum=0,
            maximum=1, 
            name='discount')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._episode_ended = False
        self._num_steps = 0

    def action_spec(self):
        return self._action_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def discount_spec(self):
        return self._discount_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return self._time_step_spec

    def reset(self):
        self._num_steps = 0
        self._episode_ended = False
        observation =  tf.random.normal(self.observation_spec().shape, mean=0, stddev=1)
        return tf_agents.trajectories.time_step.restart(np.array(observation, dtype=np.float32))

    def step(self, action):
        action_0=action[0].numpy()
        action_1=action[1].numpy()
        observation =  tf.random.normal(env.observation_spec().shape, mean=0, stddev=1)
        if self._num_steps > 10:
            self._episode_ended = True
            return tf_agents.trajectories.time_step.transition(
                np.array(observation, dtype=np.float32), reward=0, discount=0.99)
        self._num_steps += 1
        # generate randome reward between 0 and 10
        reward = np.random.randint(0,10)
        return tf_agents.trajectories.time_step.transition(
            np.array(observation, dtype=np.float32), reward=reward, discount=0.99)

env = robotaxi()
#env = tf_py_environment.TFPyEnvironment(env)

# Create a function to serialize a single example
# def serialize_example(observation, action, reward, discount):
#     feature = {
#         'action': tf.train.Feature(float_list=tf.train.FloatList(value=action.numpy().ravel())),
#         'observation': tf.train.Feature(float_list=tf.train.FloatList(value=observation.numpy().ravel())),
#         'reward': tf.train.Feature(float_list=tf.train.FloatList(value=reward.numpy().ravel())),
#         'discount': tf.train.Feature(float_list=tf.train.FloatList(value=discount.numpy().ravel()))
#     }
#     print(feature["observation"])
#     serialized_example = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
#     return serialized_example


# def create_data(num_demonstrations=100):
#     expert_data=[]
#     for _ in range(num_demonstrations):
#         expert_data.append(create_traj())
#     return expert_data

# def create_traj_batch():
#     # Define the observations and actions tensors
#     observation = tensor_spec.sample_spec_nest(
#         env.observation_spec(), outer_dims=(batch_size,))
#     action = tensor_spec.sample_spec_nest(
#         env.action_spec(), outer_dims=(batch_size,))
#     reward = tf.constant([1], dtype=tf.float32)
#     discount = tf.constant([0.99], dtype=tf.float32)
    
#     traj = trajectory.first(
#         observation=observation,
#         action=action,
#         policy_info=(),
#         reward=reward,
#         discount=discount)

#     return traj, observation, action, reward, discount

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

# def write_data(filename='trajectories.tfrecord'):
#     # Open a file for writing the serialized examples
#     with tf.io.TFRecordWriter(filename) as writer:
#         # Iterate over the examples in the dataset and write them to the file
#         traj, obs, action, reward, discount = create_traj_batch()
#         writer.write(serialize_example(obs,action,reward, discount))

# def parse_example(example_proto):
#     # Define a dictionary of feature keys and types
#     example_proto = tf.train.Example.FromString(example_proto)
#     print(example_proto)
#     feature_description = {
#         'action': tf.train.Feature(float_list=tf.train.FloatList([], tf.float32)),
#         'observation': tf.train.Feature(float_list=tf.train.FloatList([], tf.float32)),
#         'reward': tf.train.Feature(float_list=tf.train.FloatList([], tf.float32)),
#         'discount': tf.train.Feature(float_list=tf.train.FloatList([], tf.float32))
#     }

#     # Parse the example
#     example = tf.io.parse_single_example(example_proto, feature_description)

#     # Extract the feature value
#     feature_value = example["action"]
#     return feature_value

feature_description = {
    'action': tf.io.FixedLenFeature((2,), tf.float32),
    'observation': tf.io.FixedLenFeature((32,), tf.float32),
    # 'reward': tf.io.FixedLenFeature((1,), tf.float32),
    # 'discount': tf.io.FixedLenFeature((1,), tf.float32),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def read_files_from_directory(directory=''):
    """Read all files from a directory and return a list of parsed rows."""
    # get list of file in directory
    files = os.listdir(directory)
    #order list by name
    files.sort()
    # create list of full path
    files = [os.path.join(directory, file) for file in files]
    rows = []
    total_rows = 0
    for f in files:
        num_records=0
        row = None
        num_records, parsed_rows = read_data_from_file(f)
        total_rows += num_records
        rows = np.concatenate((rows, parsed_rows))
        print(f"reading {f}, length={num_records}")
    print(f"total rows {total_rows}")
    parsed_dataset_traj = convert_tfrecord_to_trajectory(rows, total_rows)
    return parsed_dataset_traj

def get_number_of_records(file_name):
    """Get the number of records in a TFRecord file."""
    return sum(1 for _ in tf.data.TFRecordDataset(file_name))

def read_data_from_file(file):
    """Read the data from a TFRecord file."""
    raw_dataset = tf.data.TFRecordDataset(file)
    
    parsed_dataset = raw_dataset.map(_parse_function)
    num_records = get_number_of_records(file)
    parsed_rows = []
    for parsed_record in parsed_dataset.take(num_records): 
        # print(repr(parsed_record))
        # if parsed_record["reward"] > 0:
        #     print(f"reached goal reward {parsed_record['reward']}")
        parsed_rows.append(parsed_record)

    print(f"num records {num_records}")
    return num_records, parsed_rows

def convert_tfrecord_to_trajectory(rows,batch_size=1000):
    """Read the data from a TFRecord file."""
    reward = tf.constant([1], dtype=tf.float32)
    discount = tf.constant([0.99], dtype=tf.float32)
    observation = np.empty((batch_size,32), dtype=np.float32)
    action = np.empty((batch_size,2), dtype=np.float32)
    i=0

    #sampled_rows = np.random.choice(rows, 10000)
    for row in rows: 
        #print(row)
        observation_val = row["observation"]
        action_val = row["action"]
        observation[i]=observation_val
        action[i]=action_val
        i=i+1

    print(action.shape)
    print(observation.shape)

    traj = trajectory.first(
        observation=observation,
        action=action,
        policy_info=(),
        reward=reward,
        discount=discount)

    return traj


parsed_dataset_traj = read_files_from_directory(directory=root_dir)
# parsed_dataset, parsed_dataset_traj = read_data('trajectories.tfrecord', batch_size)

# Create the behavioral cloning agent
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-5)

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

observation_spec, action_spec, time_step_spec = (
         spec_utils.get_tensor_specs(env))

with strategy.scope():
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=(256, 256),
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))

    observation_spec, action_spec, time_step_spec = (
            spec_utils.get_tensor_specs(env))

    agent = behavioral_cloning_agent.BehavioralCloningAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            cloning_network=actor_net,
            optimizer=optimizer)

expert_data = parsed_dataset_traj
i=0
for _ in range(training_steps):
    loss_info_after_ts = agent.train(expert_data)
    if i%100==0:
        print(f"after {i}: {loss_info_after_ts.loss}")
    i=i+1

policy = agent.policy

def get_policy_type_name(policy):
    if (isinstance(policy, str)):
        policy_type = policy
    else:
        policy_type = type(policy).__name__
    return policy_type

def get_save_dir_root(policy):
    policy_type = get_policy_type_name(policy)
    saved_models_dir = "C:/Users/benja/Documents/robots/LATEST/saved_models" #os.getenv('SAVED_MODELS_DIR')
    robot_type = "robotaxi" #os.getenv('ROBOT_TYPE')
    return os.path.join(saved_models_dir,robot_type,policy_type)

def get_save_dir_root_docker(policy):
    policy_type = get_policy_type_name(policy)
    saved_models_dir = "/saved_models" #os.getenv('SAVED_MODELS_DIR')
    robot_type = "robotaxi" #os.getenv('ROBOT_TYPE')
    return os.path.join(saved_models_dir,robot_type,policy_type)

def get_next_model_version(policy):
    path=get_save_dir_root(policy)
    file_list = os.listdir(path)
    sorted_file_list=sorted(file_list,key=str,reverse=True)
    num_dirs = len(sorted_file_list)
    next_model_version=str(num_dirs)
    return path, next_model_version

def get_save_dir_name(policy):
    path, next_dir_name=get_next_model_version(policy)
    save_dir_root_docker = get_save_dir_root_docker(policy)
    return os.path.join(path,next_dir_name), os.path.join(save_dir_root_docker,next_dir_name)

client = MongoClient('localhost:27017', 
    username='root',
    password='example')
db = client.local

def add_model(path, robot_type, model_type, training_iterations, avg_return=None, path_docker=""):
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
            "location": path_docker,
            "location_windows": path,
            "robot_type": robot_type,
            "model_type": model_type,
            "training_iterations": training_iterations,
            "notes": "NA",
            "avg_return": None
        })

def save_policy(policy, num_iterations):
    """Saves a policy to a given path."""
    tf_policy_saver = policy_saver.PolicySaver(policy)
    save_dir_name, save_dir_name_docker=get_save_dir_name(policy)
    save_dir_name = save_dir_name + "_step_" + str(1)
    save_dir_name_docker = save_dir_name_docker + "_step_" + str(1)
    save_dir_name_docker = save_dir_name_docker.replace("\\", "/")
    tf_policy_saver.save(save_dir_name)
    robot_type = "robotaxi" #os.getenv('ROBOT_TYPE')
    model_type=get_policy_type_name(policy)
    training_iterations=num_iterations
    add_model(
        save_dir_name,
        robot_type,
        model_type,
        training_iterations,
        path_docker=save_dir_name_docker)

save_policy(policy, training_steps)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not environment._episode_ended:
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return

for _ in range(10):
    avg_return = compute_avg_return(env, policy, num_episodes=10)
    print(f"avg_return: {avg_return}")