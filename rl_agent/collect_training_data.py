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
#3/4.1 job_6403b7a094b9d3d95304fcb7
#3/15 job_6412bd37f6548aa06d94eea8
root_dir = "C:\\Users\\benja\\Documents\\robots\\LATEST\\tfrecords\\job_6412bd37f6548aa06d94eea8"
# batch_size = 50000
training_steps = 10000
# Full observation width as RECORDED in the demo TFRecords. The live env writes
# its raw observation straight to disk (see robotaxi.write_trajectories_to_file),
# and the canonical/default expert corpus was recorded on the 32-wide donut
# course. This is the width we PARSE at, always - see feature_description.
FULL_OBSERVATION_SIZE = 32
# Active observation width the CURRENT job trains against. Defaults to the full
# width (classic DonutCourse). Set to 31 for the 'donut_no_hint' course, whose
# observation drops the leading dist_from_traj (angle-to-next-goal) element.
# Override via ROBOTAXI_OBSERVATION_SIZE or, per-job, set_observation_size().
# The dummy env spec and the trajectory-buffer width are derived from this.
observation_size = int(
    os.environ.get("ROBOTAXI_OBSERVATION_SIZE", str(FULL_OBSERVATION_SIZE)))
# How many LEADING recorded columns to DROP at read time to go from the
# recorded (full) width down to the active width. The only feature ever
# removed is the leading dist_from_traj (index 0), so this is 0 for donut (32)
# and 1 for donut_no_hint (31). Applied both to the full 32-wide bound lists
# (to build the active observation_spec) AND to each parsed observation row in
# convert_tfrecord_to_trajectory - so a SINGLE 32-wide demo corpus can feed
# both donut and donut_no_hint jobs with no re-collection.
_OBS_DROP_LEADING = FULL_OBSERVATION_SIZE - observation_size
action_size = 2

# Single source of truth mapping a course_type -> its observation width. Used
# by set_observation_size() (called per-job from robotaxi.do_job) so the demo
# read/write + BC pipeline width always matches the course the job runs
# against. Unknown/absent courses (e.g. 'simple', 'trainer default') fall back
# to the classic 32 via the .get(..., 32) at the call site.
COURSE_OBSERVATION_SIZES = {
    "donut": 32,
    "donut_no_hint": 31,
}

class robotaxi():
    def __init__(self):
        # Kept in sync with DonutCourse.action_spec (see
        # environments/courses/donut_course.py). This was stale at the
        # pre-2026-07-19 drive-only range [0.001, 2] until 2026-07-22, when
        # it was updated to match the course's [-0.01, 1] acceleration /
        # [-1, 1] steering range. On 2026-07-24 the acceleration floor was
        # raised again to a POSITIVE 0.05 (max kept at 1.0) to force
        # always-forward driving and kill the ~0.5 m/s crawl the [-0.01, 1]
        # range allowed - see DonutCourse.action_spec's comment for the full
        # rationale. Range is now [0.05, 1.0] accel / [-1, 1] steering.
        self._action_spec = tensor_spec.BoundedTensorSpec( #BoundedArraySpec(
            shape=(2, ), dtype=np.float32, 
            minimum=[0.05,-1], 
            maximum=[1, 1],
            name='action')
        # self._observation_spec = tensor_spec.BoundedTensorSpec( #BoundedArraySpec(
        #     shape=(32,), dtype=np.float32,
        #     minimum=0,
        #     maximum=1,
        #     name='observation')
        self._observation_spec = tensor_spec.BoundedTensorSpec(
            shape=(observation_size,), dtype=np.float32,
            minimum=np.array([
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
                ], dtype=np.float32)[_OBS_DROP_LEADING:],
            maximum=np.array([
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
                ], dtype=np.float32)[_OBS_DROP_LEADING:],
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

# NOTE: observation is parsed at the RECORDED (full) width, NOT the active
# width - the active-width slice (dropping _OBS_DROP_LEADING leading columns)
# happens in convert_tfrecord_to_trajectory. This is what lets a 32-wide demo
# corpus be read by a 31-wide (donut_no_hint) job.
feature_description = {
    'action': tf.io.FixedLenFeature((action_size,), tf.float32),
    'observation': tf.io.FixedLenFeature((FULL_OBSERVATION_SIZE,), tf.float32),
    # 'reward': tf.io.FixedLenFeature((1,), tf.float32),
    # 'discount': tf.io.FixedLenFeature((1,), tf.float32),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def set_observation_size(n):
    """Re-point the demo/BC pipeline at an observation width of ``n``.

    The demo read schema (``feature_description``), the trajectory-buffer
    width (used by ``convert_tfrecord_to_trajectory``), the drop count
    (``_OBS_DROP_LEADING``), and the module-level dummy ``env`` (whose
    observation spec is baked at construction and consumed by
    ``train_agent`` / ``train_agent_sampling``) are all derived from the
    module-global ``observation_size``. That global is fixed at import time
    from ``ROBOTAXI_OBSERVATION_SIZE``; this setter lets ``robotaxi.do_job``
    override it per job based on the job's selected course (see
    ``COURSE_OBSERVATION_SIZES``), so a single trainer process can run
    ``donut`` (32) and ``donut_no_hint`` (31) jobs back-to-back with the
    correct demo width each time.

    SAFE because jobs run strictly sequentially in the singleton trainer
    process - there is never more than one course's demo pipeline live at
    once. If concurrent multi-course jobs are ever introduced, this global
    mutation would need to become per-call plumbing instead.
    """
    global observation_size, _OBS_DROP_LEADING, env
    n = int(n)
    if n == observation_size:
        return
    observation_size = n
    _OBS_DROP_LEADING = FULL_OBSERVATION_SIZE - observation_size
    # feature_description is intentionally NOT rebuilt: demos are always parsed
    # at the recorded (full) width and sliced down in
    # convert_tfrecord_to_trajectory, so the parse schema is width-invariant.
    # Rebuild the dummy env so its (import-time-baked) observation spec matches
    # the new ACTIVE width - train_agent/train_agent_sampling read it via
    # spec_utils.get_tensor_specs(env).
    env = robotaxi()
    print(f"collect_training_data: observation_size set to {observation_size} "
          f"(recorded={FULL_OBSERVATION_SIZE}, drop_leading={_OBS_DROP_LEADING})",
          flush=True)

def get_files_from_directory(directory):
    files = os.listdir(directory)
    files.sort()
    files = [os.path.join(directory, file) for file in files]
    return files

def read_files_from_directory(directory='',sampling_fraction=1.0, shuffle=False, parsed_dataset=None):
    """Read all files from a directory and return a list of parsed rows."""
    # get list of file in directory
    files = get_files_from_directory(directory)
    rows = []
    total_rows = 0
    for f in files:
        num_records=0
        row = None
        num_records, parsed_rows = read_data_from_file(f, sampling_fraction, shuffle, parsed_dataset)
        total_rows += num_records
        rows = np.concatenate((rows, parsed_rows))
        print(f"reading {f}, length={num_records}")
    print(f"total rows {total_rows}")
    parsed_dataset_traj = convert_tfrecord_to_trajectory(rows, total_rows)
    return parsed_dataset_traj

def get_number_of_records(file_name):
    """Get the number of records in a TFRecord file."""
    return sum(1 for _ in tf.data.TFRecordDataset(file_name))

def get_parsed_dataset(file):
    """Get the raw dataset from a TFRecord file."""
    raw_dataset = tf.data.TFRecordDataset(file)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def read_data_from_file(file, sampling_fraction=1.0, shuffle=False, parsed_dataset=None):
    """Read the data from a TFRecord file."""
    if(parsed_dataset is None):
        parsed_dataset = get_parsed_dataset(file)
    num_records = get_number_of_records(file)
    batch_size = int(num_records * sampling_fraction)
    if shuffle:
        parsed_dataset = parsed_dataset.shuffle(buffer_size=num_records)
    parsed_rows = []
    for parsed_record in parsed_dataset.take(batch_size): 
        # print(repr(parsed_record))
        # if parsed_record["reward"] > 0:
        #     print(f"reached goal reward {parsed_record['reward']}")
        parsed_rows.append(parsed_record)
    parsed_rows_count = len(parsed_rows)
    print(f"num records {parsed_rows_count}")
    return parsed_rows_count, parsed_rows

def convert_tfrecord_to_trajectory(rows,batch_size=1000):
    """Read the data from a TFRecord file."""
    reward = tf.constant([1], dtype=tf.float32)
    discount = tf.constant([0.99], dtype=tf.float32)
    # Fill a FULL-width buffer with a plain direct assignment (fast: one
    # tensor->ndarray conversion per row, same as the original path), then drop
    # the leading column(s) ONCE below via a vectorized numpy slice. Doing the
    # drop per-row as `row["observation"][_OBS_DROP_LEADING:]` instead dispatches
    # a TF strided_slice op for every one of ~500k rows, which turned this
    # loop from ~seconds into many minutes (looked like a hung "job init").
    observation_full = np.empty(
        (batch_size, FULL_OBSERVATION_SIZE), dtype=np.float32)
    action = np.empty((batch_size, action_size), dtype=np.float32)
    i=0

    #sampled_rows = np.random.choice(rows, 10000)
    for row in rows: 
        #print(row)
        observation_full[i]=row["observation"]
        action[i]=row["action"]
        i=i+1

    # Drop the leading _OBS_DROP_LEADING recorded column(s) to reach the active
    # observation width (0 for donut, 1 for donut_no_hint which drops the
    # leading dist_from_traj hint). Single vectorized slice + a contiguous copy
    # so downstream tf.convert_to_tensor sees a dense C-contiguous array. This
    # is what lets the 32-wide demo corpus feed a 31-wide donut_no_hint job.
    observation = np.ascontiguousarray(
        observation_full[:, _OBS_DROP_LEADING:])

    print(action.shape)
    print(observation.shape)

    traj = trajectory.first(
        observation=observation,
        action=action,
        policy_info=(),
        reward=reward,
        discount=discount)

    return traj

def train_agent(root_dir, training_steps=10000):
    """Train the agent."""
    print(f"training agent with {training_steps} steps")
    parsed_dataset_traj = read_files_from_directory(directory=root_dir)
    # parsed_dataset, parsed_dataset_traj = read_data('trajectories.tfrecord', batch_size)

    # Create the behavioral cloning agent
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-5)

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    with strategy.scope():
        observation_spec, action_spec, time_step_spec = (
                spec_utils.get_tensor_specs(env))
        
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=(512, 512),
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

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

    save_policy(policy, training_steps)

def bc_pretrain_actor_net(
        actor_net, time_step_spec, action_spec, strategy,
        trajectory_dataset, training_steps=1000, batch_size=256,
        learning_rate=3e-5):
    """BC-supervise ``actor_net``'s weights on expert demonstrations.

    Pure SAC's actor loss is ``-E_s[Q(s, pi(s)) + alpha * H(pi)]`` where
    ``pi(s)`` is sampled from the policy itself, NOT from the replay
    buffer's action column. Expert *actions* loaded into the buffer
    therefore never enter the actor's gradient directly - they only
    influence the actor indirectly, via the critic absorbing them and
    shaping ``Q(s, a)`` over many gradient steps. That indirect signal
    is too slow: pure-SAC-from-random-init takes O(thousands) of
    iterations to approximate the expert distribution, by which point
    the buffer has diluted with on-policy data and the expert anchor
    has been Fifo-evicted.

    This function bridges that gap by running supervised BC gradient
    steps directly on ``actor_net`` before SAC takes over. Because
    ``tf_agent.actor_network is actor_net`` (same Python object),
    weight updates here are immediately visible to the SAC agent. SAC
    starts with a near-expert policy, evals well from step 0, and the
    historical "starts at 20-40 goals, drifts down as buffer dilutes"
    curve is recovered. (For the modern alternative, add a BC loss
    term to SAC's actor loss instead of running this as a separate
    phase - see TD3+BC, AWAC.)

    Caller is expected to have already imported the expert demos as a
    ``tf.data.Dataset`` of unbatched single-step ``Trajectory``
    objects, the same format that ``robotaxi.main()`` already builds
    via ``tf.data.Dataset.from_tensor_slices(aligned_trajectories)``.

    Args:
      actor_net: the same ``ActorDistributionNetwork`` instance that
        was passed as ``actor_network`` to the SAC agent.
      time_step_spec, action_spec: as returned by
        ``spec_utils.get_tensor_specs(env)`` in the caller.
      strategy: the ``MirroredStrategy`` (or whatever) returned by
        ``strategy_utils.get_strategy(...)``. The BC agent must be
        constructed under the same scope so its variables live on the
        same devices as the SAC agent's.
      trajectory_dataset: ``tf.data.Dataset`` of single-step
        ``Trajectory`` objects (no leading batch or time dim per
        element). Will be ``shuffle().batch(batch_size).repeat()``
        internally.
      training_steps: number of BC gradient updates. 1000 matches the
        historical pre-loop pretraining call before the move to this
        repo (commit dfef1f8) commented it out.
      batch_size: rows per BC gradient update. 256 matches SAC's
        batch_size so the BC and SAC training-time costs are
        comparable per-step.
      learning_rate: Adam LR for the BC optimizer. The BC optimizer is
        separate from SAC's actor optimizer - we only share the
        underlying network weights, not optimizer slots.
    """
    print(f"BC pretraining actor_net for {training_steps} steps "
          f"(batch_size={batch_size}, lr={learning_rate})...", flush=True)

    with strategy.scope():
        bc_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate)
        bc_agent = behavioral_cloning_agent.BehavioralCloningAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            cloning_network=actor_net,
            optimizer=bc_optimizer,
            num_outer_dims=1)
        bc_agent.initialize()

    # shuffle().batch().repeat(): each BC step sees a fresh batch
    # sampled (with replacement across epochs) from the expert demos.
    # buffer_size=10000 is large enough for good shuffling on the 50k
    # expert items we typically load while keeping memory bounded.
    bc_dataset = (trajectory_dataset
                  .shuffle(buffer_size=10000)
                  .batch(batch_size)
                  .prefetch(10)
                  .repeat())
    bc_iter = iter(bc_dataset)

    for bc_step in range(training_steps):
        batch = next(bc_iter)
        loss_info = bc_agent.train(batch)
        if bc_step % 100 == 0:
            print(f"  BC step {bc_step}/{training_steps}: "
                  f"loss={loss_info.loss.numpy():.4f}", flush=True)

    print(f"BC pretraining done (final loss={loss_info.loss.numpy():.4f}).",
          flush=True)


def train_agent_sampling(
        actor_net, root_dir, training_steps=10000, 
        sampling_fraction=1.0, parsed_dataset=None):
    """Train the agent."""
    print(f"training agent with {training_steps} steps")
    parsed_dataset_traj = read_files_from_directory(
            directory=root_dir,
            sampling_fraction=sampling_fraction,
            shuffle=True,
            parsed_dataset=parsed_dataset)

    # Create the behavioral cloning agent
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-5)

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    with strategy.scope():
        actor_net = actor_net

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

    # policy = agent.policy

    # save_policy(policy, training_steps)

#train_agent(root_dir, training_steps)

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
    try:
        path=get_save_dir_root_docker(policy)
    except Exception as e:
        path=get_save_dir_root_docker(policy)
        print(e)
    file_list = os.listdir(path)
    sorted_file_list=sorted(file_list,key=str,reverse=True)
    num_dirs = len(sorted_file_list)
    next_model_version=str(num_dirs)
    return path, next_model_version

def get_save_dir_name(policy):
    path, next_dir_name=get_next_model_version(policy)
    save_dir_root_docker = get_save_dir_root_docker(policy)
    return os.path.join(path,next_dir_name), os.path.join(save_dir_root_docker,next_dir_name)

client = MongoClient('mongo:27017', 
    username='root',
    password='example')
# `local` is a MongoDB-reserved, unreplicated system database (oplog etc.) -
# writes to it fail with "retryable writes is not supported for unreplicated
# ns: local.models" once retryable writes are enforced. robotaxi.py's own
# MongoClient setup uses `client.robotaxi` (see its `db = client.robotaxi`);
# this module's standalone connection was never updated to match, so any
# DEMO/BC job that reaches add_model() here (via save_policy() at the end of
# train_agent()) fails at that point even though the actual data
# collection/training leading up to it succeeded.
db = client.robotaxi

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
    tf_policy_saver.save(save_dir_name_docker)
    robot_type = "robotaxi" #os.getenv('ROBOT_TYPE')
    model_type=get_policy_type_name(policy)
    training_iterations=num_iterations
    add_model(
        save_dir_name,
        robot_type,
        model_type,
        training_iterations,
        path_docker=save_dir_name_docker)

# save_policy(policy, training_steps)

# def compute_avg_return(environment, policy, num_episodes=10):

#   total_return = 0.0
#   for _ in range(num_episodes):

#     time_step = environment.reset()
#     episode_return = 0.0

#     while not environment._episode_ended:
#       action_step = policy.action(time_step)
#       time_step = environment.step(action_step.action)
#       episode_return += time_step.reward
#     total_return += episode_return

#   avg_return = total_return / num_episodes
#   return avg_return

# for _ in range(10):
#     avg_return = compute_avg_return(env, policy, num_episodes=10)
#     print(f"avg_return: {avg_return}")