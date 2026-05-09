"""Reverb replay-buffer setup for robotaxi training.

Today (Level 1) the replay buffer is local to the training process: a
single ``reverb.Server`` lives inside the sim-controller container, and
both the actor (collect) and the learner (train) talk to it through
in-process gRPC. ``make_local_replay`` returns the four objects the
training driver needs:

  - server:    the running ``reverb.Server`` (kept alive by the caller)
  - replay:    the ``ReverbReplayBuffer`` wrapper for the table
  - dataset:   a prefetched ``tf.data.Dataset`` for sampling batches
  - observer:  a ``ReverbAddTrajectoryObserver`` to attach to actors

When we move to Level 2 (Reverb-as-service), this module is the natural
seam: a ``make_remote_replay(server_address, ...)`` variant will take the
same args and return the same shape, but with ``local_server=`` swapped
out for connecting to an external ``reverb-server`` container.
"""

import tensorflow as tf
import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

TABLE_NAME = 'uniform_table'


class _FanoutTrajectoryObserver:
    """Dispatches a batched ``Trajectory`` to N unbatched reverb observers.

    ``actor.Actor`` running on a ``ParallelPyEnvironment`` produces a
    ``Trajectory`` whose leaves all carry a leading batch dim of N (one
    row per parallel env). ``ReverbAddTrajectoryObserver`` expects
    unbatched trajectories - it will reject batched input with
    ``ValueError: Tensor of incompatible shape``.

    Each parallel env runs its own episode sequence (different start
    times, different episode lengths, different ``step_type`` flags), so
    we can't just collapse the batch dim into the table. Instead, we
    keep N independent reverb writers - each its own
    ``ReverbAddTrajectoryObserver`` with its own sequence/stride
    accumulator and trajectory writer - and route row ``i`` of every
    incoming batched trajectory to writer ``i``.

    All N writers append into the same reverb table, so the agent sees
    a flat pool of single-actor sequences regardless of how many parallel
    envs produced them.
    """

    def __init__(self, py_client, table_name, num_envs,
                 sequence_length, stride_length):
        self._observers = [
            reverb_utils.ReverbAddTrajectoryObserver(
                py_client,
                table_name,
                sequence_length=sequence_length,
                stride_length=stride_length)
            for _ in range(num_envs)
        ]

    def __call__(self, batched_trajectory):
        for i, observer in enumerate(self._observers):
            unbatched = tf.nest.map_structure(
                lambda x, i=i: x[i], batched_trajectory)
            observer(unbatched)

    def reset(self, write_cached_steps=True):
        for observer in self._observers:
            observer.reset(write_cached_steps=write_cached_steps)

    def flush(self):
        for observer in self._observers:
            observer.flush()

    def close(self):
        for observer in self._observers:
            observer.close()


def make_local_replay(collect_data_spec, capacity, sample_batch_size,
                      sequence_length=2, stride_length=1, num_envs=1):
    """Build an in-process Reverb table + server + buffer + dataset + observer(s).

    Args:
      collect_data_spec: from ``tf_agent.collect_data_spec``. This is the
        per-actor (unbatched) spec; the same spec is used regardless of
        ``num_envs`` because per-env writers each see unbatched rows.
      capacity: max number of items in the table (``replay_buffer_capacity``).
      sample_batch_size: training batch size for ``as_dataset``.
      sequence_length: number of consecutive timesteps per item. SAC uses 2.
      stride_length: how far to advance between trajectory writes.
      num_envs: number of parallel collection envs feeding this buffer.
        Default 1 (single-actor). When >1, the collect observer returned
        is a fan-out that splits batched trajectories across N independent
        per-env writers (see ``_FanoutTrajectoryObserver`` above).

    Returns:
      ``(server, replay, dataset, collect_observer, expert_observer)``.

      - ``collect_observer`` is what gets passed to ``actor.Actor`` in
        the ``observers=[...]`` list. It accepts whatever shape the env
        produces (unbatched for single-env, batched for parallel-env).
      - ``expert_observer`` always accepts a plain unbatched
        ``Trajectory`` and is what offline-data ingestion loops should
        use (e.g. loading recorded expert demonstrations directly into
        the table without going through any env).

      In single-env mode the two observers are the same instance. In
      multi-env mode they are distinct writers into the same shared
      table - so closing both via ``server.stop()`` is sufficient, but
      callers that explicitly close should close both.

      Caller is responsible for keeping ``server`` alive until training
      ends.
    """
    table = reverb.Table(
        TABLE_NAME,
        max_size=capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    server = reverb.Server([table])

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        collect_data_spec,
        sequence_length=sequence_length,
        table_name=TABLE_NAME,
        local_server=server)

    dataset = replay.as_dataset(
        sample_batch_size=sample_batch_size,
        num_steps=sequence_length).prefetch(50)

    expert_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay.py_client,
        TABLE_NAME,
        sequence_length=sequence_length,
        stride_length=stride_length)

    if num_envs <= 1:
        # Single-env mode: actor produces unbatched trajectories, so the
        # plain reverb observer can be reused for both collection and
        # expert-demo ingestion.
        collect_observer = expert_observer
    else:
        collect_observer = _FanoutTrajectoryObserver(
            replay.py_client,
            TABLE_NAME,
            num_envs=num_envs,
            sequence_length=sequence_length,
            stride_length=stride_length)

    return server, replay, dataset, collect_observer, expert_observer
