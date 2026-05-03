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

import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

TABLE_NAME = 'uniform_table'


def make_local_replay(collect_data_spec, capacity, sample_batch_size,
                      sequence_length=2, stride_length=1):
    """Build an in-process Reverb table + server + buffer + dataset + observer.

    Args:
      collect_data_spec: from ``tf_agent.collect_data_spec``.
      capacity: max number of items in the table (``replay_buffer_capacity``).
      sample_batch_size: training batch size for ``as_dataset``.
      sequence_length: number of consecutive timesteps per item. SAC uses 2.
      stride_length: how far to advance between trajectory writes.

    Returns:
      ``(server, replay, dataset, observer)``. Caller is responsible for
      keeping ``server`` alive until training ends and calling
      ``server.stop()`` and ``observer.close()`` on shutdown.
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

    observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay.py_client,
        TABLE_NAME,
        sequence_length=sequence_length,
        stride_length=stride_length)

    return server, replay, dataset, observer
