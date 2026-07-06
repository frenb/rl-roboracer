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

# Online table: receives writes from both the actor.Actor collect step
# AND, in single-table mode (demo_capacity == 0), the offline expert-
# demo prefill at job start. FIFO-evicts when full.
ONLINE_TABLE_NAME = 'uniform_table'
# Demo table: only used in two-table mode (demo_capacity > 0). Receives
# writes ONLY from the offline expert-demo prefill at job start. After
# prefill the table is never written to again, so its FIFO eviction
# never fires - the demos stay forever. Sampled alongside the online
# table during training via tf.data.Dataset.sample_from_datasets with
# the user-configured demo_sample_ratio.
DEMO_TABLE_NAME = 'demo_table'

# Back-compat alias. Older imports of TABLE_NAME from this module
# (none in-tree, but defensive) continue to resolve to the online
# table's name.
TABLE_NAME = ONLINE_TABLE_NAME


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
                      sequence_length=2, stride_length=1, num_envs=1,
                      demo_capacity=0, demo_sample_ratio=0.0,
                      checkpointing_dir=None):
    """Build an in-process Reverb table + server + buffer + dataset + observer(s).

    Two operating modes, selected by ``demo_capacity``:

    **Single-table mode** (``demo_capacity == 0``, default for back-compat):
      One Reverb table receives both the offline expert-demo prefill
      and the runtime RL collect writes. Demos eventually FIFO-evict
      as collection accumulates. Equivalent to the trainer's behavior
      before the demo-protected buffer feature.

    **Two-table mode** (``demo_capacity > 0``):
      Two Reverb tables live in the same server:
        - online table: ``capacity`` items, FIFO eviction. Only the
          actor.Actor collect step writes to it.
        - demo table:   ``demo_capacity`` items, FIFO eviction. Only
          the offline expert-demo prefill writes to it. Since it gets
          no writes after init, its FIFO never fires - the demos stay
          forever.
      The training ``dataset`` returned is a tf.data
      ``sample_from_datasets`` mix of (demo_ds, online_ds) weighted by
      ``demo_sample_ratio`` so each training batch draws
      ``demo_sample_ratio``% of its rows from demos.

    Args:
      collect_data_spec: from ``tf_agent.collect_data_spec``. This is the
        per-actor (unbatched) spec; the same spec is used regardless of
        ``num_envs`` because per-env writers each see unbatched rows.
      capacity: max items in the online table.
      sample_batch_size: training batch size for ``as_dataset``.
      sequence_length: number of consecutive timesteps per item. SAC uses 2.
      stride_length: how far to advance between trajectory writes.
      num_envs: number of parallel collection envs feeding the online
        table. Default 1 (single-actor). When >1, the collect observer
        returned is a fan-out that splits batched trajectories across
        N independent per-env writers (see ``_FanoutTrajectoryObserver``
        above).
      demo_capacity: 0 to keep single-table mode. >0 to add a protected
        demo table of this capacity. The demo table never FIFO-evicts
        in practice because no writes hit it after init.
      demo_sample_ratio: in two-table mode, fraction of each batch
        sampled from the demo table (the rest comes from the online
        table). Ignored in single-table mode. Clamped to [0, 1].

    Returns:
      ``(server, replay, dataset, collect_observer, expert_observer,
         demo_replay, demo_observer)``.

      - ``replay`` is the online ReverbReplayBuffer (always present).
      - ``dataset`` is the training-side prefetched dataset; in two-
        table mode it's the mixed dataset, in single-table mode it's
        the plain online dataset.
      - ``collect_observer`` is what gets passed to ``actor.Actor`` in
        the ``observers=[...]`` list - always writes to the online
        table.
      - ``expert_observer`` is for ingesting offline expert demos into
        the ONLINE table (used in single-table mode).
      - ``demo_replay`` / ``demo_observer`` are None in single-table
        mode. In two-table mode they wrap the demo table. The trainer
        feeds expert demos through ``demo_observer`` when it exists,
        falling back to ``expert_observer`` (single-table mode).

      checkpointing_dir: optional path for the Reverb server's on-disk
        checkpoint directory. When set, ``reverb.checkpointers.DefaultCheckpointer``
        is used so that ``server.localhost_client().checkpoint()`` always
        writes to this directory (enabling deterministic save/restore across
        pause/resume cycles). When None (default), Reverb uses a random
        temp dir and checkpointing is not used.

      Caller is responsible for keeping ``server`` alive until training
      ends.
    """
    import os  # noqa: PLC0415 (local import is fine; module is stdlib)
    online_table = reverb.Table(
        ONLINE_TABLE_NAME,
        max_size=capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    tables = [online_table]

    # Two-table mode: add the demo-protected table to the same server.
    # The Reverb server can host multiple tables; the trainer
    # references them by name from independent ReverbReplayBuffer
    # wrappers below.
    if demo_capacity > 0:
        demo_table = reverb.Table(
            DEMO_TABLE_NAME,
            max_size=demo_capacity,
            sampler=reverb.selectors.Uniform(),
            # FIFO eviction is set the same way as the online table so
            # demo_capacity acts as a hard upper bound even in pathological
            # cases (e.g., the trainer mistakenly writes to the demo table
            # after prefill). In normal use no writes hit the demo table
            # after init, so FIFO never fires.
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1))
        tables.append(demo_table)

    # Build the server with a fixed checkpointing directory when requested
    # so pause-time checkpoint() calls always write to the same path.
    if checkpointing_dir:
        os.makedirs(checkpointing_dir, exist_ok=True)
        _checkpointer = reverb.checkpointers.DefaultCheckpointer(
            path=checkpointing_dir)
        server = reverb.Server(tables, checkpointer=_checkpointer)
    else:
        server = reverb.Server(tables)

    online_replay = reverb_replay_buffer.ReverbReplayBuffer(
        collect_data_spec,
        sequence_length=sequence_length,
        table_name=ONLINE_TABLE_NAME,
        local_server=server)

    online_dataset = online_replay.as_dataset(
        sample_batch_size=sample_batch_size,
        num_steps=sequence_length).prefetch(50)

    # Build the demo half + the mixed dataset only when two-table mode
    # is active. Avoids paying any of the tf.data composition cost in
    # the back-compat path.
    demo_replay = None
    demo_observer = None
    if demo_capacity > 0:
        demo_replay = reverb_replay_buffer.ReverbReplayBuffer(
            collect_data_spec,
            sequence_length=sequence_length,
            table_name=DEMO_TABLE_NAME,
            local_server=server)
        demo_dataset = demo_replay.as_dataset(
            sample_batch_size=sample_batch_size,
            num_steps=sequence_length).prefetch(50)

        # Clamp ratio defensively. 0.0 -> only the online stream
        # contributes (demo table kept but unused, useful for ablations);
        # 1.0 -> only demos. Mixing weight=0 datasets can confuse
        # sample_from_datasets in some TF versions, so we collapse to a
        # single-stream dataset at the edges.
        r = max(0.0, min(1.0, float(demo_sample_ratio)))
        if r <= 0.0:
            dataset = online_dataset
        elif r >= 1.0:
            dataset = demo_dataset
        else:
            dataset = tf.data.Dataset.sample_from_datasets(
                [demo_dataset, online_dataset],
                weights=[r, 1.0 - r],
                seed=None,
                stop_on_empty_dataset=False)

        demo_observer = reverb_utils.ReverbAddTrajectoryObserver(
            demo_replay.py_client,
            DEMO_TABLE_NAME,
            sequence_length=sequence_length,
            stride_length=stride_length)
    else:
        dataset = online_dataset

    expert_observer = reverb_utils.ReverbAddTrajectoryObserver(
        online_replay.py_client,
        ONLINE_TABLE_NAME,
        sequence_length=sequence_length,
        stride_length=stride_length)

    if num_envs <= 1:
        # Single-env mode: actor produces unbatched trajectories, so the
        # plain reverb observer can be reused for both collection and
        # (single-table-mode) expert-demo ingestion.
        collect_observer = expert_observer
    else:
        collect_observer = _FanoutTrajectoryObserver(
            online_replay.py_client,
            ONLINE_TABLE_NAME,
            num_envs=num_envs,
            sequence_length=sequence_length,
            stride_length=stride_length)

    return (server, online_replay, dataset, collect_observer,
            expert_observer, demo_replay, demo_observer)
