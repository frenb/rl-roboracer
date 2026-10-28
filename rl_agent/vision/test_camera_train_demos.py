"""Phase 5: a camera TRAIN streams its DEMO corpus into BC and Reverb.

No gym and no ROS. CUDA is hidden by the launch command so this does
not take GPU memory from a live TRAIN job.

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= \
        -v "<repo>/rl_agent:/python_ws/src" -w /python_ws/src \
        sim_controller:latest vision/test_camera_train_demos.py

Exit 0 = all assertions passed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import tensorflow as tf

from vision.camera_record import serialize_camera_example
import collect_training_data as ctd


_passed = 0
_failed = 0


def _expect(label, predicate, detail=""):
    global _passed, _failed
    try:
        ok = bool(predicate())
    except Exception:
        ok = False
        detail = (detail + "\n" + traceback.format_exc()).strip()
    if ok:
        _passed += 1
        print("PASS  " + label, flush=True)
    else:
        _failed += 1
        extra = ("  " + detail) if detail else ""
        print("FAIL  " + label + extra, flush=True)


def _raises(label, fn, needle):
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        _expect(label, lambda: needle in text, text)
        return
    except Exception:
        _expect(label, lambda: False, traceback.format_exc())
        return
    _expect(label, lambda: False, "no exception")


def _ramp():
    ys, xs = np.meshgrid(np.arange(84), np.arange(84), indexing="ij")
    img = np.empty((84, 84, 3), dtype=np.float32)
    img[:, :, 0] = np.minimum(xs, 255) / 255.0
    img[:, :, 1] = np.minimum(ys, 255) / 255.0
    img[:, :, 2] = 128.0 / 255.0
    return img


def _write_camera(path, rows):
    with tf.io.TFRecordWriter(path) as writer:
        for obs, action, reward, discount, step in rows:
            payload, _ = serialize_camera_example(
                obs, action, reward, discount, step)
            writer.write(payload)


def _write_flat(path, obs_width):
    obs = np.linspace(-1.0, 1.0, obs_width, dtype=np.float32)
    action = np.asarray([0.2, 0.0], dtype=np.float32)
    feature = {
        "action": tf.train.Feature(
            float_list=tf.train.FloatList(value=action)),
        "observation": tf.train.Feature(
            float_list=tf.train.FloatList(value=obs)),
        "reward": tf.train.Feature(
            float_list=tf.train.FloatList(value=[1.0])),
        "discount": tf.train.Feature(
            float_list=tf.train.FloatList(value=[0.99])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(example.SerializeToString())


def _settings_checks():
    scratch = ctd.camera_train_settings(
        "donut_camera", [], 5000, 50000, 0.25, 1.0)
    _expect("no demo source starts from scratch",
            lambda: scratch["uses_demos"] is False
            and scratch["bc_pretrain_steps"] == 0
            and scratch["demo_prefill_count"] == 0
            and scratch["demo_sample_ratio"] == 0.0
            and scratch["awac_lambda"] == 0.0)
    kept = ctd.camera_train_settings(
        "donut_camera", ["/tfrecords/job_x"], 5000, 50000, 0.25, 1.0)
    _expect("a demo source keeps BC, ratio, and awac",
            lambda: kept["uses_demos"] is True
            and kept["bc_pretrain_steps"] == 5000
            and kept["demo_sample_ratio"] == 0.25
            and kept["awac_lambda"] == 1.0)
    _expect("prefill above the cap becomes 10000",
            lambda: kept["demo_prefill_count"] == 10000)
    small = ctd.camera_train_settings(
        "donut_camera_no_rays", ["/tmp/d"], 100, 2000, 0.1, 0.5)
    _expect("a smaller prefill cap is kept",
            lambda: small["demo_prefill_count"] == 2000)
    _expect("a zero prefill stays zero",
            lambda: ctd.camera_prefill_count(0) == 0)


def _bc_and_reverb(cam_dir, narrow_dir, flat_dir):
    _raises(
        "31-wide file is refused by the 2-wide course",
        lambda: ctd.validate_camera_demo_dirs(
            [cam_dir], "donut_camera_no_rays"),
        "requires 2")
    _raises(
        "a flat file is refused",
        lambda: ctd.validate_camera_demo_dirs([flat_dir], "donut_camera"),
        "no image feature")

    traj_ds, tagged = ctd.camera_train_datasets(
        [cam_dir], "donut_camera", min_accel=None)
    first_traj, first_sid = next(iter(tagged))
    obs = first_traj.observation
    _expect("streamed observation is the dict",
            lambda: set(obs.keys()) == {"vector", "image"})
    _expect("image is one frame, not a stacked corpus",
            lambda: tuple(obs["image"].shape) == (84, 84, 3))
    _expect("vector width is 31",
            lambda: tuple(obs["vector"].shape) == (31,))
    _expect("stored reward is kept",
            lambda: abs(float(first_traj.reward.numpy()) - 0.25) < 1e-6)
    _expect("source index is 0",
            lambda: int(first_sid.numpy()) == 0)
    _expect("dataset element is one step",
            lambda: tuple(first_traj.action.shape) == (2,))

    # Both directories contribute. sample_from_datasets mixes them, so
    # a short prefix is not all source 0.
    mixed, mixed_tagged = ctd.camera_train_datasets(
        [cam_dir, narrow_dir], "donut_camera", min_accel=None)
    del mixed
    seen = set()
    for _traj, sid in mixed_tagged.take(8):
        seen.add(int(sid.numpy()))
    _expect("two sources both appear in the stream",
            lambda: seen == {0, 1})

    from tf_agents.networks import actor_distribution_network
    from tf_agents.agents.sac import tanh_normal_projection_network
    from tf_agents.specs import tensor_spec
    from tf_agents.trajectories import time_step as ts
    from tf_agents.train.utils import strategy_utils
    from vision.camera_networks import (
        camera_preprocessing_combiner, camera_preprocessing_layers)

    obs_spec = {
        "vector": tensor_spec.TensorSpec((31,), tf.float32),
        "image": tensor_spec.TensorSpec((84, 84, 3), tf.float32),
    }
    action_spec = tensor_spec.BoundedTensorSpec(
        (2,), tf.float32, minimum=[0.05, -1.0], maximum=[1.0, 1.0])
    time_step_spec = ts.time_step_spec(obs_spec)
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            obs_spec,
            action_spec,
            preprocessing_layers=camera_preprocessing_layers(),
            preprocessing_combiner=camera_preprocessing_combiner(),
            fc_layer_params=(32, 32),
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))
    ctd.bc_pretrain_actor_net(
        actor_net=actor_net,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        strategy=strategy,
        trajectory_dataset=traj_ds,
        training_steps=1,
        batch_size=2,
        learning_rate=3e-5)
    _expect("BC pretrain on the Conv actor completed", lambda: True)

    from replay import make_local_replay
    element = next(iter(traj_ds))
    spec = tf.nest.map_structure(
        lambda t: tf.TensorSpec(t.shape, t.dtype), element)
    (_server, _replay, _dataset, _collect, _expert,
     demo_replay, demo_observer) = make_local_replay(
         spec, capacity=8, sample_batch_size=2, sequence_length=2,
         num_envs=1, demo_capacity=8, demo_sample_ratio=0.5)
    added = 0
    for traj, _sid in tagged.take(4):
        demo_observer(traj)
        added += 1
    demo_observer.flush()
    demo_size = demo_replay.py_client.server_info()["demo_table"].current_size
    _expect("Reverb demo table accepted streamed frames",
            lambda: added == 4 and demo_size > 0, "size=%s" % demo_size)

    from awac_sac_agent import AwacSacAgent
    from vision.camera_networks import DictObsCriticNetwork
    with strategy.scope():
        critic_net = DictObsCriticNetwork(
            (obs_spec, action_spec), joint_fc_layer_params=(32, 32))
        agent = AwacSacAgent(
            time_step_spec,
            action_spec,
            awac_lambda=0.5,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(3e-4),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(3e-4),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(3e-4))
        agent.initialize()
    agent.set_demo_iter(iter(demo_replay.as_dataset(
        sample_batch_size=2, num_steps=2)))
    batch_obs = tf.nest.map_structure(
        lambda t: tf.stack([t, t]), first_traj.observation)
    batch_ts = ts.restart(batch_obs, batch_size=2)
    awac_loss = agent.actor_loss(batch_ts)
    _expect("AWAC actor loss runs on camera dict observations",
            lambda: bool(np.isfinite(float(awac_loss.numpy()))),
            "loss=%s" % awac_loss)


def main():
    _settings_checks()
    image = _ramp()
    vector = np.linspace(-1.0, 1.0, 31, dtype=np.float32)
    narrow = np.asarray([1.5, 0.1], dtype=np.float32)
    rows = []
    for i in range(4):
        obs = {
            "vector": vector + i,
            "image": np.clip(image + (i * 0.01), 0.0, 1.0),
        }
        action = np.asarray([0.2, 0.1 * i], dtype=np.float32)
        step = 2 if i == 2 else 1
        rows.append((obs, action, 0.25, 0.9, step))
    narrow_rows = []
    for i in range(4):
        obs = {"vector": narrow, "image": image}
        # Pad the vector to 31 so this second directory is a valid
        # donut_camera file with a different source index. The 2-wide
        # refusal is covered by validate_camera_demo_dirs above.
        obs = {
            "vector": vector + 10 + i,
            "image": image,
        }
        action = np.asarray([0.3, 0.0], dtype=np.float32)
        narrow_rows.append((obs, action, 0.5, 0.9, 1))
    with tempfile.TemporaryDirectory() as root:
        cam_dir = os.path.join(root, "camera")
        narrow_dir = os.path.join(root, "other")
        flat_dir = os.path.join(root, "flat")
        os.makedirs(cam_dir)
        os.makedirs(narrow_dir)
        os.makedirs(flat_dir)
        _write_camera(os.path.join(cam_dir, "a.tfrecord"), rows)
        _write_camera(os.path.join(narrow_dir, "b.tfrecord"), narrow_rows)
        _write_flat(os.path.join(flat_dir, "flat.tfrecord"), 32)
        two_wide = os.path.join(root, "two")
        os.makedirs(two_wide)
        _write_camera(
            os.path.join(two_wide, "n.tfrecord"),
            [({"vector": narrow, "image": image},
              np.asarray([0.2, 0.0], dtype=np.float32), 0.25, 0.9, 1)])
        _raises(
            "2-wide file is refused by the 31-wide course",
            lambda: ctd.validate_camera_demo_dirs(
                [two_wide], "donut_camera"),
            "requires 31")
        _bc_and_reverb(cam_dir, narrow_dir, flat_dir)
    print("%d passed, %d failed" % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
