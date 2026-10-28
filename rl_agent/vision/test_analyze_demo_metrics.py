"""Phase 3: demo metrics read speed from vector[0] when an image is stored.

    docker compose -f docker-compose.yml -f compose/scale.yml exec -e CUDA_VISIBLE_DEVICES= sim-controller \
        sh -c "cd /python_ws/src && python vision/test_analyze_demo_metrics.py"
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import tensorflow as tf

import analyze_demo_metrics as metrics
from vision.camera_record import serialize_camera_example


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


def _args():
    return argparse.Namespace(
        goal_budget=100, goal_cap=30, speed_moving=1.0, speed_stop=0.3)


def _image():
    return np.zeros((84, 84, 3), dtype=np.float32)


def _write_camera(path, vectors):
    image = _image()
    with tf.io.TFRecordWriter(path) as writer:
        for vector in vectors:
            payload, _ = serialize_camera_example(
                {"vector": vector, "image": image},
                [0.2, 0.0], 0.0, 0.9, 1)
            writer.write(payload)


def _write_flat(path):
    obs = np.zeros(32, dtype=np.float32)
    obs[0] = 99.0
    obs[1] = 3.0
    feature = {
        "action": tf.train.Feature(
            float_list=tf.train.FloatList(value=[0.2, 0.0])),
        "observation": tf.train.Feature(
            float_list=tf.train.FloatList(value=obs)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(example.SerializeToString())


def main():
    wide = np.zeros(31, dtype=np.float32)
    wide[0] = 3.0
    wide[1] = 99.0
    wide[16] = 5.0
    narrow = np.asarray([3.0, 0.25], dtype=np.float32)

    with tempfile.TemporaryDirectory() as root:
        cam = os.path.join(root, "camera")
        abl = os.path.join(root, "ablation")
        flat = os.path.join(root, "flat")
        os.makedirs(cam)
        os.makedirs(abl)
        os.makedirs(flat)
        cam_path = os.path.join(cam, "000_g0_0000trajectories.tfrecord")
        abl_path = os.path.join(abl, "000_g0_0000trajectories.tfrecord")
        flat_path = os.path.join(flat, "000_g0_0000trajectories.tfrecord")
        _write_camera(cam_path, [wide, wide])
        _write_camera(abl_path, [narrow])
        _write_flat(flat_path)

        cam_layout = metrics.record_layout(cam_path)
        _expect("31-wide camera course", lambda: cam_layout["course"] == "donut_camera")
        _expect("31-wide speed index is 0", lambda: cam_layout["speed_idx"] == 0)
        _expect("31-wide forward ray is 16", lambda: cam_layout["fwd_idx"] == 16)
        cam_obs = metrics._read_vectors(cam, cam_layout)
        cam_stats = metrics.analyze("camera", cam_obs, _args(),
                                    cam_layout["speed_idx"], cam_layout["fwd_idx"])
        _expect("camera speed is vector[0], not the 99 decoy",
                lambda: abs(cam_stats["speed_mean"] - 3.0) < 1e-6)
        _expect("31-wide crash estimate runs",
                lambda: cam_stats["crash_ended"] is not None)

        abl_layout = metrics.record_layout(abl_path)
        _expect("2-wide course", lambda: abl_layout["course"] == "donut_camera_no_rays")
        _expect("2-wide has no forward ray", lambda: abl_layout["fwd_idx"] is None)
        abl_obs = metrics._read_vectors(abl, abl_layout)
        abl_stats = metrics.analyze("ablation", abl_obs, _args(),
                                    abl_layout["speed_idx"], abl_layout["fwd_idx"])
        _expect("2-wide speed is vector[0]",
                lambda: abs(abl_stats["speed_mean"] - 3.0) < 1e-6)
        _expect("2-wide crash estimate skipped",
                lambda: abl_stats["crash_ended"] is None)

        flat_layout = metrics.record_layout(flat_path)
        _expect("flat file is not a camera course", lambda: flat_layout["kind"] == "flat")
        _expect("flat speed index stays 1", lambda: flat_layout["speed_idx"] == 1)
        flat_obs = metrics._read_vectors(flat, flat_layout)
        flat_stats = metrics.analyze("flat", flat_obs, _args(),
                                     flat_layout["speed_idx"], flat_layout["fwd_idx"])
        _expect("flat speed is still observation[1]",
                lambda: abs(flat_stats["speed_mean"] - 3.0) < 1e-6)

    print("%d passed, %d failed" % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
