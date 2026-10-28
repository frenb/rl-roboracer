"""Phase 2: camera TFRecords stream back as {vector, image} plus the stored reward.

No gym and no ROS. Runs in sim-controller so tensorflow and the mongo
client import succeed. CUDA is hidden by the launch command so this
does not take GPU memory from a live TRAIN job.

    docker compose -f docker-compose.yml -f compose/scale.yml exec -e CUDA_VISIBLE_DEVICES= sim-controller \
        sh -c "cd /python_ws/src && python vision/test_camera_demo_dataset.py"

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


def main():
    image = _ramp()
    vector = np.linspace(-1.0, 1.0, 31, dtype=np.float32)
    wide = {
        "vector": vector,
        "image": image,
    }
    narrow = {
        "vector": vector[:2].copy(),
        "image": image,
    }

    with tempfile.TemporaryDirectory() as root:
        cam_dir = os.path.join(root, "camera")
        flat_dir = os.path.join(root, "flat")
        narrow_dir = os.path.join(root, "narrow")
        os.makedirs(cam_dir)
        os.makedirs(flat_dir)
        os.makedirs(narrow_dir)
        _write_camera(os.path.join(cam_dir, "000.tfrecord"), [
            (wide, [0.20, 0.10], 0.25, 0.9, 1),
            (wide, [0.01, -0.5], -1.0, 0.0, 2),
        ])
        _write_camera(os.path.join(cam_dir, "001.tfrecord"), [
            (wide, [0.40, 0.0], 1.5, 0.5, 1),
        ])
        _write_flat(os.path.join(flat_dir, "legacy.tfrecord"), 32)
        _write_camera(os.path.join(narrow_dir, "a.tfrecord"), [
            (narrow, [0.30, 0.0], 0.5, 0.8, 0),
        ])

        rows = list(ctd.camera_demo_dataset(cam_dir, "donut_camera", min_accel=None))
        _expect("streams 3 rows from two files", lambda: len(rows) == 3)
        first = rows[0]
        got_vec = first["observation"]["vector"].numpy()
        got_img = first["observation"]["image"].numpy()
        _expect("vector matches", lambda: np.allclose(got_vec, vector))
        _expect(
            "image within 1/255",
            lambda: got_img.shape == (84, 84, 3)
            and np.max(np.abs(got_img - image)) <= (1.0 / 255.0) + 1e-6)
        _expect(
            "stored reward and discount",
            lambda: abs(float(first["reward"].numpy()) - 0.25) < 1e-6
            and abs(float(first["discount"].numpy()) - 0.9) < 1e-6)
        _expect(
            "stored next_step_type on the second row",
            lambda: int(rows[1]["next_step_type"].numpy()) == 2
            and abs(float(rows[1]["reward"].numpy()) - (-1.0)) < 1e-6)

        kept = list(ctd.camera_demo_dataset(cam_dir, "donut_camera"))
        _expect(
            "accel floor drops the 0.01 row",
            lambda: len(kept) == 2
            and all(float(r["action"].numpy()[0]) >= 0.05 for r in kept))

        _raises(
            "31-wide file refused by the 2-wide course",
            lambda: ctd.camera_demo_dataset(cam_dir, "donut_camera_no_rays", min_accel=None),
            "requires 2")
        _raises(
            "flat file refused by the camera reader",
            lambda: ctd.camera_demo_dataset(flat_dir, "donut_camera", min_accel=None),
            "no image feature")
        _raises(
            "camera file refused by the flat reader",
            lambda: ctd.read_data_from_file(os.path.join(cam_dir, "000.tfrecord")),
            "camera TFRecord")

        count, parsed = ctd.read_data_from_file(os.path.join(flat_dir, "legacy.tfrecord"))
        _expect(
            "flat reader still returns the 32-vector",
            lambda: count == 1 and tuple(parsed[0]["observation"].shape) == (32,))

        narrow_rows = list(ctd.camera_demo_dataset(
            narrow_dir, "donut_camera_no_rays", min_accel=None))
        _expect(
            "2-wide course reads a 2-vector",
            lambda: len(narrow_rows) == 1
            and tuple(narrow_rows[0]["observation"]["vector"].shape) == (2,))

    print("%d passed, %d failed" % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
