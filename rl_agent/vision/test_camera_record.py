"""Phase 1: a camera observation becomes vector floats plus 21168 image bytes.

No gym and no ROS. Tensorflow is imported by camera_record.

    docker compose -f docker-compose.yml -f compose/scale.yml exec -e CUDA_VISIBLE_DEVICES= sim-controller \
        sh -c "cd /python_ws/src && python vision/test_camera_record.py"

Exit 0 = all assertions passed.
"""
from __future__ import annotations

import os
import sys
import traceback

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import tensorflow as tf

from vision.camera_record import (
    IMAGE_BYTES,
    camera_observation_features,
    image_to_uint8_bytes,
    is_camera_observation,
)


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


def _ramp(h, w):
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :, 0] = np.minimum(xs, 255) / 255.0
    img[:, :, 1] = np.minimum(ys, 255) / 255.0
    img[:, :, 2] = 128.0 / 255.0
    return img


def _example_from_features(features):
    example = tf.train.Example(features=tf.train.Features(feature={
        "observation": features["observation"],
        "image": features["image"],
    }))
    return tf.train.Example.FromString(example.SerializeToString())


def main():
    image = _ramp(84, 84)
    vector = np.linspace(-1.0, 1.0, 31, dtype=np.float32)
    obs = {"vector": vector, "image": image}

    _expect("dict is a camera observation", lambda: is_camera_observation(obs))
    _expect(
        "flat vector is not a camera observation",
        lambda: not is_camera_observation(vector))

    features = camera_observation_features(obs)
    _expect("vector length 31", lambda: features["vector_len"] == 31)
    _expect("image is 21168 bytes", lambda: features["image_len"] == IMAGE_BYTES)

    parsed = _example_from_features(features)
    got_vec = np.asarray(parsed.features.feature["observation"].float_list.value, dtype=np.float32)
    got_img = parsed.features.feature["image"].bytes_list.value[0]
    _expect("parsed vector matches", lambda: np.allclose(got_vec, vector))
    _expect("parsed image length", lambda: len(got_img) == IMAGE_BYTES)

    recovered = np.frombuffer(got_img, dtype=np.uint8).reshape(84, 84, 3).astype(np.float32) / 255.0
    _expect(
        "round trip within 1/255",
        lambda: np.max(np.abs(recovered - image)) <= (1.0 / 255.0) + 1e-6)

    narrow = camera_observation_features({"vector": vector[:2], "image": image})
    _expect("vector length 2", lambda: narrow["vector_len"] == 2)

    raw = (image * 255.0).round().astype(np.uint8)
    _expect(
        "uint8 image is stored unchanged",
        lambda: image_to_uint8_bytes(raw) == raw.tobytes())

    def _bad_shape():
        try:
            image_to_uint8_bytes(np.zeros((8, 8, 3), dtype=np.float32))
        except ValueError:
            return True
        return False

    _expect("wrong image shape raises", _bad_shape)

    print("%d passed, %d failed" % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
