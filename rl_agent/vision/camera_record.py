"""Encode a camera-course observation into TFRecord features.

Flat donut rows stay on the existing writer path. A camera row stores the
policy vector in ``observation`` and the CSI frame in ``image`` as uint8
rgb8 bytes (84*84*3 = 21168). On read, divide by 255 to recover the
float32 image ``image_to_obs`` produced. See
docs/camera-demo-tfrecord-plan.md.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

# Same shape as environments.courses.donut_course_camera.IMAGE_HW.
IMAGE_HW = (84, 84, 3)
IMAGE_BYTES = 84 * 84 * 3


def is_camera_observation(observation):
    """True when this step's observation is the camera-course dict."""
    return (
        isinstance(observation, dict)
        and "vector" in observation
        and "image" in observation
    )


def _as_array(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def vector_floats(vector):
    """Policy vector as a 1-D float32 array. Length is the course's (31 or 2)."""
    return _as_array(vector).astype(np.float32, copy=False).reshape(-1)


def image_to_uint8_bytes(image):
    """float32 [0, 1] image -> 21168 uint8 bytes. uint8 input is stored as-is."""
    arr = _as_array(image)
    if arr.size != IMAGE_BYTES:
        raise ValueError(
            "camera image has %d values, expected %d (%s)"
            % (arr.size, IMAGE_BYTES, IMAGE_HW))
    if np.issubdtype(arr.dtype, np.floating):
        scaled = np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255)
        arr = scaled.astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise TypeError(
            "camera image must be float32 in [0, 1] or uint8, got %s" % arr.dtype)
    return np.ascontiguousarray(arr).tobytes()


def camera_observation_features(observation):
    """TF features for the vector and the image. ``observation`` is the dict."""
    if not is_camera_observation(observation):
        raise TypeError("camera observation must be a dict with vector and image")
    vector = vector_floats(observation["vector"])
    image_bytes = image_to_uint8_bytes(observation["image"])
    return {
        "observation": tf.train.Feature(
            float_list=tf.train.FloatList(value=vector)),
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes])),
        "vector_len": int(vector.shape[0]),
        "image_len": len(image_bytes),
    }


def _flat_floats(value):
    return _as_array(value).astype(np.float32, copy=False).reshape(-1)


def _flat_ints(value):
    return _as_array(value).reshape(-1).astype(np.int64)


def serialize_camera_example(observation, action, reward, discount, next_step_type):
    """One camera TFRecord row: vector, uint8 image, and the step's reward.

    ``write_trajectories_to_file`` uses this for dict observations. Flat
    donut rows do not.
    """
    encoded = camera_observation_features(observation)
    feature = {
        "action": tf.train.Feature(
            float_list=tf.train.FloatList(value=_flat_floats(action))),
        "observation": encoded["observation"],
        "image": encoded["image"],
        "reward": tf.train.Feature(
            float_list=tf.train.FloatList(value=_flat_floats(reward))),
        "discount": tf.train.Feature(
            float_list=tf.train.FloatList(value=_flat_floats(discount))),
        "next_step_type": tf.train.Feature(
            int64_list=tf.train.Int64List(value=_flat_ints(next_step_type))),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString(), encoded
