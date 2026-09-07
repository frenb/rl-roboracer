"""Smoke-test the dict-obs network constructors (no gym).

Covers both camera courses' vector widths: donut_camera (31-D) and the
raycast-ablation donut_camera_no_rays (2-D). The towers infer the vector
width from the spec, so this is what confirms a narrowed observation still
builds and runs before committing hours of TRAIN to it.
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from vision.camera_networks import (
    DictObsCriticNetwork, camera_preprocessing_layers,
)


def _check_width(vec_dim, course):
    obs = {
        'vector': tensor_spec.BoundedTensorSpec(
            (vec_dim,), np.float32, -10, 1000, 'vector'),
        'image': tensor_spec.BoundedTensorSpec(
            (84, 84, 3), np.float32, 0, 1, 'image'),
    }
    act = tensor_spec.BoundedTensorSpec((2,), np.float32, 0.05, 1, 'action')
    net = DictObsCriticNetwork((obs, act), joint_fc_layer_params=(64, 64))
    layers = camera_preprocessing_layers()
    assert set(layers) == {'image', 'vector'}
    # create_variables runs Conv2D on GPU — the path that aborted when
    # libcudnn8 was the CUDA 12.2 package (missing libcublasLt.so.12).
    dummy = (
        {
            'vector': tf.zeros((1, vec_dim), tf.float32),
            'image': tf.zeros((1, 84, 84, 3), tf.float32),
        },
        tf.zeros((1, 2), tf.float32),
    )
    net.create_variables()
    q, _ = net(dummy)
    assert q.shape == (1,)
    print(f'OK camera_networks {net.name} vector({vec_dim},) [{course}] '
          f'q {q.numpy()}')


def main():
    _check_width(31, 'donut_camera')
    _check_width(2, 'donut_camera_no_rays')


if __name__ == '__main__':
    main()
