"""SAC actor/critic towers for dict observations (the donut_camera courses).

31/32-D jobs keep the stock MLP. This module is only imported when the
course's COURSE_OBS_KIND is "dict". Each call to
camera_preprocessing_layers() builds fresh Keras layers so actor and critic
do not share weights.

image: Nature-DQN-style Conv2D 32/64/64 on 84x84, Flatten, Dense 256
vector: Dense 64 - input width is inferred from the observation spec, so this
  serves both donut_camera (31-D) and donut_camera_no_rays (2-D) unchanged
concat -> existing fc_layer_params / TanhNormalProjectionNetwork
"""
from __future__ import annotations

import tensorflow as tf
from tf_agents.networks import encoding_network
from tf_agents.networks import network


def camera_preprocessing_layers():
    return {
        'image': tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
        ], name='csi_image_encoder'),
        'vector': tf.keras.layers.Dense(
            64, activation='relu', name='csi_vector_encoder'),
    }


def camera_preprocessing_combiner():
    return tf.keras.layers.Concatenate(axis=-1)


class DictObsCriticNetwork(network.Network):
    """Q(s, a) for a dict observation. Stock DDPG CriticNetwork only
    uses nest.flatten(obs)[0], which would drop the other modality.
    """

    def __init__(self, input_tensor_spec, joint_fc_layer_params=(512, 512),
                 name='DictObsCriticNetwork'):
        super(DictObsCriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        observation_spec, _action_spec = input_tensor_spec
        self._encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=camera_preprocessing_layers(),
            preprocessing_combiner=camera_preprocessing_combiner(),
            fc_layer_params=None,
            kernel_initializer='glorot_uniform')
        self._joint_layers = [
            tf.keras.layers.Dense(
                n, activation='relu', kernel_initializer='glorot_uniform')
            for n in (joint_fc_layer_params or ())
        ]
        self._out = tf.keras.layers.Dense(
            1, kernel_initializer='glorot_uniform')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        observations, actions = inputs
        emb, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state,
            training=training)
        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
        joint = tf.concat([emb, actions], axis=-1)
        for layer in self._joint_layers:
            joint = layer(joint, training=training)
        q = self._out(joint, training=training)
        return tf.reshape(q, [-1]), network_state
