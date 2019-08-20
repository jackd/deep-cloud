"""log-dense-exp layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from more_keras.layers import Dense
from more_keras.layers import utils as layer_utils
from more_keras import constraints as c


def _complex_angle(x):
    return tf.where(tf.greater(x, 0), tf.zeros_like(x), np.pi * tf.ones_like(x))


def _angle_to_unit_vector(angle):
    return tf.stack([tf.cos(angle), tf.sin(angle)], axis=-1)


def _zeros_if_any_small(inputs, outputs, eps=1e-12):
    return tf.where(tf.reduce_any(tf.less(tf.abs(inputs), eps), axis=-1),
                    tf.zeros_like(outputs), outputs)


def get_log_dense_exp_features(rel_coords,
                               num_complex_features=8,
                               max_order=3,
                               is_total_order=True,
                               dropout_rate=None,
                               use_batch_norm=False):
    # NOTE: probably going to be gradient problems if `not is_total_order`
    constraints = [tf.keras.constraints.NonNeg()]
    if max_order is not None:
        if is_total_order:
            order_constraint = tf.keras.constraints.MaxNorm(max_order, axis=0)
        else:
            order_constraint = c.MaxValue(max_order)
        constraints.append(order_constraint)
    constraint = c.compound_constraint(*constraints)
    dense = Dense(num_complex_features,
                  kernel_constraint=constraint,
                  kernel_initializer=tf.initializers.truncated_normal(
                      mean=0.5, stddev=0.2))

    def f(inputs, min_log_value=-10):
        x = inputs
        x = layer_utils.lambda_call(tf.abs, x)
        x = layer_utils.lambda_call(tf.math.log, x)
        x = layer_utils.lambda_call(tf.maximum, x, min_log_value)
        x = dense(x)
        x = layer_utils.lambda_call(tf.exp, x)
        angle = layer_utils.lambda_call(_complex_angle, inputs)
        angle = dense(angle)
        components = layer_utils.lambda_call(_angle_to_unit_vector, angle)
        x = layer_utils.lambda_call(tf.expand_dims, x, axis=-1)
        x = layer_utils.lambda_call(tf.multiply, x, components)
        x = layer_utils.flatten_final_dims(x, n=2)
        x = layer_utils.lambda_call(_zeros_if_any_small, inputs, x)
        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        return x

    return tf.ragged.map_flat_values(f, rel_coords)
