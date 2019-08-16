from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import functools
from keras_config.layers import VariableMomentumBatchNormalization
from keras_config.callbacks import BatchNormMomentumScheduler
from keras_config import functions
from deep_cloud.functions.decay import complementary_exponential_decay
import six
layers = tf.keras.layers


def value(dim):
    return getattr(dim, 'value', dim)  # TF-COMPAT


def transform_diff(transform):
    return (tf.matmul(transform, transform, transpose_b=True) -
            tf.eye(value(transform.shape[-1])))  # TF-COMPAT


def mlp(x,
        units,
        training=None,
        use_batch_norm=True,
        batch_norm_momentum=0.99,
        dropout_rate=0,
        activation='relu'):
    use_bias = not use_batch_norm
    for u in units:
        dense = layers.Dense(u, use_bias=use_bias)
        x = dense(x)
        if use_batch_norm:
            x = VariableMomentumBatchNormalization(
                momentum=batch_norm_momentum)(x, training=training)
        x = layers.Activation(activation)(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x, training=training)
    return x


def add_identity(x, num_dims=None):
    if num_dims is None:
        num_dims = x.shape[-1]
    return x + tf.eye(num_dims, dtype=x.dtype)


def feature_transform_net(features,
                          num_dims,
                          training=None,
                          use_batch_norm=True,
                          batch_norm_momentum=0.99,
                          local_activation='relu',
                          global_activation='relu',
                          local_units=(64, 128, 1024),
                          global_units=(512, 256),
                          reduction=tf.reduce_max):
    """
    Feature transform network.

    Args:
        inputs: (B, N, f_in) inputs features
        training: flag used in batch norm
        batch_norm_momentum: batch norm momentum
        num_dims: output dimension

    Retturns:
        (B, num_dims, num_dims) transformation matrix,
    """
    x = mlp(features,
            local_units,
            training=training,
            activation=local_activation,
            use_batch_norm=use_batch_norm,
            batch_norm_momentum=batch_norm_momentum)
    x = layers.Lambda(reduction, arguments=dict(axis=-2))(x)  # TF-COMPAT
    x = mlp(x,
            global_units,
            training=training,
            activation=global_activation,
            use_batch_norm=use_batch_norm,
            batch_norm_momentum=batch_norm_momentum)

    delta = layers.Dense(num_dims**2,
                         kernel_initializer=tf.keras.initializers.zeros(),
                         bias_initializer=tf.keras.initializers.constant(
                             np.eye(num_dims).flatten()))(x)
    delta = layers.Reshape((num_dims,) * 2)(delta)

    delta = layers.Lambda(add_identity,
                          arguments=dict(num_dims=num_dims))(delta)  # TF-COMPAT
    return delta


def apply_transform(args):
    cloud, matrix = args
    return tf.matmul(cloud, matrix)


def _divide_by_batch_size(args):  # TF-COMPAT
    value, value_with_batch_dim = args
    batch_size = tf.shape(value_with_batch_dim)[0]
    return value / tf.cast(batch_size, value.dtype)


def pointnet_classifier(
        input_spec,
        output_spec,
        training=None,
        use_batch_norm=True,
        batch_norm_momentum=0.99,
        dropout_rate=0.3,
        reduction=tf.reduce_max,
        units0=(64, 64),
        units1=(64, 128, 1024),
        global_units=(512, 256),
        transform_reg_weight=0.0005 * 32,  # account for averaging
):
    """
    Get a pointnet classifier.

    Args:
        inputs_spec: `tf.keras.layers.InputSpec` representing cloud coordinates.
        training: bool indicating training mode.
        output_spec: InputSpec (shape, dtype attrs) of the output
        use_batch_norm: flag indicating usage of batch norm.
        batch_norm_momentum: momentum value of batch norm. If this is a callable
            it is assumed to be a function of the epoch index, and the returned
            callbacks contain a callback that updates these at the end of each
            epoch. If it is a dict, it is assumed to be a serialized function.
            Ignored if use_batch_norm is False.
        dropout_rate: rate used in Dropout for global mlp.
        reduction: reduction function accepting (., axis) arguments.
        units0: units in initial local mlp network.
        units1: units in second local mlp network.
        global_units: units in global mlp network.
        transform_reg_weight: weight used in l2 regularizer. Note this should
            be half the weight of the corresponding code in the original
            implementation since we used keras, which does not half the squared
            norm. We also take the mean over the batch dimension, hence the
            factor of 32 in the default value compared to the original paper.

    Returns:
        keras model with logits as outputs and list of necessary callbacks.
    """
    inputs = tf.keras.layers.Input(shape=input_spec.shape,
                                   dtype=input_spec.dtype)
    reduction = functions.get(reduction)
    if isinstance(batch_norm_momentum, dict):
        batch_norm_momentum = functions.get(batch_norm_momentum)

    callbacks = []
    if use_batch_norm and callable(batch_norm_momentum):
        bn_fn = batch_norm_momentum
        batch_norm_momentum = 0.99  # initial momentum - irrelevant?
        callbacks.append(BatchNormMomentumScheduler(bn_fn))

    bn_kwargs = dict(use_batch_norm=use_batch_norm,
                     batch_norm_momentum=batch_norm_momentum)
    num_classes = output_spec.shape[-1]
    cloud = inputs
    transform0 = feature_transform_net(cloud, 3, training=training, **bn_kwargs)
    cloud = layers.Lambda(apply_transform)([cloud, transform0])  # TF-COMPAT
    cloud = mlp(cloud, units0, training=training, **bn_kwargs)

    transform1 = feature_transform_net(cloud, units0[-1], **bn_kwargs)
    cloud = layers.Lambda(apply_transform)([cloud, transform1])  # TF-COMPAT

    cloud = mlp(cloud, units1, training=training, **bn_kwargs)

    features = layers.Lambda(reduction,
                             arguments=dict(axis=-2))(cloud)  # TF-COMPAT
    features = mlp(features,
                   global_units,
                   training=training,
                   dropout_rate=dropout_rate,
                   **bn_kwargs)
    logits = tf.keras.layers.Dense(num_classes)(features)

    model = tf.keras.models.Model(inputs=tf.nest.flatten(inputs),
                                  outputs=logits)

    if transform_reg_weight:
        regularizer = tf.keras.regularizers.l2(transform_reg_weight)
        for transform in (transform1,):
            reg_loss = tf.keras.layers.Lambda(regularizer)(
                layers.Lambda(transform_diff)(transform))
            mean_reg_loss = tf.keras.layers.Lambda(_divide_by_batch_size)(
                [reg_loss, transform])
            model.add_loss(mean_reg_loss)  # TF-COMPAT

    return model, callbacks
