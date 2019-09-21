from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from more_keras import layers as mk_layers
from deep_cloud.models.generalized import layers as gen_layers


def in_place_bottleneck(node_features,
                        coord_features,
                        indices,
                        row_splits,
                        weights,
                        batch_norm_fn,
                        activation_fn,
                        filters=None,
                        bottleneck_factor=4,
                        dense_factory=mk_layers.Dense,
                        shortcut='identity'):
    short = node_features

    if filters is None:
        filters = node_features.shape[-1]
    # dense
    node_features = dense_factory(filters)(node_features)
    node_features = activation_fn(batch_norm_fn(node_features))
    # conv
    node_features = gen_layers.RaggedConvolution(
        filters // bottleneck_factor, dense_factory=dense_factory)([
            activation_fn(node_features), coord_features, indices, row_splits,
            weights
        ])
    node_features = activation_fn(batch_norm_fn(node_features))
    # unactivated dense
    node_features = dense_factory(filters)(node_features)
    node_features = batch_norm_fn(node_features)
    # shortcut
    if shortcut == 'conv':
        short = dense_factory(filters)(short)
        short = batch_norm_fn(short)
        node_features = node_features + short
    elif shortcut == 'identity':
        node_features = node_features + short
    elif shortcut == 'none' or shortcut is None:
        pass
    else:
        raise ValueError('Invalid shortcut value "{}". Must be in '
                         '("identity", "conv", "none")'.format(shortcut))

    node_features = activation_fn(node_features)
    return node_features


def down_sample_bottleneck(node_features,
                           coord_features,
                           indices,
                           row_splits,
                           weights,
                           sample_indices,
                           filters,
                           batch_norm_fn,
                           activation_fn,
                           filter_expansion_factor=2,
                           bottleneck_factor=4,
                           dense_factory=mk_layers.Dense,
                           shortcut='conv'):

    short = node_features
    # conv
    node_features = gen_layers.RaggedConvolution(
        filters // bottleneck_factor, dense_factory=dense_factory)([
            activation_fn(node_features), coord_features, indices, row_splits,
            weights
        ])
    node_features = activation_fn(batch_norm_fn(node_features))
    # unactivated dense
    node_features = dense_factory(filters)(node_features)
    node_features = batch_norm_fn(node_features)

    # shortcut
    if shortcut == 'none':
        return activation_fn(node_features)
    short = tf.gather(short, sample_indices)
    if shortcut == 'conv':
        short = dense_factory(filters)(short)
        short = batch_norm_fn(short)
    elif shortcut == 'identity':
        assert (filter_expansion_factor == 1)
    else:
        raise ValueError('Invalid shortcut value "{}". Must be in '
                         '("identity", "conv", "none")'.format(shortcut))
    node_features = node_features + short
    return activation_fn(node_features)
