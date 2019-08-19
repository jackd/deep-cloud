"""Classification models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
from more_keras.layers import utils
from more_keras.layers import source
from more_keras.meta_models import builder as b

from deep_cloud import neigh as n
from deep_cloud.layers import sample
from deep_cloud.models.weighpoint import core
from deep_cloud.models.weighpoint import convolvers as c
from deep_cloud.models.weighpoint import transformers as t


@gin.configurable(blacklist=['features'])
def cls_tail_activation(features,
                        activation='relu',
                        use_batch_normalization=False,
                        dropout_rate=0.5):
    return core.generalized_activation(
        features=features,
        add_bias=not use_batch_normalization,
        activation=activation,
        use_batch_normalization=use_batch_normalization,
        dropout_rate=dropout_rate)


@gin.configurable(blacklist=['features', 'num_classes'])
def cls_tail(features,
             num_classes,
             hidden_units=(),
             activation=cls_tail_activation):
    if activation is None:
        assert (len(hidden_units) == 0)
    else:
        features = activation(features)
    for u in hidden_units:
        features = tf.keras.layers.Dense(u, activation=None)(features)
        features = activation(features)
    logits = tf.keras.layers.Dense(num_classes, activation=None)(features)
    return logits


@gin.configurable(blacklist=['features'])
def cls_head_activation(features,
                        activation='relu',
                        use_batch_normalization=True,
                        dropout_rate=None):
    return core.generalized_activation(
        features=features,
        add_bias=not use_batch_normalization,
        activation=activation,
        use_batch_normalization=use_batch_normalization,
        dropout_rate=dropout_rate)


@gin.configurable(blacklist=['coords', 'normals'])
def cls_head(coords,
             normals=None,
             r0=0.1,
             initial_filters=(16,),
             initial_activation=cls_head_activation,
             filters=(32, 64, 128, 256),
             global_units='combined',
             query_fn=core.query_pairs,
             radii_fn=core.constant_radii,
             coords_transform=None,
             weights_transform=None,
             convolver=None):

    if convolver is None:
        convolver = c.ExpandingConvolver(activation=cls_head_activation)
    if coords_transform is None:
        coords_transform = t.polynomial_transformer()
    if weights_transform is None:

        def weights_transform_(*args, **kwargs):
            return None
    else:
        weights_transform_ = weights_transform

    n_res = len(filters)
    unscaled_radii2 = radii_fn(n_res)

    if isinstance(unscaled_radii2, tf.Tensor):
        assert (unscaled_radii2.shape == (n_res,))
        radii2 = utils.lambda_call(tf.math.scalar_mul, r0**2, unscaled_radii2)
        radii2 = tf.keras.layers.Lambda(tf.unstack,
                                        arguments=dict(axis=0))(radii2)
        for i, radius2 in enumerate(radii2):
            tf.compat.v1.summary.scalar('r%d' % i,
                                        tf.sqrt(radius2),
                                        family='radii')
    else:
        radii2 = unscaled_radii2 * (r0**2)

    def maybe_feed(r2):
        if isinstance(r2, (tf.Tensor, tf.Variable)):
            return b.prebatch_feed(tf.keras.layers.Lambda(tf.sqrt)(radius2))
        else:
            return np.sqrt(r2)

    features = b.as_batched_model_input(normals)
    for f in initial_filters:
        layer = tf.keras.layers.Dense(f)
        features = tf.ragged.map_flat_values(layer, features)
        features = tf.ragged.map_flat_values(initial_activation, features)

    features = utils.flatten_leading_dims(features, 2)
    global_features = []

    for i, radius2 in enumerate(radii2):
        neighbors, sample_rate = query_fn(coords,
                                          maybe_feed(radius2),
                                          name='query%d' % i)
        if not isinstance(radius2, tf.Tensor):
            radius2 = source.constant(radius2, dtype=tf.float32)
        neighborhood = n.InPlaceNeighborhood(coords, neighbors)
        features, nested_row_splits = core.convolve(features, radius2,
                                                    filters[i], neighborhood,
                                                    coords_transform,
                                                    weights_transform_,
                                                    convolver.in_place_conv)
        if global_units == 'combined':
            coord_features = coords_transform(neighborhood.out_coords, None)
            global_features.append(
                convolver.global_conv(features, coord_features,
                                      nested_row_splits[-2], filters[i]))

        if i < n_res - 1:
            sample_indices = sample.sample(
                sample_rate,
                tf.keras.layers.Lambda(lambda s: tf.size(s) // 4)(sample_rate))
            neighborhood = n.SampledNeighborhood(neighborhood, sample_indices)
            features, nested_row_splits = core.convolve(
                features, radius2, filters[i + 1], neighborhood,
                coords_transform, weights_transform_, convolver.resample_conv)

            coords = neighborhood.out_coords

    # global_conv
    if global_units == 'combined':
        features = tf.keras.layers.Lambda(
            tf.concat, arguments=dict(axis=-1))(global_features)
    else:
        coord_features = coords_transform(coords, None)
        features = convolver.global_conv(features, coord_features,
                                         nested_row_splits[-2], global_units)

    return features


@gin.configurable(blacklist=['inputs', 'output_spec'])
def cls_logits(inputs, output_spec, head_fn=cls_head, tail_fn=cls_tail):
    coords = inputs['positions']
    normals = inputs.get('normals')
    features = head_fn(coords, normals)
    logits = tail_fn(features, num_classes=output_spec.shape[-1])
    return logits
