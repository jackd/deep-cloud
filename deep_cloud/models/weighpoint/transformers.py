"""
Different coordinate transformations.

Each takes an unbatched relative coordinates tensor/squared ball search radius
and should return a batched transformed model tensor. This allows for the
transformer to decide what to preprocess and what to include in the main
network. For example, polynomial transformers do not introduce learned
parameters so can be computed during preprocessing.

Note maximizing preprocessing can backfire on systems without significant CPU
resources since high-order polynomial are non-trivial to compute on CPUs but
are massively accelerated using GPUs. Just because you can, doesn't mean you
should...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from more_keras.meta_models import builder as b
from more_keras.ops import polynomials as p
from deep_cloud.layers.lde import get_log_dense_exp_features
from deep_cloud.layers import ctg


class StaticTransformer(object):
    # can cache result
    def __init__(self, fn):
        self._fn = fn
        self._cache = {}

    def __call__(self, x, max_radius2):
        key = (x, max_radius2)
        if key not in self._cache:
            xb = b.as_batched_model_input(x)
            value = tf.ragged.map_flat_values(
                lambda x: self._fn(x, max_radius2), xb)
            self._cache[key] = value
        return self._cache[key]


class LearnedTransformer(object):
    # must batch before calling
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, x, max_radius2):
        batched_x = b.as_batched_model_input(x)
        x_features = tf.ragged.map_flat_values(
            lambda f: self._fn(f, max_radius2), batched_x)
        return x_features


def _scale(args):
    x, radius2 = args
    return x if radius2 is None else x / tf.sqrt(radius2)


def _get_scale_layer(scale_coords):
    if scale_coords:
        scale_layer = tf.keras.layers.Lambda(_scale)
    else:

        def scale_layer(args):
            return args[0]

    return scale_layer


@gin.configurable
def polynomial_transformer(max_order=3,
                           is_total_order=True,
                           base_builder=None,
                           scale_coords=False):
    layer = tf.keras.layers.Lambda(p.get_nd_polynomials,
                                   arguments=dict(max_order=max_order,
                                                  is_total_order=is_total_order,
                                                  base_builder=base_builder))
    scale_layer = _get_scale_layer(scale_coords)
    return StaticTransformer(
        lambda x, radius2: layer(scale_layer([x, radius2])))


@gin.configurable
def lde_transformer(num_complex_features=8,
                    max_order=3,
                    is_total_order=3,
                    dropout_rate=None,
                    use_batch_norm=False,
                    scale_coords=False):
    scale_layer = _get_scale_layer(scale_coords)
    return LearnedTransformer(lambda x, radius2: get_log_dense_exp_features(
        scale_layer([x, radius2]),
        num_complex_features=num_complex_features,
        max_order=max_order,
        is_total_order=is_total_order,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm))


def _dist2(x):
    return tf.reduce_sum(tf.square(x), axis=-1)


@gin.configurable
def ctg_transformer():
    layer = tf.keras.layers.Lambda(_dist2)
    squared_dists = StaticTransformer(lambda x, _: layer(x))

    def f(x, radius2):
        ctg_layer = ctg.ContinuousTruncatedGaussian()
        ctg_factor = tf.ragged.map_flat_values(
            lambda sd: ctg_layer([sd, radius2]), squared_dists(x, radius2))
        return ctg_factor

    return f
