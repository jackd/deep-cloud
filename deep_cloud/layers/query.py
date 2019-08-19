from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deep_cloud.ops import query as qops
from more_keras.ragged.layers import ragged_lambda


def _query_pairs(args, **kwargs):
    if isinstance(args, tf.Tensor):
        args = args,
    neighbors = qops.query_pairs(*args, **kwargs)
    return neighbors


def query_pairs(coords, radius, name=None, max_neighbors=None):
    if isinstance(radius, tf.Tensor):
        kwargs = {}
        args = coords, radius
    else:
        kwargs = dict(radius=radius)
        args = coords
    kwargs['max_neighbors'] = max_neighbors
    neighbors = ragged_lambda(_query_pairs, arguments=kwargs)(args)
    return neighbors


# multiple outputs bugged
# see https://github.com/tensorflow/tensorflow/issues/27639
# resolved by making the output a list... fixed in 1.14+
def _query_knn(args, **kwargs):
    if isinstance(args, tf.Tensor):
        args = args,
    neighbors, dists = qops.query_knn(*args, **kwargs)
    return [neighbors, dists]


def query_knn(coords, k, name=None):
    if isinstance(k, tf.Tensor):
        args = coords, k
        kwargs = {}
    else:
        args = coords
        kwargs = dict(k=k)
    neighbors, dists = tf.keras.layers.Lambda(_query_knn,
                                              arguments=kwargs,
                                              name=name)(args)
    return neighbors, dists


def _reverse_query_pairs(args):
    return qops.reverse_query_pairs(*args)


def reverse_query_pairs(neighbors, size):
    return ragged_lambda(_reverse_query_pairs)([neighbors, size])
