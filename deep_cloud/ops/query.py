from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from more_keras.tf_compat import dim_value
from deep_cloud.ops.np_utils import tree_utils
from more_keras.ragged import np_impl as ra


def query_pairs(coords, radius, max_neighbors=None):

    def fn(coords, radius_tensor=None):
        r = radius if radius_tensor is None else radius_tensor.numpy()
        tree = tree_utils.cKDTree(coords.numpy())
        pairs = tree_utils.query_pairs(tree, r, max_neighbors)
        return pairs.values, pairs.row_splits

    if isinstance(radius, tf.Tensor):
        args = (coords, radius)
    else:
        args = coords,

    values, row_splits = tf.py_function(fn, args, (tf.int64, tf.int64))
    size = coords.shape[0]

    if not isinstance(size, int):
        size = dim_value(coords.shape[0])
    if size is not None:
        size += 1
    row_splits.set_shape((size))
    values.set_shape((None,))
    return tf.RaggedTensor.from_row_splits(values, row_splits)


def query_knn(coords, k):

    def fn(coords, k_tensor=None):
        kv = k if k_tensor is None else k_tensor.numpy()
        tree = tree_utils.cKDTree(coords.numpy())
        values, dists = tree_utils.query_knn(tree, kv)
        return values, dists

    args = (coords, k) if isinstance(k, tf.Tensor) else (coords,)

    values, dists = tf.py_function(fn, args, (tf.int64, tf.float32))
    shape = (coords.shape[0], k)
    values.set_shape(shape)
    dists.set_shape(shape)
    return values, dists


def reverse_query_pairs(neighbors, size):
    if not isinstance(neighbors, tf.RaggedTensor):
        raise NotImplementedError
    if neighbors.ragged_rank != 1:
        raise NotImplementedError

    def fn(flat_values, row_splits, size):
        rev, rev_indices = tree_utils.reverse_query_ball(
            ra.RaggedArray.from_row_splits(flat_values.numpy(),
                                           row_splits.numpy()),
            size=size.numpy())
        # not sure why this is needed
        # rev_indices comes from a scipy.csc matrix - maybe relevant?
        rev_indices = np.array([t.item() for t in rev_indices], dtype=np.int64)
        return rev.values, rev.row_splits, rev_indices

    values, splits, rev_indices = tf.py_function(
        fn, (neighbors.values, neighbors.row_splits, size),
        (neighbors.dtype, tf.int64, tf.int64))
    values.set_shape(neighbors.values.shape)
    splits.set_shape((None,))
    rev_indices.set_shape((None,))
    return tf.RaggedTensor.from_row_splits(values, splits), rev_indices
