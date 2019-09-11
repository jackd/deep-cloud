"""
Bipartite edge reduction operations.

Most have the signature
flat_edge_values, flat_node_indices, row_splits, **kwargs
     -> a_features, b_features
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras.ops import utils as op_utils
from deep_cloud.ops.asserts import assert_flat_tensor, INT_TYPES, FLOAT_TYPES

# shared by reduce_sum, reduce_max, reduce_min, reduce_mean
_docstring = """
    Reduce edge values over nodes defined by `node_indices`.

    Args:
        flat_edge_values: [ne, f] flat tensor of edge values to be reduced.
        flat_node_indices: [ne] int tensor of indices
        row_splits: [na] node_indices row_splits.
        size: int tensor, size of set `b`.
        symmetric: if True, don't bother computing features_b

    Returns:
        features_a: [na, f] features in disjoint set `a`
        featrues_b: [size, f] features in disjoint set `b`, or None if
            `symmetric`
"""


def _reduce_simple(flat_edge_values,
                   flat_node_indices,
                   row_splits,
                   size,
                   ragged_reduction=tf.math.reduce_sum,
                   unsorted_segment_reduction=tf.math.unsorted_segment_sum,
                   symmetric=False):
    """
    Reduce edge values over nodes defined by `node_indices`.

    Args:
        flat_edge_values: [ne, f] flat tensor of edge values to be reduced.
        flat_node_indices: [ne] int tensor of indices
        row_splits: [na] node_indices row_splits.
        size: int tensor, size of set `b`.
        ragged_reduction: reduction operation that can be applied to a ragged
            tensor on the ragged dimension, e.g. `tf.math.reduce_sum`
        unsorted_segment_reduction: a `tf.math.unsorted_segment_*` function.
        symmetric: if True, features_b are not computed.

    Returns:
        `features_a` if symmetric, else `features_a, features_b`
        features_a: [na, f] features in disjoint set `a`
        featrues_b: [size, f] features in disjoint set `b`, or None if
            `symmetric`
    """
    assert_flat_tensor('flat_edge_values',
                       flat_edge_values,
                       2,
                       dtype=FLOAT_TYPES)
    assert_flat_tensor('flat_node_indices',
                       flat_node_indices,
                       1,
                       dtype=INT_TYPES)
    if not (isinstance(size, int) or
            isinstance(size, tf.Tensor) and size.shape.ndims == 0):
        raise ValueError('size must be a scalar, got {}'.format(size))
    features_a = ragged_reduction(tf.RaggedTensor.from_row_splits(
        flat_edge_values, row_splits),
                                  axis=1)
    if symmetric:
        return features_a
    else:
        features_b = unsorted_segment_reduction(flat_edge_values,
                                                flat_node_indices, size)
    return features_a, features_b


def reduce_sum(flat_edge_values,
               flat_node_indices,
               row_splits,
               size,
               symmetric=False):
    return _reduce_simple(flat_edge_values,
                          flat_node_indices,
                          row_splits,
                          size,
                          tf.math.reduce_sum,
                          tf.math.unsorted_segment_sum,
                          symmetric=symmetric)


def reduce_max(flat_edge_values,
               flat_node_indices,
               row_splits,
               size,
               symmetric=False):
    return _reduce_simple(flat_edge_values,
                          flat_node_indices,
                          row_splits,
                          size,
                          tf.math.reduce_max,
                          tf.math.unsorted_segment_max,
                          symmetric=symmetric)


def reduce_min(flat_edge_values,
               flat_node_indices,
               row_splits,
               size,
               symmetric=False):
    return _reduce_simple(flat_edge_values,
                          flat_node_indices,
                          row_splits,
                          size,
                          tf.math.reduce_min,
                          tf.math.unsorted_segment_min,
                          symmetric=symmetric)


def reduce_mean(flat_edge_values,
                flat_node_indices,
                row_splits,
                size,
                symmetric=False):
    return _reduce_simple(flat_edge_values,
                          flat_node_indices,
                          row_splits,
                          size,
                          tf.math.reduce_mean,
                          tf.math.unsorted_segment_mean,
                          symmetric=symmetric)


for fn in reduce_sum, reduce_max, reduce_mean, reduce_min:
    fn.__doc__ = _docstring

del fn


def reduce_weighted_mean(flat_edge_values,
                         flat_weights,
                         flat_node_indices,
                         row_splits,
                         size,
                         symmetric=False,
                         delta=1e-12):
    """
    Reduce edge values over nodes defined by `node_indices`.

    `flat_*` tensors conceptually all correspond to ragged tensors with the
    same `row_splits`. After raggedifying, this finds
    `sum(weights * edge_values) / sum(weights)` for one of the disjoint sets,
    and the corresponding values for the other (summed over different indices).

    Args:
        flat_edge_values: [ne, f] flat tensor of edge values to be reduced.
        flat_weights: [ne] weights applied to each edge.
        flat_node_indices: [ne] int tensor of indices.
        row_splits: [na] node_indices row_splits.
        size: int tensor, size of set `b`.
        symmetric: if True, don't bother computing features_b
        delta: small offset to avoid division by zero for isolated nodes.

    Returns:
        `features_a` if symmetric else `features_a, features_b`
        features_a: [na, f] features in disjoint set `a`
        features_b: [size, f] features in disjoint set `b`, or None if symmetric
    """
    assert_flat_tensor('flat_edge_values',
                       flat_edge_values,
                       2,
                       dtype=FLOAT_TYPES)
    assert_flat_tensor('flat_weights',
                       flat_weights,
                       1,
                       dtype=flat_edge_values.dtype)
    assert_flat_tensor('flat_node_indices',
                       flat_node_indices,
                       1,
                       dtype=INT_TYPES)
    if not isinstance(size, int) or isinstance(size,
                                               tf.Tensor) and size.shape != ():
        raise ValueError('size must be a scalar, got {}'.format(size))
    flat_edge_values = flat_edge_values * tf.reshape(flat_weights, (-1, 1, 1))
    numers = reduce_sum(flat_edge_values,
                        flat_node_indices,
                        row_splits,
                        size,
                        symmetric=symmetric)
    denoms = reduce_sum(flat_weights,
                        flat_node_indices,
                        row_splits,
                        size,
                        symmetric=symmetric)
    # check for None in case of symmetry.

    features_a, features_b = (None if numer is None else numer / (denom + delta)
                              for numer, denom in zip(numers, denoms))
    if symmetric:
        return features_a
    return features_a, features_b


def distribute_node_features(node_features_a, node_features_b,
                             flat_edge_features, flat_node_indices, row_splits):
    """
    Distribute node features amongst edges.

    Args:
        node_features_a: [n_a, f] float tensor of node features in set `a`.
        node_features_b: [n_b, f] float_tensor of node_features in set `b`.
        flat_edge_features: [n_e, f] float_tensor of edge features.
        flat_node_indices: indices of set `b` in ragged ordering of set `a`.
        row_splits: to form ragged ordering with flat_node_indices.

    Returns:
        [n_e, f] float resulting from adding the distributed node features
            to the existing edge features.
    """
    # No short-cut for symmetric version.
    from more_keras.ops import utils
    assert_flat_tensor('node_features_a', node_features_a, 2, FLOAT_TYPES)
    assert_flat_tensor('node_features_b', node_features_b, 2, FLOAT_TYPES)
    assert_flat_tensor('flat_edge_features', flat_edge_features, 2, FLOAT_TYPES)
    assert_flat_tensor('flat_node_indices', flat_node_indices, 1, INT_TYPES)
    assert_flat_tensor('row_splits', row_splits, 1, INT_TYPES)
    node_features_a = op_utils.repeat(node_features_a,
                                      utils.diff(row_splits),
                                      axis=0)
    node_features_b = tf.gather(node_features_b, flat_node_indices)
    return tf.add_n([node_features_a, node_features_b, flat_edge_features])
