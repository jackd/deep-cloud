from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras.ops import utils


def reduce_flat_mean(x, row_splits_or_k, weights, eps=1e-7):
    """
    Reduce weighted mean of ragged tensor components along axis 1.

    Conceptually this should be
    ```python
    tf.reduce_sum(x * weights, axis=1) / tf.reduce_sum(weights, axis=1)
    ```
    where x and weights either share `row_splits` or 1st dimension (`k`).

    However, since the row_splits may come from pre-batched computation,
    I -think- tensorflow does some checks to ensure things are all good when
    operating with ragged tensors which slows things down immensely. We avoid
    this by doing the multiplication using the flat representation and summing
    using ragged (or flat if row_splits_or_k is a scalar k)

    Args:
        x: [ni, f] float tensor
        row_splits_or_k: [no+1] (row splits) int or scalar int (k)
        weights: [ni,] or None for uniform weighting

    Returns:
        [no, f] float tensor representing the weighted mean.
    """
    x = tf.convert_to_tensor(x, tf.float32)
    row_splits_or_k = tf.convert_to_tensor(row_splits_or_k, tf.int64)
    if row_splits_or_k.shape.ndims == 0:
        # k
        x = utils.reshape_leading_dim(x, (-1, row_splits_or_k))
        if weights is None:
            # uniform weight
            return tf.reduce_mean(x, axis=1)
        else:
            weights = tf.expand_dims(weights, axis=-1)
            return tf.reduce_sum(x * weights, axis=1) / tf.reduce_sum(weights,
                                                                      axis=1)
    # ragged
    if weights is None:
        return tf.reduce_mean(tf.RaggedTensor.from_row_splits(
            x, row_splits_or_k),
                              axis=1)

    weights = tf.convert_to_tensor(weights, tf.float32)
    weights = tf.expand_dims(weights, axis=-1)
    numer = tf.reduce_sum(tf.RaggedTensor.from_row_splits(
        x * weights, row_splits_or_k),
                          axis=1)
    denom = tf.reduce_sum(tf.RaggedTensor.from_row_splits(
        weights, row_splits_or_k),
                          axis=1)
    if eps is not None:
        denom = denom + eps

    return numer / denom


def flat_expanding_edge_conv(node_features,
                             coord_features,
                             indices,
                             row_splits_or_k,
                             weights,
                             eps=1e-7):
    """
    Expanding edge convolution which operates on flat inputs for efficiency.

    Args:
        node_features: [pi, fi]
        coord_features: [pok, fk]
        indices: [pok] or `None`. If `None`, pi == pok must be True
        row_splits_or_k: [pok+1] int array or row_splits of node_features, or
            scalar if neighborhood sizes are constant.
        weights: [pok, fk] weights applied to coord features
        eps: used when neighborhood is empty to avoid indeterminant forms.

    Returns:
        convolved node_features, [pok, fi*fk]
    """
    if node_features is None:
        return reduce_flat_mean(coord_features, row_splits_or_k, weights)
    else:
        assert (all(
            isinstance(t, tf.Tensor) and t.shape.ndims == 2
            for t in (node_features, coord_features)))
        assert (weights is None or
                isinstance(weights, tf.Tensor) and weights.shape.ndims == 2)
        if indices is not None:
            assert (indices.shape.ndims == 1)
            node_features = tf.gather(node_features, indices)
        if weights is not None:
            coord_features = weights * coord_features
        merged = utils.outer(node_features, coord_features)
        merged = utils.flatten_final_dims(merged, 2)
        return reduce_flat_mean(merged, row_splits_or_k, weights, eps=1e-7)


def flat_expanding_global_deconv(global_features, coord_features,
                                 row_splits_or_k):
    """
    Global deconvolution operation.

    Args:
        global_features: [pi, fi]
        coord_features: [po, fk]
        row_splits_or_k: [pi+1]

    Returns:
        convolved features: [po, fi*fk]
    """
    from tensorflow.python.ops.ragged.ragged_util import repeat  # pylint: disable=no-name-in-module
    if row_splits_or_k.shape.ndims == 0:
        raise NotImplementedError

    global_features = repeat(global_features,
                             utils.diff(row_splits_or_k),
                             axis=0)
    merged = utils.outer(global_features, coord_features)
    merged = utils.flatten_final_dims(merged, 2)
    return merged
