from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras.ops import utils


def _reduce_unweighted_flat_mean(x, row_splits_or_k, eps):
    if row_splits_or_k.shape.ndims == 0:
        # constant neighborhood size
        x = utils.reshape_leading_dim(x, (-1, row_splits_or_k))
        _assert_is_rank(3, x, 'x')
        denom = tf.cast(row_splits_or_k, x.dtype)
    else:
        # ragged
        x = tf.RaggedTensor.from_row_splits(x, row_splits_or_k)
        assert (x.shape.ndims == 3)
        denom = tf.expand_dims(tf.cast(utils.diff(row_splits_or_k), x.dtype),
                               axis=-1)
        assert (denom.shape.ndims == 2)

    if eps is not None:
        denom += eps
    return tf.reduce_sum(x, axis=1) / denom


def reduce_flat_mean(x, row_splits_or_k, weights, eps=1e-7):
    """
    Reduce weighted mean of ragged tensor components along axis 1.

    Conceptually this should be
    ```python
    tf.reduce_sum(x, axis=1) / tf.reduce_sum(weights, axis=1)
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
        weights: [ni] or None for uniform weighting

    Returns:
        [no, f] float tensor representing the weighted mean.
    """
    _assert_is_rank(2, x, 'x')
    x = tf.convert_to_tensor(x, tf.float32)
    row_splits_or_k = tf.convert_to_tensor(row_splits_or_k, tf.int64)

    # unweighted
    if weights is None:
        return _reduce_unweighted_flat_mean(x, row_splits_or_k, eps)

    _assert_is_rank(1, weights, 'weights')
    weights = tf.expand_dims(weights, axis=-1)
    numer = x * weights
    if row_splits_or_k.shape.ndims == 0:
        # fixed number of neighbors k
        numer = utils.reshape_leading_dim(numer, (-1, row_splits_or_k))
        # x is no [no, k, f]
        weights = utils.reshape_leading_dim(weights, (-1, row_splits_or_k))
    else:
        numer = tf.RaggedTensor.from_row_splits(numer, row_splits_or_k)
        weights = tf.RaggedTensor.from_row_splits(weights, row_splits_or_k)

    # numer (unreduced) is [no, k?, f], weights is [no, k?, 1]
    numer = tf.reduce_sum(numer, axis=1)
    denom = tf.reduce_sum(weights, axis=1)
    if eps is not None:
        denom = denom + eps

    return numer / denom


def _assert_is_rank(rank, tensor, name):
    err_msg = '{} should be a rank {} tensor, got {}'
    if not (isinstance(tensor, tf.Tensor) and tensor.shape.ndims == rank):
        raise ValueError(err_msg.format(name, rank, tensor))


# def expanding_edge_conv(
#         node_features, coord_features, indices, weights, eps=1e-7):
#     """
#     Expanding edge convolution.

#     Args:
#         node_features: [B, n?, f_i] possibly ragged tensor of features at nodes.
#         coord_features: [B, n?, k?, p] possibly ragged tensor of features of
#             relative coordinates.
#         indices: [B, n?, k?]
#     """


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
        return reduce_flat_mean(coord_features, row_splits_or_k, weights, eps)
    else:
        for name, tensor in (('node_features', node_features),
                             ('coord_features', coord_features)):
            _assert_is_rank(2, tensor, name)
        if weights is not None:
            _assert_is_rank(1, weights, 'weights')

        if indices is not None:
            assert (indices.shape.ndims == 1)
            node_features = tf.gather(node_features, indices)

        # doing coord_features * weights before outer product might be faster?
        merged = utils.outer(node_features, coord_features)
        merged = utils.flatten_final_dims(merged, 2)
        out = reduce_flat_mean(merged, row_splits_or_k, weights, eps)
        return out


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
    if row_splits_or_k.shape.ndims == 0:
        raise NotImplementedError

    global_features = utils.repeat(global_features,
                                   utils.diff(row_splits_or_k),
                                   axis=0)
    merged = utils.outer(global_features, coord_features)
    merged = utils.flatten_final_dims(merged, 2)
    return merged
