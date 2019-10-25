from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def featureless_conv(kernel, sparse_indices, edge_weights):
    """
    Compute a point cloud convolution when there are no input features.

    Equivalent to

    out_{ip} = \\sum_{j, k} n^{(k)}_{ij}^{(k)} \\theta^{(k)}_p.

    Args:
        kernel: [K, F_out] float tensor.
        sparse_indices: [E, 2] int64 tensor of sparse indices for a sparse
            tensor with dense shape [N_out, N_in], or [E] int64 tensor
            corresponding to the first index (i.e. in [0, N_out)).
        edge_weights: [K, E] float tensor of edge weights.

    Returns:
        [N_out, F_out] node features.
    """
    with tf.name_scope('featureless_conv'):
        kernel = tf.convert_to_tensor(kernel, dtype_hint=tf.float32)
        sparse_indices = tf.convert_to_tensor(sparse_indices,
                                              dtype_hint=tf.int64)
        edge_weights = tf.convert_to_tensor(edge_weights, dtype_hint=tf.float32)
        if sparse_indices.shape.ndims == 1:
            i = sparse_indices
        else:
            assert (sparse_indices.shape.ndims == 2)
            assert (sparse_indices.shape[1] == 2)
            i = sparse_indices[:, 0]
        # edge_weights are [K, E] everywhere else in this module
        edge_weights = tf.transpose(edge_weights, (1, 0))  # [E, K]
        neigh_sum = tf.math.segment_sum(edge_weights, i)  # [N, K]
        return tf.matmul(neigh_sum, kernel)  # [N, F_out]


def get_sparse_transform(sparse_indices, dense_shape):
    if isinstance(sparse_indices, (list, tuple)):
        assert (len(sparse_indices) == 2)
        sparse_indices = tf.stack(sparse_indices, axis=-1)

    def fn(features, edge_weights):
        sp = tf.SparseTensor(sparse_indices, edge_weights, dense_shape)
        return tf.sparse.sparse_dense_matmul(sp, features)

    return fn


def get_gather_sum_transform(sparse_indices, dense_shape):
    if isinstance(sparse_indices, (list, tuple)):
        i, j = sparse_indices
    else:
        i, j = tf.unstack(sparse_indices, axis=-1)

    def fn(features, edge_weights):
        features = tf.gather(features, j)
        features = features * tf.expand_dims(edge_weights, axis=-1)
        features = tf.segment_sum(features, i)
        # if isinstance(dense_shape[0], int):
        #     features.set_shape((dense_shape[0], features.shape[1]))
        return features

    return fn


def map_conv(features,
             kernel,
             sparse_indices,
             edge_weights,
             dense_shape,
             term_impl=get_sparse_transform,
             transform_first=None):
    """
    Graph convolution backed by `tf.map_fn`.

    Args:
        features: [N_in, F_in] float tensor of input features.
        kernel: [T, F_in, F_out] float tensor of feature transformations.
        sparse_indices: [E, 2] int tensor of indices of neighbors matrix.
        edge_weights: [T, E] float tensor of sparse neighbors matrix for each
            kernel feature.
        dense_shape: dense shape of neighbors, value (N_out, N_in).
        term_impl: one of `get_sparse_transform`, `get_gather_sum_transform`.
        transform_first: bool dictating whether to transform before or after
            sparse multiplication. Defaults to the option with fewer operations
            based on the same number of points.

    Returns:
        [N_out, F_out] float tensor of features.
    """
    T, F_in, F_out = kernel.shape
    if isinstance(dense_shape, tf.Tensor):
        dense_shape = tf.unstack(dense_shape, axis=0)
    N_out, N_in = dense_shape
    if transform_first is None:
        transform_first = F_out <= F_in
    get_term_base = term_impl(sparse_indices, dense_shape)

    del T, N_in, N_out

    def get_term(kernel, edge_weights):
        if transform_first:
            f = tf.matmul(features, kernel)
        else:
            f = features
        f = get_term_base(f, edge_weights)
        if not transform_first:
            f = tf.matmul(f, kernel)
        return f

    out = tf.map_fn(lambda args: get_term(*args), (kernel, edge_weights),
                    tf.float32)
    return tf.reduce_sum(out, axis=0)


def fold_conv(features,
              kernel,
              sparse_indices,
              edge_weights,
              dense_shape,
              term_impl=get_sparse_transform,
              transform_first=None):
    """
    Graph convolution backed by `tf.foldl`.

    Args:
        features: [N_in, F_in] float tensor of input features.
        kernel: [T, F_in, F_out] float tensor of feature transformations.
        sparse_indices: [E, 2] int tensor of indices of neighbors matrix.
        edge_weights: [T, E] float tensor of sparse neighbors matrix for each
            kernel feature.
        dense_shape: dense shape of neighbors, value (N_out, N_in).
        term_impl: one of `get_sparse_transform`, `get_gather_sum_transform`.
        transform_first: bool dictating whether to transform before or after
            sparse multiplication. Defaults to the option with fewer operations
            based on the same number of points.

    Returns:
        [N_out, F_out] float tensor of features.
    """
    T, F_in, F_out = kernel.shape
    if isinstance(dense_shape, tf.Tensor):
        dense_shape = tf.unstack(dense_shape, axis=0)
    N_out, N_in = dense_shape
    if transform_first is None:
        transform_first = F_out <= F_in
    get_term_base = term_impl(sparse_indices, dense_shape)

    del T, N_in

    def get_term(kernel, edge_weights):
        if transform_first:
            f = tf.matmul(features, kernel)
        else:
            f = features
        f = get_term_base(f, edge_weights)
        if not transform_first:
            f = tf.matmul(f, kernel)
        return f

    init = tf.zeros((N_out, F_out), dtype=features.dtype)
    out = tf.foldl(lambda acc, args: acc + get_term(*args),
                   (kernel, edge_weights), init)
    return out


def unstack_conv(features,
                 kernel,
                 sparse_indices,
                 edge_weights,
                 dense_shape,
                 term_impl=get_sparse_transform,
                 transform_first=None):
    T, F_in, F_out = kernel.shape
    del T
    if transform_first is None:
        transform_first = F_out <= F_in

    kernels = tf.unstack(kernel, axis=0)
    edge_weights = tf.unstack(edge_weights, axis=0)

    term_fn = term_impl(sparse_indices, dense_shape)

    terms = []
    for k, ew in zip(kernels, edge_weights):
        if transform_first:
            term = tf.matmul(features, k)
        else:
            term = features
        term = term_fn(term, ew)
        if not transform_first:
            term = tf.matmul(term, k)
        terms.append(term)
    return tf.add_n(terms)


def block_conv(features,
               kernel,
               sparse_indices,
               edge_weights,
               dense_shape,
               term_impl=get_sparse_transform,
               transform_first=None):
    T, F_in, F_out = kernel.shape
    if isinstance(dense_shape, tf.Tensor):
        dense_shape = tf.unstack(dense_shape, axis=0)
    N_out, N_in = dense_shape
    if transform_first is None:
        transform_first = F_out <= F_in

    if transform_first:
        # concat along axis=1
        # note tf.sparse.concat doesn't seem to support gradients?
        i, j = tf.unstack(sparse_indices, axis=-1)
        i = tf.tile(tf.expand_dims(i, axis=0), (T, 1))
        j = tf.expand_dims(j, axis=0) + tf.expand_dims(
            tf.range(T, dtype=j.dtype) * N_in, axis=1)

        i = tf.reshape(i, (-1,))
        j = tf.reshape(j, (-1,))
        edge_weights = tf.reshape(edge_weights, (-1,))
        sp = tf.sparse.reorder(
            tf.SparseTensor(tf.stack((i, j), axis=-1), edge_weights,
                            (N_out, T * N_in)))
        features = tf.expand_dims(features, axis=0)
        features = tf.matmul(features, kernel)  # [T, N_in, F_out]
        features = tf.reshape(features, (T * N_in, -1))
        features = term_impl(sp.indices, (N_out, T * N_in))(features, sp.values)
        return features
    else:
        # transform last
        # concat along axis=0
        i, j = tf.unstack(sparse_indices, axis=-1)
        i = tf.expand_dims(i, axis=0) + tf.expand_dims(
            tf.range(T, dtype=i.dtype) * N_out, axis=1)
        j = tf.tile(j, (T,))

        i = tf.reshape(i, (-1,))
        j = tf.reshape(j, (-1,))
        edge_weights = tf.reshape(edge_weights, (-1,))
        features = term_impl(tf.stack((i, j), axis=-1),
                             (T * N_out, N_in))(features, edge_weights)
        features = tf.reshape(features, (T, N_out, F_in))
        features = tf.reshape(tf.transpose(features, (1, 0, 2)),
                              (N_out, T * F_in))
        kernel = tf.reshape(kernel, (T * F_in, F_out))
        return tf.matmul(features, kernel)
        # features = tf.reshape(features, (T, N_out, -1))
        # features = tf.matmul(features, kernel)
        # return tf.reduce_sum(features, axis=0)
