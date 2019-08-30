from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def sample(unweighted_probs, k, dtype=tf.int64):
    logits = tf.math.log(unweighted_probs)
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, k)
    return tf.cast(indices, dtype)


def mask(unweighted_probs, target_mean):
    probs = unweighted_probs * (target_mean / tf.reduce_mean(unweighted_probs))
    return tf.random.uniform(shape=tf.shape(probs)) < probs


def _rejection_sample_while(neighbors):
    """Rejection sampling using tf.while_loops."""
    if isinstance(neighbors, tf.RaggedTensor):
        N = neighbors.nrows(out_type=tf.int32)
    else:
        N = tf.shape(neighbors, out_type=tf.int32)[0]
    if neighbors.dtype != tf.int32:
        neighbors = tf.cast(neighbors, tf.int32)

    def consume(i, consumed):
        ni = neighbors[i]
        consumed = consumed.scatter(ni,
                                    tf.ones(shape=(tf.size(ni)), dtype=tf.bool))
        # consumed = tf.tensor_scatter_nd_update(
        #     consumed, tf.expand_dims(ni, -1),
        #     tf.ones(dtype=tf.bool, shape=tf.shape(ni)))
        return consumed

    def cond(i, index, consumed, out):
        return i < N

    def body(i, index, consumed, out):
        cond_value = consumed.read(i)
        # cond_value = consumed[i]
        out_next, consumed_next, index_next = tf.cond(
            cond_value, lambda: (out, consumed, index), lambda:
            (out.write(index, i), consume(i, consumed), index + 1))
        return i + 1, index_next, consumed_next, out_next

    out_arr = tf.TensorArray(size=N,
                             dtype=tf.int32,
                             dynamic_size=False,
                             element_shape=())

    # consumed = tf.zeros(shape=(N,), dtype=tf.bool)
    consumed = tf.TensorArray(size=N,
                              dtype=tf.bool,
                              dynamic_size=False,
                              element_shape=())

    _, size, ___, out_arr = tf.while_loop(cond, body, (0, 0, consumed, out_arr))
    return out_arr.stack()[:size]


def _rejection_sample_np_py_function(neighbors):
    """Rejection sampling via numpy with py_function."""
    from deep_cloud.ops.np_utils.sample import rejection_sample
    values = neighbors.values
    row_splits = neighbors.row_splits

    def fn(values, row_splits):
        return rejection_sample(values.numpy(), row_splits.numpy())

    out = tf.py_function(fn, [values, row_splits], tf.int32)
    out.set_shape((None,))
    return out


def _rejection_sample_tf_py_function(neighbors):
    """Rejection sampling via eager mode tensors."""
    args = neighbors.values, neighbors.row_splits
    if tf.executing_eagerly():
        return _rejection_sample_eager(*args)
    out = tf.py_function(_rejection_sample_eager, args, tf.int32)
    out.set_shape((None,))
    return out


def _rejection_sample_eager(values, row_splits):
    assert (tf.executing_eagerly())
    out = []
    N = tf.size(row_splits) - 1
    consumed = tf.zeros((N,), dtype=tf.bool)
    for i in range(N):
        if not consumed[i]:
            out.append(i)
            vals = tf.expand_dims(values[row_splits[i]:row_splits[i + 1]],
                                  axis=-1)
            consumed = tf.tensor_scatter_update(
                consumed, vals, tf.ones(shape=(tf.size(vals),), dtype=tf.bool))
    return tf.stack(out, axis=0)


def rejection_sample(neighbors):
    """
    Sample points such that each point is in a sampled points neighborhood.

    Algorithm operates by taking unconsumed points in order and then consuming
    all points in the neighborhood as consumed.

    For a random sampling, shuffle points before computing the neighborhood.

    Args:
        neighbors: [N, k?] possibly ragged int array of points and their
            neighbors. `neighbors[i] == p, q, r` indicates node `i` has
            neighborhood containing `p`, `q` and `r`. Does not affect the
            algorithm if `i` is in the neighborhood of `i`.

    Returns:
        indices: [n] int32 tensor of indices in the resulting sample.
    """
    return _rejection_sample_np_py_function(neighbors)
