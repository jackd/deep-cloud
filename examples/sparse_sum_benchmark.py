from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_data(dense_shape, mean_edges=100, seed=123):
    """Generate random sparse matrix data."""
    N, M = dense_shape
    r = np.random.RandomState(seed)

    num_edges = int(mean_edges * N)

    flat_index = (r.uniform(size=num_edges) * N * M).astype(np.int64)
    # flat_index = np.concatenate([flat_index, np.arange(0, N * M, M)], axis=0)
    flat_index = np.array(sorted(set(flat_index)), dtype=np.int64)
    i, j = np.unravel_index(flat_index, (N, M))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = np.stack((i, j), axis=-1)
    weights = np.random.uniform(size=(i.shape[0],)).astype(np.float32)
    return sparse_indices, weights


def sparse_sum(sp, axis=1):
    """sparse.reduce_sum."""
    return tf.sparse.reduce_sum(sp, axis=axis)


def sparse_sum2(sp, axis=1):
    """sparse.reduce_sum_sparse -> to_dense."""
    return tf.sparse.to_dense(tf.sparse.reduce_sum_sparse(sp, axis=axis))


def seg_sum(sp, axis=1, ordered=False):
    """
    math.(unsorted_)segment_sum.

    Args:
        sp: rank 2 sparse tensor
        axis: int, axis along which to sum
        ordered: if True, other axis indices are assumed to be ascending.

    Returns:
        rank 1 dense tensor equivalent to tf.sparse.reduce_sum(sp, axis=axis)
    """
    if sp.shape.ndims != 2:
        raise NotImplementedError
    other_axis = 0 if axis in (1, -1) else 1
    if ordered:
        return tf.math.segment_sum(sp.values, sp.indices[:, other_axis])
    else:
        return tf.math.unsorted_segment_sum(sp.values,
                                            sp.indices[:, other_axis],
                                            sp.dense_shape[other_axis])


def compare(dense_shape, axis=1, ordered=False, **kwargs):
    sparse_indices, weights = get_data(dense_shape, **kwargs)
    sparse_indices = tf.constant(sparse_indices, dtype=tf.int64)
    weights = tf.constant(weights, dtype=tf.float32)
    sp = tf.SparseTensor(sparse_indices, weights, dense_shape)
    sparse = sparse_sum(sp, axis=axis)
    sparse2 = sparse_sum2(sp, axis=axis)
    seg = seg_sum(sp, axis=axis, ordered=ordered)
    sparse_grad, = tf.gradients(sparse, weights)
    # sparse2_grad, = tf.gradients(sparse2, weights)
    seg_grad, = tf.gradients(seg, weights)
    err = tf.reduce_max(tf.abs(seg - sparse))
    shape_err = tf.reduce_max(tf.abs(tf.shape(sparse) - tf.shape(seg)))
    err2 = tf.reduce_max(tf.abs(seg - sparse2))
    shape_err2 = tf.reduce_max(tf.abs(tf.shape(sparse2) - tf.shape(seg)))
    grad_err = tf.reduce_max(tf.abs(sparse_grad - seg_grad))
    # grad_err2 = tf.reduce_max(tf.abs(sparse2_grad - seg_grad))

    with tf.Session() as sess:
        err, shape_err, err2, shape_err2, grad_err = sess.run(
            (err, shape_err, err2, shape_err2, grad_err))
    assert (err < 1e-4)
    assert (shape_err == 0)
    assert (shape_err2 == 0)
    assert (grad_err < 1e-4)
    return err


def run_benchmarks(dense_shape, axis=1, ordered=False, **kwargs):
    sparse_indices, weights = get_data(dense_shape, **kwargs)
    sparse_indices = tf.constant(sparse_indices, dtype=tf.int64)
    weights = tf.constant(weights, dtype=tf.float32)
    dense_shape = tf.constant(dense_shape, dtype=tf.int64)
    sp = tf.SparseTensor(sparse_indices, weights, dense_shape)
    # sp_transpose = tf.sparse.reorder(tf.sparse.transpose(sp, (1, 0)))
    sparse = sparse_sum(sp, axis=axis)
    sparse_grad, = tf.gradients(sparse, weights)

    seg = seg_sum(sp, axis=axis, ordered=ordered)
    seg_grad = tf.gradients(seg, weights)

    sparse2 = sparse_sum2(sp, axis=axis)
    # sparse2_grad, = tf.gradients(sparse2, weights)
    # print(sparse2_grad)
    names = []
    time = []
    mem = []

    # with tf.Session() as sess:
    #     print(sess.run(sp_transpose))
    # exit()

    def update(name, result):
        time.append(result['wall_time'])
        mem.append(result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])
        names.append(name)

    with tf.Session() as sess:
        print('------------------')
        print('----- SPARSE -----')
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (sparse, sparse_grad))
        update('sparse', result)
        # # no gradients to sparse2 - unfair comparison
        print('------------------')
        print('----- SPARSE2 -----')
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (sparse2,))
        update('sparse2', result)
        print('------------------')
        print('----- SEG ----')
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (seg, seg_grad))
        update('seg', result)

    time = np.array(time)
    i = np.argmin(time)
    best_time = time[i]
    print('Fastest:  {}, {}'.format(names[i], best_time))

    mem = np.array(mem)
    j = np.argmin(mem)
    best_mem = mem[j]
    print('Smallest: {}, {:.2f}mb'.format(names[j], best_mem / (1024**2)))

    print('rel time, rel mem, name')
    for name, t, m in zip(names, time, mem):
        print('{:.3f}, {:.3f}, {}'.format(t / best_time, m / best_mem, name))
    return time, mem, names


axis = 0
ordered = False
dense_shape = (int(1e4), int(1e6))
# compare(axis=axis, dense_shape=dense_shape, ordered=ordered)
run_benchmarks(axis=axis, dense_shape=dense_shape, ordered=ordered)
