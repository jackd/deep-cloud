from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_data(dense_shape=(int(1e3), int(1e4)), mean_edges=100, nf=16, seed=123):
    N, M = dense_shape
    r = np.random.RandomState(seed)

    num_edges = int(mean_edges * N)

    flat_index = (r.uniform(size=num_edges) * N * M).astype(np.int64)
    flat_index = np.concatenate([flat_index, np.arange(0, N * M, M)], axis=0)
    flat_index = np.array(sorted(set(flat_index)), dtype=np.int64)
    i, j = np.unravel_index(flat_index, (N, M))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = np.stack((i, j), axis=-1)
    unique, counts = np.unique(i, return_counts=True)
    row_lengths = np.zeros((N,), dtype=np.int64)
    row_lengths[unique] = counts
    row_splits = np.concatenate([[0], np.cumsum(row_lengths)], axis=0)
    features = np.random.uniform(size=(M, nf)).astype(np.float32)
    weights = np.random.uniform(size=(i.shape[0],)).astype(np.float32)
    ragged_indices = j
    return sparse_indices, ragged_indices, row_splits, features, weights


def sparse_sum(indices, weights, features, dense_shape):
    sp_weights = tf.SparseTensor(indices, weights, dense_shape=dense_shape)
    values = tf.sparse.sparse_dense_matmul(sp_weights, features)
    # norm_factor = tf.sparse.reduce_sum(weights, axis=1, keepdims=True)
    norm_factor = tf.expand_dims(tf.segment_sum(weights, indices[:, 0]),
                                 axis=-1)
    return values / norm_factor


def sparse_sum2(indices, weights, features, dense_shape):
    sp_weights = tf.SparseTensor(indices, weights, dense_shape=dense_shape)
    values = tf.matmul(tf.sparse.to_dense(sp_weights), features)
    # norm_factor = tf.sparse.reduce_sum(weights, axis=1, keepdims=True)
    norm_factor = tf.expand_dims(tf.segment_sum(weights, indices[:, 0]),
                                 axis=-1)
    return values / norm_factor


def ragged_sum(flat_indices, row_splits, weights, features):
    features = tf.gather(features, flat_indices)
    features = features * tf.expand_dims(weights, axis=1)
    segment_ids = tf.repeat(tf.range(tf.size(row_splits) - 1),
                            row_splits[1:] - row_splits[:-1],
                            axis=0)
    values = tf.math.segment_sum(features, segment_ids)
    return values / tf.expand_dims(tf.math.segment_sum(weights, segment_ids),
                                   axis=-1)
    # return tf.reduce_sum(tf.RaggedTensor.from_row_splits(features, row_splits),
    #                      axis=1)


def embedding_sum(indices, weights, features, dense_shape):
    sp_ids = tf.SparseTensor(indices,
                             tf.range(tf.size(weights)),
                             dense_shape=dense_shape)
    sp_weights = tf.SparseTensor(indices, weights, dense_shape=dense_shape)
    return tf.nn.embedding_lookup_sparse(features,
                                         sp_ids,
                                         sp_weights,
                                         combiner='sum')


def compare(dense_shape=(int(1e3), int(1e4)), **kwargs):
    sparse_indices, ragged_indices, row_splits, features, weights = get_data(
        dense_shape, **kwargs)
    features = tf.constant(features, dtype=tf.float32)
    ragged_indices = tf.constant(ragged_indices, dtype=tf.int64)
    row_splits = tf.constant(row_splits, dtype=tf.int64)
    weights = tf.constant(weights, dtype=tf.float32)
    sparse_indices = tf.constant(sparse_indices, dtype=tf.int64)

    sp = sparse_sum(sparse_indices, weights, features, dense_shape)
    rag = ragged_sum(ragged_indices, row_splits, weights, features)
    err = tf.reduce_max(sp - rag)
    with tf.Session() as sess:
        err = sess.run(err)
    print(err)
    return err


def run_benchmarks(dense_shape=(int(1e3), int(1e4)), **kwargs):
    sparse_indices, ragged_indices, row_splits, features, weights = get_data(
        dense_shape, **kwargs)
    features = tf.constant(features, dtype=tf.float32)
    ragged_indices = tf.constant(ragged_indices, dtype=tf.int64)
    row_splits = tf.constant(row_splits, dtype=tf.int64)
    weights = tf.constant(weights, dtype=tf.float32)
    sparse_indices = tf.constant(sparse_indices, dtype=tf.int64)

    sp = sparse_sum(sparse_indices, weights, features, dense_shape)
    sp_grads = tf.gradients(sp, (features, weights))
    sp2 = sparse_sum2(sparse_indices, weights, features, dense_shape)
    sp2_grads = tf.gradients(sp2, (features, weights))
    # emb = embedding_sum(sparse_indices, weights, features, dense_shape)
    # emb_grads = tf.gradients(emb, (features_tf, weights_tf))
    rag = ragged_sum(ragged_indices, row_splits, weights, features)
    rag_grads = tf.gradients(rag, (features, weights))
    names = []
    time = []
    mem = []

    def update(name, result):
        time.append(result['wall_time'])
        mem.append(result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])
        names.append(name)

    with tf.Session() as sess:
        print('------------------')
        print('----- SPARSE -----')
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (sp, sp_grads))
        update('sparse', result)
        # print('------------------')
        # print('--- EMBEDDING ----')
        # bm = tf.test.Benchmark()
        # bm.run_op_benchmark(sess, (emb, emb_grads))
        print('------------------')
        print('----- SPARSE2 ----')
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (sp2, sp2_grads))
        update('dense', result)
        print('------------------')
        print('----- RAGGED -----')
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (rag, rag_grads))
        update('ragged', result)

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


# def run_benchmark(op_fn):
#     features_tf = tf.constant(features, dtype=tf.float32)
#     flat_indices = tf.constant(indices.flat_values, dtype=tf.int64)
#     row_splits = tf.constant(indices.row_splits, dtype=tf.int64)
#     flat_weights = tf.constant(weights, dtype=tf.float32)

#     op = op_fn(flat_weights, flat_indices, row_splits, features_tf)
#     grads = tf.gradients(op, features_tf)
#     with tf.Session() as sess:
#         bm = tf.test.Benchmark()
#         bm.run_op_benchmark(sess, (op, grads))

# run_benchmark(sparse_mean)
# run_benchmark(ragged_mean)
# print(np.mean(indices.row_lengths))
# N = 64
M = int(1e3)
mean_edges = 100
times = []
mems = []
# Ns = (16, 64, 256, 1024, 2048, 1024 * 3, 4096)
# mean_edges = 100
# x = Ns
N = int(1e4)
mean_edges = (4, 8, 12, 16, 32, 64, 128)
x = mean_edges
# for N in Ns:
for me in mean_edges:
    time, mem, names = run_benchmarks(dense_shape=(N, M), mean_edges=me, nf=16)
    times.append(time)
    mems.append(mem)

times = np.array(times).T
mems = np.array(mems).T

import matplotlib.pyplot as plt


def vis(x, times, mems, names, normalize=True):
    assert (len(times) == len(names) == len(mems))
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle('normalize = {}'.format(normalize))

    if normalize:
        times = times / times[:, :1]
        mems = mems / mems[:, :1]

    for (time, mem, name) in zip(times, mems, names):
        ax0.plot(x, time, label=name)
        ax1.plot(x, mem, label=name)

    ax0.set_title('time')
    ax0.legend()
    ax1.set_title('mem')
    ax1.legend()

    # for ax in (ax0, ax1):
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')


vis(x, times, mems, names, normalize=True)
vis(x, times, mems, names, normalize=False)
compare()
plt.show()
