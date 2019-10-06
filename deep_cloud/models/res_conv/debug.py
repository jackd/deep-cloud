from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from time import time

r = np.random.RandomState(123)

F = 64
mean_edges = 9

# small
N = 40960

# # big
# N = 41287

num_edges = int(mean_edges * N)

flat_index = r.randint(0, high=N * N, size=num_edges, dtype=np.int64)
flat_index = np.sort(flat_index)
i, j = np.unravel_index(flat_index, (N, N))  # pylint: disable=unbalanced-tuple-unpacking

features = np.random.uniform(size=(N, F)).astype(np.float32)
kernel = np.random.uniform(size=(F, F)).astype(np.float32)
features = np.matmul(features, kernel)

features = tf.constant(features, dtype=tf.float32)
i = tf.constant(i, dtype=tf.int64)
j = tf.constant(j, dtype=tf.int64)
sparse_indices = (i, j)


def super_simple(features, sparse_indices):
    """Simplified version of unstack_sum."""
    i, j = sparse_indices
    features = tf.gather(features, j)
    features = tf.segment_sum(features, i)
    return features


out = super_simple(features, sparse_indices)
grads = tf.gradients(out, (features,))

with tf.Session() as sess:
    sess.run((out, grads))
    t = time()
    result = tf.test.Benchmark().run_op_benchmark(sess, (out, grads))
    dt = time() - t

print('dt:         {} ms'.format(dt * 1000))
print('Time:       {} ms'.format(result['wall_time'] * 1000))
print('GPU memory: {} Mb'.format(
    result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'] / (1024**2)))
