from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from time import time


def sparse_mul(features, indices, values, dense_shape):
    if isinstance(indices, (list, tuple)):
        indices = tf.stack(indices, axis=-1)
    sp = tf.SparseTensor(indices, values, dense_shape)
    return tf.sparse.sparse_dense_matmul(sp, features)


def gather_sum(features, indices, values, dense_shape):
    if isinstance(indices, tf.Tensor):
        indices = tf.unstack(indices, axis=-1)
    i, j = indices
    features = tf.gather(features, j) * tf.expand_dims(values, axis=-1)
    features = tf.math.segment_sum(features, i)
    return features


def sparse_conv(features, indices, v0, x_in, x_out, mul_impl=sparse_mul):
    if isinstance(indices, (list, tuple)):
        i, j = indices
        indices = tf.stack((i, j), axis=-1)
    else:
        i, j = tf.unstack(indices, axis=-1)
    dense_shape = tf.shape(x_out)[0], tf.shape(x_in)[0]
    values = v0 * tf.gather(x_in, j) * tf.gather(x_out, i)
    return mul_impl(features, indices, values, dense_shape)


def feature_conv(features, indices, v0, x_in, x_out, mul_impl=sparse_mul):
    if isinstance(indices, (list, tuple)):
        i, j = indices
        indices = tf.stack((i, j), axis=-1)
    else:
        i, j = tf.unstack(indices, axis=-1)
    dense_shape = tf.shape(x_out)[0], tf.shape(x_in)[0]
    features = features * tf.expand_dims(x_in, axis=-1)
    features = mul_impl(features, indices, v0, dense_shape)
    features = features * tf.expand_dims(x_out, axis=-1)
    return features


r = np.random.RandomState(123)

# include_grads = False
include_grads = True
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

features = r.uniform(size=(N, F)).astype(np.float32)
x_in = r.uniform(size=(N,)).astype(np.float32)
x_out = r.uniform(size=(N,)).astype(np.float32)
kernel = r.uniform(size=(F, F)).astype(np.float32)
v0 = r.uniform(size=i.shape).astype(np.float32)
features = np.matmul(features, kernel)

# features = tf.constant(features, dtype=tf.float32)
features = tf.Variable(features, dtype=tf.float32)
i = tf.constant(i, dtype=tf.int64)
j = tf.constant(j, dtype=tf.int64)
indices = (i, j)
v0 = tf.constant(v0, dtype=tf.float32)
x_in = tf.constant(x_in, dtype=tf.float32)
x_out = tf.constant(x_out, dtype=tf.float32)

names, fns = zip(*(
    ('sparse_sp', functools.partial(sparse_conv, mul_impl=sparse_mul)),
    ('sparse_gs', functools.partial(sparse_conv, mul_impl=gather_sum)),
    ('feature_sp', functools.partial(feature_conv, mul_impl=sparse_mul)),
    ('feature_gs', functools.partial(feature_conv, mul_impl=gather_sum)),
))

init_ops = []
train_ops = []
for fn in fns:
    f = fn(features, indices, v0, x_in, x_out)
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    init_ops.append(features.initializer)
    train_ops.append(optimizer.minimize(f, var_list=[features]))

times = []
dts = []
memories = []

with tf.Session() as sess:
    for i, t in zip(init_ops, train_ops):
        sess.run(i)
        for _ in range(5):
            sess.run(t)
        bm = tf.test.Benchmark()
        t_start = time()
        result = bm.run_op_benchmark(sess, t)
        dts.append(time() - t_start)
        times.append(result['wall_time'])
        memories.append(
            result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])

times = np.array(times) * 1000  # ms
ti = np.argmin(times)
tmin = times[ti]
print('Best time: {}, {} ms'.format(names[ti], tmin))
print('rel times:')
for name, t in zip(names, times):
    print('{:15s} {:.3f} {:.3f}'.format(name, t / tmin, t))

dts = np.array(dts) * 1000  # ms
dti = np.argmin(dts)
dtmin = dts[dti]
print('Best dt: {}, {} ms'.format(names[dti], dtmin))
print('dts:')
for name, dt in zip(names, dts):
    print('{:15s} {:.3f} {:.3f}'.format(name, dt / dtmin, dt))

memories = np.array(memories) / 1024**2  # Mb
mi = np.argmin(memories)
mmin = memories[mi]
print('Best memory: {}, {} Mb'.format(names[mi], mmin))
print('Memory usage')
for name, memory in zip(names, memories):
    print('{:15s} {:.03f} {}'.format(name, memory / mmin, memory))
