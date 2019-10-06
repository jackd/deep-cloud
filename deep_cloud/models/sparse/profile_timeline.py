"""
Benchmarking script for sparse convolutions.

Based on flex-conv profiling.

https://github.com/cgtuebingen/Flex-Convolution/blob/master/user_ops/profile_flexconv.py

See also:
https://fossies.org/linux/tensorflow/tensorflow/core/profiler/g3doc/python_api.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from deep_cloud.models.sparse import ops


def run_benchmark(seed=42,
                  batch_size=8,
                  mean_neighbors=9,
                  f_in=64,
                  f_out=64,
                  n_in=4096,
                  n_out=4096,
                  n_edge_features=4):
    n_in *= batch_size
    n_out *= batch_size
    profile_dir = '/tmp/tf_profiling/fold_conv'
    if not os.path.isdir(profile_dir):
        os.makedirs(profile_dir)

    timeline_path = os.path.join(profile_dir, 'timeline.json')

    np.random.seed(seed)
    tf.set_random_seed(seed)

    num_edges = int(n_in * mean_neighbors)

    flat_index = np.random.randint(0,
                                   high=n_in * n_out,
                                   size=num_edges,
                                   dtype=np.int64)
    flat_index = np.unique(flat_index)
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(flat_index, (n_out, n_in))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = np.stack((i, j), axis=-1)
    edge_weights = np.random.uniform(size=(n_edge_features,
                                           i.shape[0])).astype(np.float32)
    features = np.random.uniform(size=(n_in, f_in)).astype(np.float32)
    kernel = np.random.uniform(size=(n_edge_features, f_in,
                                     f_out)).astype(np.float32)

    sparse_indices = tf.convert_to_tensor(sparse_indices, dtype=tf.int64)
    edge_weights = tf.convert_to_tensor(edge_weights, dtype=tf.float32)
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
    dense_shape = tuple(
        tf.convert_to_tensor(n, tf.int64) for n in (n_out, n_in))

    forward_op = ops.fold_conv(features, kernel, sparse_indices, edge_weights,
                               dense_shape)
    back_prop = tf.gradients(forward_op, (features, kernel))

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # warmup
        for i in range(10):
            sess.run([forward_op, back_prop])

        run_metadata = tf.RunMetadata()
        sess.run(back_prop,
                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata)

    opts = tf.profiler.ProfileOptionBuilder(
        tf.profiler.ProfileOptionBuilder.time_and_memory())

    tf.profiler.profile(tf.get_default_graph(),
                        run_meta=run_metadata,
                        cmd='op',
                        options=opts.build())

    tf.profiler.profile(tf.get_default_graph(),
                        run_meta=run_metadata,
                        cmd='code',
                        options=opts.build())

    opts = opts.with_step(0).with_timeline_output(timeline_path)
    tf.profiler.profile(tf.get_default_graph(),
                        run_meta=run_metadata,
                        cmd='code',
                        options=opts.build())


if __name__ == '__main__':
    run_benchmark()
