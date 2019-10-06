"""
Benchmarking script for sparse convolutions.

Based on flex-conv profiling.

https://github.com/cgtuebingen/Flex-Convolution/blob/master/user_ops/profile_flexconv.py

See also:
https://fossies.org/linux/tensorflow/tensorflow/core/profiler/g3doc/python_api.md

Requires CUPTI on path
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64/
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from deep_cloud.models.sparse import ops

from absl import app
from absl import flags

flags.DEFINE_integer('seed', default=42, help='random seed')
flags.DEFINE_integer('batch_size', default=8, help='n_in/n_out multiplier')
flags.DEFINE_integer('neighbors', default=9, help='mean number of neighbors')
flags.DEFINE_integer('f_in', default=64, help='number of input channels')
flags.DEFINE_integer('f_out', default=64, help='number of output_channels')
flags.DEFINE_integer('n_in', default=4096, help='number of input points')
flags.DEFINE_integer('n_out', default=4096, help='number of output points')
flags.DEFINE_integer('n_edge', default=4, help='number of edge features')
flags.DEFINE_string('order_by', default='micros', help='"micros" or "bytes"')


def run_benchmark(seed=42,
                  batch_size=8,
                  mean_neighbors=9,
                  f_in=64,
                  f_out=64,
                  n_in=4096,
                  n_out=4096,
                  n_edge_features=4,
                  order_by='micros'):
    n_in *= batch_size
    n_out *= batch_size
    profile_dir = '/tmp/tf_profiling/fold_conv'
    # timeline_path = os.path.join(profile_dir, 'timeline.json')
    ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
    opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory())
    # opts = opts.with_step(0).with_timeline_output(timeline_path)
    # opts = opts.order_by('micros').build()
    opts = opts.order_by(order_by).build()

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

    with tf.contrib.tfprof.ProfileContext(profile_dir) as pctx:

        with tf.Session(config=tf.ConfigProto(
                log_device_placement=True)) as sess:
            # warmup
            for i in range(2):
                sess.run([forward_op, back_prop])

            # benchmark
            for i in range(10):
                pctx.trace_next_step()
                pctx.dump_next_step()
                sess.run([forward_op])
                pctx.profiler.profile_operations(options=opts)

            for i in range(10):
                pctx.trace_next_step()
                pctx.dump_next_step()
                sess.run([back_prop])
                pctx.profiler.profile_operations(options=opts)


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    def main(_):
        run_benchmark(
            seed=FLAGS.seed,
            batch_size=FLAGS.batch_size,
            mean_neighbors=FLAGS.neighbors,
            f_in=FLAGS.f_in,
            f_out=FLAGS.f_out,
            n_in=FLAGS.n_in,
            n_out=FLAGS.n_out,
            n_edge_features=FLAGS.n_edge,
            order_by=FLAGS.order_by,
        )

    app.run(main)
