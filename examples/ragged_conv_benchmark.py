from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from more_keras.ops import utils as op_utils

from deep_cloud.problems.partnet import PartnetProblem
from deep_cloud.ops.np_utils.tree_utils import pykd

for coords, labels in tfds.as_numpy(
        PartnetProblem().get_base_dataset().take(1)):
    break

tree = pykd.KDTree(coords)
dists, indices = tree.query(tree.data, 2, return_distance=True)
scale = np.mean(dists[:, 1])
assert (scale > 0)
coords *= (2 / scale)

tree = pykd.KDTree(coords)
indices = tree.query_ball_point(coords, 4, approx_neighbors=16)

c0 = coords.shape[0]
# nf = 64
nf = 16
nd = 3
features = np.random.uniform(size=(c0, nf, nd))


def run_benchmark(op_fn):
    in_coords = tf.constant(coords, dtype=tf.float32)
    out_coords = in_coords
    features_tf = tf.constant(features, dtype=tf.float32)
    flat_indices = tf.constant(indices.flat_values, dtype=tf.int64)
    row_splits = tf.constant(indices.row_splits, dtype=tf.int64)

    op = op_fn(in_coords, out_coords, features_tf, flat_indices, row_splits)
    with tf.Session() as sess:
        bm = tf.test.Benchmark()
        bm.run_op_benchmark(sess, op)


# run_benchmark(scale_reduce_broadcast)
# run_benchmark(gather_reduce)
