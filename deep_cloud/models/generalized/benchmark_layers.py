from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from deep_cloud.problems.partnet import PartnetProblem
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils.core import rejection_sample_active

from deep_cloud.models.generalized import layers
# from tensorflow.contrib.compiler import xla


def in_place_conv_benchmark(coords):
    tree = pykd.KDTree(coords)

    indices = tree.query_ball_point(coords, 4, approx_neighbors=16)

    fi = 64
    units = 64
    c0 = coords.shape[0]

    coord_features = (np.repeat(coords, indices.row_lengths, axis=0) -
                      coords[indices.flat_values])
    features = np.random.uniform(size=(c0, fi))

    node_features = tf.constant(features, dtype=tf.float32)
    coord_features = tf.constant(coord_features, dtype=tf.float32)
    flat_indices = tf.constant(indices.flat_values, dtype=tf.int64)
    row_splits = tf.constant(indices.row_splits, dtype=tf.int64)

    conv = layers.RaggedConvolution(units)
    args = [node_features, coord_features, flat_indices, row_splits]
    conv0 = conv(args)
    conv0_back, = tf.gradients(conv0, node_features)
    conv1 = layers.ragged_convolution(conv.layer,
                                      None,
                                      *args,
                                      gather_first=True)
    conv1_back, = tf.gradients(conv1, node_features)
    args = [node_features, -coord_features, flat_indices, row_splits]
    conv2 = layers.ragged_convolution_transpose(conv.layer, None, *args)
    conv2_back, = tf.gradients(conv2, node_features)

    # [conv_comp] = xla.compile(lambda args: conv(args), inputs=[args])

    # conv3 = layers.ragged_convolution_transpose(conv.layer,
    #                                             None,
    #                                             *args,
    #                                             gather_first=True)
    # conv3_back, = tf.gradients(conv3, node_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        bm = tf.test.Benchmark()
        print('base')
        bm.run_op_benchmark(sess, (conv0, conv0_back))
        print('gather_first')
        bm.run_op_benchmark(sess, (conv1, conv1_back))
        print('transpose')
        bm.run_op_benchmark(sess, (conv2, conv2_back))
        # print('base_no_grad')
        # bm.run_op_benchmark(sess, conv0)
        # print('base_xla')
        # bm.run_op_benchmark(sess, conv_comp)
        # print('transpose_gather_first')
        # bm.run_op_benchmark(sess, (conv3, conv3_back))
        print(sess.run(tf.reduce_max(tf.abs(conv0 - conv2))))

        conv = layers.RaggedConvolution(units)


def subsample_conv_benchmark(coords):
    tree = pykd.KDTree(coords)
    # subsample
    indices = rejection_sample_active(tree, coords, 4, 16)
    out_coords = coords[indices]
    tree = pykd.KDTree(out_coords)

    indices = tree.query_ball_point(coords, 4, approx_neighbors=16)
    coord_features = (np.repeat(coords, indices.row_lengths, axis=0) -
                      out_coords[indices.flat_values])

    fi = 64
    units = 64
    c0 = coords.shape[0]

    features = np.random.uniform(size=(c0, fi))

    node_features = tf.constant(features, dtype=tf.float32)
    coord_features = tf.constant(coord_features, dtype=tf.float32)
    flat_indices = tf.constant(indices.flat_values, dtype=tf.int64)
    row_splits = tf.constant(indices.row_splits, dtype=tf.int64)

    conv = layers.RaggedConvolution(units)

    args = [node_features, coord_features, flat_indices, row_splits]
    conv0 = conv(args)
    conv0_back, = tf.gradients(conv0, node_features)
    conv1 = layers.ragged_convolution(conv.layer,
                                      None,
                                      *args,
                                      gather_first=True)
    conv1_back, = tf.gradients(conv1, node_features)
    args = [node_features, -coord_features, flat_indices, row_splits]
    # conv2 = layers.ragged_convolution_transpose(conv.layer, None, *args)
    # conv2_back, = tf.gradients(conv2, node_features)

    # conv3 = layers.ragged_convolution_transpose(conv.layer,
    #                                             None,
    #                                             *args,
    #                                             gather_first=True)
    # conv3_back, = tf.gradients(conv3, node_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        bm = tf.test.Benchmark()
        print('base')
        bm.run_op_benchmark(sess, (conv0, conv0_back))
        print('gather_first')
        bm.run_op_benchmark(sess, (conv1, conv1_back))
        # bm.run_op_benchmark(sess, conv2_back)
        # bm.run_op_benchmark(sess, conv3_back)
        # print(sess.run(tf.reduce_max(tf.abs(conv0 - conv2))))


def grid_conv_benchmark():
    grid_features = tf.Variable(np.random.normal(size=(8, 64, 64, 64)),
                                dtype=tf.float32)
    grid_conv = tf.keras.layers.Conv2D(64, 3)(grid_features)
    grid_grad, = tf.gradients(grid_conv, grid_features)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        bm = tf.test.Benchmark()
        bm.run_op_benchmark(sess, (grid_conv, grid_grad))


for coords, labels in tfds.as_numpy(
        PartnetProblem().get_base_dataset(split='validation').take(1)):
    break

coords = coords[:4096]

tree = pykd.KDTree(coords)
dists, indices = tree.query(tree.data, 2, return_distance=True)
scale = np.mean(dists[:, 1])
assert (scale > 0)
coords *= (2 / scale)

print('in_place')
in_place_conv_benchmark(coords)
print('subsample')
subsample_conv_benchmark(coords)
print('grid')
grid_conv_benchmark()
