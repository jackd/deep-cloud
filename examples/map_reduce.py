from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

num_points = int(1e4)
nf = 16
nd = 3
units = 16
radius = 4

layer = tf.keras.layers.Dense(units)
layer.build((None, nf + nd))


def get_fn(features, coords):

    def fn(args):
        indices, invalid, coord = args
        rel_coords = coord - tf.gather(coords, indices)
        # rel_coords = layer(rel_coords)
        gathered = tf.gather(features, indices)
        gathered = tf.concat((gathered, rel_coords), axis=-1)
        # gathered = layer(gathered)
        return tf.reduce_sum(tf.where(invalid, tf.zeros_like(gathered),
                                      gathered),
                             axis=0)

    return fn


def map_reduce(features, coords, indices):
    indices_rect = indices.to_tensor(-1)
    invalid = tf.equal(indices_rect, -1)
    return tf.map_fn(get_fn(features, coords), (indices_rect, invalid, coords),
                     features.dtype)


def vectorized_map_reduce(features, coords, indices):
    indices_rect = indices.to_tensor(-1)
    invalid = tf.equal(indices_rect, -1)
    return tf.vectorized_map(get_fn(features, coords),
                             (indices_rect, invalid, coords))


def map_reduce_naive(features, coords, indices):
    rel_coords = tf.repeat(coords, indices.row_lengths(), axis=0) - tf.gather(
        coords, indices.flat_values)
    gathered = tf.gather(features, indices.flat_values)
    gathered = tf.concat((gathered, rel_coords), axis=-1)
    gathered = tf.RaggedTensor.from_row_splits(gathered, indices.row_splits)
    return tf.reduce_sum(gathered, axis=1)


# create points on a sphere
coords = np.random.normal(size=(num_points, nd)).astype(np.float32)
coords /= np.linalg.norm(coords, axis=-1, keepdims=True)
tree = cKDTree(coords)
dists, indices = tree.query(coords, 2)
scale_factor = np.mean(dists[:, -1])
coords /= scale_factor
tree = cKDTree(coords)
indices = tree.query_ball_point(coords, radius)
consumed = np.zeros((num_points,), dtype=np.bool)
out = []
for i in range(num_points):
    if not consumed[i]:
        consumed[indices[i]] = True
        out.append(i)
indices = np.array(out)
coords = coords[indices]
indices = cKDTree(coords)
indices = tree.query_ball_point(coords, radius)
row_lengths = [len(i) for i in indices]
flat_indices = np.concatenate(indices, axis=0)
features = np.random.normal(size=(num_points, nf)).astype(np.float32)


def run_benchmark(op_fn):
    features_tf = tf.constant(features, dtype=tf.float32)
    flat_indices_tf = tf.constant(flat_indices, dtype=tf.int32)
    row_lengths_tf = tf.constant(row_lengths, dtype=tf.int32)
    indices_tf = tf.RaggedTensor.from_row_lengths(flat_indices_tf,
                                                  row_lengths_tf)
    coords_tf = tf.constant(coords, dtype=tf.float32)

    out = op_fn(features_tf, coords_tf, indices_tf)
    grads = tf.gradients(out, [features_tf, layer.kernel])
    grads = [g for g in grads if g is not None]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        bm = tf.test.Benchmark()
        bm.run_op_benchmark(sess, [out, *grads])


def verify_values():
    features_tf = tf.constant(features, dtype=tf.float32)
    flat_indices_tf = tf.constant(flat_indices, dtype=tf.int32)
    row_lengths_tf = tf.constant(row_lengths, dtype=tf.int32)
    indices_tf = tf.RaggedTensor.from_row_lengths(flat_indices_tf,
                                                  row_lengths_tf)
    coords_tf = tf.constant(coords, dtype=tf.float32)

    vmr = vectorized_map_reduce(features_tf, coords_tf, indices_tf)
    naive = map_reduce_naive(features_tf, coords_tf, indices_tf)
    diff = tf.reduce_max(tf.abs(vmr - naive))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(diff))


# run_benchmark(map_reduce)
run_benchmark(vectorized_map_reduce)
run_benchmark(map_reduce_naive)
verify_values()
