from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin

from deep_cloud.model_builder import PipelineBuilder
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import core

DEFAULT_TREE = pykd.KDTree
SQRT_2 = np.sqrt(2.)


@gin.configurable(
    blacklist=['builder', 'py_func', 'features', 'tree', 'coords', 'radius'])
def get_conv_args(builder, py_func, tree, coords, radius, k0, in_row_starts,
                  total_out_size):
    # from deep_cloud.model_builder import PyFuncBuilder
    # py_func = PyFuncBuilder()

    def fn(tree, coords):
        indices = tree.query_ball_point(coords, radius, approx_neighbors=k0)
        rc = np.repeat(coords, indices.row_lengths,
                       axis=0) - coords[indices.flat_values]
        rc /= radius
        return rc, indices.flat_values, indices.row_lengths

    rc, flat_indices, row_lengths = py_func.output_tensor(
        py_func.node(fn, tree, coords),
        (tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None,), dtype=tf.int64),
         tf.TensorSpec(shape=(None,), dtype=tf.int64)))

    rc = builder.trained_input(builder.batch(rc, ragged=True).flat_values)
    flat_indices = builder.batch(flat_indices, ragged=True)
    flat_indices = flat_indices + tf.expand_dims(in_row_starts, axis=1)
    flat_indices = flat_indices.flat_values
    row_lengths = builder.batch(row_lengths, ragged=True).flat_values
    i = tf.repeat(tf.range(total_out_size, dtype=tf.int64), row_lengths)
    j = flat_indices
    sparse_indices = tf.stack((i, j), axis=0)
    return sparse_indices, rc


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def classifier_pipeline(input_spec,
                        output_spec,
                        depth=4,
                        tree_impl=DEFAULT_TREE,
                        k0=16):
    pipeline = PipelineBuilder()
    py_func = pipeline.py_func_builder('pre_batch')
    assert (isinstance(input_spec, tf.TensorSpec))
    coords = pipeline.pre_batch_input(input_spec)

    coords = py_func.input_node(coords)

    def f(coords):
        tree = tree_impl(coords)
        dists = tree.query(tree.data, 2, return_distance=True)[0]
        coords *= 2 / np.mean(dists[:, 1])
        return coords

    coords = py_func.node(f, coords)
    tree = py_func.input_node(tree_impl, coords)

    radii = 4 * np.power(2, np.arange(depth))

    # initial query in order to do initial rejection sample
    indices = py_func.node(
        lambda tree, coords: core.rejection_sample_active(
            tree, coords, radii[0], k0), tree, coords)

    out_coords = py_func.node(lambda coords, indices: coords[indices], coords,
                              indices)
    tree = py_func.node(tree_impl, out_coords)

    # initial large down-sample conv
    raise NotImplementedError('TODO')

    for i in range(1, depth - 1):
        raise NotImplementedError('TODO')

    pipeline.finalize()
    return pipeline
