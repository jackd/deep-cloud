from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import tensorflow as tf

from more_keras.ragged import batching as ragged_batching
from more_keras.framework.problems import get_current_problem
from more_keras.layers import utils as layer_utils
from more_keras.ops import utils as op_utils
from more_keras.ops import polynomials

from deep_cloud.meta_network import MetaNetworkBuilder
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import core

DEFAULT_TREE = pykd.KDTree

# get_nd_polynomials = gin.external_configurable(polynomials.get_nd_polynomials,
#                                                blacklist=['coords'])


@gin.configurable
def compute_edges_eager_fn(k0=16, tree_impl=DEFAULT_TREE):
    return functools.partial(compute_edges_eager, k0=k0, tree_impl=tree_impl)


def compute_edges_eager(coords, depth=5, k0=16, tree_impl=DEFAULT_TREE):
    coords = coords.numpy() if hasattr(coords, 'numpy') else coords
    tree = tree_impl(coords)
    dists, indices = tree.query(tree.data, 2, return_distance=True)
    del indices
    # closest = np.min(dists[:, 1])
    scale = np.mean(dists[:, 1])
    assert (scale > 0)
    coords *= (2 / scale)

    # coords is now a packing of barely-intersecting spheres of radius 1.
    all_coords = [coords]
    tree = tree_impl(coords)
    trees = [tree]

    radii = 4 * np.power(2, np.arange(depth))

    flat_indices = []
    row_splits = []
    rel_coords = []
    sample_indices = []

    # lines = ['---']

    def add_conv(tree, coords, radius, k0):
        indices = tree.query_ball_point(coords, radius, approx_neighbors=k0)
        rc = np.repeat(coords, indices.row_lengths,
                       axis=0) - coords[indices.flat_values]
        rc /= radius
        flat_indices.append(indices.flat_values)
        row_splits.append(indices.row_splits)
        rel_coords.append(rc)

        # n = tree.n
        # m = coords.shape[0]
        # e = indices.row_splits[-1]
        # lines.append(str((e, n, m, e / n, e / m, radius)))
        return indices

    # initial query in order to do initial rejection sample
    # indices = tree.query_ball_point(coords, radii[0], approx_neighbors=k0)
    indices = np.array(core.rejection_sample_active(tree, coords, radii[0], k0))
    # indices = np.array(core.rejection_sample_lazy(tree, coords, radii[0], k0))
    sample_indices.append(indices)
    out_coords = coords[indices]
    all_coords.append(out_coords)
    tree = tree_impl(out_coords)
    trees.append(tree)
    # initial large down-sample conv
    add_conv(tree, coords, radii[0] * 2, k0 * 4)
    coords = out_coords

    for i in range(1, depth - 1):
        # in place
        indices = add_conv(tree, coords, radii[i], k0)
        indices = np.array(core.rejection_sample_precomputed(indices),
                           dtype=np.int64)
        sample_indices.append(indices)
        out_coords = coords[indices]
        all_coords.append(out_coords)
        tree = tree_impl(out_coords)
        trees.append(tree)

        # downs sample
        add_conv(tree, coords, radii[i] * np.sqrt(2), k0)
        coords = out_coords

    # final in_place
    add_conv(tree, coords, radii[-1], k0)
    # lines.append('***')
    # print('\n'.join(lines))  # DEBUG
    return (
        tuple(flat_indices),
        tuple(rel_coords),
        tuple(row_splits),
        tuple(all_coords),
        tuple(sample_indices),
    )


def _flatten_output(fn, *args, **kwargs):
    return tf.nest.flatten(fn(*args, **kwargs))


def ragged_to_sparse_indices(rt, offset=None):
    if offset is not None:
        assert (rt.ragged_rank == 2)
        rt = (rt + offset).values
    assert (rt.ragged_rank == 1)
    assert (rt.dtype.is_integer)
    i = tf.repeat(tf.range(rt.nrows(), dtype=rt.dtype), rt.values)
    j = rt.values
    return tf.stack((i, j), axis=1)


@gin.configurable(blacklist=['feature_spec'])
def get_buiilder(feature_spec, depth=5, edge_fn=None):
    if edge_fn is None:
        edge_fn = compute_edges_eager_fn()
    builder = MetaNetworkBuilder()
    features = tf.nest.map_structure(builder.prebatch_input, feature_spec)
    if isinstance(features, dict):
        positions = features['positions']
    else:
        positions = features

    n_convs = 2 * (depth - 1)
    py_func_specs = [
        (tf.TensorSpec((None,), tf.int64),) * n_convs,  # flat_indices
        (tf.TensorSpec((None, 3), tf.float32),) * n_convs,  # flat_rel_coords
        # (tf.TensorSpec((None,), tf.float32),) * n_convs,  # feature_weights
        (
            tf.TensorSpec((None,), tf.int64),) * n_convs,  # row_splits
        (tf.TensorSpec((None, 3), tf.float32),) * depth,  # all_coords
        (tf.TensorSpec((None,), tf.int64),) * (depth - 1),  # sample_indices
    ]

    specs_flat = tf.nest.flatten(py_func_specs)
    fn = functools.partial(_flatten_output, edge_fn, depth=depth)
    out_flat = tf.py_function(fn, [positions])
    for out, spec in zip(out_flat, specs_flat):
        out.set_shape(spec.shape)

    (flat_indices, flat_rel_coords, row_splits, all_coords,
     sample_indices) = tf.nest.pack_sequence_as(py_func_specs, out_flat)
    neigh_indices = tf.nest.map_structure(
        lambda i, rs: builder.batched(tf.RaggedTensor.from_row_splits(i, rs)),
        flat_indices, row_splits)

    all_coords, sample_indices, rel_coords = tf.nest.map_structure(
        lambda x: builder.batched(x, ragged=True),
        (all_coords, sample_indices, flat_rel_coords))

    offsets = tf.nest.map_structure(
        lambda ac: tf.expand_dims(ac.row_starts(), axis=1), all_coords[:-1])

    sample_indices = tf.nest.map_structure(lambda i, o: (i + o).flat_values,
                                           sample_indices, offsets)

    offsets = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=-1),
                                    offsets)
    in_place_indices = tf.nest.map_structure(ragged_to_sparse_indices,
                                             neigh_indices[1::2], offsets)
    down_sample_indices = tf.nest.map_structure(ragged_to_sparse_indices,
                                                neigh_indices[::2], offsets)

    all_coords, rel_coords = tf.nest.map_structure(lambda x: x.flat_values,
                                                   (all_coords, rel_coords))
    raise NotImplementedError('TODO?')
    # return builder
