from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
import gin

from more_keras.ragged import batching as ragged_batching
from more_keras.framework.problems import get_current_problem
from more_keras.layers import utils as layer_utils
from more_keras.ops import utils as op_utils

from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import core

DEFAULT_TREE = pykd.KDTree
SQRT_2 = np.sqrt(2.)


@gin.configurable
def compute_edges_eager_fn(k0=16, tree_impl=DEFAULT_TREE):
    return functools.partial(compute_edges_eager, k0=k0, tree_impl=tree_impl)


def compute_edges_eager(coords, depth=6, k0=16, tree_impl=DEFAULT_TREE):
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

        # down sample
        # larger radius means number of edges remains roughly constant
        # number of ops remains constant if number of filters doubles
        # also makes more neighbors for unsampling (~4 on average vs ~2)
        add_conv(tree, coords, radii[i] * SQRT_2, k0)
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


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def pre_batch_map(features,
                  labels,
                  weights=None,
                  depth=5,
                  shuffle=True,
                  edge_fn=None):
    if isinstance(features, dict):
        positions = features['positions']
        normals = features.get('normals')
    else:
        positions = features
        normals = None
    if edge_fn is None:
        edge_fn = compute_edges_eager_fn()

    n_convs = 2 * depth - 2
    specs = [
        (tf.TensorSpec((None,), tf.int64),) * n_convs,  # flat_indices
        (tf.TensorSpec((None, 3), tf.float32),) * n_convs,  # flat_rel_coords
        # (tf.TensorSpec((None,), tf.float32),) * n_convs,  # feature_weights
        (
            tf.TensorSpec((None,), tf.int64),) * n_convs,  # row_splits
        (tf.TensorSpec((None, 3), tf.float32),) * depth,  # all_coords
        (tf.TensorSpec((None,), tf.int64),) * (depth - 1),  # sample_indices
    ]

    specs_flat = tf.nest.flatten(specs)

    fn = functools.partial(_flatten_output, edge_fn, depth=depth)
    out_flat = tf.py_function(fn, [positions], [s.dtype for s in specs_flat])
    for out, spec in zip(out_flat, specs_flat):
        out.set_shape(spec.shape)
    out = tf.nest.pack_sequence_as(specs, out_flat)
    # (flat_node_indices, flat_rel_coords, edge_weights, row_splits, all_coords,
    #  sample_indices) = out
    (flat_node_indices, flat_rel_coords, row_splits, all_coords,
     sample_indices) = out
    # flat_node_indices, row_splits, all_coords, sample_indices = out

    all_coords, sample_indices = tf.nest.map_structure(
        ragged_batching.pre_batch_ragged, (all_coords, sample_indices))

    node_indices = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
                                         flat_node_indices, row_splits)
    rel_coords = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
                                       flat_rel_coords, row_splits)

    # edge_weights = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
    #                                      edge_weights, row_splits)
    features = dict(
        all_coords=all_coords,
        rel_coords=rel_coords,
        # edge_weights=edge_weights,
        node_indices=node_indices,
        sample_indices=sample_indices,
    )
    if normals is not None:
        features['normals'] = ragged_batching.pre_batch_ragged(normals)

    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def post_batch_map(features, labels, weights=None, return_coords=False):
    all_coords, rel_coords, node_indices, sample_indices = (
        features[k]
        for k in ('all_coords', 'rel_coords', 'node_indices', 'sample_indices'))
    depth = len(all_coords)

    all_coords, sample_indices = tf.nest.map_structure(
        ragged_batching.post_batch_ragged, (all_coords, sample_indices))
    row_splits = tf.nest.map_structure(lambda ac: ac.row_splits, all_coords)
    # sample indices
    offsets = tf.nest.map_structure(
        lambda x: tf.expand_dims(x.row_starts(), axis=1), all_coords[1:])
    sample_indices = tf.nest.map_structure(lambda si, o: (si + o).values,
                                           sample_indices, offsets)

    # neighbor indices
    offsets = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=-1),
                                    offsets)

    def ragged_to_sparse_indices(rt, offset=None):
        if offset is not None:
            assert (rt.ragged_rank == 2)
            rt = (rt + offset).values
        assert (rt.ragged_rank == 1)
        assert (rt.dtype.is_integer)
        i = tf.repeat(tf.range(rt.nrows(), dtype=rt.dtype),
                      rt.row_lengths(),
                      axis=0)
        return tf.stack((i, rt.values), axis=-1)

    down_sample_indices = tf.nest.map_structure(ragged_to_sparse_indices,
                                                node_indices[::2], offsets)
    in_place_indices = tf.nest.map_structure(ragged_to_sparse_indices,
                                             node_indices[1::2], offsets)

    sizes = tuple(tf.shape(ac.flat_values)[0] for ac in all_coords)

    def up_sample_args(down_sample_indices, dense_shape):
        values = tf.range(tf.shape(down_sample_indices, out_type=tf.int64)[0])
        sp = tf.SparseTensor(down_sample_indices, values, dense_shape)
        sp = tf.sparse.reorder(tf.sparse.transpose(sp, (1, 0)))
        return sp.indices, sp.values

    dense_shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    up_sample_indices, up_sample_perms = zip(*(
        up_sample_args(dsi, ds)
        for dsi, ds in zip(down_sample_indices, dense_shapes)))

    # rel_coords
    flat_rel_coords = tf.nest.map_structure(
        lambda x: tf.transpose(x.flat_values, (1, 0)), rel_coords)

    down_sample_rel_coords = flat_rel_coords[::2]
    in_place_rel_coords = flat_rel_coords[1::2]

    assert (len(all_coords) == depth)
    assert (len(sample_indices) == depth - 1)
    assert (len(in_place_indices) == depth - 1)
    assert (len(down_sample_indices) == depth - 1)
    assert (len(up_sample_indices) == depth - 1)
    assert (len(up_sample_perms) == depth - 1)
    assert (len(in_place_rel_coords) == depth - 1)
    assert (len(down_sample_indices) == depth - 1)

    features = dict(
        # all_coords=all_coords,
        sample_indices=sample_indices,
        in_place_indices=in_place_indices,
        in_place_rel_coords=in_place_rel_coords,
        down_sample_indices=down_sample_indices,
        down_sample_rel_coords=down_sample_rel_coords,
        up_sample_indices=up_sample_indices,
        up_sample_perms=up_sample_perms,
        row_splits=row_splits,
        # edge_weights=edge_weights,
    )
    if return_coords:
        features['all_coords'] = all_coords

    problem = get_current_problem()
    if problem is not None:
        labels, weights = problem.post_batch_map(labels, weights)

    return ((features, labels) if weights is None else
            (features, labels, weights))

    # in_place = tuple(
    #     dict(sparse_indices=si,
    #          rel_coords=rc,
    #          dense_shape=(s, s),
    #          edge_weights=ew)
    #     for si, rc, s, ew in zip(sparse_indices[::2], flat_rel_coords[::2],
    #                              sizes, edge_weights[::2]))

    # down_sample = tuple(
    #     dict(sparse_indices=si,
    #          rel_coords=rc,
    #          dense_shape=(s0, s1),
    #          edge_weights=ew) for si, rc, s0, s1, ew in zip(
    #              sparse_indices[1::2], flat_rel_coords[1::2], sizes[:-1],
    #              sizes[1:], edge_weights[1::2]))

    # def transpose_kwargs(sparse_indices, rel_coords, dense_shape, edge_weights):
    #     sp = tf.SparseTensor(sparse_indices,
    #                          tf.range(tf.shape(sparse_indices)[0]), dense_shape)
    #     spt = tf.sparse.reorder(tf.sparse.transpose(sp, (1, 0)))
    #     rel_coords = tf.gather(rel_coords, spt.values, axis=1)
    #     edge_weights = tf.gather(edge_weights, spt.values)
    #     return dict(sparse_indices=spt.indices,
    #                 rel_coords=rel_coords,
    #                 dense_shape=spt.dense_shape,
    #                 edge_weights=edge_weights)

    # up_sample = tuple(transpose_kwargs(**ds) for ds in down_sample)

    # assert (len(all_coords) == depth)
    # assert (len(row_splits) == 2 * depth - 1)
    # assert (len(sample_indices) == depth - 1)
    # assert (len(in_place) == depth)
    # assert (len(down_sample) == depth - 1)
    # # assert (len(up_sample) == depth - 1)

    # features = dict(
    #     all_coords=all_coords,
    #     row_splits=row_splits,
    #     sample_indices=flat_sample_indices,
    #     in_place_kwargs=in_place,
    #     down_sample_kwargs=down_sample,
    #     # up_sample_kwargs=up_sample,
    # )

    # if normals is not None:
    #     normals = ragged_batching.post_batch_ragged(normals)
    #     normals = layer_utils.flatten_leading_dims(normals)
    #     features['normals'] = normals

    # if class_index is not None:
    #     features['class_index'] = class_index
    # problem = get_current_problem()
    # if problem is not None:
    #     labels, weights = problem.post_batch_map(labels, weights)

    # return ((features, labels) if weights is None else
    #         (features, labels, weights))


if __name__ == '__main__':
    # basic benchmarking
    from time import time
    import functools
    from deep_cloud.problems.partnet import PartnetProblem
    import tqdm
    tf.compat.v1.enable_eager_execution()
    tf.config.experimental_run_functions_eagerly(True)
    split = 'train'
    batch_size = 16
    # batch_size = 2
    num_warmup = 5
    num_batches = 10
    problem = PartnetProblem()

    def run_explicit():
        with problem:
            dataset = problem.get_base_dataset(split).map(pre_batch_map, -1)
            dataset = dataset.batch(batch_size)

        for ex in dataset:
            post_batch_map(*ex)
            break
        print('Finished single run of explicit post_batch_map')

    def run_benchmark():
        with problem:
            dataset = problem.get_base_dataset(split).map(pre_batch_map, -1)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(post_batch_map, -1).prefetch(-1)

        for i, _ in enumerate(
                tqdm.tqdm(dataset.take(num_warmup + num_batches),
                          total=num_warmup + num_batches,
                          desc='benchmarking')):
            if i == num_warmup:
                t = time()
            if i == num_warmup + num_batches - 1:
                dt = time() - t

        print('{} batches in {} s: {} ms / batch'.format(
            num_batches, dt, dt * 1000 / num_batches))

    # run_benchmark()
    run_explicit()
