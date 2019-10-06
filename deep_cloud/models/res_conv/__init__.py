from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import numpy as np
import tensorflow as tf

import collections
from more_keras import layers as mk_layers
from more_keras.ragged import batching as ragged_batching
from more_keras.framework.problems import get_current_problem

from deep_cloud.ops.np_utils.tree_utils import core
from deep_cloud.ops.np_utils.tree_utils import pykd
# from deep_cloud.ops.np_utils.tree_utils import spatial

DEFAULT_TREE = pykd.KDTree
# DEFAULT_TREE = spatial.KDTree

SQRT_2 = np.sqrt(2.)

SparseComponents = collections.namedtuple('SparseComponents',
                                          ['indices', 'values', 'dense_shape'])


@gin.configurable
def compute_edges_eager_fn(tree_impl=DEFAULT_TREE, k0=16):
    return functools.partial(compute_edges_eager, tree_impl=tree_impl, k0=k0)


def compute_edges_eager(coords, depth=5, tree_impl=DEFAULT_TREE, k0=16):
    """
    Get edges for clouds which roughly halve in size each step.

    Args:
        coords: [N, nd] float coordinates of points
        depth: int number of steps
        tree_impl:
        k0: approx_neighbors of first in-place neighborhood search.

    Returns:
        all_indices: (2 * depth - 1) list of [Ne_i, 2] sparse indices
            in row-major ordering.
        all_rel_coords: (2 * depth - 1) list of [Ne_i, nd] float relative
            coordinates normalized to range [-1, 1]
        all_rel_dists: (2 * depth - 1) list of [Ne_i] float relative distances
            normalized to range [0, 1]
        all_sample_indices: (depth - 1,) list of [N_i] int indices of sampled
            clouds, i.e. coords[i+1] = coords[i][all_sample_indices[i]]
    """
    coords = coords.numpy() if hasattr(coords, 'numpy') else coords
    tree = tree_impl(coords)
    dists, indices = tree.query(tree.data, 2, return_distance=True)
    del indices
    # closest = np.min(dists[:, 1])
    scale = np.mean(dists[:, 1])
    assert (scale > 0)
    coords *= (2 / scale)
    # coords is now a packing of barely-intersecting spheres of radius 1.

    tree = tree_impl(coords)

    # radii = [4]
    # for i in range(1, depth):
    #     radii.append(2.**(i / 4) * radii[-1])
    # print(radii)
    # print(4 * np.power(2, 3. / 4 * np.arange(depth)))
    # exit()

    # radii = 4 * np.power(2, 3. / 4 * np.arange(depth))
    # approx_neighbors = (k0 * np.power(2, np.arange(0, depth, 0.5))).astype(
    #     np.int64)
    sample_radii = 4

    all_indices = []
    all_rel_dists = []
    all_sample_indices = []
    all_coords = []

    def ragged_to_sparse(ragged_indices):
        return np.stack([
            np.repeat(np.arange(ragged_indices.leading_dim),
                      ragged_indices.row_lengths), ragged_indices.flat_values
        ],
                        axis=-1)

    def add_rel_dists(in_coords, out_coords, ragged_indices, radius):
        rel_coords = np.repeat(in_coords, ragged_indices.row_lengths,
                               axis=0) - out_coords[ragged_indices.flat_values]
        rel_dists = np.linalg.norm(rel_coords, axis=-1)
        rel_dists /= radius
        all_rel_dists.append(rel_dists)

    def add_in_place(tree, coords, radius, approx_neighbors):
        ragged_indices = tree.query_ball_point(
            coords, radius, approx_neighbors=approx_neighbors)
        all_indices.append(ragged_to_sparse(ragged_indices))
        add_rel_dists(coords, coords, ragged_indices, radius)
        # print('in-place')
        # print(np.mean(ragged_indices.row_lengths))
        return ragged_indices

    def add_down_sample(tree, coords, ragged_indices, neighbors_radius,
                        approx_neighbors):
        sample_indices = core.rejection_sample_precomputed(ragged_indices)
        # sample_indices = core.rejection_sample_lazy(tree, coords, sample_radius,
        #                                             approx_neighbors)
        sample_indices = np.array(sample_indices, dtype=np.int64)
        all_sample_indices.append(sample_indices)
        out_coords = coords[sample_indices]
        all_coords.append(out_coords)
        tree = tree_impl(out_coords)
        ragged_indices = tree.query_ball_point(
            coords, neighbors_radius, approx_neighbors=approx_neighbors)
        # print(len(sample_indices))
        # print(np.mean(ragged_indices.row_lengths))
        all_indices.append(ragged_to_sparse(ragged_indices))
        add_rel_dists(coords, out_coords, ragged_indices, neighbors_radius)
        return tree, out_coords

    for _ in range(depth - 1):
        # print(i)
        # in-place
        ragged_indices = add_in_place(tree,
                                      coords,
                                      sample_radii,
                                      approx_neighbors=k0)
        # down-sample
        # sample_radii *= SQRT_2
        sample_radii *= 2
        tree, coords = add_down_sample(tree, coords, ragged_indices,
                                       sample_radii, k0)

    # print('final')
    # final in-place
    add_in_place(tree, coords, sample_radii * 2, k0)

    return (
        tuple(all_indices),
        tuple(all_coords),
        tuple(all_rel_dists),
        tuple(all_sample_indices),
    )


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

    n_convs = 2 * depth - 1
    nd = positions.shape[1]
    specs = [
        (tf.TensorSpec((None, 2), tf.int64),) * n_convs,  # indices
        (tf.TensorSpec((None, nd), tf.float32),) * (depth - 1),  # all_coords
        (tf.TensorSpec((None,), tf.float32),) * n_convs,  # rel_dists
        (tf.TensorSpec((None,), tf.int64),) * (depth - 1),  # sample_indices
    ]

    specs_flat = tf.nest.flatten(specs)

    def flatten_output(fn, *args, **kwargs):
        return tf.nest.flatten(fn(*args, **kwargs))

    fn = functools.partial(flatten_output, edge_fn, depth=depth)
    out_flat = tf.py_function(fn, [positions], [s.dtype for s in specs_flat])
    for out, spec in zip(out_flat, specs_flat):
        out.set_shape(spec.shape)
    out_flat = tf.nest.map_structure(ragged_batching.pre_batch_ragged, out_flat)
    indices, all_coords, rel_dists, sample_indices = tf.nest.pack_sequence_as(
        specs, out_flat)
    all_coords = (ragged_batching.pre_batch_ragged(positions),) + all_coords

    features = dict(
        indices=indices,
        all_coords=all_coords,
        rel_dists=rel_dists,
        sample_indices=sample_indices,
    )

    if normals is not None:
        all_normals = [normals]
        for si in sample_indices:
            normals = normals[si]
            all_normals.append(normals)
        features['normals'] = tf.nest.map_structure(
            ragged_batching.pre_batch_ragged, tuple(all_normals))

    cloud_size = tf.shape(positions, out_type=tf.int64)[0]
    cloud_sizes = [cloud_size]
    for si in sample_indices:
        cloud_sizes.append(tf.size(si, out_type=tf.int64))
    features['cloud_sizes'] = tuple(cloud_sizes)

    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable
def get_weights(rel_dists):
    return tf.nest.map_structure(lambda rd: 1 - rd, rel_dists)


def transpose_sparse(sp):
    indices = tf.roll(sp.indices, 1, axis=1)
    # dense_shape = tf.reverse(sp.dense_shape, axis=0)
    dense_shape = sp.dense_shape[::-1]
    return tf.SparseTensor(indices, sp.values, dense_shape)


def reduce_sparse_sum(sp, axis=1, ordered=False):
    """
    Sparse sum reduction based on math.(unsorted_)segment_sum.

    https://github.com/tensorflow/tensorflow/issues/32763

    Args:
        sp: rank 2 sparse tensor
        axis: int, axis along which to sum
        ordered: if True, other axis indices are assumed to be ascending.

    Returns:
        rank 1 dense tensor equivalent to tf.sparse.reduce_sum(sp, axis=axis)
    """
    other_axis = 0 if axis in (1, -1) else 1
    if ordered:
        return tf.math.segment_sum(sp.values, sp.indices[:, other_axis])
    else:
        return tf.math.unsorted_segment_sum(sp.values,
                                            sp.indices[:, other_axis],
                                            sp.dense_shape[other_axis])


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def post_batch_map(features, labels, weights=None, weights_fn=get_weights):
    cloud_sizes = features.pop('cloud_sizes')
    features = tf.nest.map_structure(ragged_batching.post_batch_ragged,
                                     features)

    indices, all_coords, rel_dists, sample_indices = (features[k] for k in (
        'indices', 'all_coords', 'rel_dists', 'sample_indices'))
    normals = features.get('normals')
    all_coords, rel_dists = tf.nest.map_structure(lambda x: x.flat_values,
                                                  (all_coords, rel_dists))
    cloud_row_splits = [tf.pad(tf.cumsum(cs), [[1, 0]]) for cs in cloud_sizes]
    row_starts = [splits[:-1] for splits in cloud_row_splits]
    total_cloud_sizes = [splits[-1] for splits in cloud_row_splits]
    dense_shapes = []
    depth = len(sample_indices) + 1
    indices = list(indices)

    for i in range(depth - 1):
        # in-place
        offset = tf.reshape(row_starts[i], (-1, 1, 1))
        indices[2 * i] = (indices[2 * i] + offset).flat_values
        dense_shapes.append([total_cloud_sizes[i]] * 2)
        # tf.tile(tf.expand_dims(total_cloud_sizes[i], axis=0), (2,)))
        # down-sample
        offset = tf.expand_dims(tf.stack((row_starts[i], row_starts[i + 1]),
                                         axis=-1),
                                axis=-2)
        indices[2 * i + 1] = (indices[2 * i + 1] + offset).flat_values
        dense_shapes.append([total_cloud_sizes[i], total_cloud_sizes[i + 1]])

    in_place_neighbors = []
    down_sample_neighbors = []
    up_sample_neighbors = []
    dist_weights = weights_fn(rel_dists)

    def normalized_sparse_components(sp):
        norm_factor = reduce_sparse_sum(sp, axis=1, ordered=True)
        sp = sp / tf.expand_dims(norm_factor, axis=1)
        return SparseComponents(sp.indices, sp.values, sp.dense_shape)

    sparse_neighbors = tuple(
        tf.SparseTensor(i, w, s) for i, w, s in zip(
            indices[1::2], dist_weights[1::2], dense_shapes[1::2]))

    for sp in sparse_neighbors:
        down_sample_neighbors.append(normalized_sparse_components(sp))
        up_sample_neighbors.append(
            normalized_sparse_components(tf.sparse.reorder(
                transpose_sparse(sp))))

    in_place_neighbors = tuple(
        SparseComponents(*args)
        for args in zip(indices[::2], dist_weights, dense_shapes[::2]))

    features = dict(
        in_place_neighbors=in_place_neighbors,
        down_sample_neighbors=tuple(down_sample_neighbors),
        up_sample_neighbors=tuple(up_sample_neighbors),
        all_coords=all_coords,
    )
    if normals in features:
        normals = tf.nest.map_structure(lambda x: x.flat_values, normals)
        features['normals'] = normals

    labels, weights = get_current_problem().post_batch_map(labels, weights)
    return ((features, labels) if weights is None else
            (features, labels, weights))


# class LinearConvolution(tf.keras.layers.Layer):

#     def __init__(self, units, **kwargs):
#         self.units = units
#         self.coord_layer = tf.keras.layers.Dense(self.units, use_bias=False)
#         self.feature_layer = tf.keras.layers.Dense(self.units, use_bias=False)
#         super(LinearConvolution, self).__init__(**kwargs)

#     def build(self, input_shape):
#         coord_features, node_features, neighbors = input_shape
#         coord_size = coord_features.shape[-1]
#         node_size = node_features.shape[-1]
#         self.coord_layer = tf.keras.layers.Dense(self.units)

#     def call(self, inputs):
#         coord_features, nodes_features, neighbors = inputs


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def res_conv_semantic_segmenter(input_spec,
                                output_spec,
                                dense_factory=mk_layers.Dense,
                                batch_norm_impl=mk_layers.BatchNormalization,
                                activation='relu',
                                filters0=32):
    pass


if __name__ == '__main__':
    import tensorflow_datasets as tfds
    from deep_cloud.problems.partnet import PartnetProblem
    tf.compat.v1.enable_v2_tensorshape()
    problem = PartnetProblem()
    dataset = problem.get_base_dataset(split='validation')

    def test_eager():
        for positions, _ in tfds.as_numpy(dataset):
            compute_edges_eager(positions)
            break
        print('test_pre_batch_map_fn passed')

    def test_pre_batch_map_fn():
        ds = dataset.map(pre_batch_map)
        for _ in tfds.as_numpy(ds):
            break
        print('test_pre_batch_map_fn passed')

    def test_pipeline():
        ds = dataset.map(pre_batch_map)
        with problem:
            ds = ds.batch(16)
            ds = ds.map(post_batch_map)
        for _ in tfds.as_numpy(ds):
            break
        print('test_pipeline passed')

    def benchmark_pipeline():
        dataset = problem.get_base_dataset(split='train')
        # dataset = problem.get_base_dataset(split='validation')
        autotune = tf.data.experimental.AUTOTUNE
        with problem:
            dataset = dataset.map(pre_batch_map, autotune)
            dataset = dataset.batch(16)
            dataset = dataset.map(post_batch_map, autotune)
        dataset = dataset.prefetch(autotune)
        ops = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        with tf.Session() as sess:
            tf.test.Benchmark().run_op_benchmark(sess, ops)

    # test_eager()
    # test_pre_batch_map_fn()
    # test_pipeline()
    benchmark_pipeline()

    print('finished')
