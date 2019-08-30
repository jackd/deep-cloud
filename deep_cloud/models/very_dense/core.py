from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf
import functools
import gin
import itertools
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from more_keras.layers import utils as layer_utils
from more_keras.layers import Dense
from more_keras.meta_models import builder
from more_keras.models import mlp
from more_keras.ops import utils as op_utils
from more_keras.ragged import np_impl as ra
from more_keras.ragged.layers import ragged_lambda_call
from more_keras.ragged.layers import maybe_ragged_lambda_call
from deep_cloud.ops import ordering
from deep_cloud.ops.asserts import assert_flat_tensor, INT_TYPES, FLOAT_TYPES
from deep_cloud.ops.np_utils import tree_utils
from deep_cloud.ops.np_utils import cloud as np_cloud
from deep_cloud.ops.np_utils import ordering as np_ordering
from deep_cloud.layers import query
from deep_cloud.layers.cloud import get_relative_coords
from deep_cloud.models.very_dense import extractors
from deep_cloud.models.very_dense import poolers
from deep_cloud.models.very_dense import utils
from deep_cloud.models.very_dense.reshaper import Reshaper

Lambda = tf.keras.layers.Lambda

SQRT_2 = np.sqrt(2)


def get_exponential_radii(depth=4, r0=0.1, expansion_rate=2):
    return r0 * expansion_rate**np.arange(depth)


def compute_edges_eager(coords, normals, radii, pooler, reorder=False):
    """
    Recursively sample the input cloud and find edges based on ball searches.

    Args:
        coords: [N_0, num_dims] numpy array or eager tensor of cloud
            coordinates.
        normals: [N_0, num_features] numpy array of node features.
        radii: [depth] numpy array or eager tensor of radii for ball searches.
        pooler: `Pooler` instance.

    Returns:
        all_coords: [depth] list of numpy coordinate arrays of shape
            [N_i, num_dims]
        all_normals: [depth] list of numpy node features of shape
            [N_i, num_features]
        flat_rel_coords: flat values in rel_coords (see below)
        flat_node_indices: flat values in node_indices (see below)
        row_splits: row splits for rel_coords, node_indices (see below)

    The following aren't actually returned but are easier to describe this way.
        rel_coords: [depth, <=depth] list of lists of [N_i, k?, num_dims]
            floats, where k is the number of neighbors (ragged).
            `rel_coords[i][j][p]` (k?, num_dims) dives the relative coordinates
            of all points in cloud `j` in the neighborhood of point `p` in cloud
            `i`.
        node_indices: [depth, <=depth] list of lists of [N_i, k?] ints.
            see below. `node_indices[i][j][p] == (r, s, t)` indicates that
            node `p` in cloud `i` neighbors nodes `r, s, t` in cloud `j`.
    """
    # accomodate eager coords tensor, so can be used with tf.py_functions
    if hasattr(coords, 'numpy'):
        coords = coords.numpy()
    if hasattr(radii, 'numpy'):
        radii = radii.numpy()
    if hasattr(normals, 'numpy'):
        normals = normals.numpy()
    if any(isinstance(t, tf.Tensor) for t in (coords, normals, radii)):
        assert (tf.executing_eagerly())

    if reorder:
        indices = np_ordering.iterative_farthest_point_ordering(
            coords, coords.shape[0] // 2)
        coords, normals = np_ordering.partial_reorder(indices, coords, normals)

    depth = len(radii)
    node_indices = utils.lower_triangular(depth)
    trees = [None for _ in range(depth)]
    all_coords = [coords]
    all_normals = [normals]

    # TODO: The below uses `depth * (depth + 1) // 2` ball searches
    # we can do it with `depth`. When depth is 4 it's not that big a saving...

    # do the edges diagonal and build up coordinates
    for i, radius in enumerate(radii[:-1]):
        tree = trees[i] = cKDTree(coords)
        indices = node_indices[i][i] = tree_utils.query_pairs(tree, radius)
        coords, normals, indices = pooler(coords, normals, indices)
        # node_indices[i + 1][i] = indices
        all_coords.append(coords)
        all_normals.append(normals)

    # final cloud
    tree = trees[-1] = cKDTree(coords)
    node_indices[-1][-1] = tree_utils.query_pairs(tree, radii[-1])
    # We have all trees and node_indices [i, i] # not longer [i, i + 1]

    # do below the diagonal, i.e. [i, j], i > j + 1
    for i in range(1, depth):
        in_tree = trees[i]
        radius = radii[i]
        for j in range(i):
            node_indices[i][j] = tree_utils.query_ball_tree(
                trees[j], in_tree, radius)

    # TODO: Should be able to do the following in `depth` `rel_coords` calls
    # Currently uses `depth * (depth + 1) // 2`.
    flat_rel_coords = utils.lower_triangular(depth)
    for i in range(depth):
        for j in range(i + 1):
            indices = node_indices[i][j]
            flat_rel_coords[i][j] = np_cloud.get_relative_coords(
                all_coords[j],
                all_coords[i],
                indices.flat_values,
                row_lengths=indices.row_lengths)

    flat_node_indices = tf.nest.map_structure(lambda x: x.flat_values,
                                              node_indices)
    row_splits = tf.nest.map_structure(lambda x: x.row_splits, node_indices)

    return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits)


def compute_edges(coords, normals, radii, pooler, reorder=False):
    """Graph-mode wrapper for compute_edges_eager. Same inputs/outputs."""
    if tf.executing_eagerly():
        (all_coords, all_normals, flat_rel_coords, flat_node_indices,
         row_splits) = compute_edges_eager(coords,
                                           normals,
                                           radii,
                                           pooler,
                                           reorder=reorder)
    else:

        def fn(args):
            coords, normals = args
            out = compute_edges_eager(coords,
                                      normals,
                                      radii,
                                      pooler,
                                      reorder=reorder)
            return tf.nest.flatten(out)

        depth = len(radii)
        Tout = (
            [tf.float32] * depth,  # coords
            [tf.float32] * depth,  # normals
            utils.lower_triangular(depth, tf.float32),  # flat_rel_coords
            utils.lower_triangular(depth, tf.int64),  # flat_node_indices
            utils.lower_triangular(depth, tf.int64),  # row_splits
        )
        Tout_flat = tf.nest.flatten(Tout)
        out_flat = Lambda(lambda c: tf.py_function(fn, [c], Tout_flat))(
            [coords, normals])
        (all_coords, all_normals, flat_rel_coords, flat_node_indices,
         row_splits) = tf.nest.pack_sequence_as(Tout, out_flat)

    # fix sizes
    num_dims = coords.shape[1]
    num_features = normals.shape[1]
    sizes = [coords.shape[0]]
    # even if size is None, we still get rank information from the below
    for _ in range(depth - 1):
        sizes.append(pooler.output_size(sizes[-1]))

    for i in range(depth):
        size = sizes[i]
        all_coords[i].set_shape((size, num_dims))
        all_normals[i].set_shape((size, num_features))
        for rc in flat_rel_coords[i]:
            rc.set_shape((None, num_dims))
        for ni in flat_node_indices[i]:
            ni.set_shape((None,))
        for rs in row_splits[i]:
            rs.set_shape((size + 1,))

    return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits)


def batch_row_splits(row_splits):
    # we convert row_splits -> row_lengths, batch, flatten, then convert back.
    assert (row_splits.shape[0] is not None)
    row_lengths = layer_utils.diff(row_splits)
    row_lengths = builder.batched(row_lengths)
    row_lengths = layer_utils.flatten_leading_dims(row_lengths)
    row_splits = layer_utils.row_lengths_to_splits(row_lengths)
    return row_splits


def compute_batched_edges(coords, normals, radii, pooler, reorder=False):
    """
    Args:
        coords: [n, num_dims] float tensor of unbatched coords.
        normals: [n, f] float tensor of unbatched node features.
        radii: [K]-length list of ball search radii.
        pooler: `poolers.Pooler` instance.

    Returns:
        all_coords: K-length list of [N, num_dims] float tensor of
            flattened batched coordinates for each set.
        all_normals: K-length list of [N, f] float tensor of flattened
            batched node features for each set.
        flat_node_indices: [K, <=K] llist int into returned all_*
        flat_rel_coords: [K, <=K] llist
        row_splits: [K, <=K] llist of ints denoting ragged structure of flat_*.
            row_splits[i][j] is of size [N]
    """
    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        compute_edges(
            coords, normals, radii=radii, pooler=pooler, reorder=reorder)

    def flat_batch(tensor):
        if isinstance(tensor, tf.RaggedTensor):
            tensor = tensor.flat_values
        else:
            assert (tensor.shape[0] is None)

        tensor = builder.batched(tensor)
        return tensor.flat_values

    row_splits = tf.nest.map_structure(batch_row_splits, row_splits)
    flat_rel_coords = tf.nest.map_structure(flat_batch, flat_rel_coords)
    flat_node_indices = tf.nest.map_structure(flat_batch, flat_node_indices)
    sizes = builder.batched([layer_utils.leading_dim(c) for c in all_coords])

    outer_row_splits = [layer_utils.row_lengths_to_splits(s) for s in sizes]

    if not all([c.shape[0] is not None] for c in all_coords):
        raise NotImplementedError

    all_coords, all_normals = tf.nest.map_structure(builder.batched,
                                                    (all_coords, all_normals))

    return (
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        outer_row_splits,
    )


def flatten_edges(all_coords, all_normals, flat_rel_coords, flat_node_indices,
                  row_splits, outer_row_splits):
    """
    TODO: update

    Flatten batched data into a single graph.

    Args:
        all_coords: [K] [B, n?, 3] possibly ragged coordinates.
        all_normals: [K] [B, n?, 3] possibly ragged normals.
        rel_coords: [K, <=K] [B, n?, k?, 3] ragged relative coordinates.
        node_indices: [K, <=K] [B, n?, k?] ragged neighborhoods.

    Returns:
        all_coords: [K] [N, 3] float flat coordinates
        all_normals: [K] [N, 3] float flat normals
        rel_coords: [K, <=K] [M, 3] float flat relative coordinates
        node_indices: [K, <=K] [M,] int flat neighborhoords into flattened
            coords/normals.
        nested_row_splits: [K, <=K] [B+1], [N+1] int
            nested row splits from node_indices.
    """
    offsets = [layer_utils.get_row_offsets(c) for c in all_coords]
    all_coords = [layer_utils.flatten_leading_dims(c) for c in all_coords]
    all_normals = [layer_utils.flatten_leading_dims(n) for n in all_normals]

    K = len(offsets)
    for i in range(K):
        for j in range(i + 1):
            flat_node_indices[i][j] = layer_utils.apply_row_offset(
                tf.RaggedTensor.from_nested_row_splits(
                    flat_node_indices[i][j],
                    [outer_row_splits[i], row_splits[i][j]]),
                offsets[j]).flat_values

    # nested_row_splits = tf.nest.map_structure(
    #     tf.keras.layers.Lambda(lambda ni: ni.nested_row_splits), node_indices)
    # flat_rel_coords, flat_node_indices = tf.nest.map_structure(
    #     lambda ni: ni.flat_values, (rel_coords, node_indices))
    return (
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        outer_row_splits,
    )


def fps_reorder(coords, *args, sample_frac=0.25):

    def _fps_reorder(args, sample_frac=0.25):
        coords, *args = args
        num_points = op_utils.leading_dim(coords, dtype=tf.int64)
        num_points_f = tf.cast(num_points, dtype=tf.float32)
        num_samples = tf.cast(num_points_f * sample_frac, tf.int64)
        indices = ordering.iterative_farthest_point_order(coords, num_samples)
        return list(ordering.partial_reorder(indices, coords, *args))

    return tf.keras.layers.Lambda(_fps_reorder,
                                  arguments=dict(sample_frac=sample_frac))(
                                      (coords, *args))


def get_base_node_network_factories(num_factories=4,
                                    units_scale=8,
                                    unit_expansion_factor=2,
                                    network_depth=2):
    return [
        functools.partial(mlp,
                          units=[4 * units_scale * unit_expansion_factor**i] *
                          network_depth) for i in range(num_factories)
    ]


def _ragged_reduction(args, reduction):
    values, row_splits = args
    values = tf.RaggedTensor.from_row_splits(values, row_splits)
    return reduction(values, axis=1)


def apply_ragged_reduction(flat_values, row_splits_or_k, reduction):
    if row_splits_or_k.shape.ndims == 0:
        values = layer_utils.reshape_leading_dim(flat_values,
                                                 (-1, row_splits_or_k))
        return tf.keras.layers.Lambda(reduction, arguments=dict(axis=1))(values)
    else:
        return tf.keras.layers.Lambda(_ragged_reduction,
                                      arguments=dict(reduction=reduction))(
                                          [flat_values, row_splits_or_k])


def get_base_global_network(units=(512, 256), dropout_impl=None):
    return functools.partial(mlp, units=units, dropout_impl=dropout_impl)


@gin.configurable
def very_dense_classifier(
        input_spec,
        output_spec,
        sample_frac=0.25,
        resolution_depth=4,
        repeats=3,
        pooler_factory=poolers.SlicePooler,
        reorder=True,
        extractor_factory=extractors.get_base_extractor,
        node_network_factories=None,
        global_network_factory=get_base_global_network,
        dense_factory=Dense,
        concat_features=True,
        reduction=tf.reduce_max,
):

    def input_from_spec(spec):
        return builder.prebatch_input(shape=spec.shape, dtype=spec.dtype)

    if isinstance(input_spec, dict):
        coords = input_from_spec(input_spec['positions'])
        normals = input_from_spec(input_spec['normals'])
    else:
        coords = input_from_spec(input_spec)
        normals = None

    if node_network_factories is None:
        node_network_factories = get_base_node_network_factories(
            resolution_depth)

    radii = get_exponential_radii(depth=resolution_depth)

    if reorder:
        coords, normals = fps_reorder(coords, normals)

    batched_edges = compute_batched_edges(coords, normals, radii,
                                          pooler_factory(sample_frac))
    (
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        outer_row_splits,
    ) = builder.as_model_input(flatten_edges(*batched_edges))

    preds = []
    num_classes = output_spec.shape[-1]
    sizes = [layer_utils.leading_dim(c) for c in all_coords]

    def get_extractor():
        return extractor_factory(sizes, flat_node_indices, row_splits)

    inp_node_features = all_normals
    inp_edge_features = flat_rel_coords
    inp_global_features = None

    for _ in range(repeats + 1):
        node_features, edge_features = get_extractor()(
            inp_node_features,
            inp_edge_features,
            #    global_features,
            #    all_coords,
        )

        node_features = [
            network_factory()(n)  # fresh model each time
            for network_factory, n in zip(node_network_factories, node_features)
        ]
        global_features = [
            apply_ragged_reduction(n, rs, reduction)
            for (n, rs) in zip(node_features, outer_row_splits)
        ]
        if inp_global_features is not None:
            global_features.append(inp_global_features)
        global_features = layer_utils.concat(global_features, axis=-1)
        global_features = global_network_factory()(global_features)
        preds.append(Dense(num_classes, activation=None)(global_features))
        inp_node_features = node_features
        inp_edge_features = edge_features
        inp_global_features = global_features

    return builder.model(preds)


if __name__ == '__main__':
    from absl import logging

    def make_normals():
        normals = np.random.randn(n, 3).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    do_profile = False
    do_dataset_profile = False
    vis_single = False
    vis_batched = False

    # do_profile = True
    # do_dataset_profile = True
    # vis_single = True
    vis_batched = True

    depth = 4
    sample_frac = 0.25
    n = 1024
    # n = 512
    logging.set_verbosity(logging.INFO)
    base_coords = np.random.uniform(size=(n, 3)).astype(np.float32)
    base_coords[:, 2] = 0
    indices = np_ordering.iterative_farthest_point_ordering(base_coords)
    base_coords = base_coords[indices]
    scale_factor = np.mean(
        cKDTree(base_coords).query(base_coords,
                                   k=11)[0][:, -1])  # mean 10th neighbor
    base_coords /= scale_factor  # now the 10th neighbor is on average 1 unit away
    base_normals = make_normals()

    radii = get_exponential_radii(r0=1)
    # pooler = poolers.InverseDensitySamplePooler(sample_frac)
    pooler = poolers.SlicePooler(sample_frac)

    # if do_profile:
    #     # super simple profiling
    #     import tqdm
    #     from time import time
    #     num_runs = 100
    #     dt = 0
    #     logging.info('Running basic profiling')
    #     for _ in tqdm.tqdm(range(num_runs)):
    #         coords = np.random.uniform(size=(n, 3)).astype(np.float32)
    #         coords[:, 2] = 0
    #         coords /= scale_factor
    #         normals = make_normals()
    #         t = time()
    #         compute_edges_eager(coords, normals, radii=radii, pooler=pooler)
    #         dt += time() - t
    #     logging.info('Completed {} runs in {:.2f}s, {:.1f} runs / s'.format(
    #         num_runs, dt, num_runs / dt))

    # if do_dataset_profile:
    #     # simple profiling for dataset
    #     import tqdm
    #     from time import time
    #     warm_up = 10
    #     num_runs = 50
    #     batch_size = 16
    #     reorder = True
    #     logging.info('Running dataset profiling')

    #     def gen():
    #         while True:
    #             coords = np.random.uniform(size=(n, 3)).astype(np.float32)
    #             coords[:, 2] = 0
    #             coords /= scale_factor
    #             normals = make_normals()
    #             labels = np.ones((), dtype=np.int64)
    #             yield (coords, normals), labels

    #     with tf.Graph().as_default():  # pylint: disable=not-context-manager
    #         dataset = tf.data.Dataset.from_generator(
    #             gen, ((tf.float32, tf.float32), tf.int64),
    #             (((n, 3), (n, 3)), ()))
    #         with builder.MetaNetworkBuilder() as b:
    #             (coords, normals), labels = b.prebatch_inputs_from(dataset)
    #             # if reorder:
    #             #     coords, normals = fps_reorder(coords,
    #             #                                   normals,
    #             #                                   sample_frac=sample_frac)
    #             nested_out = compute_batched_edges(coords,
    #                                                normals,
    #                                                radii=radii,
    #                                                pooler=pooler,
    #                                                reorder=reorder)
    #             flat_out = tf.nest.flatten(nested_out)
    #             # model_inps = [b.as_model_input(f) for f in flat_out]
    #             labels = b.batched(labels)
    #         preprocessor = b.preprocessor((labels,))
    #         dataset = preprocessor.map_and_batch(dataset, batch_size=batch_size)
    #         out = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    #         with tf.Session() as sess:
    #             logging.info('Warming up with {} runs, batch_size {}'.format(
    #                 warm_up, batch_size))
    #             for _ in tqdm.tqdm(range(warm_up)):
    #                 sess.run(out)
    #             logging.info('Starting {} actual runs'.format(num_runs))
    #             t = time()
    #             for _ in tqdm.tqdm(range(num_runs)):
    #                 sess.run(out)
    #             dt = time() - t
    #     logging.info(
    #         'Completed {} runs with batch_size {} in {:.2f}s, {:.1f} runs / s, '
    #         '{:.1f} examples / s'.format(num_runs, batch_size, dt,
    #                                      num_runs / dt,
    #                                      num_runs * batch_size / dt))


    def vis(coords, node_indices, point_indices=None):
        import trimesh
        depth = len(coords)
        if point_indices is None:
            point_indices = [0] * depth
        for i, point_index in enumerate(point_indices):
            ci = coords[i]
            center = ci[point_index]
            for j in range(i + 1):
                cj = coords[j]
                ni = node_indices[i][j]
                ni = ra.RaggedArray.from_row_splits(ni.values, ni.row_splits)
                neighbors = ni[point_index]
                print(i, j, ci.shape, cj.shape, neighbors.shape)
                print(np.max(np.linalg.norm(cj[neighbors] - center, axis=-1)))
                scene = trimesh.Scene()
                scene.add_geometry(trimesh.PointCloud(cj, color=(0, 255, 0)))
                scene.add_geometry(trimesh.PointCloud(ci, color=(0, 0, 255)))
                scene.add_geometry(
                    trimesh.PointCloud(cj[neighbors], color=(255, 0, 0)))
                scene.add_geometry(
                    trimesh.primitives.Sphere(center=center, radius=0.2))

                scene.show(background=(0, 0, 0))

    # if vis_single:
    #     coords = base_coords
    #     normals = base_normals
    #     coords, normals = fps_reorder(coords, normals)
    #     coords, normals, flat_rel_coords, flat_node_indices, row_splits = \
    #         compute_edges(
    #             coords, normals, radii=radii, pooler=pooler)
    #     # coords = trimesh.creation.icosphere(7).vertices
    #     # coords = coords[np.random.choice(coords.shape[0], 2048)]
    #     # coords = coords[coords[:, 1] > 0]
    #     # tf.enable_eager_execution()
    #     coords, normals, flat_rel_coords, flat_node_indices, row_splits = \
    #         compute_edges(
    #             tf.constant(base_coords),
    #             tf.constant(base_normals), radii=radii, pooler=pooler)
    #     # compute_edges_eager(coords, normals, radii=radii, pooler=pooler)
    #     with tf.Session() as sess:
    #         coords, normals, flat_rel_coords, flat_node_indices, row_splits = \
    #             sess.run(
    #             (coords, normals, flat_rel_coords, flat_node_indices,
    #              row_splits))

    #     node_indices = tf.nest.map_structure(ra.RaggedArray.from_row_splits,
    #                                          flat_node_indices, row_splits)
    #     vis(coords, node_indices)

    if vis_batched:
        # coords = np.random.uniform(size=(2, n, 3)).astype(np.float32)
        # coords[0, :, 1] = 0
        # coords[1, :, 1] = 1
        # coords /= scale_factor
        coords = np.stack([base_coords, base_coords], axis=0)
        coords[1, :, 2] = 1
        normals = np.stack([base_normals, base_normals], axis=0)
        labels = np.array([0, 1], dtype=np.int64)
        dataset = tf.data.Dataset.from_tensor_slices(
            ((coords, normals), labels))

        with builder.MetaNetworkBuilder() as b:
            coords = b.prebatch_input(shape=(n, 3), dtype=tf.float32)
            normals = b.prebatch_input(shape=(n, 3), dtype=tf.float32)

            coords, normals = fps_reorder(coords, normals)

            batched_out = compute_batched_edges(coords,
                                                normals,
                                                radii=radii,
                                                pooler=pooler)

            flattened_out = flatten_edges(*batched_out)

            b.as_model_input(flattened_out)
            # b.as_model_input(batched_out)
            labels = b.batched(b.prebatch_input(shape=(), dtype=tf.int64))

        preprocessor = b.preprocessor((labels,))
        dataset = preprocessor.map_and_batch(dataset, batch_size=2)
        values, labels = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()

        with tf.Session() as sess:
            values = sess.run(values)

        (all_coords, all_normals, flat_rel_coords, flat_node_indices,
         row_splits,
         outer_row_splits) = tf.nest.pack_sequence_as(flattened_out, values)

        node_indices = tf.nest.map_structure(ra.RaggedArray.from_row_splits,
                                             flat_node_indices, row_splits)

        point_indices = [rs[1] for rs in outer_row_splits]
        vis(all_coords, node_indices, point_indices)
        # for i in range(batch_size)
        # for i, point_indices in enumerate(zip(*batch_row_offsets)):
        #     vis(coords, node_indices, point_indices)
