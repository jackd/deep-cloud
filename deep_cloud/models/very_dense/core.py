from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf
import functools
import itertools
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from more_keras.layers import utils as layer_utils
from more_keras.meta_models import builder
from more_keras.ops import utils as op_utils
from more_keras.ragged import np_impl as ra
from more_keras.ragged.layers import ragged_lambda_call
from more_keras.tf_compat import dim_value
from deep_cloud.ops.ordering import iterative_farthest_point_order
from deep_cloud.ops.asserts import assert_flat_tensor, INT_TYPES, FLOAT_TYPES
from deep_cloud.ops.np_utils import tree_utils
from deep_cloud.ops.np_utils import cloud as np_cloud
from deep_cloud.layers import query
from deep_cloud.layers.cloud import get_relative_coords
from deep_cloud.models.very_dense import extractors
from deep_cloud.models.very_dense import poolers
from deep_cloud.models.very_dense import utils

Lambda = tf.keras.layers.Lambda

SQRT_2 = np.sqrt(2)


def get_smart_radii(depth=4, r0=0.1):
    return r0 * SQRT_2**np.arange(depth)


def compute_edges_eager(coords, normals, radii, pooler):
    """
    Recursively sample the input cloud and find edges based on ball searches.

    Args:
        coords: [N_0, num_dims] numpy array or eager tensor of cloud
            coordinates.
        normals: [N_0, num_features] numpy array of node features.
        radii: [depth] numpy array or eager tensor of radii for ball searches.
        pooler: `Pooler` instance.

    Returns:
        all_coords: `depth` numpy coordinate arrays of shape [N_i, num_dims]
        all_normals: `depth` numpy node features of shape [N_i, num_features]
        flat_rel_coords: [depth, <=depth] list of lists of [e_ij, num_dims]
            floats.
        flat_node_indices: [depth, <=depth] list of lists of [e_ij] ints.
            see below. All values will be less than the number of nodes in
            set `i`.
        row_splits: [depth, <=depth] list of lists. row_splits[i][j] has
            shape [N_i+1]. See below.

    Notes:
        `flat_node_indices` and `flat_rel_coords` entries are intended to be
        interpretted in conjunction with `row_splits` as ragged arrays which we
        will refer to as `node_indices` and `rel_coords`, e.g.
        ```
        node_indices[i][j] = RaggedArray.from_row_splits(
            flat_node_indices[i][j], row_splits[i][j])
        ```

        node_indices: `node_indices[i][j][p] == (q, r, s)` indicates
            that node `p` in set `i` is connected to nodes `q, r, s` in set `j`.
        rel_coords: `rel_coords[i][j][p, q]` is displacement between element `p`
        of set `i` and element `q` of set `j`.

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

    flat_node_indices = tf.nest.map_structure(
        lambda indices: indices.flat_values, node_indices)
    row_splits = tf.nest.map_structure(lambda indices: indices.row_splits,
                                       node_indices)

    return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits)


def compute_edges(coords, normals, radii, pooler):
    """Graph-mode wrapper for compute_edges_eager. Same inputs/outputs."""
    if tf.executing_eagerly():
        return compute_edges_eager(coords, normals, radii, pooler)

    def fn(args):
        coords, normals = args
        out = compute_edges_eager(coords, normals, radii, pooler)
        return tf.nest.flatten(out)

    depth = len(radii)
    Tout = (
        [tf.float32] * depth,  # coords
        [tf.float32] * depth,  # normals
        utils.lower_triangular(depth, tf.float32),  # rel_coords
        utils.lower_triangular(depth, tf.int32),  # node indices
        utils.lower_triangular(depth, tf.int32),  # row_splits
    )
    Tout_flat = tf.nest.flatten(Tout)
    out_flat = Lambda(lambda c: tf.py_function(fn, [c], Tout_flat))(
        [coords, normals])
    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        tf.nest.pack_sequence_as(Tout, out_flat)

    # fix sizes
    size = dim_value(coords.shape[0])
    num_dims = dim_value(coords.shape[1])
    num_features = dim_value(normals.shape[1])
    sizes = [size]
    # even if size is None, we still get rank information from the below
    for i in range(depth - 1):
        size = pooler.output_size(size)
        sizes.append(size)

    for i in range(depth):
        size = sizes[i]
        if size is None:
            continue

        all_coords[i].set_shape((size, num_dims))
        all_normals[i].set_shape((size, num_features))
        for rs in row_splits[i]:
            rs.set_shape((size + 1,))
        for rc in flat_rel_coords[i]:
            rc.set_shape((None, num_dims))
        for ni in flat_node_indices[i]:
            ni.set_shape((None,))

    return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits)


def _apply_batch_offset(args):
    flat_values, row_splits, offset = args
    row_lengths = op_utils.diff(row_splits)
    return op_utils.repeat(offset, row_lengths, axis=0) + flat_values


def _apply_batch_offset_ragged(rt, offset):
    values = tf.keras.layers.Lambda(_apply_batch_offset)(
        [rt.values, rt.row_splits, offset])
    return tf.RaggedTensor.from_row_splits(values, rt.row_splits)


def compute_batched_edges(coords, normals, radii, pooler):
    """
    Args:
        coords: [n, num_dims] float tensor of unbatched coords.
        normals: [n, f] float tensor of unbatched node features.
        radii: [depth]-length list of ball search radii.
        pooler: `poolers.Pooler` instance.

    Returns:
        all_coords: depth-length list of [N, num_dims] float tensor of
            flattened batched coordinates for each set.
        all_normals: depth-length list of [N, f] float tensor of flattened
            batched node features for each set.
        flat_node_indices: [depth, <=depth] llist int.
        flat_rel_coords: [depth, <=depth] llist
    """
    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        builder.batched(
            compute_edges(coords, normals, radii=radii, pooler=pooler))
    flat_node_indices = list(flat_node_indices)
    batch_row_offsets = [
        layer_utils.get_row_offsets(c, dtype=flat_node_indices[0][0].dtype)
        for c in all_coords
    ]
    expander = Lambda(tf.expand_dims, arguments=dict(axis=-1))
    batch_row_offsets = [expander(ro) for ro in batch_row_offsets]

    for i in range(depth):
        flat_node_indices[i] = [
            _apply_batch_offset_ragged(nids, offset) if isinstance(
                nids, tf.RaggedTensor) else Lambda(
                    tf.math.add_n)([nids, offset])
            for nids, offset in zip(flat_node_indices[i], batch_row_offsets[:i +
                                                                            1])
        ]

    # flatten everything after split
    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        tf.nest.map_structure(lambda r: layer_utils.flatten_leading_dims(r), (
            all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits))

    return (
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        batch_row_offsets,
    )


def _input_from_spec(spec):
    return builder.prebatch_input(shape=spec.shape, dtype=spec.dtype)


def _iterative_farthest_point_order(args):
    points, num_samples = args
    return iterative_farthest_point_order(points, num_samples)


def _full_indices(indices, total):
    all_indices = tf.range(total, dtype=indices.dtype)
    missing = tf.sets.set_difference(all_indices, indices)
    return tf.concat([indices, missing], axis=-1)


def _as_offset(row_splits):
    return row_splits[:-1]


def very_dense_classifier(
        input_spec,
        output_spec,
        sample_frac=0.5,
        resolution_depth=4,
        repeats=4,
        pooler_factory=poolers.SlicePooler,
        reorder=True,
        extractor_factory=extractors.get_base_extractor,
        node_network_fns=None,
):
    if isinstance(input_spec, dict):
        positions = _input_from_spec(input_spec['positions'])
        normals = _input_from_spec(input_spec['normals'])
    else:
        positions = _input_from_spec(input_spec)
        normals = None

    radii = get_smart_radii(depth=resolution_depth)

    if reorder:
        # TODO: is this faster to do with numpy implementation?
        # We already go to numpy land in `compute_edges`
        num_points = layer_utils.leading_dim(positions)
        num_samples = Lambda(tf.math.multiply,
                             arguments=dict(y=sample_frac))(positions)
        indices = Lambda(_iterative_farthest_point_order)(
            [positions, num_samples])
        full_indices = layer_utils.lambda_call(_full_indices, indices,
                                               num_points)
        positions = layer_utils.gather(positions, full_indices)
        if normals is not None:
            normals = layer_utils.gather(positions, full_indices)

    (
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        batch_row_offsets,
    ) = compute_batched_edges(coords, normals, radii,
                              pooler_factory(sample_frac))

    sizes = tuple(layer_utils.leading_dim(c) for c in all_coords)
    extractor = extractor_factory(flat_node_indices, row_splits, sizes)

    nodes_features = all_normals
    edge_features = flat_rel_coords


if __name__ == '__main__':
    from absl import logging

    def make_normals():
        normals = np.random.randn(n, 3).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    do_profile = False
    vis_single = False
    vis_batch = False

    # do_profile = True
    # vis_single = True
    vis_batched = True

    depth = 4
    r0 = 1

    logging.set_verbosity(logging.INFO)
    n = 1024
    coords = np.random.uniform(size=(n, 3)).astype(np.float32)
    scale_factor = np.mean(cKDTree(coords).query(
        coords, k=11)[0][:, -1])  # mean 10th neighbor
    coords /= scale_factor  # now the 10th neighbor is on average 1 unit away

    normals = make_normals()

    radii = get_smart_radii(r0=0.2)
    # pooler = poolers.InverseDensitySamplePooler(0.5)
    pooler = poolers.SlicePooler(0.5)

    if do_profile:
        # super simple profiling
        import tqdm
        from time import time
        num_runs = 100
        dt = 0
        logging.info('Running basic profiling')
        for _ in tqdm.tqdm(range(num_runs)):
            coords = np.random.uniform(size=(1024, 3)).astype(np.float32)
            coords /= scale_factor
            normals = make_normals()
            t = time()
            compute_edges_eager(coords, normals, radii=radii, pooler=pooler)
            dt += time() - t
        logging.info('Completed {} runs in {:2f}s, {} runs / s'.format(
            num_runs, dt, num_runs / dt))

    def vis(coords, node_indices, point_index=0):
        import trimesh
        depth = len(coords)
        for i in range(depth):
            ci = coords[i]
            center = ci[point_index]
            for j in range(i + 1):
                cj = coords[j]
                neighbors = node_indices[i][j][point_index]
                print(i, j, ci.shape, cj.shape)
                print(np.max(np.linalg.norm(cj[neighbors] - center, axis=-1)))
                scene = trimesh.Scene()
                # scene.add_geometry(trimesh.PointCloud(ci, color=(0, 255, 0)))
                scene.add_geometry(trimesh.PointCloud(ci, color=(0, 0, 255)))
                scene.add_geometry(
                    trimesh.PointCloud(cj[neighbors], color=(255, 0, 0)))
                scene.add_geometry(
                    trimesh.primitives.Sphere(center=center, radius=0.02))

                scene.show(background=(0, 0, 0))

    if vis_single:
        from deep_cloud.ops.np_utils.ordering import \
            iterative_farthest_point_ordering as ifps_np
        coords, normals, flat_rel_coords, flat_node_indices, row_splits = \
            compute_edges_eager(coords, normals, radii=radii, pooler=pooler)
        # coords = trimesh.creation.icosphere(7).vertices
        # coords = coords[np.random.choice(coords.shape[0], 2048)]
        # coords = coords[coords[:, 1] > 0]
        # tf.enable_eager_execution()
        n = 1024
        coords = np.empty(shape=(n, 3), dtype=np.float32)
        coords[:, :2] = np.random.uniform(size=(n, 2))
        coords[:, 2] = 0
        coords = coords[ifps_np(coords)]
        normals = make_normals()
        coords, normals, flat_rel_coords, flat_node_indices, row_splits = \
            compute_edges(coords, normals, radii=radii, pooler=pooler)
        # compute_edges_eager(coords, normals, radii=radii, pooler=pooler)
        with tf.Session() as sess:
            coords, normals, flat_rel_coords, flat_node_indices, row_splits = \
                sess.run(
                (coords, normals, flat_rel_coords, flat_node_indices,
                 row_splits))

        node_indices = tf.nest.map_structure(ra.RaggedArray.from_row_splits,
                                             flat_node_indices, row_splits)
        vis(coords, node_indices)

    if vis_batched:
        coords = np.random.uniform(size=(2, n, 3)).astype(np.float32)
        coords /= scale_factor
        normals = np.random.randn(2, n, 3).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        labels = np.array([0, 1], dtype=np.int64)
        dataset = tf.data.Dataset.from_tensor_slices(
            ((coords, normals), labels))

        with builder.MetaNetworkBuilder() as b:
            coords = b.prebatch_input(shape=(n, 3), dtype=tf.float32)
            normals = b.prebatch_input(shape=(n, 3), dtype=tf.float32)
            nested_out = compute_batched_edges(coords,
                                               normals,
                                               radii=radii,
                                               pooler=pooler)
            flat_out = tf.nest.flatten(nested_out)
            model_inps = [b.as_model_input(f) for f in flat_out]
            labels = b.batched(b.prebatch_input(shape=(), dtype=tf.int32))

        preprocessor = b.preprocessor((labels,))
        dataset = preprocessor.map_and_batch(dataset, batch_size=2)
        out = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        values, labels = out
        print(len(tf.nest.flatten(values)))
        with tf.Session() as sess:
            values = sess.run(values)

        nested_out = tf.nest.pack_sequence_as(nested_out, values)
        for i in range(2):
            coords, normals, flat_rel_coords, flat_node_indices, row_splits, \
                batch_row_offsets = tf.nest.map_structure(
                    lambda x: x[i], nested_out)
            vis(coords, node_indices, point_index=0)
