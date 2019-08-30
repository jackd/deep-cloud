from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import gin

import tensorflow as tf
import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from more_keras.ops import utils as op_utils
from more_keras.layers import utils as layer_utils
from more_keras.layers import Dense
from more_keras.models import mlp

from deep_cloud.ops.np_utils import ordering as np_ordering
from deep_cloud.models.very_dense import utils
from deep_cloud.models.very_dense import poolers
from deep_cloud.models.very_dense import extractors
from more_keras.framework import pipelines
from more_keras import spec


@gin.configurable
def exponential_radii(depth=4, r0=0.1, expansion_rate=2):
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
    from deep_cloud.ops.np_utils import tree_utils
    from deep_cloud.ops.np_utils import cloud as np_cloud
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

        def fn(coords, normals):
            # coords, normals = args
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
        out_flat = tf.py_function(fn, [coords, normals], Tout_flat)
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


@gin.configurable
def prebatch_map(features,
                 labels,
                 weights=None,
                 radii=None,
                 pooler=None,
                 reorder=False):
    if radii is None:
        radii = exponential_radii()
    if pooler is None:
        pooler = poolers.SlicePooler()
    coords = features['positions']
    normals = features['normals']
    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        compute_edges(coords, normals, radii, pooler, reorder=reorder)
    if any(c.shape[0] is None for c in all_coords):
        raise NotImplementedError('TODO')

    rel_coords = tf.nest.map_structure(
        lambda rc, rs: tf.RaggedTensor.from_row_splits(rc, rs), flat_rel_coords,
        row_splits)
    node_indices = tf.nest.map_structure(
        lambda ni, rs: tf.RaggedTensor.from_row_splits(ni, rs),
        flat_node_indices, row_splits)

    all_coords = tuple(all_coords)
    all_normals = tuple(all_normals)
    rel_coords = utils.ttuple(rel_coords)
    node_indices = utils.ttuple(node_indices)
    features = (
        all_coords,
        all_normals,
        rel_coords,
        node_indices,
    )
    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable
def post_batch_map(features, labels, weights=None):
    all_coords, all_normals, rel_coords, node_indices = features
    offsets = [op_utils.get_row_offsets(c) for c in all_coords]
    outer_row_splits = tuple(op_utils.get_row_splits(c) for c in all_coords)
    row_splits = tf.nest.map_structure(lambda rt: rt.nested_row_splits[1],
                                       node_indices)
    all_coords, all_normals = tf.nest.map_structure(
        op_utils.flatten_leading_dims, (all_coords, all_normals))

    flat_rel_coords = tf.nest.map_structure(lambda x: x.flat_values, rel_coords)

    K = len(all_coords)
    flat_node_indices = utils.lower_triangular(K)
    for i in range(K):
        for j in range(i + 1):
            flat_node_indices[i][j] = op_utils.apply_row_offset(
                node_indices[i][j], offsets[j]).flat_values
    flat_node_indices = utils.ttuple(flat_node_indices)

    all_coords = tuple(all_coords)
    all_normals = tuple(all_normals)

    features = (all_coords, all_normals, flat_rel_coords, flat_node_indices,
                row_splits, outer_row_splits)
    return ((features, labels) if weights is None else
            (features, labels, weights))


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


@gin.configurable
def get_base_node_network_factories(num_factories=4,
                                    units_scale=4,
                                    unit_expansion_factor=2,
                                    network_depth=2):
    return [
        functools.partial(mlp,
                          units=[4 * units_scale * unit_expansion_factor**i] *
                          network_depth) for i in range(num_factories)
    ]


@gin.configurable
def get_base_global_network(units=(512, 256), dropout_impl=None):
    return functools.partial(mlp, units=units, dropout_impl=dropout_impl)


@gin.configurable
def very_dense_classifier(
        input_spec,
        output_spec,
        repeats=3,
        extractor_factory=extractors.get_base_extractor,
        node_network_factories=None,
        global_network_factory=get_base_global_network,
        dense_factory=Dense,
        concat_global_features=True,
        concat_node_features=False,
        concat_edge_features=False,
        reduction=tf.reduce_max,
):
    num_classes = output_spec.shape[-1]

    (
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        outer_row_splits,
    ) = spec.inputs(input_spec)

    if node_network_factories is None:
        node_network_factories = get_base_node_network_factories(
            len(all_coords))

    sizes = [layer_utils.leading_dim(c) for c in all_coords]

    preds = []

    inp_node_features = tuple(all_normals)
    inp_edge_features = utils.ttuple(flat_rel_coords)
    inp_global_features = None

    for _ in range(repeats + 1):
        node_features, edge_features = extractor_factory(
            sizes, flat_node_indices, row_splits)(
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
        if inp_global_features is not None and concat_global_features:
            global_features.append(inp_global_features)
        global_features = layer_utils.concat(global_features, axis=-1)
        global_features = global_network_factory()()(global_features)  # FIX
        preds.append(Dense(num_classes, activation=None)(global_features))

        node_features = tuple(node_features)
        edge_features = utils.ttuple(edge_features)
        inp_node_features = (tf.nest.map_structure(
            lambda a, b: tf.concat([a, b], axis=-1), inp_node_features,
            node_features) if concat_node_features else node_features)
        inp_edge_features = (tf.nest.map_structure(
            lambda a, b: tf.concat([a, b], axis=-1), inp_edge_features,
            edge_features) if concat_edge_features else edge_features)
        inp_global_features = global_features

    inputs = tf.nest.flatten((
        all_coords,
        all_normals,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        outer_row_splits,
    ))
    return tf.keras.Model(inputs=inputs, outputs=preds)


if __name__ == '__main__':
    from more_keras.ragged import np_impl as ra
    import functools
    depth = 4
    sample_frac = 0.25
    expansion_rate = 2
    n = 1024
    batch_size = 2

    pooler = poolers.SlicePooler(sample_frac)
    radii = exponential_radii(depth=depth, r0=1, expansion_rate=expansion_rate)

    base_coords = np.random.uniform(size=(n, 3)).astype(np.float32)
    base_coords[:, 2] = 0
    scale_factor = np.mean(
        cKDTree(base_coords).query(base_coords,
                                   k=11)[0][:, -1])  # mean 10th neighbor
    base_coords /= scale_factor
    indices = np_ordering.iterative_farthest_point_ordering(base_coords)
    base_coords = base_coords[indices]
    base_normals = np.random.randn(n, 3).astype(np.float32)
    base_normals /= np.linalg.norm(base_normals, axis=-1, keepdims=True)

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

    coords = np.stack([base_coords] * 2, axis=0)
    coords[1, :, 2] = 1
    normals = np.stack([base_normals] * 2, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((dict(positions=coords,
                                                       normals=normals), [0,
                                                                          0]))
    dataset = dataset.map(
        functools.partial(prebatch_map,
                          reorder=False,
                          radii=radii,
                          pooler=pooler)).batch(batch_size).map(post_batch_map)

    features, labels = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    with tf.Session() as sess:
        features, labels = sess.run((features, labels))

    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        features

    row_splits = utils.pack_lower_triangle(
        [b[1] for b in utils.flatten_lower_triangle(row_splits)])
    row_splits = utils.ttuple(row_splits)

    node_indices = tf.nest.map_structure(ra.RaggedArray.from_row_splits,
                                         flat_node_indices, row_splits)
    vis(all_coords, node_indices)
