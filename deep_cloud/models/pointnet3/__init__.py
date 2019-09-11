from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import tensorflow as tf
from more_keras.models import mlp
from more_keras.layers import Dropout
from more_keras import spec
from more_keras.ops import utils as op_utils
from deep_cloud.ops import edge
from deep_cloud.ops import asserts
from deep_cloud.ops import ordering
from deep_cloud.ops import cloud
from deep_cloud.ops.np_utils.tree_utils import spatial
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import truncate


def multi_scale_group(in_tree, out_tree, radii, max_neighbors):
    """
    Get neighborhoods across multiple radii.

    Based on MSG from pointnet++.

    Args:
        in_tree: input KDTree
        out_tree: output KDTree
        radii: [N] float radii to search. Assumed to be in ascending order.
        max_neighbors: [N] ints maximum number of neighbors.

    Returns:
        [N] RaggedArrays, corresponding to zipped radii/limits.
    """
    return tuple(
        out_tree.query_ball_tree(in_tree, radius, max_neighbors=mn)
        for radius, mn in zip(radii, max_neighbors))


def _query_all_pairs(
        *all_coords,
        tree_impl=None,
        radii_lists=None,
        max_neighbors_lists=None):
    trees = [tree_impl(c.numpy()) for c in all_coords]
    values = []
    row_splits = []
    for i, (radii, max_neighbors) in enumerate(zip(
            radii_lists, max_neighbors_lists)):
        n = multi_scale_group(
            trees[i], trees[i + 1], radii, max_neighbors)
        row_splits.extend(ni.row_splits for ni in n)
        values.extend(ni.values for ni in n)
    values.extend(row_splits)
    return values


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def pre_batch_map(
        features,
        labels,
        weights=None,
        reorder=False,
        sample_fracs=(0.5, 0.125),
        radii_lists=((0.1, 0.2, 0.4), (0.2, 0.4, 0.8)),
        max_neighbors_lists=((16, 32, 128), (32, 64, 128)),
        return_all_coords=False,
        tree_impl=pykd.KDTree,
):
    depth = len(radii_lists)
    if len(sample_fracs) != depth:
        raise ValueError(
            'Expected length of `sample_fracs` and `radii_lists` to be the '
            'same, but {} != {}'.format(len(sample_fracs), depth))
    if len(max_neighbors_lists) != depth:
        raise ValueError(
            'Expected length of `max_neighbors_lists` and `radii_lists` to be '
            'the same, but {} != {}'.format(len(max_neighbors_lists), depth))
    if isinstance(features, dict):
        coords = features['positions']
        normals = features['normals']
    else:
        coords = features
        normals = None

    static_shape = coords.shape[0] is not None
    if static_shape:
        num_points = tuple(int(f * coords.shape[0]) for f in sample_fracs)
    else:
        num_points = tuple(
            tf.cast(f * tf.shape(coords)[0], tf.int64) for f in sample_fracs)

    if reorder:
        f = sample_fracs[0]
        n = int(f*coords.shape[0]) if static_shape else tf.cast(
            f*tf.shape(coords)[0], tf.int64)
        indices = ordering.iterative_farthest_point_order(coords, n)
        if normals is None:
            coords = ordering.partial_reorder(indices, coords)
        else:
            coords, normals = ordering.partial_reorder(indices, coords, normals)

    all_coords = (coords, *(coords[:n] for n in num_points))

    n_out = sum(len(r) for r in radii_lists)
    fn = functools.partial(
        _query_all_pairs,
        tree_impl=tree_impl,
        radii_lists=radii_lists,
        max_neighbors_lists=max_neighbors_lists
    )
    out = tf.py_function(fn, all_coords, [tf.int64] * n_out * 2)

    flat_node_indices = out[:n_out]
    for fni in flat_node_indices:
        fni.set_shape((None,))
    flat_node_indices = tf.nest.pack_sequence_as(radii_lists, flat_node_indices)

    row_splits = out[n_out:]
    row_splits = tf.nest.pack_sequence_as(radii_lists, row_splits)

    if static_shape:
        for rsi in tf.nest.flatten(row_splits):
            rsi.set_shape((None,))
    else:
        for rs, c in zip(row_splits, all_coords[1:]):
            for rsi in rs:
                rsi.set_shape((c.shape[0] + 1,))

    node_indices = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
                                         flat_node_indices, row_splits)
    rel_coords = []
    for i, indices in enumerate(node_indices):
        rel_coords.append(
            tuple(
                cloud.get_relative_coords(all_coords[i], all_coords[i + 1], ind)
                for ind in indices))
    rel_coords = tuple(rel_coords)

    cloud_sizes = tuple(tf.shape(c, out_type=tf.int64)[0] for c in all_coords)
    if not return_all_coords:
        all_coords = all_coords[-1]
    if normals is None:
        normals = ()  # can't return None
    features = normals, rel_coords, node_indices, all_coords, cloud_sizes

    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def post_batch_map(features,
                   labels,
                   weights=None,
                   return_all_outer_row_splits=False):
    normals, rel_coords, node_indices, coords, cloud_sizes = features
    offsets = tuple(op_utils.row_lengths_to_splits(s)[:-1] for s in cloud_sizes)
    # Not sure why this is necessary?
    # for ni in tf.nest.flatten(node_indices):
    #     ni.flat_values.set_shape((None,))
    node_indices = tuple(
        tuple(op_utils.apply_row_offset(ni, offset)
              for ni in n)  # pylint: disable=not-an-iterable
        for n, offset in zip(node_indices, offsets))

    flat_normals = (
        () if normals == () else op_utils.flatten_leading_dims(normals))
    flat_rel_coords = tf.nest.map_structure(lambda rc: rc.flat_values,
                                            rel_coords)

    if return_all_outer_row_splits:
        outer_row_splits = tuple(
            ni[0].nested_row_splits[0] for ni in node_indices)
    else:
        outer_row_splits = node_indices[-1][0].nested_row_splits[0]

    row_splits = tf.nest.map_structure(lambda ni: ni.nested_row_splits[1],
                                       node_indices)
    flat_node_indices = tf.nest.map_structure(lambda ni: ni.flat_values,
                                              node_indices)

    flat_coords = tf.nest.map_structure(op_utils.flatten_leading_dims, coords)

    features = (flat_normals, flat_rel_coords, flat_node_indices, row_splits,
                flat_coords, outer_row_splits)
    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable(blacklist=['features', 'num_classes'])
def get_logits(features, num_classes, dropout_rate=0.5):
    return mlp((512, 256),
               final_units=num_classes,
               dropout_impl=functools.partial(Dropout,
                                              rate=dropout_rate))(features)


def ragged_reduce(args, reduction, **kwargs):
    value, *row_splits = args
    rt = tf.RaggedTensor.from_nested_row_splits(value, row_splits)
    out = reduction(rt, **kwargs)
    if isinstance(out, tf.Tensor):
        return out
    else:
        return [out, *out.nested_row_splits]


@gin.configurable(blacklist=[
    'node_features', 'flat_rel_coords', 'flat_node_indices', 'row_splits'])
def pointnet_block(node_features,
                   flat_rel_coords,
                   flat_node_indices,
                   row_splits,
                   edge_network_fn,
                   reduction=tf.reduce_max):
    """
    Args:
        node_features: [ni, f] float node features for entire batch.
        flat_rel_coords: [ne, num_dims] float relative coords between edges.
        flat_node_indices: [ne] int indices of points in the input cloud, or
            None. If None, ne == ni. This achieves global pooling.
        row_splits: [no+1] int tensor, ragged component of flat_*.
        edge_network_fn: list/tuple of ints.
        reduction: reduction to be applied across the neighborhood.

    Returns:
        [no, units[-1]] float output node features
    """
    if node_features is not None:
        asserts.assert_flat_tensor('node_features', node_features, 2,
                                asserts.FLOAT_TYPES)
    asserts.assert_flat_tensor('flat_rel_coords', flat_rel_coords, 2,
                               asserts.FLOAT_TYPES)
    if flat_node_indices is not None:
        asserts.assert_flat_tensor('flat_node_indices', flat_node_indices, 1,
                                   asserts.INT_TYPES)
    asserts.assert_flat_tensor('row_splits', row_splits, 1, asserts.INT_TYPES)
    asserts.assert_callable('edge_network_fn', edge_network_fn)
    asserts.assert_callable('reduction', reduction)

    if node_features is None:
        flat_edge_features = flat_rel_coords
    else:
        if flat_node_indices is None:
            flat_edge_features = node_features
        else:
            flat_edge_features = tf.gather(node_features, flat_node_indices)
        flat_edge_features = tf.concat([flat_edge_features, flat_rel_coords],
                                       axis=-1)
    flat_edge_features = edge_network_fn(flat_edge_features)
    edge_features = tf.RaggedTensor.from_row_splits(flat_edge_features,
                                                    row_splits)
    node_features = tf.keras.layers.Lambda(
        ragged_reduce, arguments=dict(reduction=reduction, axis=1))(
            [edge_features.flat_values, *edge_features.nested_row_splits])
    if isinstance(node_features, (list, tuple)):
        node_features = tf.RaggedTensor.from_nested_row_splits(
            node_features[0], node_features[1:])
    return node_features


@gin.configurable
def mlp_recurse(units, **kwargs):
    if all(isinstance(u, int) for u in units):
        return mlp(units, **kwargs)
    else:
        return tuple(mlp_recurse(u, **kwargs) for u in units)


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def pointnet3_classifier(input_spec,
                         output_spec,
                         layer_network_lists=None,
                         global_network=None,
                         logits_network=get_logits,
                         reduction=tf.reduce_max):
    if layer_network_lists is None:
        layer_network_lists = (
            (mlp((32, 32, 64)), mlp((64, 64, 128)), mlp((64, 96, 128))),
            (mlp((64, 64, 128)), mlp((128, 128, 256)), mlp((128, 128, 256))),
        )

    if not (isinstance(layer_network_lists, (list, tuple)) and
            all(isinstance(lns, (list, tuple)) and
                all(callable(ln) for ln in lns)
            for lns in layer_network_lists)):
        raise ValueError(
            'layer_networks should be a list/tuple of list/tuples of networks'
            ' got {}'.format(layer_network_lists))
    if global_network is None:
        global_network = mlp((256, 512, 1024))

    inputs = spec.inputs(input_spec)
    (flat_normals, all_rel_coords, all_node_indices, all_row_splits,
     final_coords, outer_row_splits) = inputs
    if isinstance(final_coords, tuple):
        final_coords = final_coords[-1]

    if len(layer_network_lists) != len(all_rel_coords):
        raise ValueError(
            'Expected same number of layer networks as all_rel_coords '
            'but {} != {}'.format(
                len(layer_network_lists), len(all_rel_coords)))

    node_features = None if flat_normals == () else flat_normals
    for layer_networks, rel_coords, node_indices, row_splits in zip(
            layer_network_lists, all_rel_coords, all_node_indices,
            all_row_splits):
        node_features = [
            pointnet_block(node_features, rc, ni, rs, ln, reduction)
            for (rc, ni, rs, ln) in zip(
                rel_coords, node_indices, row_splits, layer_networks)]
        node_features = tf.concat(node_features, axis=-1)

    global_features = pointnet_block(node_features,
                                     final_coords,
                                     None,
                                     outer_row_splits,
                                     global_network,
                                     reduction=reduction)

    logits = get_logits(global_features, output_spec.shape[-1])
    inputs = tf.nest.flatten(inputs)
    return tf.keras.Model(inputs=inputs, outputs=logits)


if __name__ == '__main__':
    import numpy as np
    n = 1024
    batch_size = 2
    coords = np.random.uniform(size=(batch_size, n, 3)).astype(np.float32)

    # c0 = [coords[0]]
    # max_neighbors_lists = ((32,), (64,))
    # radii_lists = ((0.1,), (0.2,))
    # for _ in max_neighbors_lists:
    #     c0.append(c0[-1][:len(c0[-1]) // 2])

    # trees = [spatial.KDTree(c) for c in c0]
    # groups = []
    # for i, (radii, max_neighbors) in enumerate(
    #         zip(radii_lists, max_neighbors_lists)):
    #     groups.append(
    #         multi_scale_group(trees[i], trees[i+1], radii, max_neighbors))
    # for g in groups:
    #     for gi in g:
    #         print(gi.values.shape, gi.row_splits.shape)

    neighbors = spatial.KDTree(coords[0]).query_ball_tree(
        spatial.KDTree(coords[1]), 0.2, max_neighbors=None)
    coords[..., 2] = 0.1 * np.reshape(np.arange(batch_size),
                                      (-1, 1))  # separate planes
    normals = np.random.normal(size=(batch_size, n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    labels = np.zeros((batch_size,), dtype=np.int64)

    with tf.Graph().as_default():    # pylint: disable=not-context-manager
        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(positions=coords, normals=normals), labels))
        dataset = dataset.map(
            functools.partial(pre_batch_map,
                              return_all_coords=True,
                              max_neighbors_lists=((32,), (64,)),
                              radii_lists=((0.1,), (0.2,)),
                              reorder=True))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            functools.partial(post_batch_map, return_all_outer_row_splits=True))
        features, labels = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()
        with tf.Session() as sess:
            features = sess.run(features)
    (flat_normals, flat_rel_coords, flat_node_indices, row_splits, flat_coords,
     outer_row_splits) = features

    flat_rel_coords, = zip(*flat_rel_coords)
    row_splits, = zip(*row_splits)
    flat_node_indices, = zip(*flat_node_indices)

    radii = (0.2, 0.4)

    def vis(coords, node_indices, row_splits, outer_row_splits):
        import trimesh
        depth = len(node_indices)

        for i in range(depth):
            scene = trimesh.scene.Scene()
            p0 = trimesh.PointCloud(coords[i], color=(255, 255, 255))
            p1 = trimesh.PointCloud(coords[i + 1], color=(0, 0, 255))
            geometries = [p0, p1]
            rs = row_splits[i]
            ni = node_indices[i]
            for j in outer_row_splits[i][:-1]:
                geometries.append(
                    trimesh.primitives.Sphere(center=coords[i + 1][j],
                                              radius=0.02,
                                              color=(255, 255, 255)))
                geometries.append(
                    trimesh.PointCloud(coords[i][ni[rs[j]:rs[j + 1]]],
                                       color=(255, 0, 0)))
            for g in geometries:
                scene.add_geometry(g)
            scene.show(background=(0, 0, 0))

    vis(flat_coords, flat_node_indices, row_splits, outer_row_splits)
