from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import gin

import tensorflow as tf
import numpy as np
from more_keras.ops import utils as op_utils
from more_keras.layers import utils as layer_utils
from more_keras.layers import Dense
from more_keras.models import mlp

from deep_cloud.ops.np_utils import ordering as np_ordering
from deep_cloud.ops.np_utils.tree_utils import spatial
from deep_cloud.models.very_dense import utils
from deep_cloud.models.very_dense import poolers
from deep_cloud.models.very_dense import extractors

from deep_cloud.ops.np_utils.tree_utils import core
from deep_cloud.ops.np_utils import cloud as np_cloud

from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import spatial
from deep_cloud.ops.np_utils.tree_utils import skkd
# from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module

from more_keras.framework import pipelines
from more_keras.framework.problems import get_current_problem
from more_keras import spec
import collections

DEFAULT_TREE = pykd.KDTree
# DEFAULT_TREE = spatial.KDTree


@gin.configurable
def exponential_radii(depth=4, r0=1., expansion_rate=2):
    return r0 * expansion_rate**np.arange(depth)


def rejection_sample_lazy(tree, points, radius, k0):
    N = points.shape[0]
    out = []
    consumed = np.zeros((N,), dtype=np.bool)
    for i in range(N):
        if not consumed[i]:
            out.append(i)
            indices = tree.query_ball_point(np.expand_dims(points[i], 0),
                                            radius,
                                            approx_neighbors=k0)
            indices = indices[0]
            consumed[indices] = True
    return out


@gin.configurable
def compute_edges_principled_eager_fn(depth=4, k0=16, tree_impl=DEFAULT_TREE):
    return functools.partial(
        compute_edges_principled_eager,
        depth=depth,
        k0=k0,
        tree_impl=tree_impl,
    )


def compute_edges_principled_eager(coords,
                                   normals,
                                   depth=4,
                                   k0=16,
                                   tree_impl=DEFAULT_TREE):

    coords = coords.numpy() if hasattr(coords, 'numpy') else coords
    normals = normals.numpy() if hasattr(normals, 'numpy') else normals

    tree = tree_impl(coords)
    dists, indices = tree.query(tree.data, 2, return_distance=True)
    # closest = np.min(dists[:, 1])
    scale = np.mean(dists[:, 1])
    assert (scale > 0)
    coords *= (2 / scale)

    # coords is now a packing of barely-intersecting spheres of radius 1.
    all_coords = [coords]
    all_normals = [normals]
    tree = tree_impl(coords)
    trees = [tree]

    base_coords = coords
    base_tree = tree
    base_normals = normals

    # perform sampling, build trees
    radii = 4 * np.power(2, np.arange(depth))
    ## Rejection sample on original cloud
    for i, radius in enumerate(radii[:-1]):
        indices = rejection_sample_lazy(base_tree,
                                        base_coords,
                                        radius,
                                        k0=k0 * 4**i)
        coords = base_coords[indices]
        tree = tree_impl(coords)

        all_coords.append(coords)
        all_normals.append(base_normals[indices])
        trees.append(tree)

    # compute edges
    flat_node_indices = utils.lower_triangular(depth)
    flat_rel_coords = utils.lower_triangular(depth)
    row_splits = utils.lower_triangular(depth)

    for i in range(depth):
        for j in range(i + 1):
            indices = trees[j].query_ball_point(all_coords[i],
                                                radii[i],
                                                approx_neighbors=k0 *
                                                4**(i - j))
            flat_node_indices[i][j] = fni = indices.flat_values.astype(np.int64)
            row_splits[i][j] = indices.row_splits.astype(np.int64)

            # compute flat_rel_coords
            # this could be done outside the py_function, but it uses np.repeat
            # which is faster than tf.repeat on cpu.
            flat_rel_coords[i][j] = np_cloud.get_relative_coords(
                all_coords[j],
                all_coords[i],
                fni,
                row_lengths=indices.row_lengths)

    return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits)


@gin.configurable
def compute_edges_eager_fn(depth=4,
                           radii=None,
                           pooler=None,
                           reorder=False,
                           tree_impl=DEFAULT_TREE):
    if radii is None:
        radii = exponential_radii(r0=0.1, depth=depth)
    else:
        assert (len(radii) == depth)
    if pooler is None:
        pooler = poolers.SlicePooler()

    return functools.partial(compute_edges_eager,
                             radii=radii,
                             pooler=pooler,
                             reorder=reorder,
                             tree_impl=tree_impl)


def compute_edges_eager(coords,
                        normals,
                        radii,
                        pooler,
                        reorder=False,
                        tree_impl=DEFAULT_TREE):
    """
    Recursively sample the input cloud and find edges based on ball searches.

    Args:
        coords: [N_0, num_dims] numpy array or eager tensor of cloud
            coordinates.
        normals: [N_0, num_features] numpy array of node features.
        radii: [depth] numpy array or eager tensor of radii for ball searches.
        pooler: `Pooler` instance.
        tree_impl: KDTree implementation.

    Returns:
        See `compute_edges` return values.
    """
    from deep_cloud.ops.np_utils import tree_utils
    from deep_cloud.ops.np_utils import cloud as np_cloud
    depth = len(radii)
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
        if normals is None:
            coords = np_ordering.partial_reorder(indices, coords)
        else:
            coords, normals = np_ordering.partial_reorder(
                indices, coords, normals)

    node_indices = utils.lower_triangular(depth)
    trees = [None for _ in range(depth)]
    all_coords = [coords]
    all_normals = [normals]

    # TODO: The below uses `depth * (depth + 1) // 2` ball searches
    # we can do it with `depth`. When depth is 4 it's not that big a saving...

    # do the edges diagonal and build up coordinates
    # print('---')
    for i, radius in enumerate(radii[:-1]):
        tree = trees[i] = tree_impl(coords)
        indices = node_indices[i][i] = tree.query_pairs(radius,
                                                        approx_neighbors=16 *
                                                        2**i)
        # print(radius, tree.n)
        # print(np.mean(indices.row_lengths))
        coords, normals, indices = pooler(coords, normals, indices)
        # node_indices[i + 1][i] = indices
        all_coords.append(coords)
        all_normals.append(normals)

    # final cloud
    tree = trees[-1] = tree_impl(coords)
    node_indices[-1][-1] = tree.query_pairs(radii[-1],
                                            approx_neighbors=16 *
                                            2**(depth - 1))

    # do below the diagonal, i.e. [i, j], i > j + 1
    for i in range(1, depth):
        in_tree = trees[i]
        radius = radii[i]
        for j in range(i):
            node_indices[i][j] = in_tree.query_ball_tree(trees[j],
                                                         radius,
                                                         approx_neighbors=16 *
                                                         2**(2 * i - j))

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
        lambda x: x.flat_values.astype(np.int64), node_indices)
    row_splits = tf.nest.map_structure(lambda x: x.row_splits.astype(np.int64),
                                       node_indices)

    if normals is None:
        return (all_coords, flat_rel_coords, flat_node_indices, row_splits)
    else:
        return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
                row_splits)


def _flatten_output(fn, *args, **kwargs):
    return tf.nest.flatten(fn(*args, **kwargs))


@gin.configurable
def compute_edges(coords, normals, depth=4, eager_fn=None):
    """
    Graph-mode wrapper for compute_edges implementation.

    Recursively sample the input cloud and find edges based on ball searches.

    Args:
        coords: [N_0, num_dims] numpy array or eager tensor of cloud
            coordinates.
        normals: [N_0, num_features] numpy array of node features.
        eager_fn: function mapping eager versions of coords, normals, depth
            to outputs described below.

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
            `rel_coords[i][j][p]` (k?, num_dims) gives the relative coordinates
            of all points in cloud `i` in the neighborhood of point `p` in cloud
            `j`.
        node_indices: [depth, <=depth] list of lists of [N_i, k?] ints.
            see below. `node_indices[i][j][p] == (r, s, t)` indicates that
            node `p` in cloud `j` neighbors nodes `r, s, t` in cloud `i`.
    """
    if eager_fn is None:
        eager_fn = compute_edges_eager_fn()
    if tf.executing_eagerly():
        return eager_fn(coords, normals)

    py_func = functools.partial(eager_fn, depth=depth)
    if normals is None:
        Tout = (
            [tf.float32] * depth,  # coords
            utils.lower_triangular(depth, tf.float32),  # flat_rel_coords
            utils.lower_triangular(depth, tf.int64),  # flat_node_indices
            utils.lower_triangular(depth, tf.int64),  # row_splits
        )
        Tout_flat = tf.nest.flatten(Tout)
        # out_flat = tf.py_function(fn, [coords, normals], Tout_flat)
        out_flat = tf.py_function(py_func, [coords], Tout_flat)

        (all_coords, flat_rel_coords, flat_node_indices,
         row_splits) = tf.nest.pack_sequence_as(Tout, out_flat)
        all_normals = None
    else:
        Tout = (
            [tf.float32] * depth,  # coords
            [tf.float32] * depth,  # normals
            utils.lower_triangular(depth, tf.float32),  # flat_rel_coords
            utils.lower_triangular(depth, tf.int64),  # flat_node_indices
            utils.lower_triangular(depth, tf.int64),  # row_splits
        )
        Tout_flat = tf.nest.flatten(Tout)
        # out_flat = tf.py_function(fn, [coords, normals], Tout_flat)
        out_flat = tf.py_function(py_func, [coords, normals], Tout_flat)

        (all_coords, all_normals, flat_rel_coords, flat_node_indices,
         row_splits) = tf.nest.pack_sequence_as(Tout, out_flat)

    # fix sizes
    num_dims = coords.shape[1]
    # num_features = normals.shape[1]
    # sizes = [coords.shape[0]]
    # even if size is None, we still get rank information from the below
    # pooler = poolers.SlicePooler()
    # for _ in range(depth - 1):
    #     sizes.append(pooler.output_size(sizes[-1]))
    sizes = [None] * 4

    for i in range(depth):
        size = sizes[i]
        all_coords[i].set_shape((size, num_dims))
        if normals is not None:
            all_normals[i].set_shape((size, normals.shape[1]))
        for rc in flat_rel_coords[i]:
            rc.set_shape((None, num_dims))
        for ni in flat_node_indices[i]:
            ni.set_shape((None,))
        for rs in row_splits[i]:
            rs.set_shape((None if size is None else (size + 1),))

    return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
            row_splits)


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def pre_batch_map(features,
                  labels,
                  weights=None,
                  shuffle=False,
                  edge_fn=compute_edges):

    def flat_values(tensor):
        if isinstance(tensor, tf.RaggedTensor):
            return tensor.flat_values
        else:
            return tensor

    if isinstance(features, dict):
        coords = flat_values(features['positions'])
        normals = flat_values(features['normals'])
    else:
        coords = flat_values(features)
        normals = None
    class_masks = (features.get('class_masks')
                   if isinstance(features, dict) else None)
    if shuffle:
        indices = tf.random.shuffle(tf.range(tf.shape(coords)[0]))
        coords = tf.gather(coords, indices)
        if normals is not None:
            normals = tf.gather(normals, indices)

    all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits = \
        edge_fn(coords, normals)

    rel_coords = tf.nest.map_structure(
        lambda rc, rs: tf.RaggedTensor.from_row_splits(rc, rs), flat_rel_coords,
        row_splits)
    node_indices = tf.nest.map_structure(
        lambda ni, rs: tf.RaggedTensor.from_row_splits(ni, rs),
        flat_node_indices, row_splits)

    all_coords = tuple(all_coords)
    if normals is not None:
        all_normals = tuple(all_normals)
    rel_coords = utils.ttuple(rel_coords)
    node_indices = utils.ttuple(node_indices)

    # make ragged so batching knows.
    all_coords, all_normals = tf.nest.map_structure(
        lambda x: None if x is None else tf.RaggedTensor.from_tensor(
            tf.expand_dims(x, axis=0)), (all_coords, all_normals))

    features = dict(
        all_coords=all_coords,
        all_normals=all_normals,
        rel_coords=rel_coords,
        node_indices=node_indices,
    )
    if normals is None:
        del features['all_normals']
    if class_masks is not None:
        features['class_masks'] = class_masks

    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable
def post_batch_map(features, labels, weights=None):
    all_coords, rel_coords, node_indices = (
        features[k] for k in ('all_coords', 'rel_coords', 'node_indices'))
    # remove redundant ragged dimension added for batching purposes.
    all_coords = tf.nest.map_structure(
        lambda x: tf.RaggedTensor.from_nested_row_splits(
            x.flat_values, x.nested_row_splits[1:]), all_coords)
    offsets = [op_utils.get_row_offsets(c) for c in all_coords]
    outer_row_splits = tuple(op_utils.get_row_splits(c) for c in all_coords)
    row_splits = tf.nest.map_structure(lambda rt: rt.nested_row_splits[1],
                                       node_indices)

    all_coords = tf.nest.map_structure(layer_utils.flatten_leading_dims,
                                       all_coords)

    flat_rel_coords = tf.nest.map_structure(lambda x: x.flat_values, rel_coords)

    K = len(all_coords)
    flat_node_indices = utils.lower_triangular(K)
    for i in range(K):
        for j in range(i + 1):
            flat_node_indices[i][j] = layer_utils.apply_row_offset(
                node_indices[i][j], offsets[j]).flat_values
    flat_node_indices = utils.ttuple(flat_node_indices)

    all_coords = tuple(all_coords)

    class_index = features.get('class_index')
    class_masks = features.get('class_masks')

    features = dict(all_coords=all_coords,
                    flat_rel_coords=flat_rel_coords,
                    flat_node_indices=flat_node_indices,
                    row_splits=row_splits,
                    outer_row_splits=outer_row_splits)

    all_normals = features.get('all_normals')
    if all_normals is not None:
        all_normals = tf.nest.map_structure(
            lambda x: tf.RaggedTensor.from_nested_row_splits(
                x.flat_values, x.nested_row_splits[1:]), all_normals)
        all_normals = tf.nest.map_structure(layer_utils.flatten_leading_dims,
                                            all_normals)
        all_normals = tuple(all_normals)
        features['all_normals'] = all_normals

    if class_index is not None:
        features['class_index'] = class_index
    if class_masks is not None:
        features['class_masks'] = class_masks

    labels, weights = get_current_problem().post_batch_map(labels, weights)
    if isinstance(labels, tf.Tensor) and labels.shape.ndims == 2:
        assert (isinstance(weights, tf.Tensor) and weights.shape.ndims == 2)
        labels = tf.reshape(labels, (-1,))
        weights = tf.reshape(weights, (-1,))

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
    return mlp(units=units, dropout_impl=dropout_impl)


EmbeddingSpec = gin.external_configurable(
    collections.namedtuple('EmbeddingSpec', ['input_dim', 'output_dim']))


def add_residual(inp, output, dense_factory=Dense):
    if inp.shape[-1] != output.shape[-1]:
        inp = dense_factory(output.shape[-1])(inp)
    return tf.add_n([inp, output])


@gin.configurable(blacklist=['inputs'])
def very_dense_features(
        inputs,
        repeats=3,
        global_embedding_spec=None,
        extractor_factory=extractors.get_base_extractor,
        node_network_factories=None,
        global_network_factory=get_base_global_network,
        residual_global_features=False,
        residual_node_features=False,
        residual_edge_features=False,
        reduction=tf.reduce_max,
        dense_factory=Dense,
):
    (
        all_coords,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        outer_row_splits,
    ) = (inputs[k] for k in (
        'all_coords',
        'flat_rel_coords',
        'flat_node_indices',
        'row_splits',
        'outer_row_splits',
    ))
    all_normals = inputs.get('all_normals')
    if node_network_factories is None:
        node_network_factories = get_base_node_network_factories(
            len(all_coords))

    sizes = [layer_utils.leading_dim(c) for c in all_coords]

    inp_node_features = None if all_normals is None else tuple(all_normals)
    inp_edge_features = utils.ttuple(flat_rel_coords)
    if global_embedding_spec is None:
        inp_global_features = None
    else:
        inp_global_features = tf.keras.layers.Embedding(
            global_embedding_spec.input_dim,
            global_embedding_spec.output_dim)(inputs['class_index'])

    all_node_features = []
    all_edge_features = []
    all_global_features = []

    for i in range(repeats + 1):
        node_features, edge_features = extractor_factory(
            sizes, flat_node_indices, row_splits, outer_row_splits)(
                inp_node_features,
                inp_edge_features,
                inp_global_features,
                all_coords if i == 0 else None,
            )

        node_features = [
            network_factory()(n)  # fresh model each time
            for network_factory, n in zip(node_network_factories, node_features)
        ]
        global_features = [
            apply_ragged_reduction(n, rs, reduction)
            for (n, rs) in zip(node_features, outer_row_splits)
        ]
        global_features = tf.concat(global_features, axis=-1)

        if global_network_factory is not None:
            global_features = global_network_factory()(global_features)

        node_features = tuple(node_features)
        edge_features = utils.ttuple(edge_features)

        if inp_global_features is not None and residual_global_features:
            global_features = add_residual(inp_global_features, global_features,
                                           dense_factory)

        if residual_node_features:
            node_features = tf.nest.map_structure(
                lambda inp, out: add_residual(inp, out, dense_factory),
                inp_node_features, node_features)
        if residual_edge_features:
            edge_features = tf.nest.map_structure(
                lambda inp, out: add_residual(inp, out, dense_factory),
                inp_edge_features, edge_features)

        inp_node_features = node_features
        inp_edge_features = edge_features
        inp_global_features = global_features

        all_node_features.append(inp_node_features)
        all_edge_features.append(inp_edge_features)
        all_global_features.append(inp_global_features)

    return all_node_features, all_edge_features, all_global_features


@gin.configurable
def very_dense_classifier(input_spec,
                          output_spec,
                          dense_factory=Dense,
                          features_factory=very_dense_features):
    num_classes = output_spec.shape[-1]
    inputs = spec.inputs(input_spec)
    node_features, edge_features, global_features = features_factory(inputs)
    del node_features, edge_features
    preds = []
    for gf in global_features:
        if gf is not None:
            preds.append(dense_factory(num_classes, activation=None)(gf))

    return tf.keras.Model(inputs=tf.nest.flatten(inputs), outputs=preds)


def _from_row_splits(args):
    return tf.RaggedTensor.from_row_splits(*args)


@gin.configurable
def very_dense_semantic_segmenter(input_spec,
                                  output_spec,
                                  dense_factory=Dense,
                                  features_factory=very_dense_features):
    num_classes = output_spec.shape[-1]
    inputs = spec.inputs(input_spec)
    class_masks = inputs.pop('class_masks', None)
    node_features, edge_features, global_features = features_factory(inputs)
    del edge_features, global_features
    node_features = [nf[0] for nf in node_features]  # high res features
    preds = [dense_factory(num_classes)(n) for n in node_features]
    if_false = tf.fill(tf.shape(preds[0]),
                       value=tf.constant(-np.inf, dtype=tf.float32))
    outer_row_splits = inputs['outer_row_splits'][0]
    outer_row_lengths = op_utils.diff(outer_row_splits)
    if class_masks is not None:
        class_masks = tf.repeat(class_masks, outer_row_lengths, axis=0)
        preds = [tf.where(class_masks, pred, if_false) for pred in preds]
    # from_row_splits = tf.keras.layers.Lambda(_from_row_splits)
    # preds = [from_row_splits([pred, outer_row_splits]) for pred in preds]
    return tf.keras.Model(inputs=tf.nest.flatten(inputs), outputs=preds)


if __name__ == '__main__':
    from more_keras.ragged import np_impl as ra
    import functools

    config = '''
    compute_edges.eager_fn = @compute_edges_principled_eager_fn()
    '''
    gin.parse_config(config)

    def vis(coords, node_indices, point_indices=None):
        import trimesh
        depth = len(coords)

        scale_factor = np.max(np.linalg.norm(coords[0], axis=-1))
        for c in coords:
            c /= scale_factor
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
                    trimesh.primitives.Sphere(center=center, radius=0.01))

                scene.show(background=(0, 0, 0))

    # n_initial = 10000
    # n = 1024
    # batch_size = 2

    # KDTree = spatial.KDTree

    # base_coords = np.random.uniform(size=(n_initial, 3)).astype(np.float32)
    # base_coords[:, 2] = 0
    # indices = np_ordering.iterative_farthest_point_ordering(base_coords, n)
    # base_coords = base_coords[indices]

    # scale_factor = np.mean(
    #     DEFAULT_TREE(base_coords).query(base_coords, 11,
    #                                     np.inf)[1][:, -1])  # mean 10th neighbor
    # base_coords /= scale_factor
    # base_normals = np.random.randn(n, 3).astype(np.float32)
    # base_normals /= np.linalg.norm(base_normals, axis=-1, keepdims=True)

    # from timeit import timeit
    # warm_up = 2
    # number = 5

    # princ_fn = functools.partial(compute_edges_principled_eager,
    #                              base_coords,
    #                              base_normals,
    #                              depth=4)
    # for _ in range(warm_up):
    #     princ_fn()
    # t = timeit(princ_fn, number=number)
    # print('princ: {}'.format(t))

    # base_fn = functools.partial(compute_edges_eager,
    #                             base_coords,
    #                             base_normals,
    #                             depth=4)
    # # warm up
    # for _ in range(warm_up):
    #     base_fn()
    # t = timeit(base_fn, number=number)
    # print('base: {}'.format(t))

    # print('done')
    # exit()

    # coords = np.stack([base_coords] * 2, axis=0)
    # coords[1, :, 2] = 1
    # normals = np.stack([base_normals] * 2, axis=0)
    # dataset = tf.data.Dataset.from_tensor_slices((dict(positions=coords,
    #                                                    normals=normals), [0,
    #                                                                       0]))

    from deep_cloud.problems.modelnet import ModelnetProblem
    from deep_cloud.problems.builders import pointnet_builder
    import tensorflow_datasets as tfds
    from time import time
    from tqdm import tqdm
    problem = ModelnetProblem(builder=pointnet_builder(2), positions_only=False)
    dataset = problem.get_base_dataset(split='validation')
    num_examples = 100
    batch_size = 2

    def profile(edge_fn, depth=4, num_examples=10):
        times = []
        for example, _ in tqdm(tfds.as_numpy(dataset.take(num_examples)),
                               total=num_examples):
            start = time()
            edge_fn(example['positions'], example['normals'], depth=depth)
            times.append(time() - start)
        return np.array(times)

    # t_base = profile(compute_edges_eager, num_examples=num_examples)
    # t_princ = profile(compute_edges_principled_eager, num_examples=num_examples)
    # print('base: {}'.format(np.mean(t_base[4:])))
    # print('princ: {}'.format(np.mean(t_princ[4:])))
    # exit()

    def profile_dataset(dataset, depth=4, num_examples=10):
        # cef = functools.partial(compute_edges, eager_fn=edge_fn, depth=depth)
        # ds = dataset.map(pre_batch_map, -1)
        # ds = dataset.map(pre_batch_map)
        # ds = ds.batch(batch_size)
        # ds = ds.map(post_batch_map)
        start = time()
        times = []
        for _ in tqdm(tfds.as_numpy(dataset.take(num_examples)),
                      total=num_examples):
            end = time()
            times.append(end - start)
            start = end

        return np.array(times)

    # with gin.unlock_config():
    #     gin.parse_config('compute_edges.eager_fn = @compute_edges_eager')
    # edge_fn = functools.partial(compute_edges,
    #                             eager_fn=compute_edges_principled_eager)
    # t_base = profile_dataset(dataset.map(pre_batch_map),
    #                          num_examples=num_examples)
    with gin.unlock_config():
        gin.parse_config(
            'compute_edges.eager_fn = @compute_edges_principled_eager_fn()')
    t_princ = profile_dataset(dataset.map(pre_batch_map),
                              num_examples=num_examples)
    # print('base: {}'.format(np.mean(t_base[num_examples // 2:])))
    print('princ: {}'.format(np.mean(t_princ[num_examples // 2:])))
    exit()

    # dataset = dataset.map(pre_batch_map)

    # import tensorflow_datasets as tfds
    # from tqdm import tqdm
    # total = 100
    # for (example, labels) in tqdm(tfds.as_numpy(dataset.take(total)),
    #                               total=total):
    #     # compute_edges_principled_eager(example['positions'], example['normals'],
    #     #                                4)
    #     pass

    # dataset = dataset.map(pre_batch_map).batch(batch_size).map(post_batch_map)

    # features, labels = tf.compat.v1.data.make_one_shot_iterator(
    #     dataset).get_next()

    # with tf.Session() as sess:
    #     features, labels = sess.run((features, labels))

    # (all_coords, all_normals, flat_rel_coords, flat_node_indices, row_splits,
    #  outer_row_splits) = features

    # node_indices = tf.nest.map_structure(ra.RaggedArray.from_row_splits,
    #                                      flat_node_indices, row_splits)
    # vis(all_coords, node_indices)
