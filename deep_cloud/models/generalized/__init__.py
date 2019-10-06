from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import numpy as np
import tensorflow as tf
from more_keras import layers as mk_layers
from more_keras import spec
from more_keras.layers import utils as layer_utils
from more_keras.ops import utils as op_utils
from more_keras.ragged import batching as ragged_batching
from more_keras.framework.problems import get_current_problem
from more_keras.models import mlp
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import core
from deep_cloud.models.generalized import layers as gen_layers
from deep_cloud.models.generalized import blocks

DEFAULT_TREE = pykd.KDTree


def get_batch_norm(batch_norm_impl=mk_layers.BatchNormalization):
    if batch_norm_impl is None:
        return lambda x: x
    else:

        def batch_norm(x):
            return batch_norm_impl()(x)

        return batch_norm


def get_activation(activation='relu'):
    if activation is None:
        return lambda x: x
    else:

        def activate(x):
            return tf.keras.activations.get(activation)(x)

        return activate


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
    weights = []

    def add_in_place(tree, coords, radius):
        indices = tree.query_ball_point(coords, radius, approx_neighbors=k0)
        rc = np.repeat(coords, indices.row_lengths,
                       axis=0) - coords[indices.flat_values]
        dist = np.linalg.norm(rc, axis=-1)
        weights.append(radius - dist)
        flat_indices.append(indices.flat_values)
        row_splits.append(indices.row_splits)
        rel_coords.append(rc)
        return indices

    for i in range(depth - 1):
        indices = add_in_place(tree, coords, radii[i])
        indices = np.array(core.rejection_sample_precomputed(indices),
                           dtype=np.int64)
        sample_indices.append(indices)
        out_coords = coords[indices]
        tree = tree_impl(out_coords)
        indices = tree.query_ball_point(coords,
                                        radii[i + 1],
                                        approx_neighbors=k0)

        rc = np.repeat(coords, indices.row_lengths,
                       axis=0) - out_coords[indices.flat_values]
        weights.append(radii[i + 1] - np.linalg.norm(rc, axis=-1))
        flat_indices.append(indices.flat_values)
        row_splits.append(indices.row_splits)
        rel_coords.append(rc)

        coords = out_coords
        all_coords.append(coords)
        trees.append(tree)

    add_in_place(tree, coords, radii[-1])

    # print([c.shape[0] for c in all_coords])

    return (
        tuple(flat_indices),
        tuple(rel_coords),
        tuple(weights),
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

    n_convs = 2 * depth - 1
    specs = [
        (tf.TensorSpec((None,), tf.int64),) * n_convs,  # flat_indices
        (tf.TensorSpec((None, 3), tf.float32),) * n_convs,  # flat_rel_coords
        (tf.TensorSpec((None,), tf.float32),) * n_convs,  # feature_weights
        (tf.TensorSpec((None,), tf.int64),) * n_convs,  # row_splits
        (tf.TensorSpec((None, 3), tf.float32),) * depth,  # all_coords
        (tf.TensorSpec((None,), tf.int64),) * (depth - 1),  # sample_indices
    ]

    specs_flat = tf.nest.flatten(specs)

    fn = functools.partial(_flatten_output, edge_fn, depth=depth)
    out_flat = tf.py_function(fn, [positions], [s.dtype for s in specs_flat])
    for out, spec in zip(out_flat, specs_flat):
        out.set_shape(spec.shape)
    out = tf.nest.pack_sequence_as(specs, out_flat)
    (flat_node_indices, flat_rel_coords, feature_weights, row_splits,
     all_coords, sample_indices) = out

    all_coords, sample_indices = tf.nest.map_structure(
        ragged_batching.pre_batch_ragged, (all_coords, sample_indices))

    node_indices = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
                                         flat_node_indices, row_splits)
    rel_coords = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
                                       flat_rel_coords, row_splits)
    feature_weights = tf.nest.map_structure(tf.RaggedTensor.from_row_splits,
                                            feature_weights, row_splits)
    features = dict(
        all_coords=all_coords,
        rel_coords=rel_coords,
        feature_weights=feature_weights,
        node_indices=node_indices,
        sample_indices=sample_indices,
    )
    if normals is not None:
        features['normals'] = ragged_batching.pre_batch_ragged(normals)

    return ((features, labels) if weights is None else
            (features, labels, weights))


@gin.configurable(blacklist=['features', 'labels', 'weights'])
def post_batch_map(
        features,
        labels,
        weights=None,
        #    include_outer_row_splits=False
):
    all_coords, rel_coords, feature_weights, node_indices, sample_indices = (
        features[k] for k in ('all_coords', 'rel_coords', 'feature_weights',
                              'node_indices', 'sample_indices'))

    row_splits = tf.nest.map_structure(lambda rt: rt.nested_row_splits[1],
                                       node_indices)

    all_coords, sample_indices = tf.nest.map_structure(
        ragged_batching.post_batch_ragged, (all_coords, sample_indices))
    offsets = [op_utils.get_row_offsets(c) for c in all_coords]
    flat_rel_coords = tf.nest.map_structure(lambda x: x.flat_values, rel_coords)
    feature_weights = tf.nest.map_structure(lambda x: x.flat_values,
                                            feature_weights)

    depth = (len(node_indices) + 1) // 2
    flat_node_indices = [
        layer_utils.apply_row_offset(node_indices[0], offsets[0]).flat_values
    ]
    flat_sample_indices = []
    for i in range(depth - 1):
        flat_node_indices.extend(
            (layer_utils.apply_row_offset(ni, offsets[i + 1]).flat_values
             for ni in node_indices[2 * i + 1:2 * i + 3]))

        flat_sample_indices.append(
            layer_utils.apply_row_offset(sample_indices[i],
                                         offsets[i]).flat_values)

    flat_node_indices = tuple(flat_node_indices)
    flat_sample_indices = tuple(flat_sample_indices)

    class_index = features.get('class_index')

    normals = features.get('normals')

    features = dict(
        all_coords=all_coords,
        flat_rel_coords=flat_rel_coords,
        flat_node_indices=flat_node_indices,
        feature_weights=feature_weights,
        row_splits=row_splits,
        sample_indices=flat_sample_indices,
    )
    # if include_outer_row_splits:
    #     features['outer_row_splits'] = tuple(
    #         op_utils.get_row_splits(c) for c in all_coords)
    if normals is not None:
        normals = ragged_batching.post_batch_ragged(normals)
        normals = layer_utils.flatten_leading_dims(normals)
        features['normals'] = normals

    if class_index is not None:
        features['class_index'] = class_index
    labels, weights = get_current_problem().post_batch_map(labels, weights)

    if isinstance(labels, tf.Tensor) and labels.shape.ndims == 2:
        assert (isinstance(weights, tf.Tensor) and weights.shape.ndims == 2)
        labels = tf.reshape(labels, (-1,))
        weights = tf.reshape(weights, (-1,))

    return ((features, labels) if weights is None else
            (features, labels, weights))


def _from_row_splits(args):
    return tf.RaggedTensor.from_row_splits(*args)


def _ones(rel_coords):
    return tf.ones(shape=(tf.shape(rel_coords)[0], 1), dtype=rel_coords.dtype)


@gin.configurable(blacklist=['rel_coords'])
def get_coord_features(rel_coords,
                       order=2,
                       include_const=True,
                       include_mixed=True):
    if include_const:
        features = [tf.keras.layers.Lambda(_ones)(rel_coords)]
    else:
        features = []
    if order > 0:
        features.append(rel_coords)
    if order > 1:
        if include_mixed:
            x, y, z = tf.unstack(rel_coords, axis=-1)
            features.append(tf.stack([x * y, x * z, y * z], axis=-1))
        features.append(tf.square(rel_coords))
    if order > 2:
        raise NotImplementedError()
    return tf.concat(features, axis=-1)


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def generalized_classifier(input_spec,
                           output_spec,
                           coord_features_fn=get_coord_features,
                           dense_factory=mk_layers.Dense,
                           batch_norm_impl=mk_layers.BatchNormalization,
                           activation='relu',
                           global_filters=(512, 256),
                           filters0=32,
                           global_dropout_impl=None):
    batch_norm_fn = get_batch_norm(batch_norm_impl)
    activation_fn = get_activation(activation)

    inputs = spec.inputs(input_spec)
    num_classes = output_spec.shape[-1]

    # class_index = inputs.get('class_index')
    # if class_index is None:
    #     global_features = None
    # else:
    #     global_features = tf.squeeze(tf.keras.layers.Embedding(
    #         num_classes, filters0, input_lenght=1)(class_index),
    #                                  axis=1)
    (
        all_coords,
        flat_rel_coords,
        flat_node_indices,
        row_splits,
        sample_indices,
        feature_weights,
        # outer_row_splits,
    ) = (
        inputs[k] for k in (
            'all_coords',
            'flat_rel_coords',
            'flat_node_indices',
            'row_splits',
            'sample_indices',
            'feature_weights',
            # 'outer_row_splits',
        ))
    # del outer_row_splits

    depth = len(all_coords)
    coord_features = tuple(coord_features_fn(rc) for rc in flat_rel_coords)
    features = inputs.get('normals')

    filters = filters0
    if features is None:
        features = gen_layers.FeaturelessRaggedConvolution(filters)(
            [flat_rel_coords[0], flat_node_indices[0], feature_weights[0]])
    else:
        raise NotImplementedError()

    activation_kwargs = dict(batch_norm_fn=batch_norm_fn,
                             activation_fn=activation_fn)
    bottleneck_kwargs = dict(dense_factory=dense_factory, **activation_kwargs)

    features = activation_fn(batch_norm_fn(features))
    res_features = []
    for i in range(depth - 1):
        # in place
        features = blocks.in_place_bottleneck(features,
                                              coord_features[2 * i],
                                              flat_node_indices[2 * i],
                                              row_splits[2 * i],
                                              weights=feature_weights[2 * i],
                                              **bottleneck_kwargs)
        res_features.append(features)
        # down sample
        filters *= 2
        features = blocks.down_sample_bottleneck(features,
                                                 coord_features[2 * i + 1],
                                                 flat_node_indices[2 * i + 1],
                                                 row_splits[2 * i + 1],
                                                 feature_weights[2 * i + 1],
                                                 sample_indices[i],
                                                 filters=filters,
                                                 **bottleneck_kwargs)

    features = blocks.in_place_bottleneck(features,
                                          flat_rel_coords[-1],
                                          flat_node_indices[-1],
                                          row_splits[-1],
                                          feature_weights[-1],
                                          filters=filters,
                                          **bottleneck_kwargs)

    # global conv
    global_coords = all_coords[-1]
    features = gen_layers.GlobalRaggedConvolution(
        global_filters[0], dense_factory=mk_layers.Dense)(
            [features, global_coords.flat_values, global_coords.row_splits])
    logits = mlp(global_filters[1:],
                 activate_first=True,
                 final_units=num_classes,
                 batch_norm_impl=batch_norm_impl,
                 activation=activation,
                 dropout_impl=global_dropout_impl)(features)
    return tf.keras.Model(tf.nest.flatten(inputs), logits)


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def generalized_semantic_segmenter(input_spec,
                                   output_spec,
                                   dense_factory=mk_layers.Dense,
                                   batch_norm_impl=mk_layers.BatchNormalization,
                                   activation='relu',
                                   filters0=32):

    batch_norm_fn = get_batch_norm(batch_norm_impl)
    activation_fn = get_activation(activation)

    inputs = spec.inputs(input_spec)
    num_classes = output_spec.shape[-1]

    # class_index = inputs.get('class_index')
    # if class_index is None:
    #     global_features = None
    # else:
    #     global_features = tf.squeeze(tf.keras.layers.Embedding(
    #         num_classes, filters0, input_lenght=1)(class_index),
    #                                  axis=1)
    (
        all_coords,
        flat_rel_coords,
        feature_weights,
        flat_node_indices,
        row_splits,
        sample_indices,
        # outer_row_splits,
    ) = (
        inputs[k] for k in (
            'all_coords',
            'flat_rel_coords',
            'feature_weights',
            'flat_node_indices',
            'row_splits',
            'sample_indices',
            # 'outer_row_splits',
        ))
    # del outer_row_splits

    depth = len(all_coords)
    features = inputs.get('normals')

    filters = filters0
    if features is None:
        features = gen_layers.FeaturelessRaggedConvolution(filters)(
            [flat_rel_coords[0], flat_node_indices[0], feature_weights[0]])
    else:
        raise NotImplementedError()

    activation_kwargs = dict(batch_norm_fn=batch_norm_fn,
                             activation_fn=activation_fn)
    bottleneck_kwargs = dict(dense_factory=dense_factory, **activation_kwargs)

    features = activation_fn(batch_norm_fn(features))
    res_features = []
    for i in range(depth - 1):
        # in place
        features = blocks.in_place_bottleneck(features, flat_rel_coords[2 * i],
                                              flat_node_indices[2 * i],
                                              row_splits[2 * i],
                                              feature_weights[2 * i],
                                              **bottleneck_kwargs)
        res_features.append(features)
        # down sample
        filters *= 2
        features = blocks.down_sample_bottleneck(features,
                                                 flat_rel_coords[2 * i + 1],
                                                 flat_node_indices[2 * i + 1],
                                                 row_splits[2 * i + 1],
                                                 feature_weights[2 * i + 1],
                                                 sample_indices[i],
                                                 filters=filters,
                                                 **bottleneck_kwargs)

    for i in range(depth - 1, 0, -1):
        # in place
        features = blocks.in_place_bottleneck(features,
                                              flat_rel_coords[2 * i],
                                              flat_node_indices[2 * i],
                                              row_splits[2 * i],
                                              feature_weights[2 * i],
                                              filters=filters,
                                              **bottleneck_kwargs)
        if i != depth - 1:
            features = features + res_features.pop()
        # up sample
        filters //= 2
        features = gen_layers.RaggedConvolutionTranspose(
            filters, dense_factory=dense_factory)([
                features, flat_rel_coords[2 * i - 1],
                flat_node_indices[2 * i - 1], row_splits[2 * i - 1],
                feature_weights[2 * i - 1]
            ])
        features = activation_fn(batch_norm_fn(features))

    # final in place
    features = blocks.in_place_bottleneck(features,
                                          flat_rel_coords[0],
                                          flat_node_indices[0],
                                          row_splits[0],
                                          feature_weights[0],
                                          filters=filters,
                                          **bottleneck_kwargs)
    features = features + res_features.pop()

    # per-point classification layer
    logits = dense_factory(num_classes, activation=None,
                           use_bias=True)(features)

    return tf.keras.models.Model(tf.nest.flatten(inputs), logits)


if __name__ == '__main__':
    from deep_cloud.problems.partnet import PartnetProblem
    tf.compat.v1.enable_v2_tensorshape()
    # tf.compat.v1.enable_eager_execution()
    problem = PartnetProblem()
    dataset = problem.get_base_dataset(split='validation')
    with problem:
        dataset = dataset.map(pre_batch_map).batch(16).map(post_batch_map)

    input_spec, label_spec = dataset.element_spec

    features, labels = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    model = generalized_semantic_segmenter(input_spec,
                                           tf.TensorSpec((None, None, 4)))
    model.summary(print_fn=print)
    out = model(tf.nest.flatten(features))
    grads = tf.gradients(out, model.weights)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        bm = tf.test.Benchmark()
        bm.run_op_benchmark(sess, grads)

    # for features, labels in dataset.take(1):
    #     break
    # print([rs[-1].numpy() for rs in features['row_splits']])
    # print([np.mean(np.diff(rs)) for rs in features['row_splits']])
    # # for coords, labels in tfds.as_numpy(
    # #     .take(1)):
    # #     break
    # # features, labels = pre_batch_map(features, labels)
