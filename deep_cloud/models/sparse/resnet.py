from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin

from more_keras.ops import polynomials

from more_keras.layers import Dense
from more_keras.layers import BatchNormalization
from deep_cloud.models.sparse import layers as sparse_layers
Activation = tf.keras.layers.Activation
Add = tf.keras.layers.Add


def initial_block(edge_features, neigh_indices, filters=64, name='initial'):
    x = sparse_layers.FeaturelessSparseCloudConvolution(
        filters, name=name + '_conv')([edge_features, neigh_indices])
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def _get_filters(filters_arg, filters_inp):
    expected_filters = filters_inp // 4
    if filters_arg is None:
        filters_arg = expected_filters
    elif filters_arg != expected_filters:
        raise ValueError(
            'filters must be None or `node_features.shape[-1] // 4`, but '
            '{} != {}'.format(filters_arg, expected_filters))
    return filters_arg


def in_place_block(node_features,
                   edge_features,
                   neigh_indices,
                   name,
                   filters=None,
                   conv_shortcut=False):
    """
    Residual block for in-place convolution.

    Analagous to image convolutions with `stride == 1, padding='SAME'`.

    Args:
        node_features: [N, F] initial node features.
        edge_features: [E, F_edge] edge features.
        neigh_indices: [E, 2] sparse indices of neighbors.
        name: string name.
        filters: number of filters in bottleneck layer. If None, uses `F // 4`.
        conv_shortcut: if True, uses a dense layer on shortcut connection,
            otherwise uses identity and `filters` must be `None` or `F // 4`,
            and `F % 4 == 0`.

    Returns:
        [N, filters * 4] output features in the same node ordering.
    """
    out_size = tf.shape(node_features, out_type=tf.int64)[0]
    if conv_shortcut:
        if filters is None:
            raise ValueError(
                'filters must be provided if conv_shortcut is True')
        shortcut = Dense(4 * filters,
                         name=name + '_shortcut_dense')(node_features)
        shortcut = BatchNormalization(name=name + '_shortcut_bn')(shortcut)
    else:
        shortcut = node_features
        filters = _get_filters(filters, node_features.shape[-1])

    x = Dense(filters * 4, name=name + '_bottleneck_0_dense')(node_features)
    x = BatchNormalization(name=name + '_bottleneck_0_bn')(x)
    x = Activation('relu', name=name + '_bottleneck_0_relu')(x)
    x = sparse_layers.SparseCloudConvolution(filters, name=name + '_conv')(
        [x, edge_features, neigh_indices, out_size])
    x = BatchNormalization(name=name + '_conv_bn')(x)
    x = Activation('relu', name=name + '_conv_relu')(x)
    x = Dense(filters * 4, name=name + '_bottleneck_1_dense')(x)
    x = BatchNormalization(name=name + '_bottleneck_1_bn')(x)
    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x


def down_sample_block(node_features,
                      edge_features,
                      neigh_indices,
                      sample_indices,
                      name,
                      filters=None,
                      conv_shortcut=False):
    shortcut = tf.gather(node_features, sample_indices)
    if conv_shortcut:
        shortcut = Dense(filters * 4, name=name + '_shortcut_dense')(shortcut)
        shortcut = BatchNormalization(name=name + '_shortcut_bn')(shortcut)
    else:
        filters = _get_filters(filters, node_features.shape[-1])
    out_size = tf.shape(sample_indices, out_type=tf.int64)[0]
    x = Dense(filters, name=name + '_bottleneck_0_dense')(node_features)
    x = BatchNormalization(name=name + '_bottleneck_0_bn')(x)
    x = Activation('relu', name=name + '_bottleneck_0_relu')(x)
    x = sparse_layers.SparseCloudConvolution(filters, name=name + '_conv')(
        [x, edge_features, neigh_indices, out_size])
    x = BatchNormalization(name=name + '_conv_bn')(x)
    x = Activation('relu', name=name + '_conv_relu')(x)
    x = Dense(filters * 4, name=name + '_bottleneck_1_dense')(x)
    x = BatchNormalization(name=name + '_bottleneck_1_bn')(x)
    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x


def up_sample_combine(upper_node_features, node_features, edge_features,
                      neigh_indices, name):
    out_size = tf.shape(upper_node_features, out_type=tf.int64)[0]
    x = sparse_layers.SparseCloudConvolution(upper_node_features.shape[-1])(
        [node_features, edge_features, neigh_indices, out_size])
    x = BatchNormalization(name=name + '_bn')(x)
    x = Add(name=name + '_add')([upper_node_features, x])
    x = Activation('relu', name=name + '_out')(x)
    return x


def final_up_sample(node_features,
                    edge_features,
                    neigh_indices,
                    out_size,
                    num_classes,
                    name='up_sample_final'):
    return sparse_layers.SparseCloudConvolution(num_classes)(
        [node_features, edge_features, neigh_indices, out_size])


get_nd_polynomials = gin.external_configurable(polynomials.get_nd_polynomials,
                                               blacklist=['coords'])


@gin.configurable(blacklist=['rel_coords', 'axis'])
def hat_weight(rel_coords, axis=0):
    dists = tf.linalg.norm(rel_coords, axis=axis)
    val = 1 - dists
    return val


def row_normalize(values, norm_values, row_indices):
    if row_indices.shape.ndims == 2:
        row_indices = row_indices[:, 0]
    row_sum = tf.gather(tf.math.segment_sum(norm_values, row_indices),
                        row_indices)
    factor = tf.where(tf.greater(row_sum, 0), norm_values / row_sum,
                      tf.zeros_like(norm_values))
    return tf.expand_dims(factor, axis=0) * values


@gin.configurable(blacklist=[
    'sample_indices', 'in_place_rel_coords', 'in_place_indices',
    'down_sample_rel_coords', 'down_sample_indices'
])
def resnet_features(sample_indices,
                    in_place_rel_coords,
                    in_place_indices,
                    down_sample_rel_coords,
                    down_sample_indices,
                    filter_scale_factor=1,
                    edge_weight_fn=hat_weight):
    num_changes = len(sample_indices)
    for x in (in_place_indices, in_place_rel_coords, down_sample_indices,
              down_sample_rel_coords):
        assert (len(x) == num_changes)

    in_place_edge_features = tf.nest.map_structure(
        lambda x: get_nd_polynomials(x, max_order=2, axis=0),
        in_place_rel_coords)
    down_sample_edge_features = [
        get_nd_polynomials(down_sample_rel_coords[0], max_order=3, axis=0)
    ]
    down_sample_edge_features.extend((get_nd_polynomials(x, max_order=2, axis=0)
                                      for x in down_sample_rel_coords[1:]))
    down_sample_edge_features = tuple(down_sample_edge_features)

    down_sample_weights = tf.nest.map_structure(edge_weight_fn,
                                                down_sample_rel_coords)
    in_place_weights = tf.nest.map_structure(edge_weight_fn,
                                             in_place_rel_coords)

    weighted_down_sample_edge_features = tf.nest.map_structure(
        row_normalize, down_sample_edge_features, down_sample_weights,
        down_sample_indices)
    weighted_in_place_edge_features = tf.nest.map_structure(
        row_normalize, in_place_edge_features, in_place_weights,
        in_place_indices)

    # initial down-sample
    features = initial_block(weighted_down_sample_edge_features[0],
                             down_sample_indices[0],
                             filters=int(64 * filter_scale_factor))
    features = in_place_block(features,
                              weighted_in_place_edge_features[0],
                              in_place_indices[0],
                              name='in_place0')
    out_features = [features]
    filters = int(32 * filter_scale_factor)
    for i in range(1, num_changes):
        # down sample
        features = down_sample_block(features,
                                     weighted_down_sample_edge_features[i],
                                     down_sample_indices[i],
                                     sample_indices[i],
                                     name='down_sample{}'.format(i),
                                     filters=filters,
                                     conv_shortcut=True)

        # in place
        features = in_place_block(features,
                                  weighted_in_place_edge_features[i],
                                  in_place_indices[i],
                                  name='in_place{}'.format(i))
        out_features.append(features)
        filters = filters * 2
    return (out_features, in_place_edge_features, in_place_weights,
            down_sample_edge_features, down_sample_weights)


@gin.configurable(blacklist=['inputs', 'num_classes'])
def semantic_segmenter_logits(inputs, num_classes, features_fn=resnet_features):

    (sample_indices, in_place_indices, in_place_rel_coords, down_sample_indices,
     down_sample_rel_coords, up_sample_indices, up_sample_perms,
     row_splits) = (inputs.get(k)
                    for k in ('sample_indices', 'in_place_indices',
                              'in_place_rel_coords', 'down_sample_indices',
                              'down_sample_rel_coords', 'up_sample_indices',
                              'up_sample_perms', 'row_splits'))

    (out_features, in_place_edge_features, in_place_weights,
     down_sample_edge_features, down_sample_weights) = features_fn(
         sample_indices, in_place_rel_coords, in_place_indices,
         down_sample_rel_coords, down_sample_indices)
    del in_place_edge_features, in_place_weights
    # out_features = [
    #     tf.debugging.check_numerics(f, 'features{}'.format(i))
    #     for i, f in enumerate(out_features)
    # ]
    num_changes = len(out_features)
    features = out_features.pop()

    down_sample_edge_features = list(down_sample_edge_features)
    down_sample_edge_features[0] = down_sample_edge_features[0][:4]
    up_sample_edge_features = tf.nest.map_structure(
        lambda ef, p: tf.gather(ef, p, axis=1),
        tuple(down_sample_edge_features), up_sample_perms)
    up_sample_weights = tf.nest.map_structure(lambda w, p: tf.gather(w, p),
                                              down_sample_weights,
                                              up_sample_perms)
    weighted_up_sample_edge_features = tf.nest.map_structure(
        row_normalize, up_sample_edge_features, up_sample_weights,
        up_sample_indices)

    # up_sample_edge_features = tuple(
    #     tf.debugging.check_numerics(f, 'up_sample_edge{}'.format(i))
    #     for i, f in enumerate(up_sample_edge_features))

    for i in range(num_changes - 1, 0, -1):
        features = up_sample_combine(out_features.pop(),
                                     features,
                                     weighted_up_sample_edge_features[i],
                                     up_sample_indices[i],
                                     name='up_sample{}'.format(i))
        # features = tf.debugging.check_numerics(features,
        #                                        'up_features{}'.format(i))
    out_size = row_splits[0][-1]
    logits = final_up_sample(features, weighted_up_sample_edge_features[0],
                             up_sample_indices[0], out_size, num_classes)
    # logits = tf.debugging.check_numerics(logits, 'logits check')
    return logits


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def semantic_segmenter(input_spec,
                       output_spec,
                       logits_fn=semantic_segmenter_logits):
    from more_keras import spec
    inputs = spec.inputs(input_spec)
    logits = logits_fn(inputs, output_spec.shape[-1])

    model = tf.keras.models.Model(tf.nest.flatten(inputs), outputs=logits)
    return model


if __name__ == '__main__':
    from time import time
    import functools
    from deep_cloud.problems.partnet import PartnetProblem
    from deep_cloud.models.sparse import preprocess as pp
    import tqdm
    # tf.compat.v1.enable_eager_execution()
    # tf.config.experimental_run_functions_eagerly(True)
    split = 'train'
    batch_size = 16
    # batch_size = 2
    num_classes = 4
    num_warmup = 5
    num_batches = 10
    problem = PartnetProblem()
    with problem:
        dataset = problem.get_base_dataset(split).map(pp.pre_batch_map, -1)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(pp.post_batch_map, -1).prefetch(-1)

    # for args in dataset.take(1):
    #     if len(args) == 3:
    #         inputs, labels, weights = args
    #     else:
    #         inputs, labels = args
    #     logits = semantic_segmenter_logits(inputs,
    #                                        problem.output_spec.shape[-1])
    # print('Successs!')
    # exit()

    input_spec = dataset.element_spec[0]
    output_spec = problem.output_spec
    model = semantic_segmenter(input_spec, output_spec)
    args = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    if len(args) == 3:
        inputs, labels, weights = args
    else:
        inputs, labels = args
    logits = model(tf.nest.flatten(inputs))
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in tqdm.tqdm(range(num_warmup + num_batches),
                           total=num_warmup + num_batches):
            sess.run(logits)
            if i == num_warmup:
                t = time()
            if i == num_warmup + num_batches - 1:
                dt = time() - t

    print('{} batches in {} s: {} ms / batch'.format(num_batches, dt,
                                                     dt * 1000 / num_batches))
