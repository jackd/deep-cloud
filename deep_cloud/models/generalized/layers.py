from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras import layers as mk_layers
from more_keras.ops import utils as op_utils
import gin

repeat = op_utils.repeat
# repeat = tf.repeat


def block_variance_scaling_initializers(fan_ins,
                                        fan_out,
                                        scale=1.0,
                                        mode="fan_in",
                                        distribution="truncated_normal",
                                        seed=None):
    """
    Get initializers for block-dense layers.

    Example usage.
    ```python
    def block_dense(inputs, units):
        fan_ins = [inp.shape[-1] for inp in inputs]
        initializers = variance_scaling_initializers(fan_ins)
        layers = [tf.keras.layers.Dense(units, kernel_initializer=init)
                  for init in initializers]]
        outputs = [layer(inp) for layer, inp in zip(layers, inputs)]
        # you might want to do something with the split outputs here.
        return tf.math.add_n(outputs)
    ```

    Args:
        fan_ins: tuple of ints/dimensions indicating the fan_in for each block.
        fan_out: number of units in the output layer.
        scale:
        mode, distribution, seed: see tf.keras.initializers.VarianceScaling

    Returns:
        tuple of `tf.keras.initializers.VarianceScalingInitializer`s with scale
        modified such that the resulting distribution would be as if `fan_in`
        was actually `sum(fan_ins)`.
    """
    if not isinstance(fan_ins, tuple):
        raise ValueError('fan_ins must be a tuple, got {}'.format(fan_ins))
    total_fan_in = sum(fan_ins)
    kwargs = dict(mode=mode, distribution=distribution, seed=seed)

    def scale_scale(fan_in):
        if mode == 'fan_in':
            return max(1., fan_in)
        elif mode == 'fan_out':
            return 1
        else:
            return max(1., (fan_in + fan_out) / 2)

    scale /= scale_scale(total_fan_in)
    return tuple(
        tf.keras.initializers.VarianceScaling(scale=scale * scale_scale(fan_in),
                                              **kwargs) for fan_in in fan_ins)


def multi_pointnet(node_layer,
                   coord_layer,
                   activation,
                   node_features,
                   coord_features,
                   indices,
                   row_splits,
                   gather_first=False):

    if gather_first:
        node_features = tf.gather(node_features, indices)
        node_features = node_layer(node_features)
    else:
        node_features = node_layer(node_features)
        node_features = tf.gather(node_features, indices)
    coord_features = coord_layer(coord_features)
    features = node_features + coord_features
    if activation is not None:
        features = activation(features)
    features = tf.RaggedTensor.from_row_splits(features, row_splits)
    features = tf.reduce_max(features, axis=1)
    return features


def multi_pointnet_transpose(node_layer,
                             coord_layer,
                             activation,
                             node_features,
                             coord_features,
                             indices,
                             row_splits,
                             gather_first=False):
    row_lengths = op_utils.diff(row_splits)
    if gather_first:
        node_features = repeat(node_features, row_lengths, axis=0)
        node_features = node_layer(node_features)
    else:
        node_features = node_layer(node_features)
        node_features = repeat(node_features, row_lengths, axis=0)
    coord_features = coord_layer(coord_features)
    features = node_features + coord_features
    if activation is not None:
        features = activation(features)
    features = tf.math.unsorted_segment_max(
        features, indices, num_segments=tf.reduce_max(indices) + 1)
    return features


@gin.configurable
class MultiPointnet(tf.keras.layers.Layer):
    """
    Pointnet-style layer, performing pointnet across multiple neighborhoods.

    Equivalent to:
    max(Dense(concat([gathered_node_features, coord_features]), neighbors)
    """

    def __init__(self,
                 units,
                 dense_factory=mk_layers.Dense,
                 activation=None,
                 **kwargs):
        self.units = units
        self.dense_factory = dense_factory
        self.activation = tf.keras.activations.get(activation)
        super(MultiPointnet, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.built:
            return
        node_features_shape, coord_features_shape = input_shape[:2]
        fan_ins = node_features_shape[-1], coord_features_shape[-1]
        initializers = block_variance_scaling_initializers(fan_ins, self.units)
        self.node_layer, self.coord_layer = tuple(
            self.dense_factory(self.units, kernel_initializer=initializer)
            for initializer in initializers)
        self.node_layer.build(node_features_shape)
        self.coord_layer.build(coord_features_shape)
        super(MultiPointnet, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((None, self.units))

    def call(self, inputs):
        return multi_pointnet(self.node_layer, self.coord_layer, *inputs)


def _ragged_conv_combine(activation,
                         node_features,
                         coord_features,
                         indices,
                         weights=None):
    features = node_features * tf.expand_dims(coord_features, (-2,))
    features = tf.reduce_sum(features, axis=-1)
    if activation is not None:
        features = activation(features)
    num_segments = tf.reduce_max(indices) + 1
    if weights is None:
        features = tf.math.unsorted_segment_sum(features,
                                                indices,
                                                num_segments=num_segments)
    else:
        weights = tf.expand_dims(weights, axis=-1)
        features = tf.math.unsorted_segment_sum(features * weights,
                                                indices,
                                                num_segments=num_segments)
        weights = tf.math.unsorted_segment_sum(weights, indices, num_segments)
        features = features / weights
    return features


def ragged_convolution(activation,
                       layer,
                       node_features,
                       coord_features,
                       indices,
                       row_splits,
                       weights=None,
                       gather_first=False):
    coord_dims = coord_features.shape[-1]
    assert (coord_dims is not None)
    units = layer.units // coord_dims
    row_lengths = op_utils.diff(row_splits)
    if gather_first:
        node_features = repeat(node_features, row_lengths, axis=0)
        node_features = layer(node_features)
    else:
        node_features = layer(node_features)
        node_features = repeat(node_features, row_lengths, axis=0)

    node_features = tf.reshape(node_features, (-1, units, coord_dims))
    return _ragged_conv_combine(activation, node_features, coord_features,
                                indices, weights)


def featureless_ragged_convolution(activation, embedding, coord_features,
                                   indices, weights):
    return _ragged_conv_combine(activation, embedding, coord_features, indices,
                                weights)


def ragged_convolution_transpose(activation,
                                 layer,
                                 node_features,
                                 coord_features,
                                 indices,
                                 row_splits,
                                 weights=None,
                                 gather_first=False):
    coord_dims = coord_features.shape[-1]
    assert (coord_dims is not None)
    units = layer.units // coord_dims
    if gather_first:
        node_features = tf.gather(node_features, indices)
        node_features = layer(node_features)
    else:
        node_features = layer(node_features)
        node_features = tf.gather(node_features, indices)

    node_features = tf.reshape(node_features, (-1, units, coord_dims))
    features = node_features * tf.expand_dims(coord_features, (-2,))
    features = tf.reduce_sum(features, axis=-1)
    if activation is not None:
        features = activation(features)
    if weights is None:
        features = tf.RaggedTensor.from_row_splits(features, row_splits)
        features = tf.reduce_sum(features, axis=1)
    else:
        weights = tf.expand_dims(weights, axis=-1)
        features = features * weights
        features = tf.RaggedTensor.from_row_splits(features, row_splits)
        features = tf.reduce_sum(features, axis=1)
        weights = tf.RaggedTensor.from_row_splits(weights, row_splits)
        weights = tf.reduce_sum(weights, axis=1)
        features = features / weights
    return features


# def ragged_convolution_transpose(layer, activation, node_features,
#                                  coord_features, flat_indices, row_splits):
#     coord_dims = coord_features.shape[-1]
#     assert (coord_dims is not None)
#     units = layer.units // coord_dims
#     node_features = layer(node_features)
#     node_features = tf.reshape(node_features, (-1, units, coord_dims))

#     indices = tf.RaggedTensor.from_row_splits(flat_indices, row_splits)
#     indices = indices.to_tensor(default_value=-1)
#     coord_features = tf.RaggedTensor.from_row_splits(
#         coord_features, row_splits).to_tensor(default_value=0)

#     def fn(args):
#         indices, coord_features = args
#         nf = tf.gather(node_features, indices)
#         features = nf * coord_features
#         features = tf.reduce_sum(features, axis=-1)
#         if activation is not None:
#             features = activation(features)
#         return tf.reduce_sum(features, axis=0)

#     return tf.vectorized_map(fn,
#                              (indices, tf.expand_dims(coord_features, axis=2)))


@gin.configurable(blacklist=['units'])
class RaggedConvolution(tf.keras.layers.Layer):
    """
    Ragged convolution operation.

    Equivalent to:
    sum(Dense(outer_product(node_features, coord_features), neighbors)

    Based on `fo_kp = sum_{i, j, n} W_ijk f_jn x_in` for `n in neighbors(p)`
    and `k in [0, self.units)`. `f` are the input node features, and `x` is
    the tensor of relative coordinate features.

    Call args:
        node_features: [Ni, Fi] float input node features.
        coord_features: [E, D] float relative coordinate features.
        indices: [E] int indices into output cloud corresponding to
            indices of the output cloud in the neighborhood of input nodes.
        row_splits: [Ni+1] int row_splits corresponding to ragged
            `coord_features` / `indices`.

    Returns:
        out_node_features: [No, self.units] float output node features.
    """

    def __init__(self,
                 units,
                 dense_factory=mk_layers.Dense,
                 activation=None,
                 **kwargs):
        self.units = units
        self.dense_factory = dense_factory
        self.activation = tf.keras.activations.get(activation)
        super(RaggedConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((None, self.units))

    def build(self, input_shape):
        if self.built:
            return
        node_features_shape, coord_features_shape = input_shape[:2]
        self.coord_features_size = coord_features_shape[-1]
        self.layer = self.dense_factory(units=self.units *
                                        self.coord_features_size)
        self.layer.build(node_features_shape)
        super(RaggedConvolution, self).build(input_shape)

    def call(self, inputs):
        return ragged_convolution(self.activation, self.layer, *inputs)


@gin.configurable
class FeaturelessRaggedConvolution(tf.keras.layers.Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        super(FeaturelessRaggedConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((None, self.units))

    def build(self, input_shape):
        if self.built:
            return
        coord_features_shape = input_shape[0]
        coord_features_size = coord_features_shape[-1]
        self.embedding = self.add_weight(
            'embedding',
            shape=(self.units, coord_features_size),
            dtype=self.dtype,
            initializer=tf.keras.initializers.get('glorot_uniform'))
        super(FeaturelessRaggedConvolution, self).build(input_shape)

    def call(self, inputs):
        return featureless_ragged_convolution(self.activation, self.embedding,
                                              *inputs)


@gin.configurable
class RaggedConvolutionTranspose(RaggedConvolution):

    def compute_output_shape(self, input_shape):
        row_splits_shape = input_shape[-1]
        n = row_splits_shape[0]
        if n is not None:
            n = n - 1
        return tf.TensorShape((n, self.units))

    def call(self, inputs):
        return ragged_convolution_transpose(self.activation, self.layer,
                                            *inputs)


@gin.configurable
class GlobalRaggedConvolution(tf.keras.layers.Layer):

    def __init__(self, units, dense_factory, **kwargs):
        self.units = units
        self.dense_factory = dense_factory
        super(GlobalRaggedConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        row_splits_shape = input_shape[-1]
        n = row_splits_shape[-1][0]
        if n is not None:
            n = n - 1
        return tf.TensorShape((n, self.units))

    def build(self, input_shape):
        if self.built:
            return
        node_features_shape, coord_features_shape = input_shape[:2]
        self.coord_features_size = coord_features_shape[-1]
        self.layer = self.dense_factory(units=self.units *
                                        self.coord_features_size)
        self.layer.build(node_features_shape)
        super(GlobalRaggedConvolution, self).build(input_shape)

    def call(self, inputs):
        node_features, coord_features, row_splits = inputs
        node_features = self.layer(node_features)
        node_features = tf.reshape(node_features,
                                   (-1, self.units, self.coord_features_size))
        features = node_features * tf.expand_dims(coord_features, (-2,))
        features = tf.reduce_sum(features, axis=-1)
        features = tf.RaggedTensor.from_row_splits(features, row_splits)
        return tf.reduce_sum(features, axis=1)
