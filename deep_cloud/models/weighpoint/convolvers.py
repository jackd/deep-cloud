"""
Classes implementing various difference convolution strategies.

See `Convolver` for the interface to be implemented and method docstrings.

`ExpandingConvolver` and `NetworkConvolver` implement convolutions according
to depthwise expansion and MLP networks respectively.

`ResnetConvolver` takes a base `Convolver` and applies the relevant
convolution operation in Resnet-inspired blocks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gin
import six
import tensorflow as tf
from deep_cloud.layers import conv
from more_keras.layers import utils


class Convolver(object):
    """Abstract base class defining the different types of convolutions."""

    @abc.abstractmethod
    def in_place_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):
        """
        Perform an in-place/stationary convolution.

        i.e. one where the input cloud is the same as the output cloud.

        Args:
            features: [N, filters_in] float32 array of flattened batched point
                features.
            coord_features: [B, n?, k?, coord_filters] float32 possibly ragged
                tensor of relative coordinate features.
            batched_neighbors: [B, n?, k?] (num_elements == E) possibly ragged
                tensor of indices defining neighborhoods of each point in the
                batched clouds.
            filters_out: number of output filters
            weights: None or [B, n?, k?, 1] float32 array of weights for each
                neighborhood relationship.

        Returns:
            [N, filters_out] flattened batched output features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def resample_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):
        """
        Perform a resampling convolution.

        i.e. one where the output cloud is different to the input cloud.
        Strictly speaking doesn't have to be a resampling, though generally is.

        Only difference with `in_place_conv` is the output filters are of a
        different size to the inputs (according to the number of points in the
        input/output clouds).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def global_conv(self, features, coord_features, row_splits, filters_out):
        """
        Perform a global point cloud convolution.

        Args:
            features: [N, filters_in] flattened float32 array of point features
            coord_features: [B, n?, coord_filters] possibly ragged float32
                array of coordinate features, num_flat_elements == N
            row_splits: do we need this?
                Can we not use coord_features.row_splits? TODO
            filters out: number of output filters

        Returns:
            [B, filters_out] float32 global features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def global_deconv(self, global_features, coord_features, row_splits,
                      filters_out):
        raise NotImplementedError


@gin.configurable
class ExpandingConvolver(Convolver):
    """
    `ExpandingConvolver`s convolve in 3 stages.

    1. Expansion: perform a pointwise flattened cartesian product on point
        features and coord features. For `pf_in` input feature filters and
        `cf_in` input coordinate features, this results in `pf_in * cf_in`
        filters for each neighborhood relationship
    2. Integral: sum over all neighbors for each point.
    3. Filter reduction: perform a pointwise convolution (Dense layer) on each
        point. This generally decreases the filter count from `pf_in * cf_in`
        to something more amenable for subsequent convolutions.
    """

    def __init__(self, activation=None, global_activation=None):
        self._activation = activation
        self._global_activation = global_activation

    def in_place_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):
        features = conv.flat_expanding_edge_conv(
            features, coord_features.flat_values, batched_neighbors.flat_values,
            batched_neighbors.nested_row_splits[-1],
            None if weights is None else weights.flat_values)
        if filters_out is not None:
            features = tf.keras.layers.Dense(filters_out)(features)
        if self._activation is not None:
            features = self._activation(features)

        return features

    def resample_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):
        return self.in_place_conv(features, coord_features, batched_neighbors,
                                  filters_out, weights)

    def global_conv(self, features, coord_features, row_splits, filters_out):
        features = conv.flat_expanding_edge_conv(
            features, utils.flatten_leading_dims(coord_features, 2), None,
            row_splits)
        if filters_out is not None:
            features = tf.keras.layers.Dense(filters_out)(features)
        if self._global_activation is not None:
            features = self._global_activation(features)
        return features

    def global_deconv(self, global_features, coord_features, row_splits,
                      filters_out):
        features = conv.flat_expanding_global_deconv(
            global_features, utils.flatten_leading_dims(coord_features, 2),
            row_splits)
        if filters_out is not None:
            features = tf.keras.layers.Dense(filters_out)(features)
        if self._activation is not None:
            features = self._activation(features)
        return features


@gin.configurable
class ResnetConvolver(Convolver):
    """
    Takes a base convolver and performs base operations in resnet-like blocks.

    Based on
    https://github.com/keras-team/keras-applications/blob/master/'
    'keras_applications/resnet50.py
    """

    def __init__(self,
                 base_convolver=None,
                 activation=tf.nn.relu,
                 combine='add'):
        if base_convolver is None:
            base_convolver = ExpandingConvolver(activation=None)
        self._base = base_convolver
        self._activation = activation
        self._combine = combine

    def in_place_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):

        def base_conv(f):
            return self._base.in_place_conv(f, coord_features,
                                            batched_neighbors, filters_out,
                                            weights)

        x = features
        for _ in range(2):
            x = base_conv(features)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(self._activation)(x)
        x = base_conv(features)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = features
        if self._combine == 'add':
            if features.shape[-1] != x.shape[-1]:
                shortcut = tf.keras.layers.Dense(filters_out)(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            x = tf.keras.layers.Add()([x, shortcut])
            return tf.keras.layers.Activation(self._activation)(x)
        elif self._combine == 'concat':
            x = tf.keras.layers.Activation(self._activation)(x)
            return tf.keras.layers.Lambda(
                tf.concat, arguments=dict(axis=-1))([x, shortcut])

    def _resample_conv(self, features, conv_fn, activate_final=True):
        x = features
        for _ in range(2):
            x = conv_fn(features)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(self._activation)(x)
        x = conv_fn(features)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = conv_fn(features)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        # could sometimes concat here...
        x = tf.keras.layers.Add()([x, shortcut])
        if activate_final:
            x = tf.keras.layers.Activation(self._activation)(x)
        return x

    def resample_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):

        def base_conv(f):
            return self._base.resample_conv(f, coord_features,
                                            batched_neighbors, filters_out,
                                            weights)

        return self._resample_conv(features, base_conv)

    def global_conv(self, features, coord_features, row_splits, filters_out):

        def base_conv(f):
            return self._base.global_conv(f, coord_features, row_splits,
                                          filters_out)

        return self._resample_conv(features, base_conv, activate_final=False)

    def global_deconv(self, features, coord_features, row_splits, filters_out):

        def base_conv(f):
            return self._base.global_deconv(f, coord_features, row_splits,
                                            filters_out)

        return self._resample_conv(features, base_conv)


def _activation(activation):
    if activation is None:
        return lambda x: x
    if isinstance(activation, six.string_types):
        return tf.keras.layers.Activation(tf.keras.activations.get(activation))
    return activation


@gin.configurable(blacklist=['x', 'filters_out'])
def simple_mlp(x,
               filters_out,
               n_hidden=1,
               filters_hidden=None,
               hidden_activation='relu',
               final_activation='relu'):
    """
    Simple multi-layer perceptron model.

    Args:
        x: [N, filters_in] float32 input features
        filters_out: python int, number of output filters
        n_hidden: python int, number of hidden layers
        filters_hidden: python int, number of filters in each hidden layer
        hidden_activation: activation applied at each hidden layer
        final_activation: activation applied at the end

    Returns:
        [N, filters_out] float32 output features
    """
    if filters_hidden is None:
        filters_hidden = x.shape[-1]
    hidden_activation = _activation(hidden_activation)
    final_activation = _activation(final_activation)
    for _ in range(n_hidden):
        x = tf.keras.layers.Dense(filters_hidden)(x)
        x = hidden_activation(x)
    x = tf.keras.layers.Dense(filters_out)(x)
    return final_activation(x)


@gin.configurable
class NetworkConvolver(Convolver):
    """
    NetworkConvolvers use `weighpoint.layers.conv.mlp_edge_conv`s.

    These concatenate node features with relative coordinate features before
    passing them through an MLP. This is similar to how pointnet(++) work.
    """

    def __init__(self,
                 local_network_fn=simple_mlp,
                 global_network_fn=simple_mlp):
        self._local_fn = local_network_fn
        self._global_fn = global_network_fn

    def in_place_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):
        features = conv.mlp_edge_conv(
            features, coord_features.flat_values, batched_neighbors.flat_values,
            batched_neighbors.nested_row_splits[-1],
            lambda features: self._local_fn(features, filters_out),
            None if weights is None else weights.flat_values)

        return features

    def resample_conv(self, features, coord_features, batched_neighbors,
                      filters_out, weights):
        return self.in_place_conv(features, coord_features, batched_neighbors,
                                  filters_out, weights)

    def global_conv(self, features, coord_features, row_splits, filters_out):
        return conv.mlp_edge_conv(
            features,
            utils.flatten_leading_dims(coord_features, 2),
            None,
            row_splits,
            lambda features: self._global_fn(features, filters_out),
            weights=None)

    def global_deconv(self, global_features, coord_features, row_splits,
                      filters_out):
        raise NotImplementedError('TODO')
        # return conv.mlp_global_deconv(
        #     global_features, utils.flatten_leading_dims(coord_features,
        #                                                 2), row_splits,
        #     lambda features, global_features: self._global_network_factory(
        #         features, global_features, filters_out))
