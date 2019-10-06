from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.models.sparse import ops
import gin

activations = tf.keras.activations
constraints = tf.keras.constraints
initializers = tf.keras.initializers
layers = tf.keras.layers
regularizers = tf.keras.regularizers
InputSpec = layers.InputSpec


def sparse_cloud_convolution(node_features, edge_features, sparse_indices,
                             dense_shape, filters, **kwargs):
    layer = (FeaturelessSparseCloudConvolution if node_features is None else
             SparseCloudConvolution)(filters, **kwargs)
    return layer(node_features, edge_features, sparse_indices, dense_shape)


@gin.configurable
class FeaturelessSparseCloudConvolution(layers.Layer):

    def __init__(self,
                 filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(FeaturelessSparseCloudConvolution, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = int(filters)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = [
            InputSpec(ndim=2),  # [S, E] edge features
            InputSpec(shape=(None, 2), dtype=tf.int64),  # [E, 2] sparse indices
        ]

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(FeaturelessSparseCloudConvolution,
                            self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.built:
            return

        es = input_shape[0]
        num_edge_features = es[0]
        self.kernel = self.add_weight('kernel',
                                      shape=[num_edge_features, self.filters],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.filters],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        self.input_spec[0] = InputSpec(shape=(num_edge_features, None))
        super(FeaturelessSparseCloudConvolution, self).build(input_shape)

    def call(self, inputs):
        edge_features, sparse_indices = (
            tf.convert_to_tensor(i) for i in inputs)
        outputs = ops.featureless_conv(self.kernel, sparse_indices,
                                       edge_features)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


@gin.configurable
class SparseCloudConvolution(layers.Layer):

    def __init__(self,
                 filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SparseCloudConvolution, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = int(filters)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = [
            InputSpec(ndim=2),  # [N_in, F_in] node features
            InputSpec(ndim=2),  # [S, E] edge features
            InputSpec(shape=(None, 2), dtype=tf.int64),  # [E, 2] sparse indices
            InputSpec(shape=(), dtype=tf.int64),  # [] out_size
        ]

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(SparseCloudConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.built:
            return
        ns, ef = input_shape[:2]
        num_node_features = ns[1]
        num_edge_features = ef[0]
        self.kernel = self.add_weight(
            'kernel',
            shape=[num_edge_features, num_node_features, self.filters],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.filters],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        self.input_spec[0] = InputSpec(shape=(None, num_node_features))
        self.input_spec[1] = InputSpec(shape=(num_edge_features, None))
        super(SparseCloudConvolution, self).build(input_shape)

    def call(self, inputs):
        node_features, edge_features, indices, out_size = (
            tf.convert_to_tensor(i) for i in inputs)
        dense_shape = (out_size, tf.shape(node_features,
                                          out_type=out_size.dtype)[0])
        outputs = ops.fold_conv(node_features, self.kernel, indices,
                                edge_features, dense_shape)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
