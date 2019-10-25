from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
from deep_cloud.model_builder.utils import assert_is_tensor_spec

PyFuncNode = collections.namedtuple('PyFuncNode', ['builder', 'index'])


class PyFuncBuilder(object):

    def __init__(self, name=None, input_callback=None, output_callback=None):
        self._name = name
        self._input_tensors = []
        self._input_nodes = []
        self._nodes = []
        self._output_indices = []
        self._output_specs = []
        self._output_tensors = []
        self._fns = []
        self._input_callback = input_callback
        self._output_callback = output_callback

    @property
    def name(self):
        return self._name

    @property
    def input_tensors(self):
        return tuple(self._input_tensors)

    @property
    def output_tensors(self):
        return tuple(self._output_tensors)

    def __str__(self):
        return 'PyFuncBuilder<{}>'.format(self.name)

    def __repr__(self):
        return 'PyFuncBuilder<{}>'.format(self.name)

    def _node(self):
        out = PyFuncNode(self, len(self._nodes))
        self._nodes.append(out)
        return out

    def input_node(self, tensor):
        if not isinstance(tensor, tf.Tensor):
            raise ValueError('tensor must be a Tensor, got {}'.format(tensor))
        node = self._node()
        if self._input_callback is not None:
            self._input_callback(tensor)
        self._input_tensors.append(tensor)
        self._input_nodes.append(node)
        return node

    def unstack(self, node, num_outputs):
        return [self.node(lambda x: x[i], node) for i in range(num_outputs)]

    def node(self, fn, *args, **kwargs):
        num_outputs_ = kwargs.get('num_outputs_', None)
        for i, arg in enumerate(args):
            self._assert_is_own_node(arg, 'arg{}'.format(i))
        for k, v in kwargs.items():
            self._assert_is_own_node(v, k)

        args = tuple(arg.index for arg in args)
        kwargs = {k: v.index for k, v in kwargs.items()}
        out = self._node()

        def wrapped_fn(builder_values):
            args_ = tuple(builder_values[arg] for arg in args)
            kwargs_ = {k: builder_values[v.index] for k, v in kwargs.items()}
            value = fn(*args_, **kwargs_)
            assert (not isinstance(value, (tf.Tensor, tf.RaggedTensor)))
            builder_values[out.index] = value

        self._fns.append(wrapped_fn)
        if num_outputs_ is not None:
            out = self.unstack(out, num_outputs_)
        return out

    def _assert_is_own_node(self, node, name='node'):
        if not isinstance(node, PyFuncNode):
            raise ValueError('{} must be a PyFuncNode, got {}'.format(
                name, node))
        elif node.builder is not self:
            raise ValueError('{}.builder must be self, got {}'.format(
                name, node.builder))

    def output_tensor(self, node, tensor_spec):
        self._assert_is_own_node(node)
        for i, spec in enumerate(tf.nest.flatten(tensor_spec)):
            assert_is_tensor_spec(spec, 'spec{}'.format(i))
        self._output_indices.append(node.index)
        self._output_specs.append(tensor_spec)
        out = tf.nest.map_structure(
            lambda spec: tf.keras.layers.Input(
                shape=spec.shape, dtype=spec.dtype, batch_size=1), tensor_spec)
        if self._output_callback is not None:
            for o in tf.nest.flatten(out):
                self._output_callback(o)
        self._output_tensors.append(out)
        out = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0), out)
        return out

    def run(self, inputs=None):

        def f(*input_values):
            input_values = tuple(v.numpy() for v in input_values)
            assert (len(input_values) == len(self._input_nodes))
            values = [None] * len(self._nodes)
            for node, value in zip(self._input_nodes, input_values):
                values[node.index] = value
            for fn in self._fns:
                fn(values)
            out = tf.nest.flatten(tuple(
                values[i] for i in self._output_indices))
            return tuple(out)

        if inputs is None:
            inputs = self._input_tensors
        elif not isinstance(inputs, list):
            inputs = list(inputs)
        # if len(inputs) == 1:
        #     inputs, = inputs
        dtypes = tuple(
            spec.dtype for spec in tf.nest.flatten(self._output_specs))
        values = tf.py_function(f, inputs, dtypes)
        # values = tf.py_func(f, inputs, dtypes, stateful=False)
        for value, spec in zip(values, tf.nest.flatten(self._output_specs)):
            value.set_shape(spec.shape)
        values = tf.nest.pack_sequence_as(self._output_specs, values)
        return values

    def model(self):
        inputs = [
            tf.keras.layers.Input(shape=i.shape, dtype=i.dtype, batch_size=1)
            for i in self._input_tensors
        ]
        inps = [tf.squeeze(i, axis=0) for i in inputs]
        # inputs = [
        #     tf.keras.layers.Input(shape=i.shape[1:],
        #                           dtype=i.dtype,
        #                           batch_size=i.shape[0])
        #     for i in self._input_tensors
        # ]
        # out = self.run(inputs)
        out = tf.keras.layers.Lambda(self.run)(inps)
        return tf.keras.models.Model(inputs=inputs, outputs=out)
