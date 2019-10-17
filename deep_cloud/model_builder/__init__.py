from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

PyFuncNode = collections.namedtuple('PyFuncNode', ['builder', 'index'])


def _assert_is_tensor_spec(spec, name='tensor_spec'):
    if not isinstance(spec, tf.TensorSpec):
        raise ValueError('{} must be a TensorSpec, got {}'.format(name, spec))


class PyFuncBuilder(object):

    def __init__(self, name=None):
        self._name = name
        self._input_tensors = []
        self._input_nodes = []
        self._nodes = []
        self._output_indices = []
        self._output_specs = []
        self._output_tensors = []
        self._fns = []

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
            args_ = tuple(builder_values[arg.index] for arg in args)
            kwargs_ = {k: builder_values[v.index] for k, v in kwargs.items()}
            builder_values[out.index] = fn(*args_, **kwargs_)

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
            _assert_is_tensor_spec(spec, 'spec{}'.format(i))
        self._output_indices.append(node.index)
        self._output_specs.append(tensor_spec)
        out = tf.nest.map_structure(
            lambda spec: tf.keras.layers.Input(
                shape=spec.shape, dtype=spec.dtype, batch_size=1), tensor_spec)
        self._output_tensors.append(out)
        out = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0), out)
        return out

    def run(self, inputs=None):

        def f(input_values):
            input_values = tuple(v.numpy() for v in input_values)
            assert (len(input_values) == len(self._input_nodes))
            values = [None] * len(self._nodes)
            for node, value in zip(self._input_nodes, input_values):
                values[node.index] = value
            for fn in self._fns:
                fn(values)
            return tuple(
                tf.nest.flatten(values[i] for i in self._output_indices))

        if inputs is None:
            inputs = self._input_tensors
        values = tf.py_function(
            f, self._input_tensors,
            tuple(spec.dtype for spec in tf.nest.flatten(self._output_specs)))
        for value, spec in zip(values, tf.nest.flatten(self._output_specs)):
            value.set_shape(spec.shape)
        values = tuple(tf.expand_dims(v, axis=0) for v in values)
        values = tf.nest.pack_sequence_as(self._output_specs, values)
        return values


class ModelBuilder(object):

    def __init__(self):
        self._inputs = []
        self._outputs = []
        self._py_func_builder = None
        self._finalized = False
        self._model = None

    def py_func_builder(self, name=None):
        if self._py_func_builder is not None:
            raise NotImplementedError('Only single py_func_builder supported')
        self._py_func_builder = PyFuncBuilder(name=name)
        return self._py_func_builder

    def _assert_finalized(self, action):
        if not self._finalized:
            raise RuntimeError(
                'Cannot {}: builder has not been finalized'.format(action))

    def _assert_not_finalized(self, action):
        if self._finalized:
            raise RuntimeError(
                'Cannot {}: builder has been finalized'.format(action))

    def add_input(self, inp):
        self._assert_not_finalized('add_input')
        self._inputs.append(inp)

    def add_output(self, tensor):
        self._assert_not_finalized('add_output')
        if not isinstance(tensor, (tf.Tensor, tf.RaggedTensor)):
            raise ValueError(
                'tensor must be a Tensor or RaggedTensor, got {}'.format(
                    tensor))
        self._outputs.append(tensor)

    def finalize(self):
        self._finalized = True
        self.model

    @property
    def model(self):
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def _create_model(self):
        self._assert_finalized('_create_model')
        if self._py_func_builder is None:
            return tf.keras.models.Model(self._inputs,
                                         self._outputs,
                                         name='simple_model')

        py_func_outputs = self._py_func_builder.run()
        inputs_copy = [
            tf.keras.layers.Input(shape=i.shape[1:],
                                  dtype=i.dtype,
                                  batch_size=i.shape[0]) for i in self._inputs
        ]
        final_model = tf.keras.models.Model(
            inputs_copy + list(self._py_func_builder.output_tensors),
            self._outputs,
            name='final_model')
        final_out = final_model(self._inputs + list(py_func_outputs))
        return tf.keras.models.Model(self._inputs,
                                     final_out,
                                     name='combined_model')

        # inputs = [
        #     tf.keras.layers.Input(shape=i.shape[1:],
        #                           dtype=i.dtype,
        #                           ragged=isinstance(i, tf.RaggedTensor),
        #                           batch_size=i.shape[0]) for i in self._inputs
        # ]

        # py_func_outputs = self._py_func_builder.run()
        # if any(x not in self._inputs
        #        for x in self._py_func_builder.input_tensors):
        #     py_func_input_model = tf.keras.Model(
        #         inputs=self._inputs,
        #         outputs=self._py_func_builder.input_tensors)
        #     py_func_inputs = py_func_input_model(inputs)
        # else:
        #     py_func_inputs = [
        #         i for i, si, pi in zip(inputs, self._inputs,
        #                                self._py_func_builder.input_tensors)
        #         if si == pi
        #     ]
        # py_func_outputs = self._py_func_builder.run(py_func_inputs)
        # final_model = tf.keras.models.Model(
        #     inputs=self._inputs + list(self._py_func_builder.output_tensors),
        #     outputs=self._outputs)
        # final_outputs = final_model(inputs + list(py_func_outputs))
        # return tf.keras.models.Model(inputs, final_outputs)


class PipelineModels(object):
    PRE_BATCH = 'pre_batch'
    POST_BATCH = 'post_batch'
    TRAINED = 'trained'

    @classmethod
    def validate(cls, id_):
        if id_ not in cls.all():
            raise ValueError('invalid PipelineModel key {}'.format(id_))

    @classmethod
    def all(cls):
        return (PipelineModels.PRE_BATCH, PipelineModels.POST_BATCH,
                PipelineModels.TRAINED)


class PipelineBuilder(object):

    def __init__(self):
        self._pre_batch_builder = ModelBuilder()
        self._post_batch_builder = ModelBuilder()
        self._trained_builder = ModelBuilder()
        self._builders = {
            PipelineModels.PRE_BATCH: self._pre_batch_builder,
            PipelineModels.POST_BATCH: self._post_batch_builder,
            PipelineModels.TRAINED: self._trained_builder,
        }

    def py_func_builder(self,
                        pipeline_model=PipelineModels.PRE_BATCH,
                        name=None):
        return self._builders[pipeline_model].py_func_builder(name)

    def pre_batch_input(self, tensor_spec):
        _assert_is_tensor_spec(tensor_spec)
        inp = tf.keras.layers.Input(shape=tensor_spec.shape,
                                    dtype=tensor_spec.dtype,
                                    batch_size=1)
        self._pre_batch_builder.add_input(inp)
        return tf.squeeze(inp, axis=0)

    def batch(self, tensor, ragged=None):
        if ragged is None:
            if isinstance(tensor, tf.RaggedTensor):
                ragged = True
            elif isinstance(tensor, tf.Tensor):
                if tensor.shape[0] is None:
                    raise ValueError(
                        'ragged must be specified if leading dimension is None')
                ragged = False
            else:
                raise ValueError(
                    'tensor must be a Tensor or RaggedTensor, got {}'.format(
                        tensor))

        if ragged is True:
            if isinstance(tensor, tf.RaggedTensor):
                self._pre_batch_builder.add_output(tensor)
                inp = tf.keras.layers.Input(shape=tensor.shape,
                                            ragged=True,
                                            dtype=tensor.dtype)
                self._post_batch_builder.add_input(inp)
                return inp
            elif isinstance(tensor, tf.Tensor):
                if tensor.shape[0] is not None:
                    raise ValueError(
                        'tensor leading dimension must be None if ragged is '
                        'True')
                output = tf.keras.layers.Lambda(
                    lambda x: tf.RaggedTensor.from_tensor(
                        tf.expand_dims(x, axis=0)))(tensor)
                self._pre_batch_builder.add_output(output)
                inp = tf.keras.layers.Input(output.shape,
                                            dtype=output.dtype,
                                            ragged=True)
                self._post_batch_builder.add_input(inp)
                # out = tf.RaggedTensor.from_nested_row_splits(
                #     inp.flat_values, inp.nested_row_splits[1:])
                out = tf.keras.layers.Lambda(
                    lambda x: tf.RaggedTensor.from_nested_row_splits(
                        x.flat_values, x.nested_row_splits[1:]))(inp)
                return out
            else:
                raise ValueError(
                    'tensor must be a Tensor or RaggedTensor, got {}'.format(
                        tensor))
        else:
            if not isinstance(tensor, tf.Tensor):
                raise ValueError(
                    'tensor must be a tensor if Ragged is False, got {}'.format(
                        tensor))
            self._pre_batch_builder.add_output(tensor)
            out = tf.keras.layers.Input(shape=tensor.shape, dtype=tensor.dtype)
            self._post_batch_builder.add_input(out)
            return out

    def _trained_input(self, tensor):
        assert (isinstance(tensor, tf.Tensor))
        assert (len(tensor.shape) > 0)
        self._post_batch_builder.add_output(tensor)
        inp = tf.keras.layers.Input(shape=tensor.shape[1:],
                                    dtype=tensor.dtype,
                                    batch_size=tensor.shape[0])
        self._trained_builder.add_input(inp)
        return inp

    def trained_input(self, tensor):
        if isinstance(tensor, tf.RaggedTensor):
            # components = (tensor.flat_values,) + tensor.nested_row_splits
            components = tf.keras.layers.Lambda(
                lambda x: [x.flat_values] + list(x.nested_row_splits))(tensor)
            components = [self._trained_input(c) for c in components]
            # return components
            return tf.RaggedTensor.from_nested_row_splits(
                components[0], components[1:])
            # return tf.keras.layers.Lambda(
            #     lambda args: tf.RaggedTensor.from_nested_row_splits(
            #         args[0], args[1:]))(components)
        elif not isinstance(tensor, tf.Tensor):
            raise ValueError('tensor must be a Tensor or RaggedTensor, got '
                             '{}'.format(tensor))
        if len(tensor.shape) == 0:
            tensor = tf.expand_dims(tensor, axis=0)
            tensor = self._trained_input(tensor)
            return tf.squeeze(tensor, axis=0)
        return self._trained_input(tensor)

    def trained_output(self, tensor):
        self._trained_builder.add_output(tensor)
        return tensor

    def finalize(self):
        self._pre_batch_builder.finalize()
        if self._pre_batch_builder.model.trainable_weights:
            raise ValueError('pre_batch_builder.model has trainable weights')
        self._post_batch_builder.finalize()
        if self._post_batch_builder.model.trainable_weights:
            raise ValueError('post_batch_builder.model has trainable weights')
        self._trained_builder.finalize()

    def pre_batch_map(self, *args):
        args = tuple(tf.expand_dims(arg, axis=0) for arg in args)
        model = self._pre_batch_builder.model
        return model(*args)

    def post_batch_map(self, *args):
        model = self._post_batch_builder.model
        return model(*args)

    @property
    def trained_model(self):
        return self._trained_builder.model


if __name__ == '__main__':
    import numpy as np
    import functools
    from deep_cloud.ops.np_utils.tree_utils import pykd
    from deep_cloud.ops.np_utils.tree_utils import core

    size = 1024
    DEFAULT_TREE = pykd.KDTree
    SQRT_2 = np.sqrt(2.)

    builder = PipelineBuilder()
    py_func = builder.py_func_builder('pre_batch')
    coords = builder.pre_batch_input(
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

    pf_coords = py_func.input_node(coords)
    pf_tree = py_func.node(DEFAULT_TREE, pf_coords)

    def rescale(tree, coords):
        dists, indices = tree.query(tree.data, 2, return_distance=True)
        del indices
        coords /= np.mean(dists[:, 1])
        return None

    py_func.node(rescale, pf_tree, pf_coords)
    coords = py_func.output_tensor(
        pf_coords, tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

    def get_conv_args(tree, coords, radius, k0):
        indices = tree.query_ball_point(coords, radius, approx_neighbors=k0)
        rc = np.repeat(coords, indices.row_lengths,
                       axis=0) - coords[indices.flat_values]
        rc /= radius
        return rc, indices.flat_values, indices.row_splits

    pf_rc, pf_fi, pf_rs = py_func.unstack(py_func.node(
        functools.partial(get_conv_args, radius=8, k0=16), pf_tree, pf_coords),
                                          num_args=3)

    rc = py_func.output_tensor(pf_rc,
                               tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
    fi = py_func.output_tensor(pf_fi,
                               tf.TensorSpec(shape=(None,), dtype=tf.int64))
    rs = py_func.output_tensor(pf_rs,
                               tf.TensorSpec(shape=(None,), dtype=tf.int64))

    rc, fi, rs = tf.nest.map_structure(lambda x: builder.batch(x, ragged=True),
                                       (rc, fi, rs))

    rc, fi, rs = tf.nest.map_structure(builder.trained_input, (rc, fi, rs))
    for x in (rc, fi, rs):
        out = tf.reduce_max(x)
        builder.trained_output(out)

    # outs = builder._post_batch_builder._outputs
    # inps = builder._trained_builder._inputs
    outs = builder._pre_batch_builder._outputs
    inps = builder._post_batch_builder._inputs
    for out, inp in zip(outs, inps):
        print('---')
        print(out.dtype, out.shape)
        print(inp.dtype, inp.shape)
        print(type(inp))
    print(len(inps), len(outs))
    # exit()
    print('woot')
    builder.finalize()
    # builder._pre_batch_builder.finalize()
    # builder._post_batch_builder.finalize()
    # builder._trained_builder.finalize()
