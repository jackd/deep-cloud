from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.model_builder.model_builder import ModelBuilder
from deep_cloud.model_builder.utils import assert_is_tensor_spec
from deep_cloud.model_builder.utils import TensorDict


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


def _rebuild(inp):
    components = tf.keras.layers.Lambda(
        lambda i: [i.flat_values, *i.nested_row_splits])(inp)
    rebuilt = tf.keras.layers.Lambda(
        lambda c: tf.RaggedTensor.from_nested_row_splits(c[0], c[1:]))(
            components)
    return rebuilt


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
        self._marks = Marks()

    def propagate_marks(self, end):
        return self._marks.propagate(end)

    def py_func_builder(self,
                        pipeline_model=PipelineModels.PRE_BATCH,
                        name=None):

        def callback(tensor):
            self._marks[tensor] = pipeline_model

        return self._builders[pipeline_model].py_func_builder(
            name, input_callback=callback, output_callback=callback)

    def pre_batch_input(self, tensor_spec):
        assert_is_tensor_spec(tensor_spec)
        inp = tf.keras.layers.Input(shape=tensor_spec.shape,
                                    dtype=tensor_spec.dtype,
                                    batch_size=1)
        self._pre_batch_builder.add_input(inp)
        self._marks[inp] = PipelineModels.PRE_BATCH
        return tf.squeeze(inp, axis=0)

    def batch(self, tensor, ragged=None):
        self._marks[tensor] = PipelineModels.PRE_BATCH
        if ragged is None:
            if isinstance(tensor, tf.RaggedTensor):
                ragged = True
            elif isinstance(tensor, tf.Tensor):
                if tensor.shape.ndims > 0 and tensor.shape[0] is None:
                    raise ValueError(
                        'ragged must be specified if leading dimension is None')
                ragged = False
            else:
                raise ValueError(
                    'tensor must be a Tensor or RaggedTensor, got {}'.format(
                        tensor))
        assert (ragged is not None)
        if ragged:
            if isinstance(tensor, tf.RaggedTensor):
                self._pre_batch_builder.add_output(tensor)
                inp = tf.keras.layers.Input(shape=tensor.shape,
                                            ragged=True,
                                            dtype=tensor.dtype)
                self._marks[inp] = PipelineModels.POST_BATCH
                self._post_batch_builder.add_input(inp)
                # return inp

                # we rebuild to make keras play nicely.
                components = tf.keras.layers.Lambda(
                    lambda i: [i.flat_values, *i.nested_row_splits])(inp)
                rebuilt = tf.keras.layers.Lambda(
                    lambda c: tf.RaggedTensor.from_nested_row_splits(
                        c[0], c[1:]))(components)
                return rebuilt

            elif isinstance(tensor, tf.Tensor):
                output = tf.keras.layers.Lambda(
                    lambda x: tf.RaggedTensor.from_tensor(
                        tf.expand_dims(x, axis=0)))(tensor)
                self._pre_batch_builder.add_output(output)
                inp = tf.keras.layers.Input(output.shape,
                                            dtype=output.dtype,
                                            ragged=True)
                self._marks[inp] = PipelineModels.POST_BATCH
                self._post_batch_builder.add_input(inp)

                # out = tf.keras.layers.Lambda(
                #     lambda x: tf.RaggedTensor.from_nested_row_splits(
                #         x.flat_values, x.nested_row_splits[1:]))(inp)
                # return out
                components = tf.keras.layers.Lambda(
                    lambda i: [i.flat_values, *i.nested_row_splits[1:]])(inp)
                rebuilt = tf.keras.layers.Lambda(
                    lambda c: tf.RaggedTensor.from_nested_row_splits(
                        c[0], c[1:]))(components)
                return rebuilt
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
        self._marks[tensor] = PipelineModels.POST_BATCH
        assert (isinstance(tensor, tf.Tensor))
        assert (len(tensor.shape) > 0)
        self._post_batch_builder.add_output(tensor)
        inp = tf.keras.layers.Input(shape=tensor.shape[1:],
                                    dtype=tensor.dtype,
                                    batch_size=tensor.shape[0])
        self._marks[inp] = PipelineModels.TRAINED
        self._trained_builder.add_input(inp)
        return inp

    def trained_input(self, tensor):
        if isinstance(tensor, tf.RaggedTensor):
            # components = (tensor.flat_values,) + tensor.nested_row_splits
            components = tf.keras.layers.Lambda(
                lambda x: [x.flat_values] + list(x.nested_row_splits))(tensor)
            components = [self._trained_input(c) for c in components]
            # return components
            rt = tf.keras.layers.Lambda(
                lambda args: tf.RaggedTensor.from_nested_row_splits(
                    args[0], args[1:]))(components)
            # rt = tf.RaggedTensor.from_nested_row_splits(components[0],
            #                                             components[1:])
            return rt
        elif not isinstance(tensor, tf.Tensor):
            raise ValueError('tensor must be a Tensor or RaggedTensor, got '
                             '{}'.format(tensor))
        if len(tensor.shape) == 0:
            tensor = tf.expand_dims(tensor, axis=0)
            tensor = self._trained_input(tensor)
            return tf.squeeze(tensor, axis=0)
        return self._trained_input(tensor)

    def trained_output(self, tensor):
        self._marks[tensor] = PipelineModels.TRAINED
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
        return model(args)

    def post_batch_map(self, *args):
        model = self._post_batch_builder.model
        return model(args)

    @property
    def trained_model(self):
        return self._trained_builder.model


def _inputs(x):
    if isinstance(x, tf.Tensor):
        return x.op.inputs
    elif isinstance(x, tf.RaggedTensor):
        return (x.flat_values,) + x.nested_row_splits
    elif isinstance(x, tf.SparseTensor):
        return x.indices, x.values
    else:
        raise ValueError('Invalid type of x: expected Tensor, RaggedTensor'
                         ' or SparseTensor, got {}'.format(x))


class Marks(object):

    def __init__(self):
        self._base = TensorDict()

    def __getitem__(self, x):
        return self._base.get(x, None)

    def __setitem__(self, x, mark):
        # check consistency
        m = self.propagate(x)
        if m is not None and m != mark:
            raise ValueError(
                'Attempted to mark x with {}, got inputs with mark {}'.format(
                    mark, m))
        # propagate marks down dependencies
        self._propagate_down(x, mark)

    def __contains__(self, x):
        return x in self._base

    def _propagate_down(self, x, mark):
        if x not in self:
            self._base[x] = mark
            for i in _inputs(x):
                self._propagate_down(i, mark)

    def propagate(self, end):
        mark = self._base.get(end)
        if mark is not None:
            return mark
        inputs = _inputs(end)
        if len(inputs) == 0:
            return None
        mark = None
        # get marks from all inputs, ensuring consistency
        for i in inputs:
            mi = self.propagate(i)
            if mi is not None:
                if mark is None:
                    mark = mi
                elif mi != mark:
                    raise ValueError(
                        'different marks detected: {} and {}'.format(mark, mi))
        if mark is None:
            return None

        # propagate mark back down input tree.
        self._propagate_down(end, mark)
        return mark


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
                                          num_outputs=3)

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
