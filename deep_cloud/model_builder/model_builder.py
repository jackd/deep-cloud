from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.model_builder.py_func_builder import PyFuncBuilder


class ModelBuilder(object):

    def __init__(self):
        self._inputs = []
        self._outputs = []
        self._py_func_builder = None
        self._finalized = False
        self._model = None

    def py_func_builder(self,
                        name=None,
                        input_callback=None,
                        output_callback=None):
        if self._py_func_builder is not None:
            raise NotImplementedError('Only single py_func_builder supported')
        self._py_func_builder = PyFuncBuilder(name=name,
                                              input_callback=input_callback,
                                              output_callback=output_callback)
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

        py_func_outputs = tf.keras.layers.Lambda(self._py_func_builder.run)(
            self._py_func_builder.input_tensors)
        # Add dummy batch dimension to make keras models play nicely
        py_func_outputs = tf.nest.map_structure(
            lambda x: tf.expand_dims(x, axis=0), py_func_outputs)

        final_model = tf.keras.models.Model(
            self._inputs + list(self._py_func_builder.output_tensors),
            self._outputs,
            name='final_model')
        final_out = final_model(self._inputs + list(py_func_outputs))
        return tf.keras.models.Model(self._inputs,
                                     final_out,
                                     name='combined_model')
