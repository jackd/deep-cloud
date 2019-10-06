from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def prebatch_map(inputs, prebatch_model):
    inputs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), inputs)
    return prebatch_model(inputs)


def _assert_not_trainable(model):
    if len(model.trainable_weights) != 0:
        raise ValueError('Expected no trainable weights, got {}'.format(
            len(model.trainable_weights)))


class MetaNetworkBuilder(object):

    def __init__(self):
        self._prebatch_inputs = []
        self._prebatch_outputs = []
        self._postbatch_inputs = []
        self._postbatch_outputs = []
        self._trainable_inputs = []
        self._trainable_outputs = []

    def prebatch_model(self):
        model = tf.keras.Model(inputs=self._prebatch_inputs,
                               outputs=self._prebatch_outputs)
        _assert_not_trainable(model)
        return model

    def postbatch_model(self):
        model = tf.keras.Model(inputs=self._postbatch_inputs,
                               outputs=self._postbatch_outputs)
        _assert_not_trainable(model)
        return model

    def trainable_model(self):
        return tf.keras.Model(inputs=self._prebatch_inputs,
                              outputs=self._trainable_outputs)

    def prebatch_input(self, spec):
        assert (isinstance(spec, tf.TensorSpec))
        inp = tf.keras.layers.Input(shape=spec.shape,
                                    dtype=spec.dtype,
                                    batch_size=1)
        self._prebatch_inputs.append(inp)
        return tf.squeeze(inp, axis=0)

    def batched(self, value, ragged=None):
        if ragged is None:
            leading_dim = value.shape[0]
            if isinstance(leading_dim, tf.Dimension):
                leading_dim = leading_dim.value
            if isinstance(value, tf.RaggedTensor):
                ragged = True
            elif leading_dim is None:
                raise ValueError(
                    'ragged must be specified if leading dimension is not '
                    'statically known.')
            else:
                ragged = False

        if ragged:
            if isinstance(value, tf.RaggedTensor):
                spec = tf.RaggedTensorSpec.from_value(value)
                batched = tf.keras.layers.Input(
                    shape=spec._shape,
                    dtype=spec._component_specs[0].dtype,
                    name=spec._component_specs[0].name,
                    ragged=True)
                self._prebatch_outputs.append(value)
                self._postbatch_inputs.append(batched.flat_values)
                self._postbatch_inputs.extend(batched.nested_row_splits)
                return batched
            else:
                assert (isinstance(value, tf.Tensor))
                value = tf.expand_dims(value, axis=0)
                value = tf.keras.layers.Lambda(
                    tf.RaggedTensor.from_tensor)(value)
                self._prebatch_outputs.append(value)
                batched = tf.keras.layers.Input(shape=value.shape,
                                                dtype=value.dtype,
                                                ragged=True)

                self._postbatch_inputs.append(batched)
                batched = tf.keras.layers.Lambda(
                    lambda x: tf.RaggedTensor.from_nested_row_splits(
                        x.flat_values, x.nested_row_splits[1:]))(batched)
                return batched

        else:
            # ragged is False
            if isinstance(value, tf.RaggedTensor):
                raise ValueError('Cannot batch RaggedTensor with ragged=False')
            batched = tf.keras.layers.Input(shape=value.shape,
                                            dtype=value.dtype)

            self._prebatch_outputs.append(value)
            self._postbatch_inputs.append(batched)
            return batched
        raise RuntimeError('Should\'t be here')

    def model_input(self, value):
        if isinstance(value, tf.RaggedTensor):
            spec = tf.RaggedTensorSpec.from_value(value)
            batch_size = spec._component_specs[1].shape[0]
            if batch_size is not None:
                batch_size = batch_size - 1
            model_inp = tf.keras.layers.Input(
                shape=spec._shape[1:],
                dtype=spec._component_specs[0].dtype,
                name=spec._component_specs[0].name,
                batch_size=batch_size,
                ragged=True)

        elif isinstance(value, tf.Tensor):
            model_inp = tf.keras.layers.Input(shape=(value.shape[1:]),
                                              dtype=value.dtype,
                                              batch_size=value.shape[0])
        else:
            raise TypeError('Unrecognized type for model_input value {} - '
                            'must the Tensor or RaggedTensor'.format(value))

        self._postbatch_outputs.append(value)
        self._trainable_inputs.append(model_inp)
        return model_inp

    def model_output(self, value):
        self._trainable_outputs.append(value)
