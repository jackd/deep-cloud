from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import tensorflow as tf


class Reshaper(object):

    @abc.abstractproperty
    def row_offset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def flatten_leading_dims(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def reshape_leading_dim(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def indices_into_flattened(self, indices):
        raise NotImplementedError

    @staticmethod
    def from_target(target):
        if isinstance(target, tf.Tensor):
            return TensorReshaper(*target.shape[:2])
        elif isinstance(target, tf.RaggedTensor):
            return RaggedReshaper(target.row_splits)
        else:
            raise TypeError(
                'Unrecognized type. Expected Tensor or RaggedTensor, got {}'.
                format(target))


class TensorReshaper(Reshaper):

    def __init__(self, leading_dim, stride):
        # stride should be target.shape[1]
        if leading_dim is None:
            leading_dim = -1
        else:
            assert (isinstance(leading_dim, int))
        assert (isinstance(stride, int))
        self.stride = stride
        self.leading_dim = leading_dim
        self.flat_size = -1 if leading_dim == -1 else leading_dim * stride

    def flatten_leading_dims(self, value):
        assert (isinstance(value, tf.Tensor))
        return tf.reshape(
            value,
            tf.concat([[self.flat_size], tf.shape(value)[2:]], axis=0))

    def reshape_leading_dim(self, value):
        return tf.reshape(
            value,
            tf.concat([[self.leading_dim, self.stride],
                       tf.shape(value)[1:]],
                      axis=0))

    def indices_into_flattened(self, indices):
        if isinstance(indices, tf.RaggedTensor):
            values = indices.values
            offset = tf.repeat(tf.range(indices.nrows(), dtype=values.dtype),
                               indices.row_lengths(),
                               axis=0)
            offset = tf.reshape(offset, (-1, *((1,) * values.shape.ndims)))
            return indices.with_values(values + offset)
        else:
            offset = tf.range(tf.shape(indices)[0]) * self.stride
            return indices + tf.reshape(offset, (-1,) + (1,) *
                                        (indices.shape.ndims - 1))


class RaggedReshaper(Reshaper):

    def __init__(self, row_splits):
        assert (row_splits.shape.ndims == 1)
        self.row_splits = row_splits

    def flatten_leading_dims(self, value):
        assert (isinstance(value, tf.RaggedTensor))
        return value.values

    def reshape_leading_dim(self, value):
        return tf.RaggedTensor.from_row_splits(value, self.row_splits)

    @property
    def row_offset(self):
        if not hasattr(self, '_row_offset'):
            self._row_offset = self.row_splits[:-1]
        return self._row_offset

    def indices_into_flattened(self, indices):
        return indices + tf.reshape(self.row_offset,
                                    (-1,) + (1,) * (indices.shape.ndims - 1))
