from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

INT_TYPES = tf.int32, tf.int64
FLOAT_TYPES = tf.int32, tf.int64


def assert_flat_tensor(name, tensor, rank=None, dtype=None):
    if not isinstance(tensor.tf.Tensor):
        raise ValueError('{} must be a flat tensor, got {}'.format(
            name, tensor))
    if rank is not None and tensor.shape.ndims != rank:
        raise ValueError('{} must be a rank {} tensor, but got shape {}'.format(
            name, rank, tensor.shape))
    if dtype is not None and not (tensor.dtype == dtype or hasattr(
            dtype, '__contains__') and tensor.dtype in dtype):
        raise ValueError('{} must be of dtype {}, got {}'.format(
            name, dtype, tensor.dtype))
