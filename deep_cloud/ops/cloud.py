from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.ops.np_utils import cloud as _cloud
from more_keras.ops import utils


def get_relative_coords(in_coords, out_coords, indices):
    nd = in_coords.shape.ndims
    assert (nd in (2, 3))
    assert (indices.shape.ndims == nd)
    assert (in_coords.shape.ndims == out_coords.shape.ndims)
    if nd == 2 and isinstance(indices, tf.RaggedTensor):
        return _get_relative_coords_py_function(in_coords, out_coords, indices)
    gather_fn = utils.map_gather if nd == 3 else tf.gather
    in_coords = gather_fn(in_coords, indices)
    out_coords = tf.expand_dims(out_coords, axis=-2)
    return in_coords - out_coords


def _get_relative_coords_py_function(in_coords, out_coords, indices):

    def fn(in_coords, out_coords, flat_indices, row_lengths):
        return _cloud.get_relative_coords(in_coords.numpy(), out_coords.numpy(),
                                          flat_indices.numpy(),
                                          row_lengths.numpy())

    flat_coords = tf.py_function(
        fn, [in_coords, out_coords, indices.values,
             indices.row_lengths()], in_coords.dtype)
    flat_coords.set_shape([None, in_coords.shape[-1]])
    return indices.with_values(flat_coords)
