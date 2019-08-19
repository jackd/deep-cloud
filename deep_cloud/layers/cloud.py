from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_cloud.ops import cloud as _cloud
from more_keras.ragged.layers import ragged_lambda


def _get_relative_coords(args):
    return _cloud.get_relative_coords(*args)


def get_relative_coords(in_coords, out_coords, indices, name=None):
    return ragged_lambda(_get_relative_coords,
                         name=name)([in_coords, out_coords, indices])
