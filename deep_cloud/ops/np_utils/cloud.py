from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_relative_coords(in_coords, out_coords, flat_indices, row_lengths):
    assert (len(in_coords.shape) == 2)
    assert (len(out_coords.shape) == 2)
    assert (len(flat_indices.shape) == 1)
    assert (len(row_lengths.shape) == 1)
    out_coords = np.repeat(out_coords, row_lengths, axis=0)
    in_coords = in_coords[flat_indices]
    return in_coords - out_coords
