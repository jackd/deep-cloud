"""pykdtree.kdtree.KDTree wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import gin
try:
    from pykdtree.kdtree import KDTree as _KDTree  # pylint: disable=no-name-in-module
except ImportError:
    raise ImportError(
        'Failed to import pykdtree. Please follow instructions at '
        'https://github.com/storpipfugl/pykdtree, e.g. '
        '`conda install -c conda-forge pykdtree` ')
from more_keras.ragged.np_impl import RaggedArray
from deep_cloud.ops.np_utils.tree_utils import core


@gin.configurable(module='tree_utils.pykd')
class KDTree(core.KDTree):

    def __init__(self, data, **kwargs):
        self.tree = _KDTree(np.asanyarray(data), **kwargs)

    @property
    def data(self):
        return np.reshape(self.tree.data, (self.tree.n, -1))

    @property
    def n(self):
        return self.tree.n

    def query(self, x, k, distance_upper_bound=np.inf, return_distance=True):
        dists, indices = self.tree.query(
            np.asanyarray(x),
            k,
            distance_upper_bound=distance_upper_bound,
            sqr_dists=not return_distance)
        if return_distance:
            return dists, indices
        else:
            return indices


# def recursive_query_ball_point(tree,
#                                points,
#                                distance_upper_bound,
#                                k0,
#                                correct_unmasked=True):
#     dists, indices = tree.query(points,
#                                 k0,
#                                 distance_upper_bound=distance_upper_bound)
#     mask = np.logical_not(np.isinf(dists))
#     row_lengths = np.count_nonzero(mask, axis=1)
#     max_row_length = np.max(row_lengths)
#     if max_row_length < k0:
#         return (
#             indices[:, :max_row_length],
#             mask[:, :max_row_length],
#             dists[:, :max_row_length],
#         )

#     invalid = row_lengths == k0
#     invalid_indices, = np.where(invalid)

#     extra_indices, extra_mask, extra_dists = recursive_query_ball_point(
#         tree,
#         points[invalid_indices],
#         distance_upper_bound,
#         2 * k0,
#         correct_unmasked=False)
#     k = extra_indices.shape[1]

#     shape = (points.shape[0], k)

#     indices_out = np.empty(shape, dtype=np.int64)
#     indices_out[:, :k0] = indices
#     indices_out[invalid_indices] = extra_indices

#     mask_out = np.zeros(shape, dtype=np.bool)
#     mask_out[:, :k0] = mask
#     mask_out[invalid_indices] = extra_mask

#     dists_out = np.empty(shape, dtype=dists.dtype)
#     dists_out[:, :k0] = dists
#     dists_out[invalid_indices] = extra_dists

#     if correct_unmasked:
#         not_masked = np.logical_not(mask_out)
#         indices_out[not_masked] = tree.n
#         dists_out[not_masked] = np.inf

#     return indices_out, mask_out, dists_out
