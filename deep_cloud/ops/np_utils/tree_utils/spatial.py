"""scipy.spatial.cKDTree wrapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree as _cKDTree  # pylint: disable=no-name-in-module
import gin
from more_keras.ragged.np_impl import RaggedArray
from deep_cloud.ops.np_utils.tree_utils import core


@gin.configurable(module='tree_utils.spatial')
class KDTree(core.KDTree):

    def __init__(self, data, **kwargs):
        self.tree = _cKDTree(data, **kwargs)

    @property
    def data(self):
        return self.tree.data

    @property
    def n(self):
        return self.tree.n

    def query(self, x, k, distance_upper_bound=np.inf, return_distance=True):
        dists, indices = self.tree.query(
            x, k, distance_upper_bound=distance_upper_bound)
        if return_distance:
            return dists, indices
        else:
            return indices

    def query_ball_point(self, x, r, max_neighbors=None, approx_neighbors=None):
        """approx_neighbors ignored."""
        ragged_lists = self.tree.query_ball_point(x, r)
        return core._maybe_clip(ragged_lists, max_neighbors, max_neighbors)

    def query_ball_tree(self,
                        other,
                        r,
                        max_neighbors=None,
                        approx_neighbors=None):
        """approx_neighbors ignored."""
        ragged_lists = self.tree.query_ball_tree(other.tree, r)
        return core._maybe_clip(ragged_lists, max_neighbors, max_neighbors)
