from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.neighbors import KDTree as _KDTree
from deep_cloud.ops.np_utils.tree_utils import core
from more_keras.ragged.np_impl import RaggedArray
import gin


@gin.configurable(module='tree_utils.skkd')
class KDTree(core.KDTree):

    def __init__(self, data, default_max_neighbors=None, **kwargs):
        data = np.asanyarray(data)
        self._n = data.shape[0]
        self.tree = _KDTree(np.asanyarray(data), **kwargs)
        self.default_max_neighbors = default_max_neighbors

    @property
    def data(self):
        return self.tree.data

    @property
    def n(self):
        return self._n

    def query(self, x, k, distance_upper_bound=np.inf, return_distance=True):
        if distance_upper_bound is None or np.isinf(distance_upper_bound):
            return self.tree.query(x, k, return_distance=return_distance)

        dists, indices = self.tree.query(x, k, return_distance=True)
        invalid = dists >= distance_upper_bound
        dists[invalid] = np.inf
        indices[invalid] = self.n
        if return_distance:
            return dists, indices
        else:
            return indices

    def query_ball_point(self, x, r, max_neighbors=None, approx_neighbors=None):
        ragged_lists, _ = self.tree.query_radius(x,
                                                 r,
                                                 return_distance=True,
                                                 sort_results=True)
        return core._maybe_clip(ragged_lists, max_neighbors,
                                self.default_max_neighbors)
