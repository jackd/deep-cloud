from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
from more_keras.ragged.np_impl import RaggedArray
from scipy import sparse as sp


def closest_neighbors(tree):
    dists, indices = tree.query(np.reshape(tree.data, (tree.n, -1)), 2)
    del indices
    return np.min(dists[:, 1])


def truncate(neighbors, limit):
    """Take only the first `limit` entries of each row of `neighbors`."""
    row_lengths = neighbors.row_lengths
    true_counts = np.minimum(row_lengths, limit)
    truncated = np.maximum(row_lengths - limit, 0)
    values = np.array([[True, False]], dtype=np.bool)
    repeats = np.stack((true_counts, truncated), axis=1)
    mask = np.repeat(
        np.tile(values, (len(row_lengths), 1)).flatten(), repeats.flatten())
    flat_values = neighbors.flat_values[mask]
    return RaggedArray.from_row_lengths(flat_values, true_counts)


def reverse_query_ball(ragged_array, size=None, data=None):
    """
    Get `query_ball_tree` for reversed in/out trees.

    Also returns data associated with the reverse, or relevant indices.

    Example usage:
    ```python
    radius = 0.1
    na = 50
    nb = 40
    r = np.random.RandomState(123)

    in_coords = r.uniform(size=(na, 3))
    out_coords = r.uniform(size=(nb, 3))

    in_tree = tree_utils.KDTree(in_coords)
    out_tree = tree_utils.KDTree(out_coords)
    arr = tree_utils.query_ball_tree(in_tree, out_tree, radius)

    rel_coords = np.repeat(out_coords, arr.row_lengths, axis=0) - \
        in_coords[arr.flat_values]
    rel_dists = np.linalg.norm(rel_coords, axis=-1)

    rev_arr, rev_indices = tree_utils.reverse_query_ball(arr, na)
    rel_coords_inv = rel_coords[rev_indices]

    arr_rev, rel_dists_inv = tree_utils.reverse_query_ball(
        arr, na, rel_dists)
    np.testing.assert_allclose(
        np.linalg.norm(rel_coords_inv, axis=-1), rel_dists)

    naive_arr_rev = tree_utils.query_ball_tree(out_tree, in_tree, radius)

    np.testing.assert_equal(
        naive_arr_rev.flat_values, rev_arr.flat_values)
    np.testing.assert_equal(
        naive_arr_rev.row_splits, rev_arr.row_splits)

    naive_rel_coords_inv = np.repeat(
        in_coords, naive_arr_rev.row_lengths, axis=0) -\
        out_coords[naive_arr_rev.flat_values]
    naive_rel_dists_inv = np.linalg.norm(naive_rel_coords_inv, axis=-1)

    np.testing.assert_allclose(rel_coords_inv, -naive_rel_coords_inv)
    np.testing.assert_allclose(rel_dists_inv, naive_rel_dists_inv)
    ```

    Args:
        ragged_array: RaggedArray instance, presumably from `query_ball_tree`
            or `query_pairs`. Note if you used `query_pairs`, the returned
            `ragged_out` will be the same as `ragged_array` input (though the
            indices may still be useful)

    Returns:
        ragged_out: RaggedArray corresponding to the opposite tree search for
            which `ragged_array` used.
        data: can be used to transform data calculated using input
            ragged_array. See above example
    """
    if data is None:
        data = np.arange(ragged_array.size, dtype=np.int64)
    # take advantage of fast scipy.sparse implementations
    mat = sp.csr_matrix(
        (data, ragged_array.flat_values, ragged_array.row_splits))
    trans = mat.transpose().tocsr()
    trans.sort_indices()
    row_splits = trans.indptr
    if size is not None:
        diff = size - row_splits.size + 1
        if diff != 0:
            tail = row_splits[-1] * np.ones((diff,), dtype=row_splits.dtype)
            row_splits = np.concatenate([row_splits, tail], axis=0)
    ragged_out = RaggedArray.from_row_splits(trans.indices, row_splits)
    return ragged_out, trans.data


class KDTree(object):

    @abc.abstractproperty
    def data(self):
        raise NotImplementedError

    @abc.abstractproperty
    def n(self):
        raise NotImplementedError

    def valid(self, indices):
        return indices < self.n

    @abc.abstractmethod
    def query(self, x, k, distance_upper_bound=np.inf, return_distance=True):
        """
        Args:
            x: [n2, m] float.
            k: int, number of neighbors
            distance_upper_bound: float, upper limit on distance.
            return_distance: bool, if True also return distances.

        Returns:
            indices, or (dists, indices) if return_distance is True
                indices: [n2, k] indices into data. Value of self.n indicates
                    an invalid index (e.g. because distance_upper_bound was
                    reached)
                dists: [n2, k] distances, or np.inf if not valid.
        """
        raise NotImplementedError

    def query_ball_point(self, x, r, max_neighbors=None, approx_neighbors=None):
        """
        Find points in the tree within `r` of `x` using only `self.query`.

        Note scipy and sklearn both implement their own versions of
        these which ignore max_neighbors and approx_neighbors arguments.

        Args:
            x: [n2, m] float.
            r: float, radius of ball search.
            max_neighbors: int, maximum number of neighbors to consider. If
                `None`, uses `approx_neighbors` and recursive strategy.
            approx_neighbors: int, approximate number of neighbors to consider
                in recursive strategy. Ignored if `max_neighbors` is given.

        Returns:
            [n2, k?] RaggedArray of indices into data.
        """
        if max_neighbors is None:
            if approx_neighbors is None:
                raise ValueError(
                    '`max_neighbors` or `approx_neighbors` must be provided for'
                    '{}.query_ball_point'.format(self.__class__.__name__))
            return self.query_ball_point_recursive(x, r, approx_neighbors)
        else:
            indices = self.query(x,
                                 max_neighbors,
                                 distance_upper_bound=r,
                                 return_distance=False)

            return RaggedArray.from_mask(indices, self.valid(indices))

    def query_ball_tree(self,
                        other,
                        r,
                        max_neighbors=None,
                        approx_neighbors=None):
        """Potential optimization of `other.query_ball_point(self.data, ...)."""
        return other.query_ball_point(self.data,
                                      r,
                                      max_neighbors=max_neighbors,
                                      approx_neighbors=approx_neighbors)

    def query_pairs(self, r, max_neighbors=None, approx_neighbors=None):
        """Potential optimization of 'self.query_ball_tree(self, ...)."""
        return self.query_ball_tree(self,
                                    r,
                                    max_neighbors=max_neighbors,
                                    approx_neighbors=approx_neighbors)

    def _query_ball_point_recursive(self, x, r, approx_neighbors):
        indices = self.query(x,
                             approx_neighbors,
                             distance_upper_bound=r,
                             return_distance=False)
        valid = indices < self.n
        row_lengths = np.count_nonzero(valid, axis=1)
        max_row_length = np.max(row_lengths)
        if max_row_length < approx_neighbors:
            return indices[:, :max_row_length], valid[:, :max_row_length]

        invalid = row_lengths == approx_neighbors
        invalid_indices, = np.where(invalid)

        extra_indices, extra_mask = self._query_ball_point_recursive(
            x[invalid_indices], r, 2 * approx_neighbors)
        k = extra_indices.shape[1]

        shape = (x.shape[0], k)

        indices_out = np.full(shape, self.n, dtype=np.int64)
        indices_out[:, :approx_neighbors] = indices
        indices_out[invalid_indices] = extra_indices

        mask_out = np.zeros(shape, dtype=np.bool)
        mask_out[:, :approx_neighbors] = valid
        mask_out[invalid_indices] = extra_mask

        return indices_out, mask_out

    def query_ball_point_recursive(self, x, r, approx_neighbors):
        """
        Query ball point using only self.query.

        Performs query using k=approx_neighbors, then repeats with double
        the number of neighbors until at least one returns an invalid flag.

        Returns:
            RaggedArray of indices
        """
        indices, valid = self._query_ball_point_recursive(
            x, r, approx_neighbors)
        return RaggedArray.from_mask(indices, valid)


def _maybe_clip(ragged_lists, max_neighbors, default_max_neighbors):
    if max_neighbors is None:
        max_neighbors = default_max_neighbors
    if max_neighbors is not None:
        ragged_lists = [rl[:max_neighbors] for rl in ragged_lists]
    return RaggedArray.from_ragged_lists(ragged_lists, dtype=np.int64)


def rejection_sample_lazy(tree, points, radius, k0):
    N = points.shape[0]
    out = []
    consumed = np.zeros((N,), dtype=np.bool)
    for i in range(N):
        if not consumed[i]:
            out.append(i)
            indices = tree.query_ball_point(np.expand_dims(points[i], 0),
                                            radius,
                                            approx_neighbors=k0)
            indices = indices[0]
            consumed[indices] = True
    return out


def rejection_sample_active(tree, points, radius, k0):
    return rejection_sample_precomputed(
        tree.query_ball_point(points, radius, approx_neighbors=k0))


def rejection_sample_precomputed(indices):
    N = indices.leading_dim
    consumed = np.zeros((N,), dtype=np.bool)
    out = []
    for i in range(N):
        if not consumed[i]:
            consumed[indices[i]] = True
            out.append(i)
    return out
