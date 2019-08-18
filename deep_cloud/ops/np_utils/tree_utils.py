from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
import scipy.sparse as sp
from more_keras.ragged.np_impl import RaggedArray


def query_ball_tree(in_tree, out_tree, radius):
    """
    Get a RaggedArray containing ball search results.

    For output `ragged`, `ragged[i, j] == k` indicates the `i`th node in
    `out_tree` is within `radius` of the `k`th node of `in_tree`.
    """
    if out_tree is in_tree:
        return query_pairs(out_tree, radius)
    else:
        ragged_lists = out_tree.query_ball_tree(in_tree, radius)
        return RaggedArray.from_ragged_lists(ragged_lists, dtype=np.int64)


def query_knn(tree, k):
    dists, indices = tree.query(tree.data, k)
    return indices, dists


def query_pairs(tree, radius, max_neighbors=None):
    """Same as `query_ball_tree` but for when `in_tree == out_tree.`"""
    pairs = tree.query_pairs(radius, output_type='ndarray')

    # take advantage of fast scipy.sparse implementations
    diag = np.repeat(np.expand_dims(np.arange(tree.n), 1), 2, axis=1)
    pairs = np.concatenate([diag, pairs, pairs[:, -1::-1]], axis=0)
    data = np.ones(shape=pairs.shape[:1], dtype=np.bool)
    mat = sp.coo_matrix((data, pairs.T)).tocsr()
    ragged = RaggedArray.from_row_splits(mat.indices,
                                         mat.indptr,
                                         values_dtype=np.int64)
    if max_neighbors is not None:
        ragged = RaggedArray.from_ragged_lists(
            [l[:max_neighbors] for l in ragged.ragged_lists])
    return ragged


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

    in_tree = tree_utils.cKDTree(in_coords)
    out_tree = tree_utils.cKDTree(out_coords)
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
