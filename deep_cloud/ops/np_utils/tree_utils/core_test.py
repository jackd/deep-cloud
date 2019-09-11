from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from deep_cloud.ops.np_utils.tree_utils import spatial
from deep_cloud.ops.np_utils.tree_utils import core
from more_keras.ragged.np_impl import RaggedArray
from deep_cloud.ops.np_utils import sample

na = 50
nb = 40


class BallNeighborhoodTest(unittest.TestCase):

    def test_reverse_query_ball(self):
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        out_coords = r.uniform(size=(nb, 3))
        in_tree = spatial.KDTree(in_coords)
        out_tree = spatial.KDTree(out_coords)
        arr = out_tree.query_ball_tree(in_tree, radius, max_neighbors=None)

        rel_coords = np.repeat(out_coords, arr.row_lengths, axis=0) - \
            in_coords[arr.flat_values]
        rel_dists = np.linalg.norm(rel_coords, axis=-1)

        rev_arr, rev_indices = core.reverse_query_ball(arr, na)
        rel_coords_inv = rel_coords[rev_indices]

        arr_rev, rel_dists_inv = core.reverse_query_ball(arr, na, rel_dists)
        del arr_rev
        np.testing.assert_allclose(np.linalg.norm(rel_coords_inv, axis=-1),
                                   rel_dists_inv)

        naive_arr_rev = in_tree.query_ball_tree(out_tree,
                                                radius,
                                                max_neighbors=None)

        np.testing.assert_equal(naive_arr_rev.flat_values, rev_arr.flat_values)
        np.testing.assert_equal(naive_arr_rev.row_splits, rev_arr.row_splits)

        naive_rel_coords_inv = (
            np.repeat(in_coords, naive_arr_rev.row_lengths, axis=0) -
            out_coords[naive_arr_rev.flat_values])
        naive_rel_dists_inv = np.linalg.norm(naive_rel_coords_inv, axis=-1)

        np.testing.assert_allclose(rel_coords_inv, -naive_rel_coords_inv)
        np.testing.assert_allclose(rel_dists_inv, naive_rel_dists_inv)

    def test_query_gather(self):
        # logic test: should be the same
        # query_pairs(in_tree, radius)[mask]
        # query_ball_tree(in_tree, cKDTree(in_tree.data[mask]), radius)
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        in_tree = spatial.KDTree(in_coords)
        neighbors = in_tree.query_pairs(radius, max_neighbors=None)
        indices = sample.inverse_density_sample(
            in_coords.shape[0] // 2, neighborhood_size=neighbors.row_lengths)
        out_coords = in_coords[indices]
        out_tree = spatial.KDTree(out_coords)
        # out_neigh = in_tree.query_ball_tree(out_tree,
        #                                     radius,
        #                                     max_neighbors=None)
        out_neigh = out_tree.query_ball_tree(in_tree,
                                             radius,
                                             max_neighbors=None)
        out_neigh2 = neighbors.gather(indices)

        np.testing.assert_equal(out_neigh.values, out_neigh2.values)
        np.testing.assert_equal(out_neigh.row_splits, out_neigh2.row_splits)

    def test_truncate(self):
        r = np.random.RandomState(123)  # pylint: disable=no-member
        upper = 10
        row_lengths = (r.uniform(size=(10,)) * upper).astype(np.int64)
        data = np.zeros((np.sum(row_lengths),), dtype=np.bool)
        ragged = RaggedArray.from_row_lengths(data, row_lengths)

        def slow_truncate(ragged, limit):
            return RaggedArray.from_ragged_lists(
                tuple(rl[:limit] for rl in ragged.ragged_lists))

        limit = upper // 2
        actual = core.truncate(ragged, limit)
        expected = slow_truncate(ragged, limit)
        np.testing.assert_equal(actual.flat_values, expected.flat_values)
        np.testing.assert_equal(actual.row_splits, expected.row_splits)


if __name__ == '__main__':
    unittest.main()
