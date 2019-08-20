from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from deep_cloud.ops.np_utils import tree_utils
from more_keras.ragged.np_impl import RaggedArray
from deep_cloud.ops.np_utils import sample

na = 50
nb = 40


# class BallNeighborhoodTest(object):
class BallNeighborhoodTest(unittest.TestCase):

    def test_query_ball_tree(self):
        radius = 1
        pa = [
            [0, 0],
            [10, 0],
            [0, 0.5],
        ]
        pb = [[0, 0.1], [-0.9, 0]]
        in_tree = tree_utils.cKDTree(pa)
        out_tree = tree_utils.cKDTree(pb)
        arr = tree_utils.query_ball_tree(in_tree, out_tree, radius)
        np.testing.assert_equal(arr.flat_values, [0, 2, 0])
        np.testing.assert_equal(arr.row_lengths, [2, 1])

    def test_query_pairs(self):
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        tree = tree_utils.cKDTree(in_coords)
        base = RaggedArray.from_ragged_lists(tree.query_ball_tree(tree, radius),
                                             dtype=np.int64)
        efficient = tree_utils.query_pairs(tree, radius)
        np.testing.assert_equal(base.flat_values, efficient.flat_values)
        np.testing.assert_equal(base.row_splits, efficient.row_splits)
        clamped = tree_utils.query_pairs(tree, radius, max_neighbors=5)
        self.assertLessEqual(np.max(clamped.row_lengths), 5)

    def test_reverse_query_ball(self):
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
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
        del arr_rev
        np.testing.assert_allclose(np.linalg.norm(rel_coords_inv, axis=-1),
                                   rel_dists_inv)

        naive_arr_rev = tree_utils.query_ball_tree(out_tree, in_tree, radius)

        np.testing.assert_equal(naive_arr_rev.flat_values, rev_arr.flat_values)
        np.testing.assert_equal(naive_arr_rev.row_splits, rev_arr.row_splits)

        naive_rel_coords_inv = (
            np.repeat(in_coords, naive_arr_rev.row_lengths, axis=0) -
            out_coords[naive_arr_rev.flat_values])
        naive_rel_dists_inv = np.linalg.norm(naive_rel_coords_inv, axis=-1)

        np.testing.assert_allclose(rel_coords_inv, -naive_rel_coords_inv)
        np.testing.assert_allclose(rel_dists_inv, naive_rel_dists_inv)

    def test_query_mask(self):
        # logic test: should be the same
        # query_pairs(in_tree, radius)[mask]
        # query_ball_tree(in_tree, cKDTree(in_tree.data[mask]), radius)
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        in_tree = tree_utils.cKDTree(in_coords)
        neighbors = tree_utils.query_pairs(in_tree, radius)
        mask = sample.inverse_density_mask(
            neighborhood_size=neighbors.row_lengths, mean_keep_rate=0.5)
        out_coords = in_coords[mask]
        out_tree = tree_utils.cKDTree(out_coords)
        out_neigh = tree_utils.query_ball_tree(in_tree, out_tree, radius)
        out_neigh2 = neighbors.mask(mask)

        np.testing.assert_equal(out_neigh.values, out_neigh2.values)
        np.testing.assert_equal(out_neigh.row_splits, out_neigh2.row_splits)

    def test_query_gather(self):
        # logic test: should be the same
        # query_pairs(in_tree, radius)[mask]
        # query_ball_tree(in_tree, cKDTree(in_tree.data[mask]), radius)
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        in_tree = tree_utils.cKDTree(in_coords)
        neighbors = tree_utils.query_pairs(in_tree, radius)
        indices = sample.inverse_density_sample(
            in_coords.shape[0] // 2, neighborhood_size=neighbors.row_lengths)
        out_coords = in_coords[indices]
        out_tree = tree_utils.cKDTree(out_coords)
        out_neigh = tree_utils.query_ball_tree(in_tree, out_tree, radius)
        out_neigh2 = neighbors.gather(indices)

        np.testing.assert_equal(out_neigh.values, out_neigh2.values)
        np.testing.assert_equal(out_neigh.row_splits, out_neigh2.row_splits)


# BallNeighborhoodTest().test_reverse_query_ball()
# BallNeighborhoodTest().test_query_pairs()
if __name__ == '__main__':
    unittest.main()
