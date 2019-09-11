from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import numpy as np
from more_keras.ragged.np_impl import RaggedArray
from deep_cloud.ops.np_utils import sample

na = 50
nb = 40


class TreeImplTest(object):

    @abc.abstractmethod
    def kd_tree(self, data):
        raise NotImplementedError

    def test_query(self):
        radius = 1
        pa = [
            [0, 0],
            [10, 0],
            [0, 0.5],
        ]
        pb = [[0, 0.1], [-0.9, 0]]
        in_tree = self.kd_tree(pa)
        dists, indices = in_tree.query(pb, 2, np.inf, return_distance=True)
        np.testing.assert_allclose(
            dists, [[0.1, 0.4], [0.9, np.sqrt(0.9**2 + 0.5**2)]])
        valid = in_tree.valid(indices)
        flat_values = indices[valid]
        np.testing.assert_equal(flat_values, [0, 2, 0, 2])
        self.assertTrue(np.all(valid))  # pylint: disable=no-member
        indices = in_tree.query(pb, 2, radius, return_distance=False)
        valid = in_tree.valid(indices)
        np.testing.assert_equal(indices.flatten()[:3], [0, 2, 0])
        np.testing.assert_equal(valid, [[True, True], [True, False]])

    def test_query_ball_point(self):
        radius = 1.
        pa = [
            [0, 0],
            [10, 0],
            [0, 0.5],
        ]
        pb = [[0, 0.1], [-0.9, 0]]
        in_tree = self.kd_tree(pa)
        ragged = in_tree.query_ball_point(pb, radius, max_neighbors=5)
        np.testing.assert_equal(ragged.flat_values, [0, 2, 0])
        np.testing.assert_equal(ragged.row_lengths, [2, 1])

    def test_query_ball_tree(self):
        radius = 1
        pa = [
            [0, 0],
            [10, 0],
            [0, 0.5],
        ]
        pb = [[0, 0.1], [-0.9, 0]]
        in_tree = self.kd_tree(pa)
        out_tree = self.kd_tree(pb)
        arr = out_tree.query_ball_tree(in_tree, radius, max_neighbors=5)
        np.testing.assert_equal(arr.flat_values, [0, 2, 0])
        np.testing.assert_equal(arr.row_lengths, [2, 1])

    def test_query_pairs(self):
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        tree = self.kd_tree(in_coords)
        base = tree.query_ball_tree(tree, radius, max_neighbors=20)
        efficient = tree.query_pairs(radius, max_neighbors=20)
        np.testing.assert_equal(base.flat_values, efficient.flat_values)
        np.testing.assert_equal(base.row_splits, efficient.row_splits)
        clamped = tree.query_pairs(radius, max_neighbors=5)
        self.assertLessEqual(np.max(clamped.row_lengths), 5)  # pylint: disable=no-member

    def test_query_mask(self):
        # logic test: should be the same
        # query_pairs(in_tree, radius)[mask]
        # query_ball_tree(in_tree, cKDTree(in_tree.data[mask]), radius)
        radius = 0.1
        r = np.random.RandomState(123)  # pylint: disable=no-member
        in_coords = r.uniform(size=(na, 3))
        in_tree = self.kd_tree(in_coords)
        neighbors = in_tree.query_pairs(radius, max_neighbors=20)
        mask = sample.inverse_density_mask(
            neighborhood_size=neighbors.row_lengths, mean_keep_rate=0.5)
        out_coords = in_coords[mask]
        out_tree = self.kd_tree(out_coords)
        out_neigh = out_tree.query_ball_tree(in_tree, radius, max_neighbors=20)
        out_neigh2 = neighbors.mask(mask)

        np.testing.assert_equal(out_neigh.values, out_neigh2.values)
        np.testing.assert_equal(out_neigh.row_splits, out_neigh2.row_splits)
