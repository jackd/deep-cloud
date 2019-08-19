from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util  # pylint: disable=no-name-in-module
from deep_cloud.ops.np_utils import tree_utils
from deep_cloud.ops import query as q
from more_keras.test_utils import RaggedTestCase


@test_util.run_all_in_graph_and_eager_modes
class QueryTest(RaggedTestCase):

    def test_query_pairs(self):
        coords = np.random.uniform(size=(100, 3))
        radius = 0.1
        tree = tree_utils.cKDTree(coords)
        expected = tree_utils.query_pairs(tree, radius)
        actual = self.evaluate(q.query_pairs(coords, radius))
        self.assertAllEqual(actual.values, expected.values)
        self.assertAllEqual(actual.row_splits, expected.row_splits)

    def test_query_knn(self):
        coords = np.random.uniform(size=(100, 3))
        k = 8
        tree = tree_utils.cKDTree(coords)
        expected = tree_utils.query_knn(tree, k)
        actual = self.evaluate(q.query_knn(coords, k))
        self.assertAllEqual(actual[0], expected[0])
        self.assertAllClose(actual[1], expected[1])


if __name__ == '__main__':
    tf.test.main()
