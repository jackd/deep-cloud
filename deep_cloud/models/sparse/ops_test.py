from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from deep_cloud.models.sparse import ops


# class SparseOpsTest(object):
class SparseOpsTest(tf.test.TestCase):

    def test_featureless_conv(self):
        kernel = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        sparse_indices = np.array(
            [[0, 0], [0, 1], [0, 3], [1, 0], [1, 1], [2, 2], [3, 0], [3, 3]],
            dtype=np.int64)
        edge_weights = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
        ],
                                dtype=np.float32).T

        weight_sum = np.array(
            [[9, 12, 15], [21, 23, 25], [15, 16, 17], [39, 41, 43]],
            dtype=np.float32)
        expected = np.matmul(weight_sum, kernel)

        out = ops.featureless_conv(kernel, sparse_indices, edge_weights)
        self.assertEqual(out.shape, tf.TensorShape((4, 2)))
        actual = self.evaluate(out)
        np.testing.assert_allclose(expected, actual)

    def test_conv(self):
        from scipy import sparse
        N_in = 5
        N_out = 3
        T = 2
        E = 5
        F_in = 7
        F_out = 11

        features = np.random.uniform(size=(N_in, F_in)).astype(np.float32)
        kernel = np.random.uniform(size=(T, F_in, F_out)).astype(np.float32)
        sparse_indices = np.array([[0, 0], [0, 1], [1, 2], [2, 3], [2, 4]],
                                  dtype=np.int64)
        edge_weights = np.random.uniform(size=(T, E)).astype(np.float32)
        dense_shape = (N_out, N_in)

        i, j = sparse_indices.T

        expected = np.zeros((N_out, F_out), dtype=np.float32)
        for t in range(T):
            N = sparse.csr_matrix(sparse.coo_matrix((edge_weights[t], (i, j))))
            theta = kernel[t]
            expected += (N @ features) @ theta

        for transform_first in (False, True):
            for fn in (ops.fold_conv, ops.unstack_conv, ops.block_conv):
                out = fn(features,
                         kernel,
                         sparse_indices,
                         edge_weights,
                         dense_shape,
                         transform_first=transform_first)
                self.assertEqual(out.shape, tf.TensorShape((N_out, F_out)))
                np.testing.assert_allclose(self.evaluate(out),
                                           expected,
                                           rtol=1e-4,
                                           atol=1e-4)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.enable_v2_tensorshape()
    tf.test.main()
    # SparseOpsTest().test_conv()
    # print('good')
