from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow.python.framework import test_util  # pylint: disable=no-name-in-module
from deep_cloud.ops import conv


@test_util.run_all_in_graph_and_eager_modes
class ConvTest(tf.test.TestCase):
    # class ConvTest(object):

    def test_flat_mean_ragged(self):
        x = np.expand_dims(np.array([2, 3, 1, 5, 6, 2], dtype=np.float32),
                           axis=1)
        weights = np.array([2, 2, 1, 1, 3, 5], dtype=np.float32)
        row_splits = [0, 4, 5, 6]

        xr = tf.RaggedTensor.from_row_splits(x, row_splits)
        wr = tf.RaggedTensor.from_row_splits(weights, row_splits)
        wr = tf.expand_dims(wr, axis=-1)

        naive = tf.reduce_sum(xr * wr, axis=1) / tf.reduce_sum(wr, axis=1)
        implemented = conv.reduce_flat_mean(x, row_splits, weights, eps=0)
        self.assertAllClose(self.evaluate(naive), self.evaluate(implemented))


if __name__ == '__main__':
    tf.test.main()
