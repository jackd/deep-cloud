from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from weighpoint.ops import cloud


def get_single_rel_coords_data():
    in_coords = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int64)
    out_coords = tf.constant([[7, 10], [9, 20]], dtype=tf.int64)
    indices = tf.RaggedTensor.from_row_lengths(
        tf.constant([0, 2, 0, 1, 2], dtype=tf.int64),
        tf.constant([2, 3], dtype=tf.int64))
    expected = tf.constant([
        [-6, -8],
        [-2, -4],
        [-8, -18],
        [-6, -16],
        [-4, -14],
    ])
    expected = tf.RaggedTensor.from_row_lengths(
        expected, tf.constant([2, 3], dtype=tf.int64))
    return in_coords, out_coords, indices, expected


@test_util.run_all_in_graph_and_eager_modes
class CloudTest(tf.test.TestCase):
    def test_single_rel_coords(self):
        with tf.device('/cpu:0'):
            in_coords, out_coords, indices, expected = \
                get_single_rel_coords_data()
            rel_coords = cloud.get_relative_coords(
                in_coords, out_coords, indices)

        self.assertAllEqual(
            self.evaluate(rel_coords.values),
            self.evaluate(expected.values))
        self.assertAllEqual(
            self.evaluate(rel_coords.row_splits),
            self.evaluate(expected.row_splits))


if __name__ == '__main__':
    tf.test.main()
