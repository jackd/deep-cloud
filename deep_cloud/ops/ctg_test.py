from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util  # pylint: disable=no-name-in-module
from deep_cloud.ops import ctg


@test_util.run_all_in_graph_and_eager_modes
class CTGTest(tf.test.TestCase):

    def test_ctg(self):
        self.assertGreater(
            self.evaluate(ctg.continuous_truncated_gaussian(0., 4.)), 0.)
        self.assertAllClose(
            self.evaluate(ctg.continuous_truncated_gaussian(4., 4.)), 0.)
        self.assertEquals(
            self.evaluate(ctg.continuous_truncated_gaussian(5., 4.)), 0.)
        self.assertGreater(
            self.evaluate(ctg.continuous_truncated_gaussian(3.8, 4.)), 0.)
        self.assertLess(
            self.evaluate(ctg.continuous_truncated_gaussian(3.8, 4.)), 1.)
        self.assertLess(
            self.evaluate(ctg.continuous_truncated_gaussian(1., 2.)), 1.)


if __name__ == '__main__':
    tf.test.main()
