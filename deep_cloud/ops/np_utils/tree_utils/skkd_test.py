from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from deep_cloud.ops.np_utils.tree_utils import test_utils
from deep_cloud.ops.np_utils.tree_utils import skkd


class SkKDTest(unittest.TestCase, test_utils.TreeImplTest):

    def kd_tree(self, data):
        return skkd.KDTree(data)


if __name__ == '__main__':
    unittest.main()
