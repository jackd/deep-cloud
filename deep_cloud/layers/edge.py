from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_cloud.ops import edge as _edge
from more_keras.layers.utils import lambda_wrapper

reduce_sum = lambda_wrapper(_edge.reduce_sum)
reduce_max = lambda_wrapper(_edge.reduce_max)
reduce_mean = lambda_wrapper(_edge.reduce_mean)
reduce_min = lambda_wrapper(_edge.reduce_min)
reduce_weighted_mean = lambda_wrapper(_edge.reduce_weighted_mean)
distribute_node_features = lambda_wrapper(_edge.distribute_node_features)
