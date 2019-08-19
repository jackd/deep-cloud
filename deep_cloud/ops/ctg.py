from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def continuous_truncated_gaussian(x_squared, max_x_squared, scale_factor=1.0):
    """relu(exp(-k*x^2) - exp(-k*xm^2))."""
    return tf.nn.relu(
        tf.exp(-scale_factor * x_squared) -
        tf.exp(-scale_factor * max_x_squared))
