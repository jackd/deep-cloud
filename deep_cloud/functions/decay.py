from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def exponential_decay(step,
                      initial_value,
                      decay_steps,
                      decay_rate,
                      min_value=None,
                      staircase=False):
    import numpy as np
    return tf.maximum(
        tf.compat.v1.train.exponential_decay(initial_value,
                                             step,
                                             decay_steps,
                                             decay_rate,
                                             staircase=staircase))
    ## numpy version
    # import numpy as np
    # exponent = step / decay_steps
    # if staircase:
    #     exponent = np.floor(exponent)
    # value = initial_value * decay_rate**exponent
    # if min_value is not None:
    #     value = max(value, min_value)
    # return value


def complementary_exponential_decay(step,
                                    initial_value,
                                    decay_steps,
                                    decay_rate,
                                    max_value=0.99,
                                    staircase=False):
    return 1 - exponential_decay(step,
                                 1 - initial_value,
                                 decay_steps,
                                 decay_rate,
                                 None if max_value is None else 1 - max_value,
                                 staircase=staircase)
