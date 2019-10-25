from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

if not hasattr(tf, 'repeat'):
    from tensorflow.python.ops.ragged.ragged_util import repeat  # pylint: disable=import-error
    tf.repeat = repeat
