from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin


@gin.configurable
def steps_in_examples(num_examples, batch_size=None):
    if batch_size is None:
        # hopefully there's a macro
        batch_size = gin.query_parameter('batch_size/macro.value')
    return num_examples // batch_size
