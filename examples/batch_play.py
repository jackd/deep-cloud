from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def gen():
    yield np.zeros((5, 4), dtype=np.float32)
    yield np.zeros((3, 4), dtype=np.float32)


dataset = tf.data.Dataset.from_generator(gen,
                                         output_types=tf.float32,
                                         output_shapes=(None, 4))

dataset = dataset.map(
    lambda x: tf.RaggedTensor.from_tensor(tf.expand_dims(x, axis=0)))
dataset = dataset.batch(2)
print(dataset.element_spec)
dataset = dataset.map(lambda rt: tf.RaggedTensor.from_nested_row_splits(
    rt.flat_values, rt.nested_row_splits[1:]))
print(dataset.element_spec)

for x in dataset:
    # print(x)
    print(x.ragged_rank)
    print(x.flat_values.shape)
    print([s.shape for s in x.nested_row_splits])
