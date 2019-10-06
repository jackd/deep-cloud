from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from deep_cloud.meta_network import MetaNetworkBuilder
from deep_cloud.meta_network import prebatch_map
tf.compat.v1.enable_eager_execution()
# tf.compat.v1.enable_v2_tensorshape()


def test_tensor():

    def gen():
        yield np.ones(5)
        yield np.ones(5)

    dataset = tf.data.Dataset.from_generator(gen, tf.float64, (5,))
    builder = MetaNetworkBuilder()
    a_spec = dataset.element_spec
    a_inp = builder.prebatch_input(a_spec)
    batched_a_inp = builder.batched(2 * a_inp)
    builder.model_input(3 * batched_a_inp)

    dataset = dataset.map(
        lambda args: prebatch_map(args, builder.prebatch_model()))
    dataset = dataset.batch(2)
    dataset = dataset.map(builder.postbatch_model())
    for example in dataset:
        print(example.numpy())


def test_ragged():

    def gen():
        yield np.ones(5)
        yield np.ones(3)

    dataset = tf.data.Dataset.from_generator(gen, tf.float64, (None,))
    builder = MetaNetworkBuilder()
    a_spec = dataset.element_spec
    a_inp = builder.prebatch_input(a_spec)
    a_inp = tf.keras.layers.Lambda(lambda x: x * 2)(a_inp)
    batched_a_inp = builder.batched(a_inp, ragged=True)
    x = tf.keras.layers.Lambda(lambda x: x * 3)(batched_a_inp)
    builder.model_input(x)

    dataset = dataset.map(
        lambda args: prebatch_map(args, builder.prebatch_model()))
    dataset = dataset.batch(2)
    dataset = dataset.map(builder.postbatch_model())
    for example in dataset:
        print(example.flat_values.numpy())
        print(example.row_splits.numpy())
        print(example)
        print(example.shape)


def test_mixed():

    def gen():
        yield (np.ones(5), np.ones(3))
        yield (np.ones(5), np.ones(7))

    batch_size = 2

    dataset = tf.data.Dataset.from_generator(gen, (tf.float64, tf.float64),
                                             ((5,), (None,)))

    builder = MetaNetworkBuilder()
    a_spec, b_spec = dataset.element_spec
    a_inp = builder.prebatch_input(a_spec)
    b_inp = builder.prebatch_input(b_spec)

    batched_a_inp = builder.batched(a_inp * 2) * 3
    # batched_b_inp = builder.batched(b_inp)
    batched_b_inp = builder.batched(b_inp, ragged=True)
    batched_b_inp = tf.keras.layers.Lambda(lambda x: x * 3)(batched_b_inp)

    builder.model_input(batched_a_inp)
    builder.model_input(batched_b_inp)

    dataset = dataset.map(
        lambda *args: prebatch_map(args, builder.prebatch_model()))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda *args: tuple(builder.postbatch_model()(args)))

    for example in dataset:
        print(example)
        tf.nest.map_structure(lambda x: print(x.shape), example)
        # print(tf.nest.map_structure(lambda x: x.numpy(), example))


# test_tensor()
# test_ragged()
test_mixed()
