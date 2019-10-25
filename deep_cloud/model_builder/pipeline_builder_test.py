from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from deep_cloud.model_builder import pipeline_builder as pl_lib


class Scaler(tf.keras.layers.Layer):

    def __init__(self, initializer, **kwargs):
        self._initializer = initializer
        super(Scaler, self).__init__(**kwargs)

    def build(self, input_shape):
        self._tensor = self.add_weight('scalar',
                                       initializer=self._initializer,
                                       shape=(),
                                       dtype=tf.float32)

    def call(self, inputs):
        return inputs * self._tensor


# def tree_dfs(starts, neighbors_fn):
#     stack = list(starts)
#     while len(stack) > 0:
#         out = stack.pop()
#         stack.extend(neighbors_fn(out))
#         yield out


# class PipelineBuilderTest(object):
class PipelineBuilderTest(tf.test.TestCase):

    # def evaluate(self, x):
    #     return tf.nest.map_structure(lambda x: x.numpy(), x)

    # def assertEqual(self, a, b):
    #     assert (a == b)

    def test_single_io(self):
        pl = pl_lib.PipelineBuilder()
        batch_size = 2
        num_elements = 5
        x_np = np.reshape(np.arange(batch_size * num_elements),
                          (batch_size, num_elements)).astype(np.float32)
        scale_factor = 5.

        def gen():
            return x_np

        def pre_batch_map(x):
            return 2 * x

        def post_batch_map(x):
            return 3 * x

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (5,))

        inp = pl.pre_batch_input(dataset.element_spec)
        self.assertEqual(inp.shape, dataset.element_spec.shape)
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        batched = post_batch_map(pl.batch(pre_batch_map(inp)))
        trained = pl.trained_input(batched)
        pl.trained_output(
            Scaler(tf.keras.initializers.constant(scale_factor))(trained))
        pl.finalize()
        model = pl.trained_model
        dataset = dataset.map(pl.pre_batch_map).batch(batch_size).map(
            pl.post_batch_map)

        expected_batched = post_batch_map(
            np.array([pre_batch_map(x) for x in gen()]))
        expected_output = scale_factor * expected_batched

        example = None
        output = None
        for example in dataset:
            output = model(example)
            break

        np.testing.assert_allclose(self.evaluate(example), expected_batched)
        np.testing.assert_allclose(self.evaluate(output), expected_output)

    def test_ragged(self):
        pl = pl_lib.PipelineBuilder()
        x_np = [
            np.arange(5).astype(np.float32),
            np.arange(5, 12).astype(np.float32)
        ]
        scale_factor = 5.

        def gen():
            return x_np

        def pre_batch_map(x):
            if isinstance(x, list):
                return [xi * 2 for xi in x]
            elif isinstance(x, tf.RaggedTensor):
                return tf.ragged.map_flat_values(lambda x: x * 2, x)
            else:
                return x * 2

        def post_batch_map(x):
            if isinstance(x, list):
                return [xi * 3 for xi in x]
            else:
                return x * 3

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (None,))

        inp = pl.pre_batch_input(dataset.element_spec)
        self.assertEqual(inp.shape[0], dataset.element_spec.shape[0])
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        batched = pl.batch(pre_batch_map(inp), ragged=True)
        batched = tf.keras.layers.Lambda(post_batch_map)(batched)
        trained = pl.trained_input(batched)
        trained = Scaler(tf.keras.initializers.constant(scale_factor))(trained)
        pl.trained_output(trained.flat_values)
        pl.finalize()

        model = pl.trained_model
        dataset = dataset.map(pl.pre_batch_map).batch(2).map(pl.post_batch_map)

        expected_batched = post_batch_map([pre_batch_map(x) for x in gen()])
        expected_output = [scale_factor * x for x in expected_batched]

        example = None
        output = None
        for example in dataset:
            output = model(example)
            break

        expected_output = np.concatenate(expected_output, axis=0)

        np.testing.assert_allclose(self.evaluate(output), expected_output)

    def test_marks(self):
        pl = pl_lib.PipelineBuilder()
        batch_size = 2
        num_elements = 5
        x_np = np.reshape(np.arange(batch_size * num_elements),
                          (batch_size, num_elements)).astype(np.float32)
        scale_factor = 5.

        def gen():
            return x_np

        def pre_batch_map(x):
            return 2 * x

        def post_batch_map(x):
            return 3 * x

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (5,))

        inp = pl.pre_batch_input(dataset.element_spec)
        self.assertEqual(inp.shape, dataset.element_spec.shape)
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        batched = post_batch_map(pl.batch(pre_batch_map(inp)))
        trained = pl.trained_input(batched)
        trained_out = Scaler(
            tf.keras.initializers.constant(scale_factor))(trained)
        pl.trained_output(trained_out)
        pl.propagate_marks(trained_out)
        pl.finalize()

        mod = pl_lib.PipelineModels

        self.assertEqual(pl.propagate_marks(inp), mod.PRE_BATCH)
        self.assertEqual(pl.propagate_marks(batched), mod.POST_BATCH)
        self.assertEqual(pl.propagate_marks(trained), mod.TRAINED)
        self.assertEqual(pl.propagate_marks(trained_out), mod.TRAINED)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.enable_v2_tensorshape()
    tf.test.main()
    # PipelineBuilderTest().test_marks()
    # print('good')
