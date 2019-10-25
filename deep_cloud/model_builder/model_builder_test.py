from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from deep_cloud.model_builder.model_builder import ModelBuilder


# class ModelBuilderTest(object):
class ModelBuilderTest(tf.test.TestCase):

    # def evaluate(self, x):
    #     return tf.nest.map_structure(lambda x: x.numpy(), x)

    def test_single_io(self):
        mb = ModelBuilder()
        pf = mb.py_func_builder()

        x_inp = tf.keras.layers.Input(shape=(), dtype=tf.float32)
        mb.add_input(x_inp)

        x_pf = pf.input_node(x_inp)

        def f(x):
            return 2 * x

        y_pf = pf.node(f, x_pf)
        y_out = pf.output_tensor(y_pf,
                                 tf.TensorSpec(shape=(None,), dtype=tf.float32))

        mb.add_output(y_out)
        mb.finalize()
        model = mb.model

        x = np.arange(10).astype(np.float32)
        expected = f(x)
        x_tf = tf.constant(x, dtype=tf.float32)
        y_tf = model([x_tf])

        np.testing.assert_allclose(self.evaluate(y_tf), expected)

    def test_multi_io(self):
        mb = ModelBuilder()
        pf = mb.py_func_builder()

        x_inp = tf.keras.layers.Input(shape=(), dtype=tf.float32)
        y_inp = tf.keras.layers.Input(shape=(), dtype=tf.float32)

        x_pf = pf.input_node(x_inp)
        y_pf = pf.input_node(y_inp)

        def f(x, y):
            return x + y

        def g(x, y):
            return x * y

        def h(x, y):
            return x**2 + y

        s_pf = pf.node(f, x_pf, y_pf)
        t_pf = pf.node(g, x_pf, s_pf)
        s_tf = pf.output_tensor(s_pf, tf.TensorSpec(shape=(5,),
                                                    dtype=tf.float32))
        t_tf = pf.output_tensor(t_pf, tf.TensorSpec(shape=(5,),
                                                    dtype=tf.float32))
        u_tf = h(s_tf, t_tf)

        x = np.arange(5).astype(np.float32)
        y = np.arange(5, 10).astype(np.float32)
        s_expected = f(x, y)
        t_expected = g(x, s_expected)
        u_expected = h(s_expected, t_expected)

        for inp in (x_inp, y_inp):
            mb.add_input(inp)

        for out in (s_tf, t_tf, u_tf):
            mb.add_output(out)

        mb.finalize()
        model = mb.model

        s_tf, t_tf, u_tf = model([x, y])

        np.testing.assert_allclose(self.evaluate(s_tf), s_expected)
        np.testing.assert_allclose(self.evaluate(t_tf), t_expected)
        np.testing.assert_allclose(self.evaluate(u_tf), u_expected)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
    # ModelBuilderTest().test_single_io()
