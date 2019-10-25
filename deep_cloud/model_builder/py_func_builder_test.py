from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from deep_cloud.model_builder.py_func_builder import PyFuncBuilder


# class PyFuncBuilderTest(object):
class PyFuncBuilderTest(tf.test.TestCase):

    # def evaluate(self, x):
    #     return tf.nest.map_structure(lambda x: x.numpy(), x)

    def test_single_io(self):
        pf = PyFuncBuilder()
        x = np.arange(10).astype(np.float32)
        x_tf = tf.constant(x, dtype=tf.float32)
        x_pf = pf.input_node(x_tf)

        def f(x):
            return 2 * x

        y_pf = pf.node(f, x_pf)
        pf.output_tensor(y_pf, tf.TensorSpec(shape=(10,), dtype=tf.float32))
        expected = f(x)

        y_tf, = pf.run()

        np.testing.assert_allclose(self.evaluate(y_tf), expected)

    def test_multi_io(self):
        pf = PyFuncBuilder()
        x = np.arange(5).astype(np.float32)
        y = np.arange(5, 10).astype(np.float32)

        x_tf = tf.constant(x, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)
        x_pf = pf.input_node(x_tf)
        y_pf = pf.input_node(y_tf)

        def f(x, y):
            return x + y

        def g(x, y):
            return x * y

        s_pf = pf.node(f, x_pf, y_pf)
        t_pf = pf.node(g, x_pf, s_pf)
        pf.output_tensor(s_pf, tf.TensorSpec(shape=(5,), dtype=tf.float32))
        pf.output_tensor(t_pf, tf.TensorSpec(shape=(5,), dtype=tf.float32))
        s_expected = f(x, y)
        t_expected = g(x, s_expected)

        s_tf, t_tf = pf.run()

        np.testing.assert_allclose(self.evaluate(s_tf), s_expected)
        np.testing.assert_allclose(self.evaluate(t_tf), t_expected)

    def test_single_model(self):
        pf = PyFuncBuilder()
        x_inp = tf.keras.layers.Input(shape=(10,), dtype=tf.float32)
        x_pf = pf.input_node(x_inp)

        def f(x):
            return 2 * x

        y_pf = pf.node(f, x_pf)
        pf.output_tensor(y_pf, tf.TensorSpec(shape=(None, 10),
                                             dtype=tf.float32))
        model = pf.model()

        x = np.reshape(np.arange(20), (2, 10)).astype(np.float32)
        xi = np.expand_dims(x, axis=0)
        # xi = x
        y_tf = model(xi)
        expected = f(x)

        np.testing.assert_allclose(self.evaluate(y_tf), expected)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
    # PyFuncBuilderTest().test_single_model()
