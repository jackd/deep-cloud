from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def get_pca_xy_angle(positions, rotation_axis=2):
    from sklearn.decomposition import PCA

    def get_pca(xy):
        pca = PCA(n_components=1)
        xy = xy.numpy()
        pca.fit_transform(xy)
        pca_vec = tf.squeeze(pca.components_, axis=0)
        return pca_vec

    shift = 2 - rotation_axis
    if shift != 0:
        positions = tf.roll(positions, shift, axis=-1)

    xy, _ = tf.split(positions, [2, 1], axis=-1)
    pca_vec = tf.py_function(get_pca, [xy], positions.dtype)
    pca_vec.set_shape((2,))
    x, y = tf.unstack(pca_vec, axis=0)
    angle = tf.atan2(y, x)
    return angle
