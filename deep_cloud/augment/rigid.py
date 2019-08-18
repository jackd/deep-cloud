from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf


def _pack_rotation_matrix(c, s, rotation_dim=2):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    if rotation_dim == 0:
        return [1, 0, 0, 0, c, -s, 0, s, c]
    elif rotation_dim == 1:
        return [c, 0, s, 0, 1, 0, -s, 0, c]
    elif rotation_dim == 2:
        return [c, -s, 0, s, c, 0, 0, 0, 1]
    else:
        raise ValueError(
            'rotation_dim must be 0, 1 or 2, got {}'.format(rotation_dim))


def _rotate(positions, normals=None, angle=None, rotation_dim=2, impl=tf):
    """
    Randomly rotate the point cloud about the z-axis.

    Args:
        positions: (n, 3) float array
        normals (optional): (n, 3) float array
        angle: float scalar. If None, a uniform random angle in [0, 2pi) is
            used.
        rotation_dim: int denoting x (0), y (1), or z (2) axis about which to
            rotate
        impl: tf or np

    Returns:
        rotated (`positions`, `normals`). `normals` will be None if not
        provided. shape and dtype is the same as provided.
    """
    dtype = positions.dtype
    if angle is None:
        angle = tf.random.uniform((), dtype=dtype) * (2 * np.pi)

    if normals is not None:
        assert (normals.dtype == dtype)
    c = impl.cos(angle)
    s = impl.sin(angle)
    # multiply on right, use non-standard rotation matrix (-s and s swapped)
    rotation_matrix = impl.reshape(
        impl.stack(_pack_rotation_matrix(c, s, rotation_dim=rotation_dim)),
        (3, 3))

    positions = impl.matmul(positions, rotation_matrix)
    if normals is None:
        return positions, None
    else:
        return positions, impl.matmul(normals, rotation_matrix)


def rotate(positions, normals=None, angle=None, rotation_dim=2):
    """See _rotate. `angle` may also be 'pca-xy'."""
    if angle != 0:
        positions, normals = _rotate(positions,
                                     normals,
                                     angle,
                                     rotation_dim=rotation_dim,
                                     impl=tf)
    if normals is None:
        return positions
    else:
        return positions, normals


def rotate_np(positions, normals=None, angle=None, rotation_dim=2):
    return _rotate(positions,
                   normals,
                   angle,
                   rotation_dim=rotation_dim,
                   impl=np)


def reflect(xyz, dim=0, axis=-1):
    values = tf.unstack(xyz, axis=axis)
    values[dim] *= -1
    return tf.stack(values, axis=axis)


def random_rigid_transform_matrix(stddev=0.02, clip=None, dim=3):
    dim = getattr(dim, 'value', dim)  # TF-COMPAT
    offset = tf.random.normal(shape=(dim, dim), stddev=stddev)
    if clip:
        offset = tf.clip_by_value(offset, -clip, clip)  # pylint: disable=invalid-unary-operand-type
    return tf.eye(dim) + offset


def rotate_by_scheme(positions, normals=None, scheme='random', rotation_dim=2):
    """scheme should be in ("random", "pca-xy", "none")."""
    if scheme == 'none':
        angle = 0
    elif scheme == 'pca-xy':
        from deep_cloud.augment import pca
        angle = pca.get_pca_xy_angle(positions, rotation_dim=rotation_dim)
    elif scheme == 'random':
        angle = tf.random.uniform(shape=(), dtype=positions.dtype) * (2 * np.pi)
    else:
        raise ValueError('Unrecognized scheme "%s"' % scheme)
    return rotate(positions, normals, angle, rotation_dim=rotation_dim)


def random_rigid_transform(points, normals=None, stddev=0.02, clip=None):
    transform = random_rigid_transform_matrix(stddev, clip, points.shape[-1])
    points = tf.matmul(points, transform)
    if normals is None:
        return points
    else:
        raise NotImplementedError('Normal rigid transform not implemented')


def maybe_reflect(positions, axis=-1, dim=0, prob=0.5):
    should_reflect = tf.random.uniform(shape=(), dtype=tf.float32) > prob
    return tf.cond(should_reflect,
                   lambda: reflect(positions, dim=dim, axis=axis),
                   lambda: positions)


def random_scale(positions, stddev):
    scale = tf.random.truncated_normal(shape=(), mean=1.0, stddev=stddev)
    return positions * scale
