"""Free form deformation with bernstein decomposition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import comb
import tensorflow as tf


def _mesh(ranges):
    return tf.stack(tf.meshgrid(*ranges, indexing='ij'), axis=-1)


def bernstein_poly(n, v, stu):
    coeff = comb(n, v)
    weights = coeff * ((1 - stu)**(n - v)) * (stu**v)
    return weights


def trivariate_bernstein(stu, lattice):
    if len(lattice.shape) != 4 or lattice.shape[3] != 3:
        raise ValueError('lattice must have shape (L, M, N, 3)')
    l, m, n = (d - 1 for d in lattice.shape[:3])
    lmn = np.array([l, m, n], dtype=np.int32)
    v = np.stack(np.meshgrid(np.arange(l, dtype=np.int32),
                             np.arange(m, dtype=np.int32),
                             np.arange(n, dtype=np.int32),
                             indexing='ij'),
                 axis=-1)
    stu = np.reshape(stu, (-1, 1, 1, 1, 3))
    weights = bernstein_poly(n=lmn, v=v, stu=stu)
    weights = tf.reduce_prod(weights, axis=-1, keepdims=True)
    return tf.reduce_sum(weights * lattice, axis=(1, 2, 3))


def xyz_to_stu(xyz, origin, stu_axes):
    if stu_axes.shape == (3,):
        stu_axes = tf.linalg.diag(stu_axes)
        # raise ValueError(
        #     'stu_axes should have shape (3,), got %s' % str(stu_axes.shape))
    # s, t, u = np.diag(stu_axes)
    assert (stu_axes.shape == (3, 3))
    s, t, u = tf.unstack(stu_axes, axis=0)
    tu = tf.linalg.cross(t, u)
    su = tf.linalg.cross(s, u)
    st = tf.linalg.cross(s, t)

    diff = xyz - origin

    # TODO: vectorize? np.dot(diff, [tu, su, st]) / ...
    stu = tf.stack([
        tf.reduce_sum(diff * tu, axis=-1) / tf.reduce_sum(s * tu, axis=-1),
        tf.reduce_sum(diff * su, axis=-1) / tf.reduce_sum(t * su, axis=-1),
        tf.reduce_sum(diff * st, axis=-1) / tf.reduce_sum(u * st, axis=-1)
    ],
                   axis=-1)
    return stu


def stu_to_xyz(stu_points, stu_origin, stu_axes):
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points * stu_axes


def get_stu_control_points(dims):
    mesh = _mesh((tf.linspace(0., 1., d) for d in dims))
    return tf.cast(tf.reshape(mesh, (-1, 3)), tf.float32)


def get_control_points(dims, stu_origin, stu_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, stu_origin, stu_axes)
    return xyz_points


def get_stu_deformation_matrix(stu, dims):
    v = np.stack(np.meshgrid(*(np.arange(0, d, dtype=np.int32) for d in dims),
                             indexing='ij'),
                 axis=-1)
    v = np.reshape(v, (-1, 3))

    weights = bernstein_poly(n=np.array(dims, dtype=np.int32),
                             v=v,
                             stu=tf.expand_dims(stu, axis=-2))

    b = tf.reduce_prod(weights, axis=-1)
    return b


def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(stu, dims)


def get_ffd(xyz, dims, stu_origin=None, stu_axes=None):
    """
    Get free form deformation using Bernstein basis.

    Args:
        xyz: num_points, 3
        others: ignore?

    Returns:
        b: [num_points, prod(d for d in dims)] decomposition
        p: [prod(d + 1 for d in dims), 3] float.

    tf.matmul(b, p) == xyz
    """
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(xyz)
    b = get_deformation_matrix(xyz, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(xyz):
    minimum, maximum = tf.reduce_min(xyz, axis=0), tf.reduce_max(xyz, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes


def random_ffd(xyz, grid_shape=(4, 4, 4), stddev=0.2):
    if isinstance(grid_shape, int):
        num_dims = xyz.shape.as_list()[-1]  # TF-COMPAT
        grid_shape = (grid_shape,) * num_dims
    b, p = get_ffd(xyz, grid_shape)
    dp = tf.random.normal(shape=p.shape, stddev=stddev)
    return tf.matmul(b, p + dp)
