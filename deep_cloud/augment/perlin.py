"""
Perlin noise generator

https://en.wikipedia.org/wiki/Perlin_noise
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from more_keras.ops import interp


def dot_grid(coords, corner_coords, gradients):
    """
    Args:
        coords: (num_points, num_dims) point coordinates
        corner_coords: (num_points, 2**num_dims, num_dims) int coordinates of corners
        gradients: (nx, ny, ..., num_dims) gradients used in perlin noise

    Returns:
        (num_points, 2**num_dims) float of dot products
    """
    diff = (tf.expand_dims(coords, axis=-2) -
            tf.cast(corner_coords, coords.dtype))
    grads = tf.gather_nd(gradients, corner_coords)
    # num_points, 2**num_dims, num_dims
    return tf.reduce_sum(grads * diff, axis=-1)


def perlin_interp(grid_values, coords):
    """
    Args:
        grid_values: (nx, ny, ..., num_dims) gradients at grid coordinates
        coords: (num_points, num_dims)
            coordinates in range ([0, nx], [0, ny], ...)

    Returns:
        (num_points,) perlin-interpolated grid values.
    """
    corner_coords, factors = interp.get_linear_coords_and_factors(coords)
    values = dot_grid(coords, corner_coords, grid_values)
    return tf.reduce_sum(values * factors, axis=-1)


def scale_to_grid(coords, grid_shape, eps=1e-5):
    # eps ensures we don't end up interpolating on the upper boundary.
    # causes issues with calculating corners according to floor and floor + 1
    shift = tf.reduce_min(coords, axis=0)
    coords = coords - shift
    scale = tf.reduce_max(coords) / (tf.cast(grid_shape, tf.float32) - 1)
    scale = scale + eps
    coords = coords / scale

    def rescale(c):
        return c * scale + shift

    return coords, rescale


def add_perlin_noise(coords,
                     grid_shape=(4, 4, 4),
                     stddev=0.25,
                     eps=1e-5,
                     rescale=True):
    if stddev == 0 or stddev is None:
        return coords
    if isinstance(grid_shape, float):
        if int(np.round(grid_shape)) - grid_shape < 1e-4:
            grid_shape = int(np.round(grid_shape))
        else:
            raise ValueError(
                'grid_shape must be integer, got {}'.format(grid_shape))
    if isinstance(grid_shape, int):
        num_dims = coords.shape[-1]
        num_dims = getattr(num_dims, 'value', num_dims)  # TF-COMPAT
        grid_shape = (grid_shape,) * num_dims
    else:
        num_dims = len(grid_shape)
    shifts = []
    if rescale:
        coords, rescale_fn = scale_to_grid(coords, grid_shape, eps=eps)

    # TODO: vectorize?
    for _ in range(num_dims):
        gradients = tf.random.normal(shape=grid_shape + (num_dims,),
                                     stddev=stddev)
        shifts.append(perlin_interp(gradients, coords))
    coords = coords + tf.stack(shifts, axis=-1)
    if rescale:
        coords = rescale_fn(coords)
    return coords
