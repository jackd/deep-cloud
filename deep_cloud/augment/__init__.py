from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import tensorflow as tf
from deep_cloud.augment.ffd import random_ffd
from deep_cloud.augment.jitter import jitter_positions
from deep_cloud.augment.jitter import jitter_normals
from deep_cloud.augment.perlin import add_perlin_noise
from deep_cloud.augment.rigid import random_rigid_transform
from deep_cloud.augment.rigid import random_scale
from deep_cloud.augment.rigid import rotate_by_scheme
from deep_cloud.augment.rigid import maybe_reflect
from deep_cloud.augment.rigid import random_rotation


@gin.configurable(blacklist=['inputs', 'labels'])
def augment_cloud(
        inputs,
        labels,
        weights=None,
        jitter_stddev=None,
        # jitter_normals=None,
        jitter_clip=None,
        scale_stddev=None,
        uniform_scale_range=None,
        rigid_transform_stddev=None,
        maybe_reflect_x=False,
        perlin_grid_shape=None,
        perlin_stddev=None,
        rotate_scheme='none',
        angle_stddev=None,
        angle_clip=None,
        rotation_dim=2,
):
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = inputs['normals']
        positions_only = False
    else:
        positions = inputs
        positions_only = True
        normals = None
    if jitter_stddev is not None:
        positions = jitter_positions(positions,
                                     stddev=jitter_stddev,
                                     clip=jitter_clip)

    if scale_stddev is not None:
        positions = random_scale(positions, stddev=scale_stddev)

    if uniform_scale_range is not None:
        positions = random_scale(positions, uniform_range=uniform_scale_range)

    if rigid_transform_stddev is not None:
        positions, normals = random_rigid_transform(
            positions, normals, stddev=rigid_transform_stddev)

    if maybe_reflect_x:
        positions, normals = maybe_reflect(positions, normals)

    if perlin_grid_shape is not None:
        positions = add_perlin_noise(positions,
                                     perlin_grid_shape,
                                     stddev=perlin_stddev)

    if rotate_scheme:
        positions, normals = rotate_by_scheme(positions,
                                              normals,
                                              scheme=rotate_scheme,
                                              rotation_dim=rotation_dim)

    if angle_stddev:
        positions, normals = random_rotation(positions,
                                             normals,
                                             angle_stddev=angle_stddev,
                                             angle_clip=angle_clip)

    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)

    if weights is None:
        return inputs, labels
    else:
        return inputs, labels, weights
