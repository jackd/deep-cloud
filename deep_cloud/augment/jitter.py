from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def jitter_positions(positions, stddev=0.02, clip=None):
    """
    Randomly jitter points independantly by normally distributed noise.

    Args:
        positions: float array, any shape
        stddev: standard deviation of jitter
        clip: if not None, jittering is clipped to this
    """
    if stddev == 0 or stddev is None:
        return positions
    jitter = tf.random.normal(shape=tf.shape(positions), stddev=stddev)
    if clip is not None:
        jitter = tf.clip_by_norm(jitter, clip, axes=[-1])
    # scale by max norm
    max_val = tf.reduce_max(tf.linalg.norm(positions, axis=-1))
    jitter = jitter * max_val
    return positions + jitter


def jitter_normals(normals, stddev=0.02, clip=None, some_normals_invalid=False):
    if stddev == 0 or stddev is None:
        return normals
    normals = jitter_positions(normals, stddev, clip)
    norms = tf.linalg.norm(normals, axis=-1, keepdims=True)
    if some_normals_invalid:
        # some normals might be invalid, in which case they'll initially be 0.
        thresh = 0.1 if clip is None else 1 - clip
        return tf.where(tf.less(norms, thresh), tf.zeros_like(normals),
                        normals / norms)
    else:
        return normals / norms
