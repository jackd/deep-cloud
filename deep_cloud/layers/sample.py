from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.ops import sample as _s


def _sample(args, **kwargs):
    return _s.sample(*args, **kwargs)


def sample(unweighted_probs, k):
    if isinstance(k, tf.Tensor):
        args = unweighted_probs, k
        kwargs = {}
    else:
        args = unweighted_probs
        kwargs = dict(k=k)
    return tf.keras.layers.Lambda(_sample, arguments=kwargs)(args)


def _mask(args, **kwargs):
    return _s.mask(*args, **kwargs)


def mask(unweighted_probs, target_mean):
    if isinstance(target_mean, tf.Tensor):
        args = unweighted_probs, target_mean
        kwargs = {}
    else:
        args = unweighted_probs
        kwargs = dict(target_mean=target_mean)
    return tf.keras.layers.Lambda(_mask, arguments=kwargs)(args)
