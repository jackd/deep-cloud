from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def sample(unweighted_probs, k, dtype=tf.int64):
    logits = tf.math.log(unweighted_probs)
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, k)
    return tf.cast(indices, dtype)


def mask(unweighted_probs, target_mean):
    probs = unweighted_probs * (target_mean / tf.reduce_mean(unweighted_probs))
    return tf.random.uniform(shape=tf.shape(probs)) < probs
