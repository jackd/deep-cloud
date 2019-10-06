from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

N = 10000
K = 3
P = 16
Q = 32

first = np.random.uniform(size=(K, P, Q)).astype(np.float32)
second = np.random.uniform(size=(N, Q)).astype(np.float32)
third = np.random.uniform(size=(N, K)).astype(np.float32)
args = (first, second, third)

# pattern = 'kpq,nq,nk->np'


def fn(args, einsum):
    return einsum('kpq,nq,nk->np', *args)
    # a, b, c = args
    # tmp = einsum('kpq,nq->nkp', a, b)
    # return tmp
    # return einsum('nkp,nk->np', tmp, c)


def run_tf(args, device='/gpu:0'):
    with tf.device(device):
        args = [tf.constant(arg, dtype=tf.float32) for arg in args]
        value = fn(args, tf.einsum)
        with tf.Session() as sess:
            return sess.run(value)


def run_np(args):
    return fn(args, np.einsum)


expected = run_np(args)
devices = ('/cpu:0', '/gpu:0')
errs = [np.max(np.abs(run_tf(args, device) - expected)) for device in devices]

for device, err in zip(devices, errs):
    print(device, err)
