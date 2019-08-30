"""
Jax implementation of iterative farthest point sampling.

Significantly slower than python version... but maybe I'm doing something dumb?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.ops
import jax.numpy as np
from numpy import random


@jax.jit
def iterative_farthest_point_ordering(points, num_samples=None, first=None):
    if num_samples is None:
        num_samples = points.shape[0]
    index = int(random.uniform() * num_samples) if first is None else first
    # out = [index]
    out = np.empty(shape=(num_samples,), dtype=np.int64)
    out = jax.ops.index_update(out, 0, index)

    dist2 = np.sum((points - points[index])**2, axis=-1)
    for i in range(1, num_samples):
        index = np.argmax(dist2)
        next_dist2 = np.sum((points - points[index])**2, axis=-1)
        dist2 = np.minimum(dist2, next_dist2)
        # dist2 = jax.ops.index_min(dist2, jax.ops.index[:], next_dist2)
        # out.append(index)
        out = jax.ops.index_update(out, i, index)
    # out.append(np.argmax(dist2))
    jax.ops.index_update(out, -1, np.argmax(dist2))
    return out
    # return np.array(out, dtype=np.int32)


def partial_reorder(indices, points, *args):
    total = points.shape[0]
    mask = np.ones(shape=(total,), dtype=bool)
    mask[indices] = False
    out = tuple(
        np.concatenate((inp[indices], inp[mask]), axis=0)
        for inp in ((points,) + args))
    return out[0] if len(args) == 0 else out


@jax.jit
def fps_reorder(points, *args, sample_frac=0.5):
    indices = iterative_farthest_point_ordering(points,
                                                int(points.shape[0] *
                                                    sample_frac),
                                                first=0)
    return partial_reorder(indices, points, *args)


if __name__ == '__main__':
    from absl import logging
    from time import time
    from tqdm import tqdm
    warmup = 10
    runs = 100

    logging.set_verbosity(logging.INFO)
    logging.info('started')

    def do_run():
        t = time()
        points = random.uniform(size=(1024, 3)).astype(np.float32)
        iterative_farthest_point_ordering(points)
        dt = time() - t
        print('done')
        return dt

    logging.info('Warming up...')
    for _ in tqdm(range(warmup)):
        do_run()
    dt = 0
    logging.info('Doing runs...')
    for _ in tqdm(range(runs)):
        dt += do_run()
    logging.info('Completed {} runs in {:2f} s, {:2f} runs / sec'.format(
        runs, dt, runs / dt))
