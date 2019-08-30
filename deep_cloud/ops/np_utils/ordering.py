"""
Numpy implementation of iterative farthest point sampling.

Note that while CPU bound (i.e. during preprocessing) this is ~3x faster
than the tensorflow implementation - though maybe it's using multiple cores
and tensorflow is trying to do it with just 1?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def iterative_farthest_point_ordering(points, num_samples=None, first=None):
    if num_samples is None:
        num_samples = points.shape[0]
    index = int(np.random.uniform() * num_samples) if first is None else first
    out = np.empty(shape=(num_samples,), dtype=np.int64)
    out[0] = index
    dist2 = np.sum((points - points[index])**2, axis=-1)
    for i in range(1, num_samples):
        index = np.argmax(dist2)
        next_dist2 = np.sum((points - points[index])**2, axis=-1)
        dist2[:] = np.minimum(dist2, next_dist2)
        out[i] = index
    out[-1] = np.argmax(dist2)
    return out


def partial_reorder(indices, points, *args):
    total = points.shape[0]
    mask = np.ones(shape=(total,), dtype=np.bool)
    mask[indices] = False
    out = tuple(
        np.concatenate((inp[indices], inp[mask]), axis=0)
        for inp in ((points,) + args))
    return out[0] if len(args) == 0 else out


def fps_reorder(points, *args, sample_frac=0.5):
    indices = iterative_farthest_point_ordering(
        points, int(points.shape[0] * sample_frac))
    return partial_reorder(indices, points, *args)


if __name__ == '__main__':
    from time import time
    from tqdm import tqdm
    warmup = 10
    runs = 100

    def do_run():
        t = time()
        points = np.random.uniform(size=(1024, 3)).astype(np.float32)
        iterative_farthest_point_ordering(points)
        dt = time() - t
        return dt

    for _ in range(warmup):
        do_run()
    dt = 0
    for _ in tqdm(range(runs)):
        dt += do_run()
    print('Completed {} runs in {:2f} s, {:2f} runs / sec'.format(
        runs, dt, runs / dt))
