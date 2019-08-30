from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow as tf
# from deep_cloud.ops import sample
from deep_cloud.ops.sample import rejection_sample
from deep_cloud.ops.ordering import iterative_farthest_point_order
from deep_cloud.ops import query


@tf.function
def get_indices(x, r):
    pairs = query.query_pairs(x, r)
    indices = rejection_sample(pairs)
    return indices


def get_cloud(num_points=1024, num_dims=3, ordered=False):
    if ordered:
        points = tf.random.uniform((2 * num_points, num_dims), dtype=tf.float32)
        order = iterative_farthest_point_order(points, num_points)
        points = tf.gather(points, order)
    else:
        points = tf.random.uniform((num_points, num_dims), dtype=tf.float32)
    return points


def profile(num_points=1024,
            num_dims=3,
            radius=0.1,
            ordered=False,
            num_warmup=2,
            num_runs=500):
    from time import time
    from tqdm import tqdm
    assert (tf.executing_eagerly())

    def g():
        with tf.device('/cpu:0'):
            points = get_cloud(num_points, num_dims, ordered)
            pairs = query.query_pairs(points, radius)
            t = time()
            rejection_sample(pairs)
            # print(tf.size(out).numpy())
            dt = time() - t
        return dt

    for _ in tqdm(range(num_warmup), desc='warming up'):
        g()

    t = 0
    for _ in tqdm(range(num_runs), desc='Profiling...'):
        t += g()
    logging.info('Completed {} runs in {:.2f} s: {:.2f} runs / sec'.format(
        num_runs, t, num_runs / t))


def get_multi_plot_data(radii=None,
                        num_points=1024,
                        num_dims=3,
                        ordered=False,
                        num_runs=10):
    from tqdm import tqdm
    assert (tf.executing_eagerly())
    with tf.device('/cpu:0'):
        logging.info('Creating points...')
        points = [
            get_cloud(num_points, num_dims, ordered) for _ in range(num_runs)
        ]

        all_sizes = []
        for radius in tqdm(radii, 'computing indices'):
            sizes = [tf.size(get_indices(x, radius)) for x in points]
            all_sizes.append(sizes)
        return all_sizes


def split_plot_data(all_sizes, num_points, lower_frac=0.1, upper_frac=0.9):
    num_runs = len(all_sizes[0])
    lower = int(lower_frac * num_runs)
    upper = int(upper_frac * num_runs)
    mid = int(0.5 * num_runs)

    all_sizes = [tf.sort(s).numpy() / num_points for s in all_sizes]
    means = [np.mean(s) for s in all_sizes]
    mins = [s[0] for s in all_sizes]
    lowers = [s[lower] for s in all_sizes]
    medians = [s[mid] for s in all_sizes]
    uppers = [s[upper] for s in all_sizes]
    maxs = [s[-1] for s in all_sizes]

    return means, medians, mins, maxs, lowers, uppers


def plot_sizes(radii=None,
               num_points=1024,
               num_dims=3,
               num_runs=2,
               lower_frac=0.1,
               upper_frac=0.9):
    import numpy as np
    import matplotlib.pyplot as plt
    assert (tf.executing_eagerly())
    if radii is None:
        radii = np.linspace(0., 0.2, 51)[1:]
        # radii = np.logspace(0.05, 0.2, 11)

    ax = plt.gca()
    for ordered, color in ((False, 'b'), (True, 'r')):
        all_sizes = get_multi_plot_data(radii, num_points, num_dims, ordered,
                                        num_runs)

        means, medians, mins, maxs, lowers, uppers = split_plot_data(
            all_sizes, num_points, lower_frac, upper_frac)

        logging.info('Extents:\n{}'.format(np.stack([mins, maxs], axis=-1)))

        ax.fill_between(radii, mins, maxs, color=color, alpha=0.1)
        ax.fill_between(radii, lowers, uppers, color=color, alpha=0.2)
        ax.plot(radii, means, color=color)
        ax.plot(radii, medians, color=color, linestyle='dashed')
    # ax.set_xscale('log')
    plt.show()


def vis_simple():
    r = 0.1
    N = 1024
    x = tf.random.uniform((N, 2))
    indices = get_indices(x, r)
    sampled = tf.gather(x, indices)

    x = x.numpy()
    sampled = sampled.numpy()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection

    ax = plt.gca()
    collection = PatchCollection([Circle(xy, r, fc=None) for xy in sampled],
                                 alpha=0.2,
                                 ec=(1, 0, 0))
    ax.add_collection(collection)
    ax.scatter(*(x.T), s=0.4)
    ax.scatter(*(sampled.T), marker='x')
    ax.set_aspect('equal', 'box')
    print(len(indices), N)
    plt.show()


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    logging.set_verbosity(logging.INFO)
    # vis_simple()
    profile()
    # plot_sizes()
