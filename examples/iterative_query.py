import numpy as np
import tensorflow_datasets as tfds
from deep_cloud.problems.builders import pointnet_builder
from deep_cloud.ops.np_utils import sample
from deep_cloud.ops.np_utils.tree_utils.pykd import recursive_query_ball_point
from deep_cloud.ops.np_utils.tree_utils.core import closest_neighbors
from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module
from scipy import spatial
import matplotlib.pyplot as plt
from timeit import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm


def profile(points, radius=4, leafsize=4):
    points = points[:1024]
    closest = closest_neighbors(KDTree(points))
    points *= 2 / closest
    # points += np.random.normal(scale=1e-2 * (2 / closest))
    tree = KDTree(points, leafsize=leafsize)
    k = np.arange(2, 40)
    times = []
    for k0 in tqdm(k):
        times.append(
            timeit(lambda: recursive_query_ball_point(tree, points, radius, k0),
                   number=20))
    _, mask = recursive_query_ball_point(tree, points, radius, 10)
    spt = spatial.cKDTree(points, leafsize=leafsize)  # pylint: disable=no-member
    t = timeit(lambda: spt.query_ball_tree(spt, radius), number=20)
    times = np.array(times) / t

    neighbors = np.count_nonzero(mask, axis=1)
    ax = plt.gca()
    ax.plot(k, times)
    # ax.plot([k[0], k[-1]], [t, t], linestyle='dashed')
    ax.plot([k[0], k[-1]], [0, 0], linestyle='dashed', color='k')
    ax.plot([k[0], k[-1]], [1, 1], linestyle='dashed', color='k')
    ax.set_xlabel('k')
    ax.set_ylabel('t')
    ax.set_title('mean = {}, max = {}'.format(np.mean(neighbors),
                                              mask.shape[1]))
    print(times[14] / np.min(times))
    # ax.set_yscale('log')
    plt.show()


def profile_multi(radius=4, leafsize=4):
    builder = pointnet_builder(pointnet_version=2)
    for features, _ in tfds.as_numpy(
            builder.as_dataset(split='train', as_supervised=True)):
        profile(features['positions'], radius=radius, leafsize=leafsize)


def get_ks(points, radii, leafsize=4):
    r0 = 0.04
    indices = sample.rejection_sample_lazy(points, r0)
    points = points[indices]
    points *= (2 / r0)

    # points = points[:1024]
    # closest = closest_neighbors(KDTree(points))
    # points *= 2 / closest
    # points += np.random.normal(scale=1e-2 * (2 / closest))
    tree = KDTree(points, leafsize=leafsize)
    indices, masks, dists = recursive_query_ball_point(tree, points,
                                                       np.max(radii), 32)
    del indices, masks
    return np.array([np.count_nonzero(dists < r, axis=1) for r in radii],
                    dtype=np.int64)


def plot_ks():
    builder = pointnet_builder(pointnet_version=2)

    means = []
    maxs = []
    radii = np.linspace(2, 10, 51)
    total = 100
    for features, _ in tqdm(tfds.as_numpy(
            builder.as_dataset(split='train', as_supervised=True).take(total)),
                            total=total):
        ks = get_ks(features['positions'], radii)
        means.append(np.mean(ks, axis=1))
        maxs.append(np.max(ks, axis=-1))

    ax = plt.gca()
    ax.plot(radii, np.mean(means, axis=0))
    ax.plot(radii, np.mean(maxs, axis=0))
    ax.legend(['mean', 'max'])
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


def full_run(points, leafsize=4):
    # r0 = 0.04
    # indices = sample.rejection_sample_lazy(points, r0)
    # points = points[indices]
    # points *= (2 / r0)

    points = points[:1024]
    closest = closest_neighbors(KDTree(points))
    points *= 2 / closest

    print('************')
    print(points.shape[0])
    tree = KDTree(points, leafsize=leafsize)
    _, masks, _ = recursive_query_ball_point(tree, points, 4, k0=16)
    ks = np.count_nonzero(masks, axis=1)
    print('{}, {:.2f}, {}, {}'.format(points.shape[0], np.mean(ks), np.max(ks),
                                      np.sum(ks)))

    indices = sample.rejection_sample_lazy(points, 4.)
    points = points[indices]
    tree = KDTree(points, leafsize=leafsize)

    _, masks, _ = recursive_query_ball_point(tree, points, 8, k0=16)
    ks = np.count_nonzero(masks, axis=1)
    print('{}, {:.2f}, {}, {}'.format(points.shape[0], np.mean(ks), np.max(ks),
                                      np.sum(ks)))

    indices = sample.rejection_sample_lazy(points, 8.)
    points = points[indices]
    tree = KDTree(points, leafsize=leafsize)

    _, masks, _ = recursive_query_ball_point(tree, points, 16, k0=16)
    ks = np.count_nonzero(masks, axis=1)
    print('{}, {:.2f}, {}, {}'.format(points.shape[0], np.mean(ks), np.max(ks),
                                      np.sum(ks)))

    indices = sample.rejection_sample_lazy(points, 16.)
    points = points[indices]
    tree = KDTree(points, leafsize=leafsize)

    _, masks, _ = recursive_query_ball_point(tree, points, 32, k0=16)
    ks = np.count_nonzero(masks, axis=1)
    print('{}, {:.2f}, {}, {}'.format(points.shape[0], np.mean(ks), np.max(ks),
                                      np.sum(ks)))


def multi_full_run(num_runs=10):
    builder = pointnet_builder(pointnet_version=2)
    for features, _ in tfds.as_numpy(
            builder.as_dataset(split='train',
                               as_supervised=True).take(num_runs)):
        full_run(features['positions'])


if __name__ == '__main__':
    # profile_multi()
    # plot_ks()
    multi_full_run()
