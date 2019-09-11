from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow_datasets as tfds
from deep_cloud.problems.builders import pointnet_builder
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
import matplotlib.pyplot as plt

builder = pointnet_builder(pointnet_version=2)

PACKING_RATIOS = {
    2:
        np.pi * np.sqrt(3) / 6,  # https://en.wikipedia.org/wiki/Circle_packing
    3:
        np.pi /
        (3 * np.sqrt(2)
        )  # https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
}


def approx_max_neighbors(radius, num_dims):
    return PACKING_RATIOS[num_dims] * (1 + radius)**num_dims


def closest_neighbors(tree):
    if isinstance(tree, np.ndarray):
        tree = KDTree(tree)
    distances, indices = tree.query(tree.data, 2)
    del indices
    return np.min(distances[:, 1])


def analyse(points, radii):
    points = points[:1024]
    closest = closest_neighbors(points)
    points *= 2 / closest
    tree = KDTree(points)
    indices = tree.query_ball_point(points, np.max(radii))
    dists = [
        np.linalg.norm(points[ind] - points[i], axis=-1)
        for i, ind in enumerate(indices)
    ]
    del indices
    means = []
    maxs = []
    for r in radii:
        counts = np.array([np.count_nonzero(d < r) for d in dists])
        means.append(np.mean(counts))
        maxs.append(np.max(counts))
    ax = plt.gca()
    ax.plot(radii, means)
    ax.plot(radii, maxs)
    ax.plot(radii, approx_max_neighbors(radii, 2))
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend(['mean', 'max', '2D approx'])
    ax.set_xlabel('$r$')
    ax.set_ylabel('$n$')
    plt.show()


radii = np.linspace(2, 10, 21)
for features, label in tfds.as_numpy(
        builder.as_dataset(split='train', as_supervised=True)):
    analyse(features['positions'], radii)
