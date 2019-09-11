from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
from more_keras.ragged.np_impl import RaggedArray

PACKING_RATIOS = {
    2:
        np.pi * np.sqrt(3) / 6,  # https://en.wikipedia.org/wiki/Circle_packing
    3:
        np.pi /
        (3 * np.sqrt(2)
        )  # https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
}

enclosing_radii_2d = np.array([
    1,
    2,
    1 + 2 / np.sqrt(3),
    1 + np.sqrt(2),
    1 + np.sqrt(2 * (1 + 1. / np.sqrt(5))),
    3,
    3,
    1 + 1. / np.sin(np.pi / 7),
    1 + np.sqrt(2 * (2 + np.sqrt(2))),
    3.813,
    1 + 1. / np.sin(np.pi / 9),
    4.029,
    2 + np.sqrt(5),
    4.328,
    1 + np.sqrt(6 + 2 / np.sqrt(5) + 4 * np.sqrt(1 + 2 / np.sqrt(5))),
    4.615,
    4.792,
    1 + np.sqrt(2) + np.sqrt(6),
    1 + np.sqrt(2) + np.sqrt(6),
    5.122,
])  # https://en.m.wikipedia.org/wiki/Circle_packing_in_a_circle

inner_radii_3d = np.array([
    1,
    1. / 2,
    2 * np.sqrt(3) - 3,
    np.sqrt(6) - 2,
    np.sqrt(2) - 1,
    np.sqrt(2) - 1,
    0.3859,
    0.3780,
    0.3660,
    0.3530,
    (np.sqrt(5) - 3) / 2 + np.sqrt(5 - 2 * np.sqrt(5)),
    (np.sqrt(5) - 3) / 2 + np.sqrt(5 - 2 * np.sqrt(5)),
])  # https://en.m.wikipedia.org/wiki/Sphere_packing_in_a_sphere
enclosing_radii_3d = 1. / inner_radii_3d


def approx_minimum_radius(num_neighbors, num_dims):
    return np.power(num_neighbors / PACKING_RATIOS[num_dims], 1. / num_dims) - 1


def approx_max_neighbors(radius, num_dims):
    return PACKING_RATIOS[num_dims] * (1 + radius)**num_dims


def rejection_sample_with_tree(tree, radius):
    N = tree.n
    consumed = np.zeros((N,), dtype=np.bool)
    out = []
    points = tree.data
    for i in range(N):
        if not consumed[i]:
            out.append(i)
            neighbors = tree.query_ball_point(points[i], radius)
            consumed[neighbors] = True

    return out


def closest_neighbors(tree):
    if isinstance(tree, np.ndarray):
        tree = KDTree(tree)
    distances, indices = tree.query(tree.data, 2)
    del indices
    return np.min(distances[:, 1])


def mean_neighbors(tree, radius):
    if isinstance(tree, np.ndarray):
        tree = KDTree(tree)
    indices = tree.query_ball_tree(tree, radius)
    num_neighbors = [len(i) for i in indices]
    return np.mean(num_neighbors)


def get_experimental_results(radii, r0=0.04, num_dims=3, num_samples=100000):
    points = (np.random.uniform(size=(num_samples, num_dims)) - 0.5) * 4
    indices = rejection_sample_with_tree(KDTree(points), 2 * r0)
    points = points[indices]
    points /= r0
    tree = KDTree(points)

    dist, index = tree.query(np.zeros(num_dims,), 1)
    del dist
    center = points[index]
    indices = tree.query_ball_point(center, np.max(radii))
    neighbors = points[indices]
    dists = np.linalg.norm(neighbors - center, axis=-1)
    n = np.array([np.count_nonzero(dists < r) for r in radii])
    return n


import matplotlib.pyplot as plt


def plot_enclosing_radii(enclosing_radii, num_dims, ax, max_radius=None):
    if max_radius is None:
        max_radius = 10 * np.max(enclosing_radii)
    n = np.arange(1, len(enclosing_radii) + 1)
    ax.plot(enclosing_radii, n)
    r = np.linspace(1, max_radius, 101)
    phi = (1 + np.sqrt(5)) / 2
    experimental = get_experimental_results(r, num_dims=num_dims)
    approx = approx_max_neighbors(r, num_dims)
    ax.plot(r, experimental)
    ax.plot(r, approx)
    ax.plot(r, approx / (phi**(num_dims - 1)))
    ax.legend([
        'optimal', 'experimental', '$p_k(1 + r)^k$',
        '$\\frac{p_k(1 + r)^k}{\\phi^{k-1}}$'
    ])
    ax.set_xlabel('$r$')
    ax.set_ylabel('$n$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('{}-D'.format(num_dims))


fig, (ax0, ax1) = plt.subplots(1, 2)
plot_enclosing_radii(enclosing_radii_2d, 2, ax0)
plot_enclosing_radii(enclosing_radii_3d, 3, ax1)
plt.show()
