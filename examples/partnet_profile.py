from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_cloud.ops.np_utils import cloud as np_cloud
import numpy as np
from time import time
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import spatial
from deep_cloud.models.very_dense import utils
import tensorflow_datasets as tfds

DEFAULT_TREE = pykd.KDTree
# DEFAULT_TREE = spatial.KDTree


def exponential_radii(depth=4, r0=1., expansion_rate=2):
    return r0 * expansion_rate**np.arange(depth)


def rejection_sample_lazy(tree, points, radius, k0):
    N = points.shape[0]
    out = []
    consumed = np.zeros((N,), dtype=np.bool)
    for i in range(N):
        if not consumed[i]:
            out.append(i)
            indices = tree.query_ball_point(np.expand_dims(points[i], 0),
                                            radius,
                                            approx_neighbors=k0)
            indices = indices[0]
            consumed[indices] = True
    return out


def rejection_sample_active(tree, points, radius, k0):
    N = points.shape[0]
    out = []
    consumed = np.zeros((N,), dtype=np.bool)
    indices = tree.query_ball_point(points, radius, approx_neighbors=k0)
    for i in range(N):
        if not consumed[i]:
            consumed[indices[i]] = True
            out.append(i)
    return out


def compute_edges_principled_eager(coords,
                                   normals,
                                   depth=4,
                                   k0=16,
                                   tree_impl=DEFAULT_TREE):

    timer.restart()
    coords = coords.numpy() if hasattr(coords, 'numpy') else coords
    normals = normals.numpy() if hasattr(normals, 'numpy') else normals

    tree = tree_impl(coords)
    dists, indices = tree.query(tree.data, 2, return_distance=True)
    # closest = np.min(dists[:, 1])
    scale = np.mean(dists[:, 1])
    assert (scale > 0)
    coords *= (2 / scale)
    timer.tic('rescale')

    # coords is now a packing of barely-intersecting spheres of radius 1.
    all_coords = [coords]
    if normals is not None:
        all_normals = [normals]
    tree = tree_impl(coords)
    trees = [tree]

    # base_coords = coords
    # base_tree = tree
    # base_normals = normals

    # perform sampling, build trees
    radii = 4 * np.power(2, np.arange(depth))
    ## Rejection sample on original cloud
    for i, radius in enumerate(radii[:-1]):
        # indices = rejection_sample_active(base_tree,
        #                                   base_coords,
        #                                   radius,
        #                                   k0=k0 * 4**i)
        # coords = base_coords[indices]
        # if normals is not None:
        #     all_normals.append(base_normals[indices])

        indices = rejection_sample_active(tree, coords, radius, k0=k0 * 4**i)
        coords = coords[indices]
        if normals is not None:
            all_normals.append(normals[indices])

        tree = tree_impl(coords)
        all_coords.append(coords)

        trees.append(tree)

    timer.tic('trees')

    # compute edges
    flat_node_indices = utils.lower_triangular(depth)
    flat_rel_coords = utils.lower_triangular(depth)
    row_splits = utils.lower_triangular(depth)

    for i in range(depth):
        for j in range(i + 1):
            indices = trees[i].query_ball_point(all_coords[j],
                                                radii[i],
                                                approx_neighbors=k0)
            # indices = trees[j].query_ball_point(all_coords[i],
            #                                     radii[i],
            #                                     approx_neighbors=k0 *
            #                                     4**(i - j))
            flat_node_indices[i][j] = fni = indices.flat_values.astype(np.int64)
            print(trees[i].data.shape[0], all_coords[j].shape[0], len(indices))
            row_splits[i][j] = indices.row_splits.astype(np.int64)

            # # compute flat_rel_coords
            # # this could be done outside the py_function, but it uses np.repeat
            # # which is faster than tf.repeat on cpu.
            flat_rel_coords[i][j] = np_cloud.get_relative_coords(
                all_coords[i],
                all_coords[j],
                fni,
                row_lengths=indices.row_lengths)
        timer.tic('edges{}'.format(i))
    print('---')
    timer.report()

    if normals is None:
        return (all_coords, flat_rel_coords, flat_node_indices, row_splits)
    else:
        return (all_coords, all_normals, flat_rel_coords, flat_node_indices,
                row_splits)


class Timer(object):

    def restart(self):
        self.t0 = time()
        self.times = []
        self.descs = []

    def tic(self, desc=None):
        self.times.append(time() - self.t0)
        self.descs.append(desc)

    def report(self):
        times = np.diff(np.concatenate([[0], self.times]))
        for t, d in zip(times, self.descs):
            print('{:.4f}: {}'.format(t, d))
        print('total: {}'.format(self.times[-1]))


from deep_cloud.problems import partnet
from deep_cloud.problems import modelnet
from deep_cloud.problems.builders import pointnet_builder
timer = Timer()
times = []

problem = partnet.PartnetProblem()
# problem = modelnet.ModelnetProblem(builder=pointnet_builder(2))
# problem.builder.download_and_prepare()
dataset = problem.get_base_dataset(split='validation')

num_runs = 10
depth = 4

for args in tfds.as_numpy(dataset.take(num_runs)):
    if len(args) == 2:
        coords, labels = args
    else:
        coords, labels, weights = args
    if isinstance(coords, dict):
        coords = coords['positions']
    out = compute_edges_principled_eager(coords, None, depth=depth)
    times.append(timer.times)

times = np.mean(np.array(times), axis=0)
timer.times = times
print('***')
print('Mean over {} runs'.format(num_runs))
timer.report()
row_splits = out[-1]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(depth, depth)
for i, rs in enumerate(row_splits):
    for j, r in enumerate(rs):
        neigh = np.diff(r)
        print(i, j, np.mean(neigh), len(r), r[-1])
        ax[i][j].hist(neigh)

plt.show()
