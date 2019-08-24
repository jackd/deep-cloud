from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
import numpy as np
# from shape_tfds.shape.modelnet import Modelnet40
# from shape_tfds.shape.modelnet import pointnet as pn
from shape_tfds.shape.modelnet import pointnet2 as pn2
# from shape_tfds.shape.modelnet import CloudConfig
from deep_cloud.problems.builders import pointnet_builder

num_points = 1024
# base = Modelnet40(config=CloudConfig(2048))
# builder = pn.Pointnet()
builder = pn2.Pointnet2(config=pn2.CONFIG40)

# for example in base.as_dataset(as_supervised=True)

for cloud, label in tfds.as_numpy(
        builder.as_dataset(as_supervised=True, split='test')):
    positions = cloud['positions']
    break

p0 = np.array(positions[:num_points])
np.random.shuffle(positions)
p1 = positions[:num_points]


def num_neighbors_within(points, radius=0.1):
    tree = cKDTree(points)
    counts = np.zeros((points.shape[0],), dtype=np.int64)
    for i, j in tree.query_pairs(radius):
        counts[i] += 1
        counts[j] += 1
    return counts


def nearest_neighbor_dist(points, k=1):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k + 1)
    return dists[..., -1]


metric = num_neighbors_within

m0 = num_neighbors_within(p0), num_neighbors_within(p1)
m1 = nearest_neighbor_dist(p0), nearest_neighbor_dist(p1)

import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(1, 2)
colors = 'red', 'blue'
labels = 'first', 'random'
kwargs = dict(color=colors, label=labels, normed=True)
ax0.hist(np.stack(m0, axis=-1), cumulative=False, **kwargs)
ax0.set_title('neighbors within radius')
ax0.legend(prop={'size': 10})
ax1.hist(np.stack(m1, axis=-1), cumulative=True, **kwargs)
ax1.set_title('nearest neighbor dist')
ax1.legend(prop={'size': 10})

plt.show()
