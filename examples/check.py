from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow_datasets as tfds
from deep_cloud.problems.builders import pointnet_builder
from deep_cloud.problems.modelnet import ModelnetProblem
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
import tqdm
import trimesh

builder = pointnet_builder(2, uniform_density=False)
split = 'train'
problem = ModelnetProblem(builder, positions_only=False)
# dataset = builder.as_dataset(split=split, as_supervised=True)
dataset = problem.get_base_dataset(split=split)

# all_labels = []
# for data, label in tfds.as_numpy(dataset):
#     all_labels.append(label)

# labels = np.array(all_labels)
# minval = np.min(labels)
# maxval = np.max(labels)
# print(minval, maxval)
# import matplotlib.pyplot as plt
# plt.hist(labels, normed=True, bins=40)
# plt.show()
k = 10

means = []
class_names = builder.info.features['label'].names


def get_counts(tree, radius):
    counts = np.zeros((positions.shape[0],), dtype=np.int64)
    for i, j in tree.query_pairs(radius):
        counts[i] += 1
        counts[j] += 1
    return counts


for cloud, label in tqdm.tqdm(tfds.as_numpy(dataset),
                              total=problem.examples_per_epoch(split)):
    positions = cloud['positions']
    print(positions.shape)
    print(np.max(np.sum(positions**2, axis=-1), axis=0))

    tree = cKDTree(positions)
    counts = get_counts(tree, 0.1)
    mean = np.mean(counts)
    means.append(mean)
    if mean > 15:
        print('{}: {}, {}'.format(label, class_names[label], mean))
        # c2 = np.mean(get_counts(tree, 0.2))
        # c4 = np.mean(get_counts(tree, 0.4))
        # print(c2, c4)
        trimesh.PointCloud(positions).show()

import matplotlib.pyplot as plt
plt.hist(means)
plt.show()
