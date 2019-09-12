from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from deep_cloud.problems import partnet
from deep_cloud.models.very_dense import core
import trimesh

colors = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
])

tf.compat.v1.enable_eager_execution()

problem = partnet.PartnetProblem(category='chair')
problem.builder.download_and_prepare()
dataset = problem.get_base_dataset(split='train')

for coords, labels in dataset.take(10):
    (all_coords, flat_rel_coords, flat_node_indices,
     row_splits) = core.compute_edges_principled_eager(coords, normals=None)

    print('-----------------')
    print('coords')
    for c in all_coords:
        print(c.shape)

    print('node_indices')
    for i, fni in enumerate(flat_node_indices):
        for j, f in enumerate(fni):
            print(i, j, f.shape)

    print('row_splits')
    for i, rs in enumerate(row_splits):
        for j, r in enumerate(rs):
            print(i, j, r[-1], r.shape, np.mean(np.diff(r)))

    coords = coords.numpy()
    c = colors[labels.numpy()]
    trimesh.PointCloud(coords, colors=c).show()
    # coords[..., 2] = 0
    # trimesh.PointCloud(coords, colors=c).show()
