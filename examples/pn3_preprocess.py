from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.problems.modelnet import ModelnetProblem
from deep_cloud.problems.builders import pointnet_builder
from deep_cloud.models.pointnet3 import multi_scale_group
from deep_cloud.models.pointnet3 import pre_batch_map, post_batch_map
from deep_cloud.ops.np_utils import tree_utils
from deep_cloud.augment import augment_cloud
import functools
from tqdm import tqdm
from time import time

tf.compat.v1.enable_eager_execution()
N = 500
batch_size = 16
problem = ModelnetProblem(builder=pointnet_builder(2), positions_only=False)

map_fn = functools.partial(augment_cloud,
                           angle_stddev=0.06,
                           angle_clip=0.18,
                           uniform_scale_range=(0.8, 1.25),
                           rotate_scheme='random',
                           jitter_stddev=1e-2,
                           jitter_clip=5e-2)

# map2 = pre_batch_map
map2 = functools.partial(pre_batch_map,
                         radii_lists=((0.1,), (0.2,)),
                         max_neighbors_lists=((16,), (32,)))

with tf.device('/cpu:0'):
    dataset = problem.get_base_dataset('train').map(map_fn, -1).map(map2, -1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(post_batch_map, -1)
    dataset = dataset.prefetch(-1)

    t = time()
    for features, labels in tqdm(dataset.take(N)):
        pass

# dataset = problem.get_base_dataset('train').map(map_fn)

# radii_lists = ((0.1, 0.2, 0.4), (0.2, 0.4, 0.8))
# max_neighbors_lists = ((16, 32, 128), (32, 64, 128))

# t = time()
# for features, labels in tqdm(dataset.take(N), total=N):
#     coords = features['positions']
#     all_coords = [coords, coords[:512], coords[:128]]
#     trees = [tree_utils.KDTree(c) for c in all_coords]
#     for i, (radii, limits) in enumerate(zip(
#           radii_lists, max_neighbors_lists)):
#         multi_scale_group(trees[i], trees[i + 1], radii, limits)

dt = time() - t
print('Finished {} runs in {:.2f} sec, {:.2f} iterations per sec'.format(
    N, dt, N / dt))
