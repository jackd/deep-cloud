from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from time import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from deep_cloud.models.very_dense import simple
from deep_cloud.augment import augment_cloud
import gin

config = '''

train/augment_cloud.angle_stddev = 0.06
train/augment_cloud.angle_clip = 0.18
train/augment_cloud.uniform_scale_range = (0.8, 1.25)
train/augment_cloud.rotate_scheme = 'random'
train/augment_cloud.jitter_stddev = 1e-2
train/augment_cloud.jitter_clip = 5e-2
train/augment_cloud.angle_stddev = None
train/augment_cloud.angle_clip = None
train/augment_cloud.uniform_scale_range = None
'''
gin.parse_config(config)

# depth = 4
# num_tri = (depth * (depth + 1)) // 2
# out_types = (
#     (tf.float32,) * depth,  # coords
#     (tf.float32,) * depth,  # normals
#     (tf.float32,) * num_tri,  # flat_rel_coords
#     (tf.int64,) * num_tri,  # flat_node_indices
#     (tf.int64,) * num_tri,  # row_splits
# )
# flat_out_types = tuple(tf.nest.flatten(out_types))

# def compute_edges(coords, normals, eager_fn=simple.compute_edges_eager):
#     fn = functools.partial(simple._flatten_output, eager_fn, depth=4)
#     return tuple(tf.py_function(fn, [coords, normals], flat_out_types))

# def map_fn(features, labels, edge_fn=simple.compute_edges):
#     return tuple(edge_fn(features['positions'], features['normals'])), labels


def get_base_dataset(num_examples=100, **kwargs):
    # return tf.data.Dataset.from_tensor_slices(
    #   get_base_data(num_exmaples=num_examples, **kwargs))
    from deep_cloud.problems.modelnet import ModelnetProblem
    from deep_cloud.problems.builders import pointnet_builder
    problem = ModelnetProblem(builder=pointnet_builder(2), positions_only=False)
    with gin.config_scope('train'):
        dataset = problem.get_base_dataset('validation').map(augment_cloud)
    return dataset


def profile_dataset(dataset, num_examples):
    times = []
    start = time()
    for _ in tqdm(tfds.as_numpy(dataset.take(num_examples)),
                  total=num_examples):
        end = time()
        times.append(end - start)
        start = end
    return times


num_examples = 100
dataset_fn = functools.partial(get_base_dataset, num_examples=num_examples)
# mapped = dataset_fn().map(functools.partial(tf_map_fn, eager_fn=eager_fn))
mapped = dataset_fn().map(simple.pre_batch_map)
# hack_mapped = hack_map(dataset_fn, py_map_fn, flat_out_types + (tf.int64,))

t0 = profile_dataset(mapped, num_examples)
# t1 = profile_dataset(hack_mapped, num_examples)
print('mapped: {}'.format(np.mean(t0[num_examples // 2:])))
# print('hacked: {}'.format(np.mean(t1[num_examples // 2:])))
