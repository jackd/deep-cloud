from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deep_cloud.problems import partnet
from deep_cloud.problems import modelnet
from deep_cloud.problems.builders import pointnet_builder
from deep_cloud.models.very_dense import core
import functools

tf.compat.v1.enable_eager_execution()

problem = partnet.PartnetProblem()
# problem = modelnet.ModelnetProblem(builder=pointnet_builder(2))
# problem.builder.download_and_prepare()
with problem:
    dataset = problem.get_base_dataset(split='train')
    dataset = dataset.map(
        functools.partial(core.pre_batch_map,
                          edge_fn=functools.partial(
                              core.compute_edges,
                              eager_fn=core.compute_edges_principled_eager)))

    dataset = dataset.batch(2)
    dataset = dataset.map(core.post_batch_map)

    for inputs, label in dataset:
        break

    print('creating')
    with tf.device('/cpu:0'):
        core.very_dense_features(inputs, repeats=1)
    print('done')
