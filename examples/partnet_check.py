from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from deep_cloud.problems import partnet
from deep_cloud.models.very_dense import core
from tqdm import tqdm
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

split = 'train'
category = 'chair'
problem = partnet.PartnetProblem(category='chair')
problem.builder.download_and_prepare()
dataset = problem.get_base_dataset(split=split)

for coords, labels in tqdm(dataset, total=problem.examples_per_epoch(split)):
    assert (tf.reduce_all(labels < problem.output_spec.shape[-1]).numpy())
