from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
from keras_config.pipelines import Pipeline
from keras_config.trainers import Trainer
from deep_cloud.models import pointnet
from deep_cloud.problems import modelnet
from deep_cloud.models.pointnet import pointnet_classifier
from deep_cloud.functions.augment import augment_cloud
from keras_config.schedules import ClippedExponentialDecay
from keras_config.session_options import SessionOptions

SessionOptions().configure_session()

BATCH_SIZE = 32
EPOCHS = 250

problem = modelnet.ModelnetProblem()
train_pipeline = Pipeline(
    batch_size=BATCH_SIZE,
    repeats=None,
    shuffle_buffer=problem.examples_per_epoch('train'),
    map_fn=functools.partial(
        augment_cloud,
        rotate_scheme='random',
        jitter_stddev=2e-2,
        #    jitter_clip=5e-2,
    ))
validation_pipeline = Pipeline(
    batch_size=BATCH_SIZE,
    repeats=None,
    shuffle_buffer=problem.examples_per_epoch('validation'),
    map_fn=functools.partial(augment_cloud, rotate_scheme='none'))

optimizer = tf.keras.optimizers.Adam(learning_rate=ClippedExponentialDecay(
    1e-3,
    decay_steps=20 * problem.examples_per_epoch('train') / BATCH_SIZE,
    decay_rate=0.5,
    #   decay_steps=200000 / BATCH_SIZE,
    #   decay_rate=0.7,
    min_value=1e-5,
    staircase=True))

chkpt_dir = '~/deep-cloud-models/pointnet-cls/base'
trainer = Trainer(problem=problem,
                  train_pipeline=train_pipeline,
                  validation_pipeline=validation_pipeline,
                  model_fn=pointnet_classifier,
                  optimizer=optimizer,
                  chkpt_dir=chkpt_dir)

trainer.train(EPOCHS, verbose=True)
