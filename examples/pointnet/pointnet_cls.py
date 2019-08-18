from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import functools
import tensorflow as tf
from more_keras.callbacks.gin_config import GinConfigSaver
from more_keras.session import SessionOptions
from more_keras.schedules import ExponentialDecayTowards
from more_keras.framework import train as train_lib
from more_keras.framework.pipelines import Pipeline

from deep_cloud.models import pointnet
from deep_cloud.problems import modelnet
from deep_cloud.models.pointnet import pointnet_classifier
from deep_cloud.augment import augment_cloud
import gin

SessionOptions = gin.external_configurable(SessionOptions)
pointnet_classifier = gin.external_configurable(pointnet_classifier)

flags.DEFINE_multi_string('config_files',
                          default=['$HERE/configs/no-jitter.gin'],
                          help='List of paths to the config files.')
flags.DEFINE_multi_string('bindings', None,
                          'Newline separated list of Gin parameter bindings.')


@gin.configurable(blacklist=['steps_per_epoch'])
def learning_rate(
        steps_per_epoch,
        initial_value=1e-3,
        decay_epochs=20,
        decay_rate=0.5,  # code is 0.7
        clip_value=1e-5):
    return ExponentialDecayTowards(
        initial_value,
        decay_steps=decay_epochs * steps_per_epoch,
        decay_rate=decay_rate,
        clip_value=clip_value,
    )


@gin.configurable
def params(batch_size=32, epochs=250):
    return dict(batch_size=batch_size, epochs=epochs)


@gin.configurable
def augment_params(
        jitter_stddev=None,
        jitter_clip=None,
        scale_stddev=None,
        maybe_reflect_x=False,
        rotate_scheme='random',
):
    return dict(
        jitter_stddev=jitter_stddev,
        jitter_clip=jitter_clip,
        scale_stddev=scale_stddev,
        maybe_reflect_x=maybe_reflect_x,
        rotate_scheme=rotate_scheme,
    )


@gin.configurable
def chkpt_dir(root_dir='$HERE/models', name='default'):
    return os.path.join(root_dir, name)


@gin.configurable
def train(batch_size=32,
          epochs=250,
          name='default',
          train_steps=None,
          validation_steps=None):
    SessionOptions().configure_session()

    problem = modelnet.ModelnetProblem()
    with gin.config_scope('train'):
        train_params = augment_params()
    with gin.config_scope('validation'):
        validation_params = augment_params()

    train_pipeline = Pipeline(
        batch_size=batch_size,
        repeats=None,
        shuffle_buffer=problem.examples_per_epoch('train'),
        map_fn=functools.partial(
            augment_cloud,
            **train_params,
        ),
        output_spec=problem.input_spec)
    validation_pipeline = Pipeline(
        batch_size=batch_size,
        repeats=None,
        shuffle_buffer=problem.examples_per_epoch('validation'),
        map_fn=functools.partial(augment_cloud, **validation_params),
        output_spec=problem.input_spec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate(
        problem.examples_per_epoch('train') / batch_size))

    root_dir = chkpt_dir()
    train_lib.train(problem=problem,
                    train_pipeline=train_pipeline,
                    validation_pipeline=validation_pipeline,
                    model_fn=pointnet_classifier,
                    optimizer=optimizer,
                    epochs=epochs,
                    chkpt_dir=root_dir,
                    extra_callbacks=[
                        GinConfigSaver(
                            os.path.join(root_dir, 'operative-config.gin'))
                    ],
                    train_steps=train_steps,
                    validation_steps=validation_steps)


def main(_):
    os.environ['HERE'] = os.path.realpath(os.path.dirname(__file__))
    FLAGS = flags.FLAGS
    config_files = [
        os.path.expanduser(os.path.expandvars(f)) for f in FLAGS.config_files
    ]
    gin.parse_config_files_and_bindings(config_files, FLAGS.bindings)
    train()


if __name__ == '__main__':
    app.run(main)
