from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import six
import tensorflow as tf
import tensorflow_datasets as tfds
from more_keras.framework.problems.tfds import TfdsProblem
from shape_tfds.shape import modelnet

FULL = 'FULL'


def _base_modelnet_map(inputs,
                       labels,
                       positions_only=True,
                       num_points=None,
                       random_points=False,
                       normalize=False,
                       up_dim=2,
                       repeats=None):
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = None if positions_only else inputs['normals']
    else:
        positions = inputs
        normals = None

    # sample points
    if num_points is not None:
        if positions_only:
            if random_points:
                positions = tf.random.shuffle(positions)
            positions = positions[:num_points]
        else:
            if random_points:
                indices = tf.range(tf.shape(positions)[0])
                indices = tf.random.shuffle(indices)[:num_points]
                positions = tf.gather(positions, indices)
                normals = tf.gather(positions, indices)
            else:
                positions = positions[:num_points]
                normals = normals[:num_points]

    # make up-axis dim 2
    if up_dim != 2:
        shift = 2 - up_dim
        positions = tf.roll(positions, shift, axis=-1)
        if normals is not None:
            normals = tf.roll(normals, shift, axis=-1)

    if normalize:
        positions = positions - tf.reduce_mean(positions, axis=0, keepdims=True)
        dist2 = tf.reduce_max(tf.reduce_sum(tf.square(positions), axis=-1))
        positions = positions / tf.sqrt(dist2)

    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    if repeats is not None:
        labels = (labels,) * (repeats + 1)
    return inputs, labels


def _repeat_configurable(configurable, num_repeats, name='{orig}-{index}'):
    """Get `num_repeats + 1` versions of configurable."""
    config = configurable.get_config()
    cls = type(configurable)
    orig = config.pop('name')
    out = []
    for i in range(num_repeats + 1):
        config['name'] = name.format(orig=orig, index=i)
        out.append(cls.from_config(config))
    return out


@gin.configurable
class ModelnetProblem(TfdsProblem):

    def __init__(
            self,
            builder=None,
            num_points=1024,
            random_points=False,
            positions_only=True,
            loss=None,
            metrics=None,
            objective=None,
            train_split=FULL,  # 'full' or integer percent
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            normalize=False,
            repeated_outputs=None,
    ):
        if builder is None:
            builder = modelnet.Pointnet()
        elif isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        if loss is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)
        if metrics is None:
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(),
                tf.keras.metrics.SparseCategoricalCrossentropy(
                    from_logits=True),
            ]
        if repeated_outputs is not None:
            loss = _repeat_configurable(loss, repeated_outputs)
            metrics = tuple(
                zip(*(_repeat_configurable(m, repeated_outputs)
                      for m in metrics)))

        self._train_split = train_split
        if train_split == FULL:
            split_map = dict(validation='test')
        else:
            assert (isinstance(train_split, float))
            split_map = dict(
                train={'train': (0, train_split)},
                validation={'train': (train_split, 1)},
            )
        self._base_map_kwargs = dict(
            num_points=num_points,
            random_points=random_points,
            positions_only=positions_only,
            up_dim=builder.up_dim,
            normalize=normalize,
            repeats=repeated_outputs,
        )
        self.num_parallel_calls = num_parallel_calls
        input_spec = tf.TensorSpec(shape=(num_points, 3), dtype=tf.float32)
        if not positions_only:
            input_spec = dict(
                positions=input_spec,
                normals=input_spec,
            )
        labels_spec = tf.TensorSpec(shape=(), dtype=tf.int64)
        element_spec = (input_spec, labels_spec)
        if repeated_outputs:
            labels_spec = [labels_spec] * (repeated_outputs + 1)
        super(ModelnetProblem, self).__init__(
            builder=builder,
            loss=loss,
            metrics=metrics,
            objective=objective,
            element_spec=element_spec,
            as_supervised=True,
            split_map=split_map,
        )

    def _get_base_dataset(self, split):
        dataset = super(ModelnetProblem, self)._get_base_dataset(split)
        return dataset.map(
            functools.partial(_base_modelnet_map, **self._base_map_kwargs),
            self.num_parallel_calls)

    def get_config(self):
        config = super(ModelnetProblem, self).get_config()
        for k in ('input_spec', 'output_spec', 'as_supervised', 'split_map'):
            del config[k]
        config.update(self._base_map_kwargs)
        config['train_split'] = self._train_split
        config['num_parallel_calls'] = self.num_parallel_calls
        return config


@gin.configurable
class SampledModelnetProblem(ModelnetProblem):

    def __init__(self, num_points_base=2048, num_classes=40, **kwargs):
        builder = {
            10: modelnet.Modelnet10,
            40: modelnet.Modelnet40,
        }[num_classes](config=modelnet.CloudConfig(num_points_base))
        super(SampledModelnetProblem, self).__init__(builder, **kwargs)

    def get_config(self):
        config = super(SampledModelnetProblem, self).get_config()
        del config['builder']
        config['num_points_base'] = self.builder.num_points
        config['num_classes'] = self.builder.num_classes
        return config
