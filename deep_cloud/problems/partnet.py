from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gin
import tensorflow as tf

from shape_tfds.shape.shapenet import partnet
from more_keras.framework.problems.tfds import TfdsProblem
from deep_cloud.problems.utils import repeat_configurable


@gin.configurable
def partnet_model_dir(base_dir='~/deep-cloud-models/partnet',
                      problem='sem_seg',
                      name='default',
                      category='table',
                      run=0):
    return os.path.join(base_dir, problem, name, category,
                        'run-{:02d}'.format(run))


@gin.configurable
class PartnetProblem(TfdsProblem):

    def __init__(self,
                 level=1,
                 category='table',
                 objective=None,
                 repeated_outputs=None,
                 inverse_density_weights=False):
        if inverse_density_weights:
            raise NotImplementedError('TODO')

        builder = partnet.Partnet(config=category, level=level)
        num_classes = builder.num_classes[level]

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum_over_batch_size')
        metrics = [
            # tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.MeanIoU(num_classes=num_classes),
        ]
        self.inverse_density_weights = inverse_density_weights

        if repeated_outputs is not None:
            loss = (loss,) * (1 + repeated_outputs)
            metrics = (metrics,) * (1 + repeated_outputs)
            if objective is None:
                objective = metrics[-1][-1].name
        else:
            if objective is None:
                objective = metrics[-1].name
        super(PartnetProblem,
              self).__init__(builder=builder,
                             loss=loss,
                             metrics=metrics,
                             objective=objective,
                             as_supervised=True,
                             output_spec=tf.TensorSpec(shape=(None,
                                                              num_classes)))

    @property
    def class_weights(self):
        raise NotImplementedError

    def post_batch_map(self, labels, weights=None):
        if weights is not None:
            if weights.shape.ndims == 1:
                weights = tf.tile(tf.expand_dims(weights, axis=1),
                                  (1, tf.shape(labels)[1]))
            weights = tf.reshape(weights, (-1,))

        labels = tf.reshape(labels, (-1,))
        return labels, weights
