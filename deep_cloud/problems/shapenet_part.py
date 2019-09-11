from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from more_keras.framework.problems.tfds import TfdsProblem
from shape_tfds.shape.shapenet import part
from deep_cloud.problems.utils import repeat_configurable
from more_keras.ragged import batching as rb
from more_keras.flat_packer import FlatPacker

# def part_sparse_categorical_crossentropy(label_packer,
#                                          y_true,
#                                          y_pred,
#                                          from_logits=True):
#     y_true = tf.reshape(y_true, (-1,))
#     y_true = label_packer.unpack(y_true)
#     labels = y_true['point_labels']
#     assert (labels.ragged_rank == 1)
#     class_masks = y_true['class_masks']
#     if isinstance(y_pred, tf.RaggedTensor):
#         y_pred = y_pred.flat_values
#     row_lengths = labels.row_lengths()
#     class_masks = tf.repeat(class_masks, row_lengths, axis=0)
#     y_pred = tf.where(class_masks, y_pred,
#                       tf.fill(tf.shape(y_pred), -np.inf if from_logits else 0))
#     loss = tf.keras.backend.sparse_categorical_crossentropy(
#         labels.flat_values, y_pred, from_logits=from_logits)
#     return tf.RaggedTensor.from_row_splits(loss, labels.row_splits)

# class PartSparseCategoricalCrossentropy(tf.keras.losses.Loss):

#     def __init__(self, label_packer, from_logits=True):
#         self.label_packer = label_packer
#         self.from_logits = from_logits

#     def __call__(self, y_true, y_pred, sample_weight):
#         if sample_weight is None:
#             raise NotImplementedError()
#         if isinstance(sample_weight, tf.RaggedTensor):
#             sample_weight = sample_weight.flat_values
#         loss = part_sparse_categorical_crossentropy(self.label_packer, y_true,
#                                                     y_pred, self.from_logits)
#         batch_size = tf.size(loss.row_lengths, out_type=tf.float32) - 1
#         loss = tf.reduce_sum(loss.flat_values * sample_weight)
#         return loss / batch_size


def mean_iou(cm, dtype=tf.float32):
    """Compute the mean intersection-over-union via the confusion matrix."""
    # based on tf.keras.metrics.MeanIoU.result()
    sum_over_row = tf.cast(tf.reduce_sum(cm, axis=0), dtype=dtype)
    sum_over_col = tf.cast(tf.reduce_sum(cm, axis=1), dtype=dtype)
    true_positives = tf.cast(tf.diag_part(cm), dtype=dtype)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(
        tf.cast(tf.not_equal(denominator, 0), dtype=dtype))

    iou = tf.div_no_nan(true_positives, denominator)

    return tf.div_no_nan(tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)


class BlockMeanIoU(tf.keras.metrics.MeanIoU):

    def __init__(self, row_splits, name=None, dtype=None):
        self.row_splits = row_splits
        self.num_blocks = len(self.row_splits) - 1
        super(BlockMeanIoU, self).__init__(num_classes=row_splits[-1],
                                           name=name,
                                           dtype=dtype)

    def result(self):
        out = []
        for i in range(self.num_blocks):
            start, end = self.row_splits[i:i + 2]
            block_cm = self.total_cm[start:end, start:end]
            out.append(mean_iou(block_cm, self.dtype))
        out.append(np.mean(out))
        out.append(mean_iou(self.total_cm))
        return np.array(out)

    def get_config(self):
        config = super(BlockMeanIoU, self).get_config()
        config['row_splits'] = self.row_splits
        return config


@gin.configurable
class ShapenetPartProblem(TfdsProblem):

    def __init__(self,
                 objective=None,
                 repeated_outputs=None,
                 inverse_density_weights=False):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.MeanIoU(num_classes=part.NUM_PART_CLASSES)
        ]
        if repeated_outputs is not None:
            loss = repeat_configurable(loss, repeated_outputs)
            metrics = tuple(
                zip(*(repeat_configurable(m, repeated_outputs)
                      for m in metrics)))
        builder = part.ShapenetPart2017()
        self.inverse_density_weights = inverse_density_weights

        if repeated_outputs is not None:
            loss = (loss,) * (1 + repeated_outputs)
            metrics = (metrics,) * (1 + repeated_outputs)
            if objective is None:
                objective = metrics[-1][-1].name
        else:
            if objective is None:
                objective = metrics[-1].name
        super(ShapenetPartProblem, self).__init__(
            builder=builder,
            loss=loss,
            metrics=metrics,
            objective=objective,
            as_supervised=False,
            output_spec=tf.TensorSpec(shape=(None, part.NUM_PART_CLASSES)))

    @property
    def num_object_classes(self):
        return part.NUM_OBJECT_CLASSES

    @property
    def class_weights(self):
        if not self.inverse_density_weights:
            return None

        if not hasattr(self, '_class_weights'):
            freq = part.POINT_CLASS_FREQ
            class_weights = np.mean(freq) / freq
            class_weights = class_weights.astype(np.float32)
            self._class_weights = class_weights
        return self._class_weights

    def _get_base_dataset(self, split):
        base = super(ShapenetPartProblem, self)._get_base_dataset(split)
        label_masks_np = np.zeros(
            (part.NUM_OBJECT_CLASSES, part.NUM_PART_CLASSES), dtype=bool)
        for i in range(part.NUM_OBJECT_CLASSES):
            label_masks_np[part.LABEL_SPLITS[i]:part.LABEL_SPLITS[i + 1]] = True

        def map_fn(kwargs):
            label_masks = tf.constant(label_masks_np, dtype=tf.bool)
            cloud = kwargs['cloud']
            labels, positions, normals = (rb.pre_batch_ragged(cloud[k])
                                          for k in ('labels', 'positions',
                                                    'normals'))
            row_lengths = tf.size(labels.flat_values)
            class_label = kwargs['label']
            features = dict(
                positions=positions,
                normals=normals,
                class_label=class_label,
                class_masks=label_masks[class_label],
            )
            weights = 1. / tf.cast(row_lengths, tf.float32)
            if self.inverse_density_weights:
                weights = weights * tf.constant(self.class_weights,
                                                tf.float32)[labels.flat_values]
            weights = rb.pre_batch_ragged(weights)

            return features, labels, weights

        return base.map(map_fn,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def post_batch_map(self, labels, weights=None):
        batch_size = tf.size(labels.row_splits, out_type=tf.float32) - 1
        return labels.flat_values, weights.flat_values / batch_size
