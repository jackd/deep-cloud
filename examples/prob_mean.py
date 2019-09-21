from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from more_keras.metrics.prob_mean_iou import ProbMeanIoU

tf.compat.v1.enable_eager_execution()

num_classes = 4
y_true = np.arange(num_classes, dtype=np.int64)
y_pred = np.array(
    [[0.5, 0.2, 0.3, 0], [1.0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32)
# y_pred = np.eye(num_classes).astype(np.float32)

metric = ProbMeanIoU(num_classes=num_classes)
metric.update_state(y_true, y_pred)
print(metric.result())

m2 = tf.keras.metrics.MeanIoU(num_classes=num_classes)
y_pred = np.argmax(y_pred, axis=1)
cm = m2.update_state(y_true, y_pred)
print(cm.numpy())
print(m2.result())
