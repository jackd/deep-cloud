from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from more_keras.losses import continuous_mean_iou_loss

num_classes = 3
labels = np.array([2, 1, 1, 2, 0], dtype=np.int64)
preds = np.random.normal(size=(labels.shape[0], num_classes)).astype(np.float32)

print(preds)
loss = continuous_mean_iou_loss(labels, preds, from_logits=True)
print(loss.numpy())
preds[0, 0] += 1
preds[1, 0] += 1
preds[2, 0] += 1
preds[3, 0] += 1
preds[4, 1] += 1
# preds[0, 2] += 10
# preds[1, 1] += 10
# preds[2, 1] += 10
# preds[3, 2] += 10
# preds[4, 0] += 10
print(preds)
loss = continuous_mean_iou_loss(labels, preds, from_logits=True)
print(loss.numpy())
