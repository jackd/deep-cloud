from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# @tf.function
# def iterative_farthest_point_sample2(points, num_samples=None):
#     if num_samples is None:
#         num_samples = tf.shape(points)[0]
#     index = tf.random.uniform((), 0, num_samples, dtype=tf.int32)
#     dist2 = tf.reduce_sum(tf.math.squared_difference(points, points[index]),
#                           axis=-1)
#     acc = tf.TensorArray(dtype=tf.int32,
#                          size=num_samples,
#                          dynamic_size=False,
#                          clear_after_read=True,
#                          element_shape=())
#     acc = acc.write(0, index)
#     # for i in tf.data.Dataset.range(1, num_samples - 1):
#     for i in tf.range(1, num_samples - 1, dtype=tf.int32):
#         # tf.range is slightly faster...
#         index = tf.argmax(dist2, output_type=tf.int32)
#         acc = acc.write(i, index)
#         next_dist2 = tf.reduce_sum(tf.math.squared_difference(
#             points, points[index]),
#                                    axis=-1)
#         dist2 = tf.minimum(dist2, next_dist2)
#     index = tf.argmax(dist2, output_type=tf.int32)
#     acc = acc.write(num_samples - 1, index)
#     return acc.stack()


def iterative_farthest_point_order(points, num_samples=None):

    def cond(count, index, dist2, acc):
        return count < num_samples - 1

    def step(count, index, dist2, acc):
        index = tf.argmax(dist2, output_type=tf.int32)
        acc = acc.write(count, index)
        next_dist2 = tf.reduce_sum(tf.math.squared_difference(
            points, points[index]),
                                   axis=-1)
        dist2 = tf.minimum(dist2, next_dist2)
        return count + 1, index, dist2, acc

    size = tf.shape(points)[0]
    index = tf.random.uniform((), 0, size, dtype=tf.int32)
    dist2 = tf.reduce_sum(tf.math.squared_difference(points, points[index]),
                          axis=-1)
    if num_samples is None:
        num_samples = size
    acc = tf.TensorArray(dtype=tf.int32,
                         size=num_samples,
                         dynamic_size=False,
                         clear_after_read=True,
                         element_shape=())
    acc = acc.write(0, index)

    count, index, dist2, acc = tf.while_loop(cond, step, (1, index, dist2, acc))
    index = tf.argmax(dist2, output_type=tf.int32)
    acc = acc.write(count, index)
    return acc.stack()


if __name__ == '__main__':
    from time import time
    from tqdm import tqdm
    warmup = 10
    runs = 100
    with tf.device('/cpu:0'):
        points = tf.random.uniform(shape=(1024, 3), dtype=tf.float32)
        indices = iterative_farthest_point_order(points, 512)

    with tf.Session() as sess:
        for _ in range(warmup):
            sess.run(indices)
        t = time()
        for _ in tqdm(range(runs)):
            sess.run(indices)
        dt = time() - t
        print('Completed {} runs in {:2f} s, {:2f} runs / sec'.format(
            runs, dt, runs / dt))
