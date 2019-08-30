from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# @tf.function
# def iterative_farthest_point_sample2(points, num_samples=None):
#     if num_samples is None:
#         num_samples = tf.shape(points)[0]
#     index = tf.random.uniform((), 0, num_samples, dtype=tf.int64)
#     dist2 = tf.reduce_sum(tf.math.squared_difference(points, points[index]),
#                           axis=-1)
#     acc = tf.TensorArray(dtype=tf.int64,
#                          size=num_samples,
#                          dynamic_size=False,
#                          clear_after_read=True,
#                          element_shape=())
#     acc = acc.write(0, index)
#     # for i in tf.data.Dataset.range(1, num_samples - 1):
#     for i in tf.range(1, num_samples - 1, dtype=tf.int64):
#         # tf.range is slightly faster...
#         index = tf.argmax(dist2, output_type=tf.int64)
#         acc = acc.write(i, index)
#         next_dist2 = tf.reduce_sum(tf.math.squared_difference(
#             points, points[index]),
#                                    axis=-1)
#         dist2 = tf.minimum(dist2, next_dist2)
#     index = tf.argmax(dist2, output_type=tf.int64)
#     acc = acc.write(num_samples - 1, index)
#     return acc.stack()


def iterative_farthest_point_order(points, num_samples=None, first=None):
    if num_samples is None:
        out_size = points.shape[0]
    elif isinstance(num_samples, int):
        out_size = num_samples
    else:
        out_size = None

    def cond(count, index, dist2, acc):
        return count < num_samples - 1

    def step(count, index, dist2, acc):
        index = tf.argmax(dist2, output_type=tf.int64)
        acc = acc.write(count, index)
        next_dist2 = tf.reduce_sum(tf.math.squared_difference(
            points, points[index]),
                                   axis=-1)
        dist2 = tf.minimum(dist2, next_dist2)
        return count + 1, index, dist2, acc

    size = tf.shape(points, out_type=tf.int64)[0]
    index = tf.random.uniform(
        (), 0, size, dtype=tf.int64) if first is None else first
    dist2 = tf.reduce_sum(tf.math.squared_difference(points, points[index]),
                          axis=-1)
    if num_samples is None:
        num_samples = size
    num_samples = tf.cast(num_samples, tf.int32)
    acc = tf.TensorArray(dtype=tf.int64,
                         size=num_samples,
                         dynamic_size=False,
                         clear_after_read=True,
                         element_shape=())

    acc = acc.write(0, index)

    count, index, dist2, acc = tf.while_loop(cond,
                                             step, (1, index, dist2, acc),
                                             back_prop=False)
    index = tf.argmax(dist2, output_type=tf.int64)
    acc = acc.write(count, index)
    out = acc.stack()
    if out_size is not None:
        out.set_shape((out_size,))
    return out


def partial_reorder(indices, points, *args):
    total = tf.shape(points)[0]
    mask = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                         tf.ones(shape=tf.shape(indices), dtype=tf.bool),
                         (total,))
    mask = tf.logical_not(mask)
    inputs = (points, *args)
    out = tuple(
        tf.concat((tf.gather(inp, indices), tf.boolean_mask(inp, mask)), axis=0)
        for inp in inputs)
    # fix shapes
    for o, i in zip(out, inputs):
        o.set_shape(i.shape)
    return out[0] if len(args) == 0 else out


if __name__ == '__main__':
    do_profile = False
    do_vis = False

    do_profile = True
    do_vis = True

    if do_profile:
        from time import time
        from tqdm import tqdm
        warmup = 10
        runs = 100
        with tf.Graph().as_default():  # pylint: disable=not-context-manager
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

    if do_vis:
        with tf.Graph().as_default():  # pylint: disable=not-context-manager
            num_samples = 512
            points = tf.random.uniform(shape=(1024, 2), dtype=tf.float32)
            indices = iterative_farthest_point_order(points, num_samples)
            points = partial_reorder(indices, points)

            with tf.Session() as sess:
                points = sess.run(points)

        import matplotlib.pyplot as plt
        xo, yo = points[:num_samples].T
        xr, yr = points[num_samples:].T
        plt.scatter(xo, yo, marker='x')
        plt.scatter(xr, yr, marker='o', alpha=0.2)
        plt.show()
