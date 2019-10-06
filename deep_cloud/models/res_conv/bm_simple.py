from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from deep_cloud.models.res_conv import simple as si

from absl import app
from absl import flags
from time import time

flags.DEFINE_bool('naive', default=False, help='run naive implementation')
flags.DEFINE_string('params', default='small', help='param set keys')
flags.DEFINE_float('size_scale', default=1, help='Size scaling factor')
flags.DEFINE_integer('repeats', default=1, help='number of repeats')
flags.DEFINE_boolean('norm', default=False, help='normalize after each conv')
flags.DEFINE_float('edges', default=None, help='mean number of edges')
flags.DEFINE_integer('t', default=4, help='number of spatial features')

FLAGS = flags.FLAGS


def super_simple(features,
                 kernel,
                 sparse_indices,
                 edge_weights,
                 dense_shape,
                 transform_first=None):
    """Simplified version of unstack_sum."""
    T, F_in, F_out = kernel.shape
    N_out, N_in = dense_shape
    del T, N_out, N_in
    if transform_first is None:
        transform_first = F_out <= F_in

    if isinstance(sparse_indices, (list, tuple)):
        i, j = sparse_indices
    else:
        i, j = tf.unstack(sparse_indices, axis=-1)
    edge_weights = tf.unstack(edge_weights, axis=0)
    kernels = tf.unstack(kernel, axis=0)
    terms = []
    for k, ew in zip(kernels, edge_weights):
        f = features
        if transform_first:
            f = tf.matmul(f, k)
        f = tf.gather(f, j)
        f *= tf.expand_dims(ew, axis=-1)
        f = tf.segment_sum(f, i)
        if not transform_first:
            f = tf.matmul(f, k)
        terms.append(f)
    return tf.add_n(terms)


def get_data(n_in, n_out, f_in, f_out, t=4, mean_edges=9, seed=123):
    r = np.random.RandomState(seed)

    num_edges = int(mean_edges * n_out)

    flat_index = r.randint(0, high=n_in * n_out, size=num_edges, dtype=np.int64)
    flat_index = np.unique(flat_index)
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(flat_index, (n_out, n_in))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = np.stack((i, j), axis=-1)
    edge_weights = r.uniform(size=(t, i.shape[0])).astype(np.float32)
    features = r.uniform(size=(n_in, f_in)).astype(np.float32)
    if FLAGS.norm:
        features -= np.mean(features, axis=0)
        features /= np.sqrt(np.mean(features**2, axis=0))
    kernel = r.uniform(size=(t, f_in, f_out)).astype(np.float32)
    dense_shape = (n_out, n_in)

    # features = tf.constant(features, name='input_features')
    # kernel = tf.constant(kernel, name='kernel')
    # sparse_indices = tf.constant(sparse_indices, name='sparse_indices')
    # edge_weights = tf.constant(edge_weights, name='edge_weights')

    return features, kernel, sparse_indices, edge_weights, dense_shape


def run_benchmarks(names, fns, run_name=None):
    if run_name is None:
        run_name = 'BENCHMARKS'

    times = []
    memories = []
    vals = []
    dts = []

    for name, fn in zip(names, fns):
        print('---------------')
        print(name)
        print('---------------')
        graph = tf.Graph()
        with graph.as_default():
            init, f, ops = fn()

        with tf.Session(graph=graph) as sess:
            sess.run(init)
            vals.append(sess.run(f))
            sess.run(ops)

            bm = tf.test.Benchmark()
            t = time()
            result = bm.run_op_benchmark(sess, ops)
            dts.append(time() - t)
            times.append(result['wall_time'])
            memories.append(
                result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])

    print('Errs w.r.t {}'.format(names[0]))
    # errs = [
    #     tf.nest.map_structure(lambda x, y: np.max(np.abs(x - y)), v, vals[0])
    #     for v in vals[1:]
    # ]
    errs = [np.max(np.abs(v[0] - vals[0][0])) for v in vals[1:]]
    for name, err in zip(names[1:], errs):
        print(err)

    print('*************')
    print('** {} **'.format(run_name))
    print('*************')

    times = np.array(times) * 1000  # ms
    ti = np.argmin(times)
    tmin = times[ti]
    print('Best time: {}, {} ms'.format(names[ti], tmin))
    print('rel times:')
    for name, t in zip(names, times):
        print('{:15s} {:.3f} {:.3f}'.format(name, t / tmin, t))

    dts = np.array(dts) * 1000  # ms
    dti = np.argmin(dts)
    dtmin = dts[dti]
    print('Best dt: {}, {} ms'.format(names[dti], dtmin))
    print('dts:')
    for name, dt in zip(names, dts):
        print('{:15s} {:.3f} {:.3f}'.format(name, dt / dtmin, dt))

    memories = np.array(memories) / 1024**2  # Mb
    mi = np.argmin(memories)
    mmin = memories[mi]
    print('Best memory: {}, {} Mb'.format(names[mi], mmin))
    print('Memory usage')
    for name, memory in zip(names, memories):
        print('{:15s} {:.03f} {}'.format(name, memory / mmin, memory))


def run_conv_benchmarks(n_in, n_out, skip_naive=False, run_name=None, **kwargs):
    import functools

    sp = si.get_sparse_transform
    gs = si.get_gather_sum_transform

    names, fns = zip(*(
        # ('super_simple', super_simple),
        ('unstack_sparse', functools.partial(si.unstack_conv, term_impl=sp)),
        ('unstack_sum', functools.partial(si.unstack_conv, term_impl=gs)),
        ('map_sparse', functools.partial(si.map_conv, term_impl=sp)),
        ('map_sum', functools.partial(si.map_conv, term_impl=gs)),
        ('fold_sparse', functools.partial(si.fold_conv, term_impl=sp)),
        ('fold_sum', functools.partial(si.fold_conv, term_impl=gs)),
        # ('block_sparse', functools.partial(si.block_conv, term_impl=sp)),
        # ('block_sum', functools.partial(si.block_conv, term_impl=gs)),
    ))

    features_, kernel_, sparse_indices_, edge_weights_, dense_shape_ = get_data(
        n_in, n_out, **kwargs)

    def map_fn(fn):

        def out_fn():
            features = tf.Variable(features_, name='input_features')
            kernel = tf.Variable(kernel_, name='kernel')
            sparse_indices = tf.constant(sparse_indices_, name='sparse_indices')
            edge_weights = tf.constant(edge_weights_, name='edge_weights')
            dense_shape = [tf.constant(d, dtype=tf.int64) for d in dense_shape_]

            val = features
            for _ in range(FLAGS.repeats):
                val = fn(val, kernel, sparse_indices, edge_weights, dense_shape)
                if FLAGS.norm:
                    val = val - tf.reduce_mean(val, axis=0)
                    val = val / tf.sqrt(tf.reduce_mean(val**2, axis=0))

            init = features.initializer, kernel.initializer
            train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(
                val, var_list=[features, kernel])
            return init, val, train_op

        return out_fn

    fns = [map_fn(fn) for fn in fns]
    run_benchmarks(names, fns, run_name=run_name)


if __name__ == '__main__':

    def main(argv):
        tf.compat.v1.enable_v2_tensorshape()
        batch_size = 16
        param_sets = dict(
            small=dict(
                n_in=11,
                n_out=13,
                f_in=3,
                f_out=5,
                mean_edges=3,
            ),
            small2=dict(
                n_in=11,
                n_out=13,
                f_in=5,
                f_out=3,
                mean_edges=3,
            ),
            paper=dict(
                n_in=4096 * 8,
                n_out=4096 * 8,
                f_in=64,
                f_out=64,
                mean_edges=9,
            ),
            in_place=dict(
                n_in=10000 * batch_size,
                n_out=10000 * batch_size,
                f_in=32,
                f_out=32,
                mean_edges=12,
            ),
            in_place2=dict(
                n_in=2500 * batch_size,
                n_out=2500 * batch_size,
                f_in=64,
                f_out=64,
                mean_edges=12,
            ),
            down_sample=dict(
                n_in=10000 * batch_size,
                n_out=2500 * batch_size,
                f_in=32,
                f_out=64,
                mean_edges=12,
            ),
            up_sample=dict(
                n_in=2500 * batch_size,
                n_out=10000 * batch_size,
                f_in=64,
                f_out=32,
                mean_edges=12,
            ),
        )

        k = FLAGS.params
        # fn = run_multi_benchmark if FLAGS.multi else run_conv_benchmarks
        kwargs = param_sets[k]
        kwargs['n_in'] = int(FLAGS.size_scale * kwargs['n_in'])
        kwargs['n_out'] = int(FLAGS.size_scale * kwargs['n_out'])
        if FLAGS.edges is not None:
            kwargs['mean_edges'] = FLAGS.edges
        if FLAGS.t is not None:
            kwargs['t'] = FLAGS.t
        run_conv_benchmarks(**kwargs, run_name=k)

        # run_conv_benchmarks(**param_sets[k],
        #                     run_name=k,
        #                     skip_naive=not FLAGS.naive)

        # run_multi_benchmark(10000 * batch_size, 32, repeats=11)
        # run_multi_benchmark(4096 * 8, 64, mean_edges=9, repeats=11)

    app.run(main)
    print('done')
