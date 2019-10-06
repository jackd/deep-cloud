from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def naive_conv(sparse_neighbors, x_in, x_out, features_in, bias, kernels):
    """
    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of weighted values.
            Rows should sum to 1 (though this is not ef_inorced). Assumed to be
            ordered (if not, use `tf.sparse.reorder(sparse_neighbors)`).
        x_in: [D, N_in] float tensor of input cloud coordinates.
        x_out: [D, N_out] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        biase: [F_in, F_out] float tensor.
        kernel: [D, F_in, F_out] float tensor of kernel.

    Returns:
        [N_out, F_out] float output features.
    """
    i, j = tf.unstack(sparse_neighbors.indices, axis=-1)
    dx = tf.gather(x_out, i, axis=1) - tf.gather(x_in, j, axis=1)

    gathered_features = tf.gather(features_in, j)

    def get_kth_term(k, dx):
        f = tf.matmul(gathered_features, k)
        f = f * tf.expand_dims(dx * sparse_neighbors.values, axis=-1)
        reduced = tf.math.segment_sum(f, i)
        return reduced

    init = tf.zeros((sparse_neighbors.dense_shape[0], bias.shape[-1]),
                    dtype=features_in.dtype)
    dxs = [tf.ones_like(dx), dx]
    kernels = (tf.expand_dims(bias, axis=0),) + tuple(kernels)
    kernels = tf.concat(kernels, axis=0)
    # construct higher order powers
    for _ in range(len(kernels) - 2):
        dxs.append(dxs[-1] * dx)
    dx = tf.concat(dx, axis=0)

    return tf.foldl(lambda acc, args: acc + get_kth_term(*args), (kernels, dx),
                    init)


def segment_sum(values, segments):
    return tf.math.unsorted_segment_sum(values, segments,
                                        tf.reduce_max(segments) + 1)


def sparse_reduce_sum(sp, axis=-1):
    assert (axis in (0, 1, -1))
    assert (sp.indices.shape[-1] == 2)
    segments = sp.indices[:, 0 if axis in (1, -1) else 1]
    return segment_sum(sp.values, segments)


def _check_arg_shapes(sparse_neighbors, x_in, x_out, features_in, kernel):
    D, F_in, F_out = kernel.shape
    compat_msg = '{} and {} shapes incompatible: {} and {}'
    for (x, name) in ((x_in, 'x_in'), (x_out, 'x_out')):
        if x is None:
            continue
        if x.shape.ndims != 2:
            raise ValueError('{} must be rank 2, but shape is {}'.format(
                name, x.shape))
        if x.shape[0] != D:
            raise ValueError(
                compat_msg.format('kernel', name, kernel.shape, x.shape))
    if features_in.shape.ndims != 2:
        raise ValueError('features_in should be rank 2, got {}'.format(
            features_in.shape))
    if features_in.shape[1] != F_in:
        raise ValueError(
            compat_msg.format('kernel', 'features_in', kernel.shape,
                              features_in.shape))
    if x_in is not None and features_in.shape[
            0] is not None and features_in.shape[0] != x_in.shape[1]:
        raise ValueError(
            compat_msg.format('features_in', 'x_in', features_in.shape,
                              x_in.shape))

    for n, x, name in zip(sparse_neighbors.shape, (x_out, x_in),
                          ('x_out', 'x_in')):
        if n is not None and x.shape[1] != n:
            raise ValueError(
                compat_msg.format('sparse_neighbors', name,
                                  sparse_neighbors.shape, x.shape))
    return D, F_in, F_out


def combined_term(sparse_neighbors, x_in, x_out, features_in, kernel,
                  impl=None):
    if impl is None:
        D, F_in, F_out = _check_arg_shapes(sparse_neighbors, x_in, x_out,
                                           features_in, kernel)
        del D
        if F_in < F_out:
            impl = combined_term_v1
        else:
            impl = combined_term_v2
    return impl(sparse_neighbors, x_in, x_out, features_in, kernel)


def combined_term_v1(sparse_neighbors, x_in, x_out, features_in, kernel):
    """
    Implements a single term in convolution with feature transform LAST.

    This version is intended for use in DOWN-sampling, i.e. where
    N_in > N_out and/or F_in < F_out.

    \\sum_{t,q,j} n_{ij} theta_{tqp} f_{jq} xout_{ti} xin_{tj}
    = xout_{i} \\sum_q theta_{qp} \\sum_j n_{ij} f_{qj} * xin_{j}
    """
    D, F_in, F_out = _check_arg_shapes(sparse_neighbors, x_in, x_out,
                                       features_in, kernel)
    del D, F_in

    def get_kth_term(kernel, x_in=None, x_out=None):
        fx = features_in
        if x_in is not None:
            fx = fx * tf.expand_dims(x_in, axis=-1)
        fx = tf.sparse.sparse_dense_matmul(sparse_neighbors, fx)
        fx = tf.matmul(fx, kernel)
        if x_out is not None:
            fx = fx * tf.expand_dims(x_out, axis=-1)
        return fx

    args = [kernel]
    names = ['kernel']
    if x_in is not None:
        args.append(x_in)
        names.append('x_in')
    if x_out is not None:
        args.append(x_out)
        names.append('x_out')

    def accumulate(s, args):
        return s + get_kth_term(**dict(zip(names, args)))

    init = tf.zeros((sparse_neighbors.dense_shape[0], F_out),
                    dtype=features_in.dtype)
    return tf.foldl(accumulate, args, init)


def combined_term_v2(sparse_neighbors, x_in, x_out, features_in, kernel):
    """
    Implements a single term in convolution with feature transform FIRST.

    This version is intended for use in UP-sampling, i.e. where
    N_in > N_out and/or F_in < F_out.

    \\sum_{k,q,j} n_{ij} theta_{kqp} f_{jq} xout_{ki} xin_{kj}
    = \\sum_{k} xout_{ki} \\sum_j n_{ij} xin_{kj} \\sum_q theta_{kqp} f_{qj}

    For the case where x_out is None (implied to be ones) the sum over k and j
    are switched, i.e.
    \\\\sum_j n_{ij} sum_{k} xin_{kj} \\sum_q theta_{kqp} f_{qj}.
    """
    D, F_in, F_out = _check_arg_shapes(sparse_neighbors, x_in, x_out,
                                       features_in, kernel)
    del D, F_in

    def get_kth_term(kernel, x_in=None, x_out=None):
        fx = features_in
        fx = tf.matmul(fx, kernel)
        if x_in is not None:
            fx = fx * tf.expand_dims(x_in, axis=-1)
        if x_out is not None:
            # we do sparse multiplication outside foldl if x_out is None
            fx = tf.sparse.sparse_dense_matmul(sparse_neighbors, fx)
            fx = fx * tf.expand_dims(x_out, axis=-1)
        return fx

    args = [kernel]
    names = ['kernel']
    if x_in is not None:
        args.append(x_in)
        names.append('x_in')
    if x_out is not None:
        args.append(x_out)
        names.append('x_out')

    def accumulate(s, args):
        return s + get_kth_term(**dict(zip(names, args)))

    init = tf.zeros((sparse_neighbors.dense_shape[0], F_out),
                    dtype=features_in.dtype)
    fx = tf.foldl(accumulate, args, init)
    if x_out is None:
        fx = tf.sparse.sparse_dense_matmul(sparse_neighbors, fx)
    return fx


def _binom_row(prev_row):
    n = len(prev_row)
    out = [1]
    for i in range(n - 1):
        out.append(prev_row[i] + prev_row[i + 1])
    out.append(1)
    return out


def stacked_conv(sparse_neighbors,
                 x_in,
                 x_out,
                 features_in,
                 bias,
                 kernels,
                 impl=None):
    if not isinstance(kernels, (list, tuple)):
        raise ValueError(
            'kernels should be a list or tuple, got {}'.format(kernels))
    dtype = features_in.dtype
    order = len(kernels)
    N_out, N_in = tf.unstack(sparse_neighbors.dense_shape)

    in_powers = [tf.ones_like(x_in), x_in]
    out_powers = [tf.ones_like(x_out), x_out]

    # create powers
    for _ in range(order - 1):
        in_powers.append(in_powers[-1] * x_in)
        out_powers.append(out_powers[-1] * x_out)

    kernel_stack = []
    x_in_stack = []
    x_out_stack = []
    binom_coeffs = []

    # bias / zeroth order
    kernel_stack.append(tf.expand_dims(bias, axis=0))
    x_in_stack.append(tf.ones((1, N_in), dtype=dtype))
    x_out_stack.append(tf.ones((1, N_out), dtype=dtype))
    for ii, kernel in enumerate(kernels):
        i = ii + 1
        binom_coeffs = _binom_row(binom_coeffs)
        for j, c in enumerate(binom_coeffs):
            coeff = -c if j % 2 == 1 else c
            if coeff == 1:
                kernel_stack.append(kernel)
            else:
                kernel_stack.append(coeff * kernel)
            x_in_stack.append(in_powers[i - j])
            x_out_stack.append(out_powers[j])
    kernel = tf.concat(kernel_stack, axis=0)
    x_in = tf.concat(x_in_stack, axis=0)
    x_out = tf.concat(x_out_stack, axis=0)

    return combined_term(sparse_neighbors,
                         x_in,
                         x_out,
                         features_in,
                         kernel,
                         impl=impl)


def get_data(
        n_in,
        n_out,
        mean_edges=10,
        f_in=16,
        f_out=32,
        dims=3,
        order=2,
        seed=123,
):
    """
    Returns:
        sparse_indices: [e, 2] int sparse indices for [n_out, n_in] tensor.
        sparse_weights: [e] float corresponding to sparse_indices.
        features: [n_in, f_in] float features.
        x_in: [n_in, dims] float coords.
        x_out: [n_out, dims] float coords.
        bias: [f_in, f_out] kernel bias
        kernel: list of length order [dims, f_in, f_out] float kernel weights.
    """
    import numpy as np
    r = np.random.RandomState(seed)

    num_edges = int(mean_edges * n_out)

    flat_index = r.randint(0, high=n_in * n_out, size=num_edges, dtype=np.int64)
    flat_index = np.unique(flat_index)
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(flat_index, (n_out, n_in))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = np.stack((i, j), axis=-1)
    sparse_weights = r.uniform(size=(i.shape[0],)).astype(np.float32)
    features = r.uniform(size=(n_in, f_in)).astype(np.float32)
    x_in = r.uniform(size=(dims, n_in)).astype(np.float32)
    x_out = r.uniform(size=(dims, n_out)).astype(np.float32)
    bias = r.uniform(size=(f_in, f_out)).astype(np.float32)
    kernels = tuple(
        r.uniform(size=(dims, f_in, f_out)).astype(np.float32)
        for _ in range(order))

    return sparse_indices, sparse_weights, features, x_in, x_out, bias, kernels


def get_tf_data(n_in, n_out, **kwargs):
    (
        sparse_indices,
        sparse_weights,
        features_in,
        x_in,
        x_out,
        bias,
        kernels,
    ) = get_data(n_in, n_out, **kwargs)
    sparse_indices = tf.constant(sparse_indices, dtype=tf.int64)
    sparse_weights = tf.constant(sparse_weights, dtype=tf.float32)
    features_in = tf.constant(features_in, dtype=tf.float32)
    x_in = tf.constant(x_in, tf.float32)
    x_out = tf.constant(x_out, tf.float32)
    kernels = tuple(tf.constant(k, dtype=tf.float32) for k in kernels)
    bias = tf.constant(bias, dtype=tf.float32)
    dense_shape = (n_out, n_in)
    # dense_shape = (tf.shape(x_out)[0], tf.shape(x_in)[0])
    sparse_neighbors = tf.SparseTensor(sparse_indices, sparse_weights,
                                       dense_shape)
    norm_factor = sparse_reduce_sum(sparse_neighbors)
    # norm_factor = tf.expand_dims(norm_factor, axis=-1)
    # sparse_neighbors = sparse_neighbors / norm_factor
    sparse_weights = sparse_weights / tf.gather(norm_factor,
                                                sparse_neighbors.indices[:, 0])
    sparse_neighbors = tf.SparseTensor(sparse_indices, sparse_weights,
                                       dense_shape)
    sparse_neighbors = tf.sparse.reorder(sparse_neighbors)  # necessary?
    return sparse_neighbors, x_in, x_out, features_in, bias, kernels


def run_benchmarks(names, fns, args, grad_args, run_name=None):
    if run_name is None:
        run_name = 'BENCHMARKS'
    vals = []
    grads = []
    for fn in fns:
        val = fn(*args)
        grads.append(tf.gradients(val, grad_args))
        vals.append(val)

    errs = [tf.reduce_max(tf.abs(v - vals[0])) for v in vals[1:]]
    with tf.Session() as sess:

        times = []
        memories = []

        for name, val, gs in zip(names, vals, grads):
            bm = tf.test.Benchmark()
            print('---------------')
            print(name)
            print('---------------')
            result = bm.run_op_benchmark(sess, (val, gs))
            times.append(result['wall_time'])
            memories.append(
                result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])

        err_vals = sess.run(errs)
        print('err relative to {}'.format(names[0]))
        for name, err in zip(names[1:], err_vals):
            print('{}: {}'.format(name, err))

    print('*************')
    print('** {} **'.format(run_name))
    print('*************')

    times = np.array(times)
    ti = np.argmin(times)
    tmin = times[ti]
    print('Best time: {}, {}s'.format(names[ti], tmin))
    print('rel times:')
    for name, time in zip(names, times):
        print('{:15s} {:.3f}'.format(name, time / tmin))
    memories = np.array(memories)
    mi = np.argmin(memories)
    mmin = memories[mi]
    print('Best memory: {}, {} Mb'.format(names[mi], mmin / 1024**2))
    for name, memory in zip(names, memories):
        print('{:15s} {:.3f}'.format(name, memory / mmin))


def run_conv_benchmarks(n_in, n_out, skip_naive=False, run_name=None, **kwargs):
    import functools
    args = get_tf_data(n_in, n_out, **kwargs)
    grad_args = tf.nest.flatten(args[1:])
    names, fns = ([], []) if skip_naive else (['naive'], [naive_conv])

    names_, fns_ = zip(*(
        ('v1', functools.partial(stacked_conv, impl=combined_term_v1)),
        ('v2', functools.partial(stacked_conv, impl=combined_term_v2)),
    ))
    names.extend(names_)
    fns.extend(fns_)
    run_benchmarks(names, fns, args, grad_args, run_name=run_name)


def run_multi_benchmark(n, f, repeats=5, impl=None, run_name=None, **kwargs):
    args = get_tf_data(n_in=n, n_out=n, f_in=f, f_out=f, **kwargs)
    grad_args = tf.nest.flatten(args[1:])
    sparse_neighbors, x_in, x_out, features_in, bias, kernels = args
    all_features = []
    for _ in range(repeats):
        all_features.append(features_in)
        features_in = stacked_conv(sparse_neighbors,
                                   x_in,
                                   x_out,
                                   features_in,
                                   bias,
                                   kernels,
                                   impl=impl)

    grad = tf.gradients(features_in, grad_args)
    f1 = all_features[1]
    g1 = tf.gradients(f1, grad_args)

    with tf.Session() as sess:
        bm = tf.test.Benchmark()
        result1 = bm.run_op_benchmark(sess, (f1, g1))
        result = bm.run_op_benchmark(sess, (features_in, grad))
    if run_name is None:
        run_name = 'run_multi_benchmark'
    print('**** {} ****'.format(run_name))
    print('{:15s}: {} s'.format('time', result['wall_time']))
    print('{:15s}: {} mb'.format(
        'memory',
        result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'] / 1024**2))
    print('multi-benchmark with n = {}, f = {}, repeats = {}'.format(
        n, f, repeats))
    mean_time = (result['wall_time'] - result1['wall_time']) / (repeats - 1)
    print('Mean wall time: {} ms'.format(mean_time * 1000))
    size = result['extras']['allocator_maximum_num_bytes_GPU_0_bfc']
    size1 = result1['extras']['allocator_maximum_num_bytes_GPU_0_bfc']
    mean_size = (size - size1) / ((repeats - 1))
    print('Mean memory   : {} Mb'.format(mean_size / 1024**2))
    return result


if __name__ == '__main__':
    from absl import app
    from absl import flags

    flags.DEFINE_bool('naive', default=False, help='run naive implementation')
    flags.DEFINE_string('params', default='small', help='param set keys')
    flags.DEFINE_integer('batch_size', default=16, help='batch size')
    flags.DEFINE_bool('multi', default=False, help='run multi')
    flags.DEFINE_string('impl', default=None, help='v1 or v2')
    flags.DEFINE_integer('order', default=2, help='kernel order')

    FLAGS = flags.FLAGS

    def main(argv):
        tf.compat.v1.enable_v2_tensorshape()
        batch_size = FLAGS.batch_size
        param_sets = dict(
            small=dict(
                n_in=12,
                n_out=123,
                f_in=4,
                f_out=5,
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

        # k = 'small'
        # k = 'paper'
        # k = 'in_place'
        # k = 'in_place2'
        # k = 'down_sample'
        # k = 'up_sample'

        k = FLAGS.params
        fn = run_multi_benchmark if FLAGS.multi else run_conv_benchmarks
        kwargs = param_sets[k]
        if FLAGS.multi:
            kwargs['n'] = kwargs.pop('n_in')
            del kwargs['n_out']
            kwargs['f'] = kwargs.pop('f_in')
            del kwargs['f_out']

            if FLAGS.impl is not None:
                kwargs['impl'] = {
                    'v1': combined_term_v1,
                    'v2': combined_term_v2
                }[FLAGS.impl]
        else:
            kwargs['skip_naive'] = not FLAGS.naive
        kwargs['order'] = FLAGS.order
        fn(**kwargs, run_name=k)

        # run_conv_benchmarks(**param_sets[k],
        #                     run_name=k,
        #                     skip_naive=not FLAGS.naive)

        # run_multi_benchmark(10000 * batch_size, 32, repeats=11)
        # run_multi_benchmark(4096 * 8, 64, mean_edges=9, repeats=11)

    app.run(main)
