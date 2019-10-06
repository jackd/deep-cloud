"""
matvec should be better, but issues
https://github.com/tensorflow/tensorflow/issues/32877
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _check_kernel_shape(x, features_in, kernel):
    D = x.shape[-1]
    F_in = features_in.shape[1]
    kernel_shape = kernel.shape
    F_out = kernel.shape[2]
    if not (kernel_shape.ndims == 3 and kernel.shape[0] == F_in and
            kernel.shape[1] == D):
        raise ValueError(
            'kernel input shape must be [F_in, D, F_out] = {}, got {}'.format(
                [F_in, D, F_out], kernel_shape))
    return F_in, D, F_out


def sparse_dense_matmul(sp, tensor):
    return tf.sparse.sparse_dense_matmul(sp, tensor)
    # return tf.matmul(tf.sparse.to_dense(sp), tensor, a_is_sparse=True)


def segment_sum(values, segments):
    return tf.math.unsorted_segment_sum(values, segments,
                                        tf.reduce_max(segments) + 1)


def sparse_reduce_sum(sp, axis=-1):
    assert (axis in (0, 1, -1))
    assert (sp.indices.shape[-1] == 2)
    segments = sp.indices[:, 0 if axis in (1, -1) else 1]
    return segment_sum(sp.values, segments)


def naive_conv_np(neighbor_vals,
                  neighbor_indices,
                  x_in,
                  x_out,
                  features_in,
                  kernel,
                  bias=None):
    F_in, D, F_out = kernel.shape
    i, j = neighbor_indices.T
    dx = x_out[i] - x_in[j]
    assert (bias is None)
    kfj = np.reshape(
        np.matmul(features_in, np.reshape(kernel, (F_in, D * F_out))),
        (-1, D, F_out))
    kfj = kfj[j]
    kfjx = np.sum(kfj * np.expand_dims(dx, axis=-1), axis=-2)
    # return kfjx
    edge_values = np.expand_dims(neighbor_vals, axis=-1) * kfjx
    out = np.zeros(shape=(np.max(i) + 1, F_out), dtype=np.float32)
    for index, value in zip(i, edge_values):
        out[index] += value
    return out


def naive_conv(sparse_neighbors, x_in, x_out, features_in, kernel, bias=None):
    """
    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of weighted values.
            Rows should sum to 1 (though this is not ef_inorced). Assumed to be
            ordered (if not, use `tf.sparse.reorder(sparse_neighbors)`).
        x_in: [N_in, D] float tensor of input cloud coordinates.
        x_out: [N_out, D] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        kernel: [F_in, D, F_out] float tensor of kernel.

    Returns:
        [N_out, F_out] float output features.
    """

    F_in, D, F_out = _check_kernel_shape(x_in, features_in, kernel)
    i, j = tf.unstack(sparse_neighbors.indices, axis=-1)
    dx = tf.gather(x_out, i) - tf.gather(x_in, j)
    if bias is not None:
        D += 1
        kernel = tf.concat([kernel, tf.expand_dims(bias, axis=1)], axis=1)
        dx = tf.concat([dx, tf.ones(tf.shape(dx)[0], 1)], axis=1)

    kfj = tf.reshape(
        tf.matmul(features_in, tf.reshape(kernel, (F_in, D * F_out))),
        (-1, D, F_out))
    kfj = tf.gather(kfj, j)
    # kfjx = tf.linalg.matvec(kfj, dx, transpose_a=True)
    kfjx = tf.reduce_sum(kfj * tf.expand_dims(dx, axis=-1), axis=-2)
    edge_values = tf.expand_dims(sparse_neighbors.values, axis=-1) * kfjx

    out = segment_sum(edge_values, i)
    return out


def _merge_kernel_and_bias(x_out, kernel, bias):
    if bias is not None:
        x_out = tf.concat([x_out, tf.ones(tf.shape(x_out)[0], 1)], axis=-1)
        kernel = tf.concat([kernel, tf.expand_dims(bias, axis=1)], axis=1)
    return x_out, kernel


def out_term_v1(sparse_neighbors, x_out, features_in, kernel, bias=None):
    """
    Implements first term of convolution.

    term_{ip} = \\sum_{k,q,j} n_{ij} (theta_{qkp} xout_{ik} + b_{qp}) f_{jq}
              = \\sum_k xout_{ik} (\\sum_q theta_{qkp}) \\sum_j n_{ij} f_{jq}

    Largest tensor: [N, D + 1, F_out]. Good when F_in < F_out.

    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of normalized
            weighted values.
        x_out: [N_out, D] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        kernel: [F_in, D, F_out] float tensor corresponding to theta.
        bias: [F_in, F_out] flota tensor corresponding to b.

    Returns:
        [N_out, F_out] float output features.
    """
    x_out, kernel = _merge_kernel_and_bias(x_out, kernel, bias)
    F_in, D, F_out = _check_kernel_shape(x_out, features_in, kernel)
    f = sparse_dense_matmul(sparse_neighbors, features_in)  # N_out, F_in

    kf = tf.matmul(f, tf.reshape(kernel, (F_in, D * F_out)))
    return tf.linalg.matvec(tf.reshape(kf, (-1, D, F_out)),
                            x_out,
                            transpose_a=True)


def out_term_v2(sparse_neighbors, x_out, features_in, kernel, bias=None):
    """
    Implements x_out term of convolution.

    term_{ip} = \\sum_{k,q,j} n_{ij} (theta_{qkp} xout_{ik} + b_{qp}) f_{jq}
              = \\sum_k cxout_{ik} \\sum_j n_{ij} \\sum_q ctheta_{qkp} f_{jq}

    where cxout is xout padded with an extra 1 and ctheta_qkp is theta
    concatenated with b along dimension 1.

    Largest tensor: [N, D + 1, F_out]. Good when F_out < F_in.

    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of normalized
            weighted values.
        x_out: [N_out, D] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        kernel: [F_in, D, F_out] float tensor corresponding to theta.
        bias: [F_in, F_out] float tensor corresponding to b.

    Returns:
        [N_out, F_out] float output features.
    """
    x_out, kernel = _merge_kernel_and_bias(x_out, kernel, bias)
    F_in, D, F_out = _check_kernel_shape(x_out, features_in, kernel)
    fk = tf.matmul(features_in, tf.reshape(kernel, (F_in, D * F_out)))
    fk = tf.sparse.sparse_dense_matmul(sparse_neighbors, fk)
    fk = tf.reshape(fk, (-1, D, F_out))
    return tf.linalg.matvec(fk, x_out, transpose_a=True)


def in_term_v1(sparse_neighbors, x_in, features_in, kernel):
    """
    Implements x_in term of convolution.

    term_{ip} = \\sum_{k,q,j} n_{ij} theta_{pqk} xin_{jk} f_{jq}
              = \\sum_{q, k} theta_{qkp} \\sum_j n_{ij} f_{jq} x_{jk}

    Largest tensor: [max(N_in, N_out), D, F_in]. Good if F_in < F_out.

    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of normalized
            weighted values.
        x_in: [N_in, D] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        kernel: [F_in, D, F_out] float tensor corresponding to theta.

    Returns:
        [N_out, F_out] float output features.
    """
    F_in, D, F_out = _check_kernel_shape(x_in, features_in, kernel)
    # fx = tf.einsum('jq,jk->jqk',features_in, x_in)
    fx = tf.expand_dims(features_in, axis=-1) * tf.expand_dims(x_in, axis=-2)
    assert (fx.shape[1:] == (F_in, D))
    fx = tf.reshape(fx, (-1, F_in * D))
    fx = sparse_dense_matmul(sparse_neighbors, fx)  # [N_in, F_in * D]
    return tf.matmul(fx, tf.reshape(kernel, (F_in * D, F_out)))


def in_term_v2(sparse_neighbors, x_in, features_in, kernel):
    """
    Implements x_in term of convolution.

    term_{ip} = \\sum_{k,q,j} n_{ij} theta_{pqk} xin_{jk} f_{jq}
              = \\sum_j n_{ij} \\sum_{k} xin_{jk} \\sum_{q} theta_{pqk} f_{jq}

    Largest tensor: [N, D, F_out]. Good if F_out < F_in.

    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of normalized
            weighted values.
        x_in: [N_in, D] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        kernel: [F_in, D, F_out] float tensor corresponding to theta.

    Returns:
        [N_out, F_out] float output features.
    """
    F_in, D, F_out = _check_kernel_shape(x_in, features_in, kernel)
    kf_flat = tf.matmul(features_in, tf.reshape(kernel, (F_in, D * F_out)))
    kf = tf.reshape(kf_flat, (-1, D, F_out))
    kfx = tf.linalg.matvec(kf, x_in, transpose_a=True)
    return sparse_dense_matmul(sparse_neighbors, kfx)


def in_term_v3(sparse_neighbors, x_in, features_in, kernel):
    """
    Implements x_in term of convolution.

    term_{ip} = \\sum_{k,q,j} n_{ij} theta_{pqk} xin_{jk} f_{jq}
              = \\sum_j n_{ij} \\sum_{q} f_{jq} \\sum_{k} theta_{pqk} xin_{jk}

    Largest tensor: [N, D, F_in]. Good if F_in or F_out < D (unlikely?).

    Args:
        sparse_neighbors: [N_out, N_in] sparse float tensor of normalized
            weighted values.
        x_in: [N_in, D] float tensor of output cloud coordinates.
        features_in: [N_in, F_in] float tensor of input features.
        kernel: [F_in, D, F_out] float tensor corresponding to theta.

    Returns:
        [N_out, F_out] float output features.
    """
    F_in, D, F_out = _check_kernel_shape(x_in, features_in, kernel)
    del F_in, D
    kx = tf.einsum('nd,qdp->nqp', x_in, kernel)
    fkx = tf.linalg.matvec(kx, features_in, transpose_a=True)
    assert (fkx.shape[1] == F_out)
    return sparse_dense_matmul(sparse_neighbors, fkx)


def better_conv(sparse_neighbors,
                x_in,
                x_out,
                features_in,
                kernel,
                bias=None,
                in_impl=None,
                out_impl=None):
    F_in, D, F_out = kernel.shape
    if bias is not None:
        if bias.shape != (F_in, F_out):
            raise ValueError(
                'kernel and bias shapes incompatible, {} and {}'.format(
                    kernel.shape, bias.shape))
    for x, name in ((x_in, 'x_in'), (x_out, 'x_out')):
        if x.shape[-1] != D:
            raise ValueError(
                '{} and kernel shapes incompatible {} and {}'.format(
                    name, x.shape, kernel.shape))
    if features_in.shape[-1] != F_in:
        raise ValueError(
            'features_in shape {} not consistent with kernel shape {}'.format(
                features_in.shape, kernel.shape))

    # choose sensible default implementations based on static sizes.
    if in_impl is None:
        in_impl = in_term_v2
    if out_impl is None:
        out_impl = out_term_v1
    in_term = in_impl(sparse_neighbors, x_in, features_in, kernel)
    out_term = out_impl(sparse_neighbors, x_out, features_in, kernel, bias)
    return out_term - in_term


def get_data(n_in, n_out, mean_edges=10, f_in=16, f_out=32, seed=123, dims=3):
    """
    Returns:
        sparse_indices: [e, 2] int sparse indices for [n_out, n_in] tensor.
        sparse_weights: [e] float corresponding to sparse_indices.
        features: [n_in, f_in] float features.
        x_in: [n_in, dims] float coords.
        x_out: [n_out, dims] float coords.
        kernel: [f_in, dims, f_out] float kernel weights.
    """
    r = np.random.RandomState(seed)

    num_edges = int(mean_edges * n_out)

    # flat_index = r.choice(n_in * n_out, size=num_edges,
    #                       replace=False).astype(np.int64)
    flat_index = r.randint(0, high=n_in * n_out, size=num_edges, dtype=np.int64)
    flat_index = np.unique(flat_index)
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(flat_index, (n_out, n_in))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = np.stack((i, j), axis=-1)
    sparse_weights = r.uniform(size=(i.shape[0],)).astype(np.float32)
    features = r.uniform(size=(n_in, f_in)).astype(np.float32)
    x_in = r.uniform(size=(n_in, dims)).astype(np.float32)
    x_out = r.uniform(size=(n_out, dims)).astype(np.float32)
    kernel = r.uniform(size=(f_in, dims, f_out)).astype(np.float32)

    return sparse_indices, sparse_weights, features, x_in, x_out, kernel


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
    print('Best memory: {}, {}mb'.format(names[mi], mmin / 1024**2))
    for name, memory in zip(names, memories):
        print('{:15s} {:.3f}'.format(name, memory / mmin))


def get_tf_data(n_in, n_out, **kwargs):
    (sparse_indices, sparse_weights, features_in, x_in, x_out,
     kernel) = get_data(n_in, n_out, **kwargs)
    sparse_indices = tf.constant(sparse_indices, dtype=tf.int64)
    sparse_weights = tf.constant(sparse_weights, dtype=tf.float32)
    features_in = tf.constant(features_in, dtype=tf.float32)
    x_in = tf.constant(x_in, tf.float32)
    x_out = tf.constant(x_out, tf.float32)
    kernel = tf.constant(kernel, dtype=tf.float32)
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

    return sparse_neighbors, x_in, x_out, features_in, kernel


def run_in_term_benchmarks(n_in, n_out, **kwargs):
    sparse_neighbors, x_in, x_out, features_in, kernel = get_tf_data(
        n_in, n_out, **kwargs)
    del x_out
    names = ('v1', 'v2', 'v3')
    fns = (in_term_v1, in_term_v2, in_term_v3)
    args = sparse_neighbors, x_in, features_in, kernel
    grad_args = args[1:]
    run_benchmarks(names, fns, args, grad_args, 'in_term')


def run_out_term_benchmarks(n_in, n_out, **kwargs):
    sparse_neighbors, x_in, x_out, features_in, kernel = get_tf_data(
        n_in, n_out, **kwargs)
    del x_in

    names = ('v1', 'v2')
    fns = (out_term_v1, out_term_v2)
    args = sparse_neighbors, x_out, features_in, kernel
    grad_args = args[1:]
    run_benchmarks(names, fns, args, grad_args, 'out_term')


def run_conv_benchmarks(n_in, n_out, skip_naive=False, run_name=None, **kwargs):
    import functools
    args = get_tf_data(n_in, n_out, **kwargs)
    grad_args = args[1:]
    names, fns = ([], []) if skip_naive else (['naive'], [naive_conv])

    out_impls = [out_term_v1, out_term_v2]
    in_impls = [in_term_v1, in_term_v2, in_term_v3]
    for i, out_impl in enumerate(out_impls):
        for j, in_impl in enumerate(in_impls):
            names.append('better_{}{}'.format(i + 1, j + 1))
            fns.append(
                functools.partial(better_conv,
                                  in_impl=in_impl,
                                  out_impl=out_impl))
    run_benchmarks(names, fns, args, grad_args, run_name=run_name)


def run_multi_benchmark(n, f, repeats=5, **kwargs):
    sparse_neighbors, x_in, x_out, features_in, kernel = get_tf_data(n_in=n,
                                                                     n_out=n,
                                                                     f_in=f,
                                                                     f_out=f,
                                                                     **kwargs)
    for _ in range(repeats):
        features_in = better_conv(sparse_neighbors, x_in, x_out, features_in,
                                  kernel)
    grad, = tf.gradients(features_in, kernel)

    with tf.Session() as sess:
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(sess, (features_in, grad))
    print('{:15s}: {}s'.format('time', result['wall_time']))
    print('{:15s}: {}mb'.format(
        'memory',
        result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'] / 1024**2))
    print('multi-benchmark with n = {}, f = {}, repeats = {}'.format(
        n, f, repeats))


# batch_size = 16

# in_place = dict(
#     n_in=10000 * batch_size,
#     n_out=10000 * batch_size,
#     f_in=32,
#     f_out=32,
#     mean_edges=12,
# )
# small = dict(
#     n_in=1234,
#     n_out=123,
#     f_in=16,
#     f_out=32,
# )
# # tf.compat.v1.enable_eager_execution()
# sparse_neighbors, x_in, x_out, features_in, kernel = get_tf_data(**in_place)
# tf_val = naive_conv(sparse_neighbors, x_in, x_out, features_in, kernel)
# with tf.Session() as sess:
#     args, tf_val = sess.run([(sparse_neighbors.values, sparse_neighbors.indices,
#                               x_in, x_out, features_in, kernel), tf_val])
# np_val = naive_conv_np(*args)

# print(np.max(np.abs(np_val - tf_val)))
# exit()

if __name__ == '__main__':
    from absl import app
    from absl import flags

    flags.DEFINE_bool('naive', default=False, help='run naive implementation')
    flags.DEFINE_string('params', default='small', help='param set keys')
    flags.DEFINE_integer('batch_size', default=16, help='batch size')
    FLAGS = flags.FLAGS

    def main(argv):
        tf.compat.v1.enable_v2_tensorshape()
        batch_size = FLAGS.batch_size
        param_sets = dict(
            small=dict(
                n_in=1234,
                n_out=123,
                f_in=16,
                f_out=32,
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

        # run_out_term_benchmarks(**param_sets[k])
        # run_in_term_benchmarks(**param_sets[k])
        k = FLAGS.params
        run_conv_benchmarks(**param_sets[k],
                            run_name=k,
                            skip_naive=not FLAGS.naive)

        # run_multi_benchmark(10000 * batch_size, 32, repeats=11)
        # run_multi_benchmark(4096 * 8, 64, mean_edges=9, repeats=11)

    app.run(main)
