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
                    init,
                    parallel_iterations=20)


def _check_arg_shapes(sparse_neighbors, x_in, x_out, features_in, kernel):
    F_in, F_out = kernel.shape
    compat_msg = '{} and {} shapes incompatible: {} and {}'
    for (x, name) in ((x_in, 'x_in'), (x_out, 'x_out')):
        if x is None:
            continue
        if x.shape.ndims != 1:
            raise ValueError('{} must be rank 2, but shape is {}'.format(
                name, x.shape))
    if features_in.shape.ndims != 2:
        raise ValueError('features_in should be rank 2, got {}'.format(
            features_in.shape))
    if features_in.shape[1] != F_in:
        raise ValueError(
            compat_msg.format('kernel', 'features_in', kernel.shape,
                              features_in.shape))
    if x_in is not None and features_in.shape[
            0] is not None and features_in.shape[0] != x_in.shape[0]:
        raise ValueError(
            compat_msg.format('features_in', 'x_in', features_in.shape,
                              x_in.shape))

    if sparse_neighbors is None:
        assert (x_out is None)
    else:
        for n, x, name in zip(sparse_neighbors.shape, (x_out, x_in),
                              ('x_out', 'x_in')):
            if n is not None and x is not None and x.shape[0] != n:
                raise ValueError(
                    compat_msg.format('sparse_neighbors', name,
                                      sparse_neighbors.shape, x.shape))
    return F_in, F_out


def inner(sparse_neighbors, x_in, x_out, features_in, kernel, impl=None):
    """
    Inner summation with sensible default implementation.

    \\sum_{q,j} n_{ij} theta_{qp} f_{jq} xout_{i} xin_{j}

    If not provided, picks impl from `inner_downsample` or `inner_upsample`
    based on size of kernel.
    """
    if impl is None:
        F_in, F_out = _check_arg_shapes(sparse_neighbors, x_in, x_out,
                                        features_in, kernel)
        if F_in < F_out:
            impl = inner_downsample
        else:
            impl = inner_upsample
    return impl(sparse_neighbors, x_in, x_out, features_in, kernel)


def inner_downsample(sparse_neighbors, x_in, x_out, features_in, kernel):
    """
    Implements a single term in convolution with feature transform LAST.

    This version is intended for use in DOWN-sampling, i.e. where
    N_in > N_out and/or F_in < F_out.

    \\sum_{q,j} n_{ij} theta_{qp} f_{jq} xout_{i} xin_{j}
    = \\xout_{i} \\sum_q theta_{qp} \\sum_j n_{ij} f_{qj} * xin_{j}

    Args:
        sparse_neighbors: [N_out, N_in] sparse tensor
        x_in: [N_in] float tensor
        x_out: [N_out] float tensor
        features_in: [N_in, F_in] float tensor
        kernel: [F_in, F_out] float tensor

    Returns:
        [N_out, F_out] float tensor
    """
    _check_arg_shapes(sparse_neighbors, x_in, x_out, features_in, kernel)
    fx = features_in
    if x_in is not None:
        fx = fx * tf.expand_dims(x_in, axis=-1)
    fx = tf.sparse.sparse_dense_matmul(sparse_neighbors, fx)
    fx = tf.matmul(fx, kernel)
    if x_out is not None:
        fx = fx * tf.expand_dims(x_out, axis=-1)
    return fx


def inner_upsample(sparse_neighbors, x_in, x_out, features_in, kernel):
    """
    Implements a single term in convolution with feature transform FIRST.

    This version is intended for use in UP-sampling, i.e. where
    N_in > N_out and/or F_in < F_out.

    \\sum_{q,j} n_{ij} theta_{qp} f_{jq} xout_{i} xin_{j}
    = xout_{ki} \\sum_j n_{ij} xin_{j} \\sum_q theta_{qp} f_{qj}

    If sparse_neighbors is None, x_out must also be None and the sum over j
    is not performed, i.e. the result is
    \\xin_{kj} \\sum_q theta_{kqp} f_{qj}.

    Args:
        sparse_neighbors: [N_out, N_in] sparse tensor
        x_in: [N_in] float tensor
        x_out: [N_out] float tensor
        features_in: [N_in, F_in] float tensor
        kernel: [F_in, F_out] float tensor

    Returns:
        [N_out, F_out] float tensor, or [N_in, F_out] if both `x_out` and
        `sparse_neighbors` are None.
    """
    _check_arg_shapes(sparse_neighbors, x_in, x_out, features_in, kernel)
    fx = features_in
    fx = tf.matmul(fx, kernel)
    if x_in is not None:
        fx = fx * tf.expand_dims(x_in, axis=-1)
    if sparse_neighbors is None:
        assert (x_out is None)
    else:
        fx = tf.sparse.sparse_dense_matmul(sparse_neighbors, fx)
    if x_out is not None:
        fx = fx * tf.expand_dims(x_out, axis=-1)
    return fx


def _binom_row(prev_row):
    n = len(prev_row)
    out = [1]
    for i in range(n - 1):
        out.append(prev_row[i] + prev_row[i + 1])
    out.append(1)
    return out


def _get_polynomial_stacks(x_in, x_out, bias, kernels):
    kernel_stack = []
    x_in_stack = []
    x_out_stack = []
    binom_coeffs = []
    order = len(kernels)

    in_powers = [None, x_in]
    out_powers = [None, x_out]
    # create powers
    for _ in range(order - 1):
        in_powers.append(in_powers[-1] * x_in)
        out_powers.append(out_powers[-1] * x_out)

    if bias is not None:
        x_out_stack.append(
            tf.ones(shape=(1, tf.shape(x_out)[1]), dtype=x_out.dtype))
        x_in_stack.append(None)
        kernel_stack.append(tf.expand_dims(bias, axis=0))

    for ii, kernel in enumerate(kernels):
        i = ii + 1
        binom_coeffs = _binom_row(binom_coeffs)
        for j, c in enumerate(binom_coeffs):
            coeff = -c if j % 2 == 1 else c
            if coeff == 1:
                kernel_stack.append(kernel)
            else:
                kernel_stack.append(coeff * kernel)
            x_out_stack.append(out_powers[i - j])
            x_in_stack.append(in_powers[j])
    return x_in_stack, x_out_stack, kernel_stack


def manual_conv(sparse_neighbors,
                x_in,
                x_out,
                features_in,
                bias,
                kernels,
                impl=None):
    if not isinstance(kernels, (list, tuple)):
        raise ValueError(
            'kernels should be a list or tuple, got {}'.format(kernels))
    order = len(kernels)
    if order > 2:
        raise NotImplementedError()
    # bias and kernels in same loop
    F_in, F_out = bias.shape
    del F_in

    x_in_stack, x_out_stack, kernel_stack = _get_polynomial_stacks(
        x_in, x_out, bias=None, kernels=kernels)
    # merge bias into first kernel with no x_in
    terms = []
    for i, xi in enumerate(x_in_stack):
        if xi is None:
            x_out_stack[i] = tf.pad(x_out_stack[i], [[0, 1], [0, 0]],
                                    constant_values=1)
            kernel_stack[i] = tf.concat(
                [kernel_stack[i], tf.expand_dims(bias, axis=0)], axis=0)

            break
    else:
        terms.append(
            inner(sparse_neighbors,
                  None,
                  tf.ones((1, tf.shape(x_out)[1]), dtype=x_out.dtype),
                  features_in,
                  bias,
                  impl=impl))

    init = tf.zeros((tf.shape(x_out)[1], F_out), dtype=features_in.dtype)
    base_kwargs = dict(impl=impl,
                       features_in=features_in,
                       sparse_neighbors=sparse_neighbors)
    for xi, xo, k in zip(x_in_stack, x_out_stack, kernel_stack):
        args = [k]
        kwargs = base_kwargs.copy()
        names = ['kernel']
        if xi is None:
            kwargs['x_in'] = None
        else:
            args.append(xi)
            names.append('x_in')

        if xo is None:
            kwargs['x_out'] = None
        else:
            args.append(xo)
            names.append('x_out')

        if xo is None:
            kwargs['sparse_neighbors'] = None
            term = tf.foldl(
                lambda acc, args: acc + inner(**kwargs, **dict(zip(names, args))
                                             ), args, init)
            term = tf.sparse.sparse_dense_matmul(sparse_neighbors, term)
        else:
            term = tf.foldl(
                lambda acc, args: acc + inner(**kwargs, **dict(zip(names, args))
                                             ), args, init)
        terms.append(term)
    return tf.add_n(terms)


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
    N_out, _ = tf.unstack(sparse_neighbors.dense_shape)
    _, F_out = bias.shape

    x_in_stack, x_out_stack, kernel_stack, = _get_polynomial_stacks(
        x_in, x_out, bias, kernels)

    # replace `None`s with ones
    in0 = tf.ones_like(x_in)
    out0 = tf.ones_like(x_out)
    x_in_stack = [in0 if x is None else x for x in x_in_stack]
    x_out_stack = [out0 if x is None else x for x in x_out_stack]
    x_in = tf.concat(x_in_stack, axis=0)
    x_out = tf.concat(x_out_stack, axis=0)
    kernel = tf.concat(kernel_stack, axis=0)
    init = tf.zeros(shape=(N_out, F_out), dtype=features_in.dtype)
    return tf.foldl(
        lambda acc, args: inner(
            sparse_neighbors, args[0], args[1], features_in, args[
                2], impl=impl), (x_in, x_out, kernel), init)
