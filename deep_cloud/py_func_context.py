from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class PyFuncContext(object):
    _CACHE = {}
    _MAX_INT = np.iinfo(np.int64).max

    def _get_key(self):
        from uuid import uuid4
        key = str(uuid4())
        while key in self._keys:
            key = str(uuid4())
        return key

    def __init__(self):
        self._keys = set()

        def f():
            id_ = np.random.randint(PyFuncContext._MAX_INT)
            while id_ in PyFuncContext._CACHE:
                id_ = np.random.randint(PyFuncContext._MAX_INT)
            PyFuncContext._CACHE[id_] = {}
            return id_

        self._id = tf.py_func(f, [], tf.int64)

    def py_function(self, fn, out_specs, *args, **kwargs):
        nested_args = (args, kwargs)
        flat_args = tf.nest.flatten(nested_args)
        flat_out_specs = tf.nest.flatten(out_specs)
        key_out_indices = tuple(
            i for i, spec in enumerate(flat_out_specs) if spec is None)
        keys = tuple(self._get_key() for _ in key_out_indices)
        tensor_args = [self._id]
        tensor_arg_indices = []
        key_args = []
        key_arg_indices = []

        for i, arg in enumerate(flat_args):
            if isinstance(arg, int):
                key_arg_indices.append(i)
                key_args.append(arg)
            else:
                tensor_arg_indices.append(i)
                tensor_args.append(arg)

        def load_from_cache(cache, arg):
            if arg.dtype == tf.string:
                argn = arg.numpy().decode()
                if argn in cache:
                    return cache[argn]
            return arg

        def flat_fn(*args):
            *args, id_ = args
            cache = PyFuncContext._CACHE[int(id_.numpy())]
            args = tuple(load_from_cache(cache, arg) for arg in args)
            args, kwargs = tf.nest.pack_sequence_as(nested_args, args)
            out = fn(*args, **kwargs)
            tf.nest.assert_same_structure(out, out_specs)
            out_flat = tf.nest.flatten(out)
            for key_index, i in enumerate(key_out_indices):
                key = keys[key_index]
                cache[key] = out_flat[i]
                out_flat[i] = key
            return out_flat

        out_types = tuple(tf.string if spec is None else spec.dtype
                          for spec in flat_out_specs)
        flat_args.append(self._id)
        out_vals = tf.py_func(flat_fn, flat_args, out_types)
        for val, spec in zip(out_vals, flat_out_specs):
            if spec is not None:
                val.set_shape(spec.shape)
        return tf.nest.pack_sequence_as(out_specs, out_vals)

    def clean_up(self):

        def f(id_):
            del PyFuncContext._CACHE[id_]
            return True

        return tf.py_func(f, self._id, tf.bool)


if __name__ == '__main__':

    def run_simple():
        context = PyFuncContext()
        float_spec = tf.TensorSpec(dtype=tf.float32, shape=())
        x = context.py_function(lambda: 3., None)
        y = context.py_function(lambda: 4., None)
        out = context.py_function(lambda x, y: (x, y, x * y), (float_spec,) * 3,
                                  x, y)
        with tf.Session() as sess:
            print(sess.run(out))
            print(sess.run((x, y)))

    def run_full_benchmark():
        from deep_cloud.ops.np_utils.tree_utils import pykd
        from deep_cloud.ops.np_utils.tree_utils import core

        from deep_cloud.problems.partnet import PartnetProblem
        from tqdm import tqdm
        from time import time

        depth = 6
        num_warmup = 5
        num_iterations = 10
        k0 = 16
        tree_impl = pykd.KDTree
        SQRT_2 = np.sqrt(2.)
        context = PyFuncContext()
        problem = PartnetProblem()

        def f(coords, labels):
            tree = context.py_function(tree_impl, None, coords)
            dists, indices = context.py_function(
                lambda tree: tree.query(tree.data, 2, return_distance=True),
                (tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                 tf.TensorSpec(shape=(None,), dtype=tf.int64)), tree)

            scale = tf.reduce_mean(dists[:, 1])
            coords = coords * (2. / scale)

            all_coords = [coords]
            trees = [tree]

            radii = 4 * np.power(2, np.arange(depth))

            flat_indices = []
            row_splits = []
            rel_coords = []
            sample_indices = []

            def add_conv(tree, coords, radius, k0):

                def fn(tree, coords):
                    indices = tree.query_ball_point(coords,
                                                    radius,
                                                    approx_neighbors=k0)
                    rc = np.repeat(coords, indices.row_lengths,
                                   axis=0) - coords[indices.flat_values]
                    rc /= radius
                    return flat_indices, row_splits, rel_coords

                fi, rs, rc = context.py_function(
                    fn, (tf.TensorSpec(shape=(None,), dtype=tf.int64),
                         tf.TensorSpec(shape=(None,), dtype=tf.int64),
                         tf.TensorSpec(shape=(None, 3), dtype=tf.float32)),
                    (tree, coords))
                flat_indices.append(fi)
                row_splits.append(rs)
                rel_coords.append(rc)

                # n = tree.n
                # m = coords.shape[0]
                # e = indices.row_splits[-1]
                # lines.append(str((e, n, m, e / n, e / m, radius)))
                return fi, rs

            # initial query in order to do initial rejection sample
            # indices = tree.query_ball_point(coords, radii[0], approx_neighbors=k0)
            indices = context.py_function(
                lambda tree, coords: np.array(
                    core.rejection_sample_active(tree, coords.numpy(), radii[0],
                                                 k0)),
                tf.TensorSpec(shape=(None,), dtype=tf.int64), tree, coords)
            # indices = np.array(core.rejection_sample_lazy(tree, coords, radii[0], k0))
            sample_indices.append(indices)
            out_coords = tf.gather(coords, indices)
            all_coords.append(out_coords)
            tree = context.py_function(tree_impl, None, out_coords)
            trees.append(tree)
            # initial large down-sample conv
            add_conv(tree, coords, radii[0] * 2, k0 * 4)
            coords = out_coords

            def rejection_sample(flat_indices, row_splits):
                from more_keras.ragged.np_impl import RaggedArray
                ra = RaggedArray.from_row_splits(flat_indices, row_splits)
                return np.array(core.rejection_sample_precomputed(ra),
                                dtype=np.int64)

            for i in range(1, depth - 1):
                # in place
                indices_comp = add_conv(tree, coords, radii[i], k0)
                indices = context.py_function(
                    rejection_sample,
                    tf.TensorSpec(shape=(None,), dtype=tf.int64), *indices_comp)
                sample_indices.append(indices)
                out_coords = tf.gather(coords, indices)
                all_coords.append(out_coords)
                tree = context.py_function(tree_impl, None, out_coords)
                trees.append(tree)

                # down sample
                # larger radius means number of edges remains roughly constant
                # number of ops remains constant if number of filters doubles
                # also makes more neighbors for unsampling (~4 on average vs ~2)
                add_conv(tree, coords, radii[i] * SQRT_2, k0)
                coords = out_coords

            # final in_place
            add_conv(tree, coords, radii[-1], k0)
            # lines.append('***')
            # print('\n'.join(lines))  # DEBUG
            return (
                tuple(flat_indices),
                tuple(rel_coords),
                tuple(row_splits),
                tuple(all_coords),
                tuple(sample_indices),
            )

        # out = dists, indices

        dataset = problem.get_base_dataset('validation')
        dataset = dataset.map(f, -1)

        out = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        with tf.Session() as sess:
            for _ in tqdm(range(num_warmup), desc='warming up'):
                sess.run(out)

            t = time()
            for _ in tqdm(range(num_iterations), desc='profiling'):
                sess.run(out)
            dt = time() - t
            print('Ran {} iterations in {} s: {} ms / iteration'.format(
                num_iterations, dt, dt * 1000 / num_iterations))

    run_full_benchmark()
    print('done')
