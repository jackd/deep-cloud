from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from deep_cloud.model_builder import PipelineBuilder
from deep_cloud.models.sparse import builder
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import core
from more_keras.ragged.np_impl import RaggedArray

tree_impl = pykd.KDTree


def sort_ragged(ra):
    rl = ra.ragged_lists
    for r in rl:
        r.sort()
    return RaggedArray.from_ragged_lists(rl, dtype=ra.dtype)


def _get_output(pipeline, base_dataset, batch_size=2):
    dataset = base_dataset.map(pipeline.pre_batch_map)
    dataset = dataset.batch(2)
    model = pipeline.trained_model
    # for example in dataset:
    #     batched = pipeline.post_batch_map(example)
    #     model(batched)
    #     break
    # exit()
    dataset = dataset.map(pipeline.post_batch_map)
    if tf.executing_eagerly():
        for example in dataset:
            out = model(example)
            return out
        raise RuntimeError('dataset must have at least 1 entry')
    else:
        example = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        out = model(example)
        return out


class SparseBuilderTest(tf.test.TestCase):

    def test_cloud_coords(self):
        pipeline = PipelineBuilder()
        py_func = pipeline.py_func_builder('pre_batch')

        coords_np = (np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                              dtype=np.float32),
                     np.array([[0.5, 0, 0], [0.5, 0, 1]], dtype=np.float32))
        inp = pipeline.pre_batch_input(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
        coords_pf = py_func.input_node(inp)
        cloud = builder.Cloud(pipeline, py_func, coords_pf)
        trained_coords = cloud.trained_coords.flat_values
        pipeline.trained_output(trained_coords)
        pipeline.finalize()

        def gen():
            return coords_np

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (None, 3))
        actual = self.evaluate(_get_output(pipeline, dataset, batch_size=2))
        expected = np.concatenate(coords_np, axis=0)

        np.testing.assert_equal(expected, actual)

    def test_cloud_sample(self):
        pipeline = PipelineBuilder()
        py_func = pipeline.py_func_builder('pre_batch')

        coords_np = (np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                              dtype=np.float32),
                     np.array([[0.5, 0, 0], [0.5, 0, 1]], dtype=np.float32))
        sample_indices = np.array([0, 2], np.int64), np.array([1],
                                                              dtype=np.int64)

        coords_inp = pipeline.pre_batch_input(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
        sample_indices_inp = pipeline.pre_batch_input(
            tf.TensorSpec(shape=(None,), dtype=tf.int64))
        coords_pf = py_func.input_node(coords_inp)
        sample_indices_pf = py_func.input_node(sample_indices_inp)

        cloud = builder.Cloud(pipeline, py_func, coords_pf)
        sampled = cloud.sample(sample_indices_pf)

        trained_coords = sampled.trained_coords
        pipeline.trained_output(trained_coords.flat_values)
        pipeline.finalize()

        def gen():
            return zip(coords_np, sample_indices)

        dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int64),
                                                 ((None, 3), (None,)))
        actual = self.evaluate(_get_output(pipeline, dataset, batch_size=2))
        expected = np.array([coords_np[0][0], coords_np[0][2], coords_np[1][1]])
        np.testing.assert_equal(expected, actual)

    def test_in_place_neighborhood(self):
        pipeline = PipelineBuilder()
        py_func = pipeline.py_func_builder('pre_batch')

        coords_np = (np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                              dtype=np.float32),
                     np.array([[0.5, 0, 0], [0.5, 0, 1]], dtype=np.float32))

        coords_inp = pipeline.pre_batch_input(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

        coords_pf = py_func.input_node(coords_inp)
        cloud = builder.Cloud(pipeline, py_func, coords_pf)
        neigh_indices_pf = py_func.node(
            lambda tree: sort_ragged(
                tree.query_ball_point(tree.data, 1.1, approx_neighbors=3)),
            cloud.tree)

        in_place = cloud.in_place_neighborhood(
            neigh_indices_pf, lambda x: tf.transpose(x, (1, 0)),
            lambda x: tf.ones(tf.shape(x)[0], dtype=x.dtype))

        pipeline.trained_output(in_place.edge_features)
        pipeline.trained_output(in_place.weighted_edge_features)
        pipeline.trained_output(in_place.sparse_indices)
        pipeline.finalize()

        def gen():
            yield coords_np[0]
            yield coords_np[1]

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (None, 3))

        edge_features, normalized_edge_features, sparse_indices = _get_output(
            pipeline, dataset, batch_size=2)

        # sparse_indices = _get_output(pipeline, dataset, batch_size=2)

        edge_features_expected = np.array([
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 0],
        ],
                                          dtype=np.float32).T

        normalized_edge_features_expected = edge_features_expected.copy()
        normalized_edge_features_expected[:, :2] /= 2
        normalized_edge_features_expected[:, 2:5] /= 3
        normalized_edge_features_expected[:, 5:] /= 2

        sparse_indices_expected = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [3, 3],
             [3, 4], [4, 3], [4, 4]],
            dtype=np.int64)

        np.testing.assert_allclose(self.evaluate(sparse_indices),
                                   sparse_indices_expected)
        np.testing.assert_allclose(self.evaluate(edge_features),
                                   edge_features_expected)
        np.testing.assert_allclose(self.evaluate(normalized_edge_features),
                                   normalized_edge_features_expected)

    def test_sampled_neighborhood(self):
        pipeline = PipelineBuilder()
        py_func = pipeline.py_func_builder('pre_batch')

        coords_np = (np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                              dtype=np.float32),
                     np.array([[0.5, 0, 0], [0.5, 0, 1]], dtype=np.float32))
        samples_np = (np.array([1, 2],
                               dtype=np.int64), np.array([1], dtype=np.int64))

        coords_inp = pipeline.pre_batch_input(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
        samples_inp = pipeline.pre_batch_input(
            tf.TensorSpec(shape=(None,), dtype=tf.int64))

        coords_pf = py_func.input_node(coords_inp)
        samples_pf = py_func.input_node(samples_inp)
        in_cloud = builder.Cloud(pipeline, py_func, coords_pf)
        out_cloud = in_cloud.sample(samples_pf)
        neigh_indices_pf = py_func.node(
            lambda tree, coords: sort_ragged(
                tree.query_ball_point(coords, 1.1, approx_neighbors=3)),
            in_cloud.tree, out_cloud.coords_pf)

        neighborhood = in_cloud.neighborhood(
            out_cloud, neigh_indices_pf, lambda x: tf.transpose(x, (1, 0)),
            lambda x: tf.ones(tf.shape(x)[0], dtype=x.dtype))

        edge_features = neighborhood.edge_features
        normalized_edge_features = neighborhood.weighted_edge_features
        pipeline.trained_output(edge_features)
        pipeline.trained_output(normalized_edge_features)
        pipeline.trained_output(neighborhood.sparse_indices)
        pipeline.trained_output(pipeline.trained_input(in_cloud.row_starts))
        pipeline.finalize()

        def gen():
            yield coords_np[0], samples_np[0]
            yield coords_np[1], samples_np[1]

        dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int64),
                                                 ((None, 3), (None,)))

        (edge_features, normalized_edge_features, sparse_indices,
         row_starts) = _get_output(pipeline, dataset, batch_size=2)

        edge_features_expected = np.array(
            [[0, 0, 1], [0, 0, 0], [0, 0, -1], [0, 0, 1], [0, 0, 0], [0, 0, 1],
             [0, 0, 0]],
            dtype=np.float32).T

        normalized_edge_features_expected = edge_features_expected.copy()
        normalized_edge_features_expected[:, :3] /= 3
        normalized_edge_features_expected[:, 3:5] /= 2
        normalized_edge_features_expected[:, 5:7] /= 2

        sparse_indices_expected = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [1, 2],
            [2, 3],
            [2, 4],
        ],
                                           dtype=np.int64)
        np.testing.assert_equal(self.evaluate(row_starts), [0, 3])
        np.testing.assert_allclose(self.evaluate(sparse_indices),
                                   sparse_indices_expected)
        np.testing.assert_allclose(self.evaluate(edge_features),
                                   edge_features_expected)
        np.testing.assert_allclose(self.evaluate(normalized_edge_features),
                                   normalized_edge_features_expected)

    def test_neighborhood_transpose(self):
        pl = PipelineBuilder()
        py_func = pl.py_func_builder('pre_batch')

        c0_inp = pl.pre_batch_input(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

        c1_inp = pl.pre_batch_input(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

        c0_pf = py_func.input_node(c0_inp)
        c1_pf = py_func.input_node(c1_inp)
        c0 = builder.Cloud(pl, py_func, c0_pf)
        c1 = builder.Cloud(pl, py_func, c1_pf)

        neigh_indices_pf = py_func.node(
            lambda tree, coords: sort_ragged(
                tree.query_ball_point(coords, 1.1, approx_neighbors=3)),
            c0.tree, c1.coords_pf)

        rev_neigh_indices_pf = py_func.node(
            lambda tree, coords: sort_ragged(
                tree.query_ball_point(coords, 1.1, approx_neighbors=3)),
            c1.tree, c0.coords_pf)

        n0 = c0.neighborhood(c1, neigh_indices_pf,
                             lambda x: tf.transpose(x, (1, 0)),
                             lambda x: tf.ones(tf.shape(x)[0], dtype=x.dtype))
        n0t = n0.transpose
        n1 = c1.neighborhood(c0, rev_neigh_indices_pf,
                             lambda x: tf.transpose(-x, (1, 0)),
                             lambda x: tf.ones(tf.shape(x)[0], dtype=x.dtype))

        pl.trained_output(n0t.weighted_edge_features)
        pl.trained_output(n1.weighted_edge_features)
        # pl.trained_output(tf.shape(n0.sparse_indices))
        # pl.trained_output(tf.shape(n0.edge_features))
        # pl.trained_output(tf.shape(n0.edge_weights))
        # pl.trained_output(c0.trained_total_size)
        # pl.trained_output(c1.trained_total_size)
        pl.finalize()

        c0_np = (np.random.uniform(size=(5, 3)).astype(np.float32),
                 np.random.uniform(size=(2, 3)).astype(np.float32))
        c1_np = (np.random.uniform(size=(3, 3)).astype(np.float32),
                 np.random.uniform(size=(5, 3)).astype(np.float32))

        def gen():
            return zip(c0_np, c1_np)

        dataset = tf.data.Dataset.from_generator(gen, (tf.float32,) * 2,
                                                 ((None, 3),) * 2)
        out = self.evaluate(_get_output(pl, dataset, batch_size=2))
        # for o in out:
        #     print(o)

        base, rev = out
        np.testing.assert_allclose(base, rev)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.enable_v2_tensorshape()
    tf.test.main()
    # SparseBuilderTest().test_cloud_coords()
    # SparseBuilderTest().test_in_place_neighborhood()
    # SparseBuilderTest().test_neighborhood_transpose()
    # print('good')
