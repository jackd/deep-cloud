from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import test_util
from more_keras.test_utils import RaggedTestCase
from more_keras.meta_models import builder as b
from deep_cloud.layers import query as q
from deep_cloud import neigh as n


# @test_util.run_all_in_graph_and_eager_modes
class NeighborhoodTest(RaggedTestCase):
    # class NeighborhoodTest(object):

    def test_sampled_consistent(self):
        with b.MetaNetworkBuilder():
            self._test_sampled_consistent()

    def _test_sampled_consistent(self):
        batch_size = 3

        with tf.device('/cpu:0'):

            def generator():
                data = [
                    np.random.uniform(size=(100, 3)).astype(np.float32),
                    np.random.uniform(size=(200, 3)).astype(np.float32),
                    np.random.uniform(size=(50, 3)).astype(np.float32),
                ]
                labels = np.array([0, 1, 2], dtype=np.int64)
                indices = [
                    np.array([0, 5, 2, 7, 10], dtype=np.int64),
                    np.array([1, 4, 3, 2], dtype=np.int64),
                    np.array([10, 15, 3], dtype=np.int64),
                ]
                yield (data[0], indices[0]), labels[0]
                yield (data[1], indices[1]), labels[1]
                yield (data[2], indices[2]), labels[2]

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=((tf.float32, tf.int64), tf.int64),
                output_shapes=((tf.TensorShape(
                    (None, 3)), tf.TensorShape((None,))), tf.TensorShape(())))

            coords = b.prebatch_input((None, 3), tf.float32)
            sample_indices = b.prebatch_input((None,), dtype=tf.int64)
            labels = b.prebatch_input((), dtype=tf.int64)
            neighbors = q.query_pairs(coords, 0.1)
            in_place_neighborhood = n.InPlaceNeighborhood(coords, neighbors)
            sampled_neighborhood = n.SampledNeighborhood(
                in_place_neighborhood, sample_indices)

            simple_neighborhood = n.Neighborhood(
                sampled_neighborhood.in_coords, sampled_neighborhood.out_coords,
                sampled_neighborhood.neighbors)

            simple_obn = simple_neighborhood.offset_batched_neighbors
            sampled_obn = sampled_neighborhood.offset_batched_neighbors
            out = [[o.flat_values, o.nested_row_splits]
                   for o in (simple_obn, sampled_obn)]

            # out = in_place_neighborhood.offset_batched_neighbors.flat_values
            # out = b.as_batched_model_input(coords).flat_values
            flat_out = tf.nest.flatten(out)

            model = b.model(flat_out)
            preprocessor = b.preprocessor(b.batched(labels))

            dataset = preprocessor.map_and_batch(dataset, batch_size)
            # if tf.executing_eagerly():
            #     for data, label in dataset.take(1):
            #         pass
            # else:
            data, label = tf.compat.v1.data.make_one_shot_iterator(
                dataset).get_next()
            del label

            flat_out = model(tf.nest.flatten(data))
            if not isinstance(flat_out, (list, tuple)):
                flat_out = flat_out,
            out = tf.nest.pack_sequence_as(out, flat_out)

            simple_obn, sampled_obn = (
                tf.RaggedTensor.from_nested_row_splits(*o) for o in out)
            simple_obn, sampled_obn = self.evaluate((simple_obn, sampled_obn))
            self.assertRaggedEqual(simple_obn, sampled_obn)

    def test_transposed_consistent(self):
        with b.MetaNetworkBuilder():
            self._test_transposed_consistent()

    def _test_transposed_consistent(self):
        batch_size = 3

        with tf.device('/cpu:0'):
            np_data = [
                np.random.uniform(size=(100, 3)).astype(np.float32),
                np.random.uniform(size=(200, 3)).astype(np.float32),
                np.random.uniform(size=(50, 3)).astype(np.float32),
            ]

            def generator():

                labels = np.array([0, 1, 2], dtype=np.int64)
                indices = [
                    np.array([0, 5, 2, 7, 10], dtype=np.int64),
                    np.array([1, 4, 3, 2], dtype=np.int64),
                    np.array([10, 15], dtype=np.int64),
                ]
                yield (np_data[0], indices[0]), labels[0]
                yield (np_data[1], indices[1]), labels[1]
                yield (np_data[2], indices[2]), labels[2]

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=((tf.float32, tf.int64), tf.int64),
                output_shapes=((tf.TensorShape(
                    (None, 3)), tf.TensorShape((None,))), tf.TensorShape(())))

            coords = b.prebatch_input((None, 3), tf.float32)
            sample_indices = b.prebatch_input((None,), dtype=tf.int64)
            labels = b.prebatch_input((), dtype=tf.int64)
            neighbors = q.query_pairs(coords, 0.1)
            in_place_neighborhood = n.InPlaceNeighborhood(coords, neighbors)
            sampled_neighborhood = n.SampledNeighborhood(
                in_place_neighborhood, sample_indices)

            transposed = sampled_neighborhood.transpose
            simple = n.Neighborhood(sampled_neighborhood.out_coords,
                                    sampled_neighborhood.in_coords,
                                    transposed.neighbors)

            simple_obn = simple.offset_batched_neighbors
            trans_obn = transposed.offset_batched_neighbors
            simple_out = b.as_batched_model_input(simple.in_coords)
            trans_out = b.as_batched_model_input(transposed.in_coords)

            out = [[o.flat_values, o.nested_row_splits]
                   for o in (simple_obn, trans_obn, simple_out, trans_out)]
            flat_out = tf.nest.flatten(out)

            model = b.model(flat_out)
            preprocessor = b.preprocessor(b.batched(labels))

            dataset = preprocessor.map_and_batch(dataset, batch_size)
            # if tf.executing_eagerly():
            #     for data, label in dataset.take(1):
            #         pass
            # else:
            data, label = tf.compat.v1.data.make_one_shot_iterator(
                dataset).get_next()

            flat_out = model(tf.nest.flatten(data))
            if not isinstance(flat_out, (list, tuple)):
                flat_out = flat_out,
            out = tf.nest.pack_sequence_as(out, flat_out)

            out = [tf.RaggedTensor.from_nested_row_splits(*o) for o in out]
            out, label = self.evaluate((out, label))
            simple_obn, trans_obn, simple_coords, trans_coords = out
            self.assertRaggedEqual(simple_obn, trans_obn)
            self.assertRaggedEqual(simple_coords, trans_coords)
            self.assertEqual(simple_obn.nested_row_splits[-1][-1],
                             sum(d.shape[0] for d in np_data))


if __name__ == '__main__':
    tf.test.main()
    # NeighborhoodTest().test_sampled_consistent()
