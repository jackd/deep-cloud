from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin

from more_keras.ops import polynomials
from more_keras import layers
from deep_cloud.model_builder import PipelineBuilder
from deep_cloud.model_builder import PipelineModels
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import core

from deep_cloud.models.sparse import layers as sparse_layers
from typing import Any, Optional
DEFAULT_TREE = pykd.KDTree

SQRT_2 = np.sqrt(2.)


class memoized_property(property):  # pylint: disable=invalid-name
    """Descriptor that mimics @property but caches output in member variable."""

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        # See https://docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            # cached = self.fget(obj)
            cached = super(memoized_property, self).__get__(obj, objtype)
            setattr(obj, attr, cached)
        return cached


def row_normalize(weights, row_indices):
    row_sums = tf.math.segment_sum(weights, row_indices)
    row_sums = tf.gather(row_sums, row_indices)
    return weights / row_sums


def _ragged_csr_to_sparse_indices(args):

    ragged_csr = tf.RaggedTensor.from_nested_row_splits(args[0], args[1:])
    assert (isinstance(ragged_csr, tf.RaggedTensor))
    i = tf.repeat(tf.range(ragged_csr.nrows(out_type=tf.int64)),
                  ragged_csr.row_lengths(),
                  axis=0)
    j = ragged_csr.values
    return tf.stack((i, j), axis=-1)


def ragged_csr_to_sparse_indices(ragged_csr):
    if ragged_csr.shape.ndims != 2:
        raise ValueError('ragged_csr must be rank 2 but got shape {}'.format(
            ragged_csr.shape))
    if isinstance(ragged_csr, tf.RaggedTensor):
        return tf.keras.layers.Lambda(_ragged_csr_to_sparse_indices)(
            [ragged_csr.values, *ragged_csr.nested_row_splits])
    else:
        n, m = tf.unstack(tf.shape(ragged_csr, out_type=tf.int64))
        i = tf.tile(tf.expand_dims(tf.range(n)[0], axis=-1), (1, m))
        j = tf.reshape(ragged_csr, (-1,))
        return tf.stack((i, j), axis=-1)


def _get_rel_coords(in_coords, out_coords, neigh_indices):
    return (np.repeat(out_coords, neigh_indices.row_lengths, axis=0) -
            in_coords[neigh_indices.flat_values])


def _from_row_splits(args):
    return tf.RaggedTensor.from_row_splits(*args)


def _edge_features(pipeline, py_func, in_coords_pf, out_coords_pf,
                   neigh_indices_pf, edge_features_fn, edge_weights_fn):
    rel_coords = py_func.node(_get_rel_coords, in_coords_pf, out_coords_pf,
                              neigh_indices_pf)
    rel_coords = py_func.output_tensor(
        rel_coords, tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
    rel_coords = pipeline.trained_input(
        pipeline.batch(rel_coords, ragged=True).values)
    edge_features = edge_features_fn(rel_coords)
    edge_weights = edge_weights_fn(rel_coords)
    return edge_features, edge_weights


@gin.configurable
class Cloud(object):

    def __init__(self,
                 pipeline,
                 py_func,
                 coords_pf,
                 coords=None,
                 size=None,
                 tree_impl=DEFAULT_TREE,
                 input_cloud=None,
                 sample_indices=None):
        """
        Args:
            pipeline: PipelineBuilder instance.
            py_func: PyFuncBuilder instance.
            coords_pf:
                pre-batch py_func node corresponding to coordinates, (N, 3),
            coords: optional tensor correspondign to coords_pf.
            size: int scalar of single size.
            tree_impl: KD-tree implementation.
            input_cloud: for sampled clouds, this is the original cloud from
                which this cloud is taken.
            sample_indices: for sampled clouds, these are the indices of the
                original cloud from which this cloud is taken.
        """
        self._pipeline = pipeline
        self._py_func = py_func
        self._coords_pf = coords_pf
        self._tree_impl = tree_impl
        self.__cached_size = size
        self.__cached_coords = coords
        self._input_cloud = input_cloud
        self._sample_indices = sample_indices

    def sample(self, indices_pf):
        coords_pf = self._py_func.node(lambda c, i: c[i], self._coords_pf,
                                       indices_pf)
        sample_indices = self._py_func.output_tensor(
            indices_pf, tf.TensorSpec(shape=(None,), dtype=tf.int64))
        return Cloud(self._pipeline,
                     self._py_func,
                     coords_pf,
                     input_cloud=self,
                     sample_indices=sample_indices)

    @property
    def input_cloud(self):
        return self._input_cloud

    @property
    def sample_indices(self):
        """Indices of input_cloud making up this cloud - possibly `None`."""
        return self._sample_indices

    @property
    def coords_pf(self):
        return self._coords_pf

    @memoized_property
    def tree(self):
        return self._py_func.node(self._tree_impl, self.coords_pf)

    @memoized_property
    def coords(self):
        return self._py_func.output_tensor(
            self._coords_pf, tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

    @memoized_property
    def batched_coords(self):
        return self._pipeline.batch(self.coords, ragged=True)

    @memoized_property
    def trained_coords(self):
        return self._pipeline.trained_input(self.batched_coords)

    @memoized_property
    def global_sparse_indices(self):
        coords: tf.RaggedTensor = self.batched_coords
        i = tf.repeat(tf.range(coords.nrows(out_type=tf.int64)),
                      coords.row_lengths())
        j = tf.ragged.range(coords.row_lengths())
        j = j + tf.expand_dims(coords.row_starts())
        sparse_indices = tf.stack((i, j), axis=-1)
        return sparse_indices

    @memoized_property
    def trained_global_sparse_indices(self):
        return self._pipeline.trained_input(self.global_sparse_indices)

    @memoized_property
    def size(self):
        return self._py_func.output_tensor(
            self._py_func.node(lambda x: x.shape[0], self._coords_pf),
            tf.TensorSpec(shape=(), dtype=tf.int64))

    @memoized_property
    def row_lengths(self):
        return self._pipeline.batch(self.size)

    @memoized_property
    def row_splits(self):
        return tf.pad(tf.cumsum(self.row_lengths), [[1, 0]])

    @memoized_property
    def _row_starts_and_total(self):
        row_starts, total = tf.split(self.row_splits, (-1, 1), axis=0)
        total_size = tf.squeeze(total, axis=0)
        return row_starts, total_size

    @property
    def row_starts(self):
        return self._row_starts_and_total[0]

    @property
    def total_size(self):
        return self._row_starts_and_total[1]

    @memoized_property
    def trained_total_size(self):
        return self._pipeline.trained_input(self.total_size)

    def block_diagonalize(self, indices):
        """
        Get ragged CSR indices corresponding to block diagonalized tensor.

        The adjacency matrix is defined implicitly as
        ```
        adjacency[i, j] == j in indices[i]
        ```
        For general sparse adjacency matrix, `indices` is a ragged tensor, but
        can be a regular tensor if each neighborhood is the same size (e.g. when
        using k-nearest neighbors) and the number of points in each batch
        element is the same. We call this 'ragged CSR'.

        Args:
            indices: [B, N?, k?] possibly ragged tensor of indices of this
                this cloud. `indices[b, n]` is the set of neighbors of the `n`th
                point in the `b`th batch element.

        Returns:
            [T, k?] possibly ragged tensor representing block-diagonalized
                indices. `t`th row corresponds to the set of neighbors of the
                `t`th point in the entire batch.
        """
        if indices.shape.ndims != 3:
            raise ValueError('indices must be rank 3, got shape {}'.format(
                indices.shape))
        if isinstance(indices, tf.RaggedTensor):
            components = tf.keras.layers.Lambda(
                lambda i: [i.flat_values, *i.nested_row_splits])(indices)

            def f(args):
                flat_values, *nested_splits, row_starts = args
                ind = tf.RaggedTensor.from_nested_row_splits(
                    flat_values, nested_splits)
                out = ind + tf.reshape(row_starts, (-1, 1, 1))
                return [out.flat_values, *out.nested_row_splits]

            flat_values, *nested_splits = tf.keras.layers.Lambda(f)(
                [*components, self.row_starts])
            if len(nested_splits) == 2:
                return tf.keras.layers.Lambda(
                    lambda args: tf.RaggedTensor.from_row_splits(*args))(
                        [flat_values, nested_splits[1]])
            else:
                return flat_values
        else:
            indices = tf.keras.layers.Lambda(
                lambda args: args[0] + tf.reshape(args[1], (-1, 1, 1)))(
                    [indices, self.row_starts])
            return tf.reshape(indices, (-1, tf.shape(indices)[2]))
        # if isinstance(indices, tf.RaggedTensor):
        #     row_starts = self.row_starts
        #     i = indices
        #     while isinstance(i, tf.RaggedTensor):
        #         row_starts = tf.repeat(row_starts, i.row_lengths(), axis=0)
        #         i = i.values

        #     if indices.ragged_rank == 1:
        #         return tf.reshape(i + tf.expand_dims(row_starts, axis=-1),
        #                           (-1, tf.shape(indices)[2]))
        #     else:
        #         assert (indices.ragged_rank == 2)
        #         return tf.RaggedTensor.from_row_splits(
        #             i + row_starts, indices.nested_row_splits[0])
        # else:
        #     indices = tf.keras.layers.Lambda(
        #         lambda args: args[0] + tf.reshape(args[1], (-1, 1, 1)))(
        #             [indices, self.row_starts])
        #     return tf.reshape(indices, (-1, tf.shape(indices)[2]))

    def block_sparse_indices(self, indices_pf):
        """
        Convert ragged CSR example indices to batched sparse indices.

        Args:
            indices_pf: PyFuncNode corresponding to per-example ragged CSR
                indices of this cloud.

        Returns:
            [E, 2] int64 tensor of block-diagonalized batch sparse indices.
        """
        flat_neigh_indices = self._py_func.node(lambda ni: ni.flat_values,
                                                indices_pf)
        flat_neigh_indices = self._py_func.output_tensor(
            flat_neigh_indices, tf.TensorSpec(shape=(None,), dtype=tf.int64))
        row_splits = self._py_func.node(lambda ni: ni.row_splits, indices_pf)
        row_splits = self._py_func.output_tensor(
            row_splits, tf.TensorSpec(shape=(None,), dtype=tf.int64))

        neigh_indices = tf.keras.layers.Lambda(_from_row_splits)(
            [flat_neigh_indices, row_splits])
        neigh_indices = self._pipeline.batch(neigh_indices)
        block_indices = self.block_diagonalize(neigh_indices)
        sparse_indices = ragged_csr_to_sparse_indices(block_indices)
        return sparse_indices

    def in_place_neighborhood(self, neigh_indices_pf, edge_features_fn,
                              edge_weights_fn):
        sparse_indices = self.block_sparse_indices(neigh_indices_pf)
        sparse_indices = self._pipeline.trained_input(sparse_indices)
        edge_features, edge_weights = _edge_features(
            self._pipeline, self._py_func, self.coords_pf, self.coords_pf,
            neigh_indices_pf, edge_features_fn, edge_weights_fn)
        return Neighborhood(self._pipeline, self, self, sparse_indices,
                            edge_features, edge_weights)

    def neighborhood(self, out_cloud, neigh_indices_pf, edge_features_fn,
                     edge_weights_fn):
        sparse_indices = self.block_sparse_indices(neigh_indices_pf)
        sparse_indices = self._pipeline.trained_input(sparse_indices)
        edge_features, edge_weights = _edge_features(
            self._pipeline, self._py_func, self.coords_pf, out_cloud.coords_pf,
            neigh_indices_pf, edge_features_fn, edge_weights_fn)
        return Neighborhood(self._pipeline, self, out_cloud, sparse_indices,
                            edge_features, edge_weights)

    def global_conv(self, node_features, edge_features_fn, edge_weights_fn,
                    **layer_kwargs):
        coords = tf.transpose(self.trained_coords, (1, 0))
        edge_features = edge_features_fn(coords)
        edge_weights = edge_weights_fn(coords)
        sparse_indices = self.trained_global_sparse_indices
        edge_weights = row_normalize(edge_weights,
                                     tf.gather(sparse_indices, 0, axis=1))
        edge_features = edge_features * tf.expand_dims(edge_weights, axis=0)
        return sparse_layers.SparseCloudConvolution(**layer_kwargs)([
            node_features, edge_features, sparse_indices,
            self.trained_coords.nrows(out_type=tf.int64)
        ])


def _sparse_transpose(args):
    indices, values, n, m, weights = args

    # transpose a sparse tensor with range values and gather.
    val_indices = tf.range(tf.shape(weights, out_type=tf.int64)[0])
    sp = tf.SparseTensor(indices, val_indices, dense_shape=(n, m))
    sp = tf.sparse.transpose(sp, (1, 0))
    sp = tf.sparse.reorder(sp)
    return [
        sp.indices,
        tf.gather(values, sp.values, axis=1),
        tf.gather(weights, sp.values)
    ]


class Neighborhood(object):

    def __init__(self,
                 pipeline,
                 in_cloud,
                 out_cloud,
                 sparse_indices,
                 edge_features,
                 edge_weights,
                 transpose=None):
        """
        Args:
            pipeline: PipelineBuilder instance.
            in_cloud: input Cloud instance.
            out_cloud: output Cloud instance.
            sparse_indices: [E, 2] int64 train model block-diagonalized sparse
                indices.
            edge_features: [T, E] float train model edge features
            edge_weights: [E] float weights.
            transpose: (optional) transposed neighborhood.
        """

        self._pipeline = pipeline
        self._in_cloud = in_cloud
        self._out_cloud = out_cloud
        self._sparse_indices = sparse_indices
        self._edge_features = edge_features
        self._edge_weights = edge_weights
        for name, x in (
            ('sparse_indices', sparse_indices),
            ('edge_features', edge_features),
            ('edge_weights', edge_weights),
        ):
            mark = self._pipeline.propagate_marks(x)
            if mark != PipelineModels.TRAINED:
                raise ValueError('{} must be marked with {} but got {}'.format(
                    name, PipelineModels.TRAINED, mark))
        if in_cloud is out_cloud:
            if transpose is not None:
                raise ValueError('transpose must be None if symmetric is True')
            self.__cached_transpose = self

    @property
    def in_cloud(self):
        return self._in_cloud

    @property
    def out_cloud(self):
        return self._out_cloud

    @memoized_property
    def transpose(self):
        if self._in_cloud is self._out_cloud:
            return self
        sparse_indices, edge_features, edge_weights = tf.keras.layers.Lambda(
            _sparse_transpose)([
                self.sparse_indices,
                self.edge_features,
                self._out_cloud.trained_total_size,
                self._in_cloud.trained_total_size,
                self.edge_weights,
            ])
        return Neighborhood(self._pipeline,
                            self._out_cloud,
                            self._in_cloud,
                            sparse_indices,
                            edge_features,
                            edge_weights,
                            transpose=self)

    @property
    def sparse_indices(self):
        return self._sparse_indices

    @property
    def edge_features(self):
        return self._edge_features

    @property
    def edge_weights(self):
        return self._edge_weights

    @memoized_property
    def normalized_edge_weights(self):
        return row_normalize(self.edge_weights,
                             tf.gather(self.sparse_indices, 0, axis=1))

    @memoized_property
    def weighted_edge_features(self):
        return self.edge_features * tf.expand_dims(self.normalized_edge_weights,
                                                   axis=0)

    def conv(self, node_features, **layer_kwargs):
        edge_features = self.weighted_edge_features

        if node_features is None:
            layer = sparse_layers.FeaturelessSparseCloudConvolution(
                **layer_kwargs)
            return layer([edge_features, self.sparse_indices])
        else:
            layer = sparse_layers.SparseCloudConvolution(**layer_kwargs)
            return layer([
                node_features, edge_features, self.sparse_indices,
                self._out_cloud.trained_total_size
            ])


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def classifier_pipeline(input_spec,
                        output_spec,
                        depth=4,
                        tree_impl=DEFAULT_TREE,
                        k0=16):
    pipeline = PipelineBuilder()
    py_func = pipeline.py_func_builder('pre_batch')
    assert (isinstance(input_spec, tf.TensorSpec))
    coords = pipeline.pre_batch_input(input_spec)

    coords_pf = py_func.input_node(coords)

    logits = None
    pipeline.trained_output(logits)

    pipeline.finalize()
    return pipeline
