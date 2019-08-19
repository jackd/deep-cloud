"""
Uniform interfaces for various different types of neighborhoods.

See `Neighborhood` docstrings for method interpretations.

Neighborhood: generic neighborhood for arbitrary input and output clouds

InPlaceNeighborhood: output point cloud is the same as the input

SampledNeighborhood: output point cloud is a sampling of the input

TransposedNeighborhood: inverse neighborhood, i.e.
    TransposedNeighborhood(Neighborhood(
        in_coords, out_coords, neighbors_fn(in_coords, out_coords))) ==
    Neighborhood(
        out_coords,in_coords, neighborhood_fn(out_coords, in_coords))

In general, use neighborhood.transpose rather than TransposedNeighborhood
directly.

All input arguments should be unbatched, and all outputs other than
`offset_batched_neighbors` are also unbatched.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras.layers import utils
from more_keras.meta_models import builder as b
from deep_cloud.layers import query
from deep_cloud.layers import cloud


def _dist2(x, axis=-1, keepdims=False):
    return tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims)


class Neighborhood(object):

    def __init__(self, in_coords, out_coords, neighbors, transpose=None):
        self._in_coords = in_coords
        self._out_coords = out_coords
        self._neighbors = neighbors
        self._transpose = None
        self._rel_coords = None
        self._offset_batched_neighbors = None
        self._dist2 = None

    @property
    def in_coords(self):
        """[n_in, num_dims] float32 coordinates of input (unbatched)."""
        return self._in_coords

    @property
    def out_coords(self):
        """[n_out, num_dims] float32 coordinates of output (unbatched)."""
        return self._out_coords

    @property
    def neighbors(self):
        """[n_out, k?] int64 neighborhood indices (unbatched)."""
        return self._neighbors

    @property
    def rel_coords(self):
        """[n_out, k?, 3] float32 relative coordinates (unbatched)."""
        if self._rel_coords is None:
            self._rel_coords = cloud.get_relative_coords(
                self.in_coords, self.out_coords, self.neighbors)
        return self._rel_coords

    @property
    def dist2(self):
        """[n_out, k?] float32 squared relative distances."""
        if self._dist2 is None:
            layer = tf.keras.layers.Lambda(_dist2,
                                           arguments=dict(axis=-1,
                                                          keepdims=True))
            self._dist2 = tf.ragged.map_flat_values(layer, self.rel_coords)
        return self._dist2

    @property
    def transpose(self):
        """Get the transposed neighborhood."""
        if self.is_in_place:
            return self
        if not hasattr(self, '_transpose') or self._transpose is None:
            self._transpose = TransposedNeighborhood(self)
        return self._transpose

    @property
    def is_in_place(self):
        """True if the input coordinates are the same as output coordinates."""
        return self.in_coords is self.out_coords

    @property
    def offset_batched_neighbors(self):
        """
        [B, n?, k?] offset index into flattened input batched cloud features.

        For point cloud features of size [B, n?, ...],
        `tf.gather(features.values, offset_batched_neighbors)` will give the
        same result as
        ```
        tf.map(lambda features, batched_neighbors:
                tf.gather(features, batched_neighbors))
        ```

        The returned value is a model input.
        """
        if self._offset_batched_neighbors is None:
            row_length = utils.leading_dim(self.in_coords)
            row_lengths = b.batched(row_length)
            batched_neighbors = b.batched(self.neighbors)
            # row_lengths = b.as_batched_model_input(row_length)
            # batched_neighbors = b.as_batched_model_input(self.neighbors)
            offset = tf.keras.layers.Lambda(
                tf.math.cumsum, arguments=dict(exclusive=True))(row_lengths)
            offset_batched_neighbors = utils.apply_row_offset(
                batched_neighbors, offset)
            self._offset_batched_neighbors = b.as_model_input(
                offset_batched_neighbors)
        return self._offset_batched_neighbors


def _shape0(x):
    return tf.shape(x)[0]


class TransposedNeighborhood(Neighborhood):

    def __init__(self, base):
        self._base = base
        size = tf.keras.layers.Lambda(_shape0)(self._base.in_coords)
        self._neighbors, self._rev_indices = query.reverse_query_pairs(
            self._base.neighbors, size=size)
        self._offset_batched_neighbors = None
        self._rel_coords = None
        self._dist2 = None

    @property
    def in_coords(self):
        return self._base.out_coords

    @property
    def out_coords(self):
        return self._base.in_coords

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def rev_indices(self):
        return self._rev_indices

    def _rev_values(self, base_values):
        assert (isinstance(base_values, tf.RaggedTensor))
        assert (base_values.ragged_rank == 1)
        return tf.RaggedTensor.from_row_splits(
            utils.gather(base_values.flat_values, self.rev_indices),
            self.neighbors.row_splits)

    @property
    def rel_coords(self):
        if self._rel_coords is None:
            self._rel_coords = self._rev_values(self._base.rel_coords)
        return self._rel_coords

    @property
    def dist2(self):
        if self._dist2 is None:
            self._dist2 = self._rev_values(self._base.dist2)
        return self._dist2

    @property
    def transpose(self):
        return self._base


class InPlaceNeighborhood(Neighborhood):

    def __init__(self, coords, neighbors):
        self._coords = coords
        self._neighbors = neighbors
        self._rel_coords = None
        self._offset_batched_neighbors = None

    @property
    def in_coords(self):
        return self._coords

    @property
    def out_coords(self):
        return self._coords

    @property
    def transpose(self):
        return self

    @property
    def is_in_place(self):
        return True

    @property
    def offset_batched_neighbors(self):
        if self._offset_batched_neighbors is None:
            batched_neighbors = b.as_batched_model_input(self.neighbors)
            offset = utils.get_row_offsets(batched_neighbors)
            self._offset_batched_neighbors = utils.apply_row_offset(
                batched_neighbors, offset)
        return self._offset_batched_neighbors


class SampledNeighborhood(Neighborhood):

    def __init__(self, base, sample_indices):
        self._base = base
        self._sample_indices = sample_indices
        self._neighbors = None
        self._rel_coords = None
        self._transpose = None
        self._dist2 = None
        self._out_coords = None
        self._offset_batched_neighbors = None

    @property
    def in_coords(self):
        return self._base.out_coords

    def _gather(self, values):
        return utils.gather(values, self._sample_indices)

    @property
    def out_coords(self):
        if self._out_coords is None:
            self._out_coords = self._gather(self._base.out_coords)
        return self._out_coords

    @property
    def neighbors(self):
        if self._neighbors is None:
            self._neighbors = self._gather(self._base.neighbors)
        return self._neighbors

    @property
    def rel_coords(self):
        if self._rel_coords is None:
            self._rel_coords = self._gather(self._base.rel_coords)
        return self._rel_coords

    @property
    def dist2(self):
        if self._dist2 is None:
            self._dist2 = self._gather(self._base.dist2)
        return self._dist2

    @property
    def offset_batched_neighbors(self):
        if self._offset_batched_neighbors is None:
            batched_sample_indices = b.as_batched_model_input(
                self._sample_indices)
            self._offset_batched_neighbors = utils.map_gather(
                self._base.offset_batched_neighbors, batched_sample_indices)
        return self._offset_batched_neighbors
