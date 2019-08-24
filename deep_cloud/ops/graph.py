raise NotImplementedError('Might come back to this...')

# """Graph representation using tf.RaggedTensor."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import tensorflow as tf

# class RaggedGraph(object):

#     def __init__(self,
#                  row_splits,
#                  flat_edge_indices,
#                  flat_edge_features,
#                  flat_rev_edge_features=None,
#                  is_symmetric=None):
#         """
#         `flat_*` inputs should have the same ragged
#         structure. To enforce this, we only accept regular tensors
#         along with the ragged structure in `row_splits`. The ragged versions
#         are
#         ```python
#         edge_indices = tf.ragged.RaggedTensor.from_row_splits(
#             flat_edge_indices, row_splits)
#         edge_features = tf.ragged.RaggedTensor.from_row_splits(
#             flat_edge_features, row_splits)
#         rev_edge_features = tf.ragged.RaggedTensor.from_row_splits(
#             flat_rev_edge_features, row_splits)
#         ```

#         After raggedifying, `edge_indices[i] == [p, q, r]` indicates that
#         node `i` is connected to node `p`, `q` and `r`. `edge_features[i, 1]`
#         would correspond to the features associated with the edge between nodes
#         `i` and `q`.

#         In the following, `n` denotes the number of nodes in the graph, `e`
#         the number of edges and `f` the number of features. We use `?` to denote
#         a dimension that can vary across examples (even within the same batch).

#         Args:
#             row_splits: (n?,) int tensor denoting the start/end index of each
#                 row of flat tensors.
#             flat_edge_indices: (e?,) int tensor of indices of each edge.
#                 edge_indices[i] == [p, q, r] indicates node `i` is connected to
#                 nodes `p`, `q` and `r`. Converting this to a sparse matrix with
#                 uniform values should give a symmetric matrix
#                 (though this is not tested).
#             flat_edge_features: (e?, f) float tensor of edge features.
#             rev_flat_edge_features: (e?, f) float tensor of edge
#                 features in reverse graph. Calculated as the reordered version
#                 of `flat_edge_features` if not provided.
#             is_symmetric: if True, `edge_indices` is assumed to
#                 correspond to a symmetric graph (though this isn't tested).
#         """
#         self._row_splits = row_splits
#         self._flat_edge_indices = flat_edge_indices
#         self._flat_edge_features = flat_edge_features
#         self._flat_rev_edge_features = flat_rev_edge_features
#         if is_symmetric:
#             self._transpose = self
#         self._is_symmetric = is_symmetric

#     @property
#     def is_symmetric(self):
#         """
#         Flag whether or not this graph is symmetric.

#         None indicates this wasn't specified in the constructor, so it may be,
#         but shouldn't be assumed to be.
#         """
#         return self._is_symmetric

#     @property
#     def row_splits(self):
#         return self._row_splits

#     @property
#     def flat_edge_indices(self):
#         return self._flat_edge_indices

#     @property
#     def flat_edge_features(self):
#         return self._flat_edge_features

#     @property
#     def edge_indices(self):
#         if not hasattr(self, '_edge_indices'):
#             self._edge_indices = tf.RaggedTensor.from_row_splits(
#                 self._flat_edge_indices, self._row_splits)
#         return self._edge_indices

#     @property
#     def edge_features(self):
#         if not hasattr(self, '_edge_features'):
#             self._edge_features = tf.RaggedTensor.from_row_splits(
#                 self._flat_edge_features, self._row_splits)
#         return self._edge_features

#     @property
#     def transpose(self):
#         if not hasattr(self, '_transpose'):
#             if self._flat_rev_edge_features is not None:
#                 rev_indices = query.reverse_query_pairs(self.edge_indices,
#                                                         size=self.size)
#             self._transpose._transpose = self

#         return self._transpose
