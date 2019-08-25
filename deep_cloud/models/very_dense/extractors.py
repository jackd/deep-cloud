from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from more_keras.layers import Dense
from more_keras.layers import utils as layer_utils
from deep_cloud.ops.asserts import assert_flat_tensor, INT_TYPES, FLOAT_TYPES
from deep_cloud.layers import edge
from deep_cloud.models.very_dense import utils


def block_variance_scaling_initializers(fan_ins,
                                        fan_out,
                                        scale=1.0,
                                        mode="fan_in",
                                        distribution="truncated_normal",
                                        seed=None):
    """
    Get initializers for block-dense layers.

    Example usage.
    ```python
    def block_dense(inputs, units):
        fan_ins = [inp.shape[-1] for inp in inputs]
        initializers = variance_scaling_initializers(fan_ins)
        layers = [tf.keras.layers.Dense(units, kernel_initializer=init)
                  for init in initializers]]
        outputs = [layer(inp) for layer, inp in zip(layers, inputs)]
        # you might want to do something with the split outputs here.
        return tf.keras.layers.Lambda(tf.math.add_n)(outputs)
    ```

    Args:
        fan_ins: tuple of ints/dimensions indicating the fan_in for each block.
        fan_out: number of units in the output layer.
        scale:
        mode, distribution, seed: see tf.keras.initializers.VarianceScaling

    Returns:
        tuple of `tf.keras.initializers.VarianceScalingInitializer`s with scale
        modified such that the resulting distribution would be as if `fan_in`
        was actually `sum(fan_ins)`.
    """
    if not isinstance(fan_ins, tuple):
        raise ValueError('fan_ins must be a tuple, got {}'.format(fan_ins))
    total_fan_in = sum(fan_ins)
    total_fan_in = getattr(total_fan_in, 'value', total_fan_in)  # TF-COMPAT
    kwargs = dict(mode=mode, distribution=distribution, seed=seed)

    def scale_scale(fan_in):
        if mode == 'fan_in':
            return max(1., fan_in)
        elif mode == 'fan_out':
            return 1
        else:
            return max(1., (fan_in + fan_out) / 2)

    scale /= scale_scale(total_fan_in)
    return tuple(
        tf.keras.initializers.VarianceScaling(scale=scale * scale_scale(fan_in),
                                              **kwargs) for fan_in in fan_ins)


class BiPartiteFeatureExtractor(object):
    """
    Class for managing feature extraction over the edges of a bipartite graph.

    Args:
        flat_node_indices: [n_e] int tensor of indices defining
            connections between the disjoint sets. Combined with
            `row_splits` to form a ragged tensor, `edge_indices[i] == p, q, r`
            indicates that node `i` from set `b` is connected to nodes
            `p`, `q` and `r` from set `a`.
        row_splits: ragged row splits for `flat_node_indices`.
        size: size of set `a`.
        initial_units: number of features node features and edge features are
            mapped to before adding them up and passing into the
            `edge_network_fn`.
        edge_network_fn: dense network that operates on each edge independently
            (except for possibly batch-norm?). Input is the sum of
        edge_reduction_fn: function mapping edge_features to a pair of node
            feature tensors. See `deep_cloud.ops.edge`.
        dense_factory: `Dense` layer implementation. Called with `units` and
        `kernel_regularizer` kwargs.
    """

    def __init__(self,
                 flat_node_indices,
                 row_splits,
                 size,
                 initial_units,
                 edge_network_fn,
                 edge_reduction_fn,
                 dense_factory=None):
        if dense_factory is None:
            dense_factory = Dense
        assert_flat_tensor('flat_node_indices',
                           flat_node_indices,
                           1,
                           dtype=INT_TYPES)
        assert_flat_tensor('row_splits', row_splits, 1, dtype=INT_TYPES)
        assert_flat_tensor('size', size, 0, dtype=INT_TYPES)
        self.flat_node_indices = flat_node_indices
        self.row_splits = row_splits
        self.size = size
        if not isinstance(initial_units, int):
            raise ValueError(
                'initial_units must be an int, got {}'.format(initial_units))
        self.initial_units = initial_units
        if not callable(edge_network_fn):
            raise ValueError('edge_network_fn must be callable, got {}'.format(
                edge_network_fn))
        self.edge_network_fn = edge_network_fn
        if not callable(edge_reduction_fn):
            raise ValueError(
                'edge_reduction_fn must be callable, got {}'.format(
                    edge_reduction_fn))
        self.edge_reduction_fn = edge_reduction_fn
        if not callable(dense_factory):
            raise ValueError(
                'dense_factory must be callable, got {}'.format(dense_factory))
        self._dense_factory = dense_factory

    def _block_dense(self, *features):
        fan_ins = [f.shape[-1] for f in features]
        initializers = block_variance_scaling_initializers(
            fan_ins, self.initial_units)
        return [
            self._dense_factory(self.initial_units, kernel_initializer=init)(f)
            for init, f, in zip(initializers, features)
        ]

    def _prepare_edge_features(self,
                               node_features_a,
                               node_features_b,
                               edge_features,
                               symmetric=False):
        assert_flat_tensor('edge_features', edge_features, 2, FLOAT_TYPES)
        if node_features_a is None:
            if node_features_b is not None:
                raise NotImplementedError
            # no node features - just use edge features
            return self._dense_factory(
                self.initial_units,
                kernel_initializer=tf.keras.initializers.VarianceScaling())(
                    edge_features)

        # we do have node features
        assert_flat_tensor('node_features_a', node_features_a, 2, FLOAT_TYPES)
        if symmetric:
            # ignore b
            node_features_a, edge_features = self._block_dense(
                node_features_a, edge_features)
            node_features_b = node_features_a
        else:
            assert_flat_tensor('node_features_b', node_features_b, 2,
                               FLOAT_TYPES)
            node_features_a, node_features_b, edge_features = self._block_dense(
                node_features_a, node_features_b, edge_features)

        edge_features = edge.distribute_node_features(
            node_features_a,
            node_features_b,
            edge_features,
            flat_node_indices=self.flat_node_indices,
            row_splits=self.row_splits)
        return edge_features

    def __call__(self,
                 node_features_a,
                 node_features_b,
                 edge_features,
                 symmetric=False):
        edge_features = self._prepare_edge_features(node_features_a,
                                                    node_features_b,
                                                    edge_features,
                                                    symmetric=symmetric)
        edge_features = self.edge_network_fn(edge_features)
        node_features_a, node_features_b = self.edge_reduction_fn(
            edge_features,
            self.flat_node_indices,
            self.row_splits,
            self.size,
            symmetric=symmetric)
        return node_features_a, node_features_b, edge_features


class KPartiteFeatureExtractor(object):

    def __init__(self, bipartite_extractors):
        if not isinstance(bipartite_extractors, (list, tuple)):
            raise ValueError(
                'bipartite_extractors must be a list/tuple, got {}'.format(
                    bipartite_extractors))
        for i, bs in enumerate(bipartite_extractors):
            if not isinstance(bs, (list, tuple)):
                raise ValueError(
                    'entries of bipartite_extractors must be lists/tuples, '
                    'but element {} is {}'.format(i, bs))
            if len(bs) != i:
                raise ValueError(
                    '{}th entry of bipartite_extractors should have length {} '
                    'but has length {}'.format(i, i, len(bs)))
            for j, b in enumerate(bs):
                if not isinstance(b, BiPartiteFeatureExtractor):
                    raise ValueError(
                        'All entry entries of bipartite_extractors must be a '
                        'BiPartiteFeatureExtractor, but ({}, {})th element is '
                        '{}'.format(i, j, b))
        self.bipartite_extractors = tuple(
            tuple(bs) for bs in bipartite_extractors)

    def __call__(self, node_features, edge_features):
        # input validation
        for i, efs in enumerate(edge_features):
            if not isinstance(efs, (list, tuple)):
                raise ValueError(
                    'entries of edge_features must be lists/tuples, '
                    'but element {} is {}'.format(i, efs))
            if len(efs) != i:
                raise ValueError(
                    '{}th entry of bipartite_extractors should have length {} '
                    'but has length {}'.format(i, i, len(efs)))
            for j, ef in enumerate(efs):
                assert_flat_tensor('edge_features[{}][{}]'.format(i, j), ef, 2,
                                   FLOAT_TYPES)
        extractors = self.bipartite_extractors
        K = len(extractors)
        if len(node_features) != K:
            raise ValueError('Expected {} node features, got {}'.format(
                K, len(node_features)))
        for i, nf in enumerate(node_features):
            assert_flat_tensor('node_features[{}]'.format(i), nf, 2,
                               FLOAT_TYPES)
        # -------------------------
        # finished validating inputs
        # -------------------------
        all_out_edge_features = utils.lower_triangular(K)
        all_out_node_features = [[] for __ in range(K)]
        for i in range(K):
            for j in range(i):
                af, bf, ef = extractors[i][j](node_features[i],
                                              node_features[j],
                                              edge_features[i][j])
                all_out_node_features[i].appennd(af)
                all_out_node_features[j].append(bf)
                all_out_edge_features[i][j] = ef
            # symmetric cloud
            af, bf, ef = extractors[i][i](node_features[i],
                                          None,
                                          edge_features[i][i],
                                          symmetric=True)
            assert (bf is None)
            all_out_node_features[i] = af
            all_out_edge_features[i][i] = ef

        # concatenate node features
        for i in range(K):
            all_out_node_features[i] = layer_utils.concat(
                all_out_node_features[i], axis=-1)
        return all_out_node_features, all_out_edge_features


def get_base_extractor():
    raise NotImplementedError
