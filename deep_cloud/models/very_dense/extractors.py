from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import gin
import numpy as np
import tensorflow as tf
from more_keras.layers import Dense
from more_keras.layers import utils as layer_utils
from more_keras.ops import utils as op_utils
from more_keras.models import mlp
from deep_cloud.ops.asserts import assert_flat_tensor
from deep_cloud.ops.asserts import assert_callable
from deep_cloud.ops.asserts import INT_TYPES
from deep_cloud.ops.asserts import FLOAT_TYPES
from deep_cloud.layers import edge
from deep_cloud.models.very_dense import utils

SQRT_2 = np.sqrt(2)


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
        return tf.math.add_n(outputs)
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


def block_dense(dense_factory, units, *features):
    fan_ins = tuple(f.shape[-1] for f in features)
    initializers = block_variance_scaling_initializers(fan_ins, units)
    return [
        dense_factory(units, kernel_initializer=init)(f)
        for init, f, in zip(initializers, features)
    ]


class GlobalBipartiteFeatureExtractor(object):

    def __init__(self,
                 row_splits_or_k,
                 initial_units,
                 network_fn,
                 dense_factory=Dense):
        self.is_ragged = row_splits_or_k.shape.ndims
        if self.is_ragged:
            self.row_lengths = op_utils.diff(row_splits_or_k)
        else:
            self.k = row_splits_or_k
        self.network_fn = network_fn
        self.initial_units = initial_units
        self.dense_factory = dense_factory

    def __call__(self, global_features, local_features, edge_features):
        """
        Args:
            global_features: [B, fg] float global features. Can be `None`.
            local_features: [N, fl] float flat local features, e.g. normals
            edge_features: [N, fe] float flat edge features or None.
                Since all nodes are connected here to a single 'global' node,
                this will have the same number of rows as `local_features` -
                and it is concatenated onto `local_features` immediately. It
                could be thought of differently however, e.g. as coordinates
                (possibly relative to the mean).

        Returns:
            flat local features, output of `network_fn`.
        """
        if edge_features is not None:
            if local_features is not None:
                local_features = tf.concat([local_features, edge_features],
                                           axis=-1)
            else:
                local_features = edge_features
        elif local_features is None:
            raise ValueError(
                'At least one of `local_features` or `edge_features` must be '
                'non-None')
        if global_features is None:
            local_features = self.dense_factory(
                self.initial_units)(local_features)
        else:
            local_features, global_features = block_dense(
                self.dense_factory, self.initial_units, local_features,
                global_features)

            if self.is_ragged:
                global_features = layer_utils.repeat(global_features,
                                                     self.row_lengths,
                                                     axis=0)

                local_features = tf.add_n([local_features, global_features])
            else:
                local_features = layer_utils.reshape_leading_dim(
                    local_features, (-1, self.k))
                global_features = tf.expand_dims(global_features, axis=-2)
                local_features = tf.math.add_n(
                    [local_features, global_features])
                local_features = layer_utils.flatten_leading_dims(
                    local_features)
        local_features = self.network_fn(local_features)

        return local_features


class BiPartiteFeatureExtractor(object):
    """
    Class for managing feature extraction over the edges of a bipartite graph.

    Args:
        flat_node_indices: [n_e] int tensor of indices defining
            connections between the disjoint sets. Combined with
            `row_splits` to form a ragged tensor, `edge_indices[i] == p, q, r`
            indicates that node `i` from set `a` is connected to nodes
            `p`, `q` and `r` from set `b`.
        row_splits: ragged row splits for `flat_node_indices`.
        size: size of set `b`.
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
                 dense_factory=Dense):
        assert_callable('dense_factory', dense_factory)
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
        self.dense_factory = dense_factory

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
            return self.dense_factory(
                self.initial_units,
                kernel_initializer=tf.keras.initializers.VarianceScaling())(
                    edge_features)

        # we do have node features
        assert_flat_tensor('node_features_a', node_features_a, 2, FLOAT_TYPES)
        if symmetric:
            # ignore b
            node_features_a, edge_features = block_dense(
                self.dense_factory, self.initial_units, node_features_a,
                edge_features)
            node_features_b = node_features_a
        else:
            assert_flat_tensor('node_features_b', node_features_b, 2,
                               FLOAT_TYPES)
            node_features_a, node_features_b, edge_features = block_dense(
                self.dense_factory, self.initial_units, node_features_a,
                node_features_b, edge_features)

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
        out = self.edge_reduction_fn(edge_features,
                                     self.flat_node_indices,
                                     self.row_splits,
                                     self.size,
                                     symmetric=symmetric)
        if symmetric:
            node_features_a = out
            node_features_b = None
        else:
            node_features_a, node_features_b = out
        return node_features_a, node_features_b, edge_features


class KPartiteFeatureExtractor(object):
    """
    Args:
        bipartite_extractors: [K, <=K]
            llist of `BipartiteExtractor`s or `None`s.

        global_extractors: [K]
            list of `GlobalBipartiteExtractor`s or `None`s.
    """

    def __init__(
            self,
            bipartite_extractors,
            global_extractors,
    ):
        if not isinstance(bipartite_extractors, (list, tuple)):
            raise ValueError(
                'bipartite_extractors must be a list/tuple, got {}'.format(
                    bipartite_extractors))
        for i, bs in enumerate(bipartite_extractors):
            if not isinstance(bs, (list, tuple)):
                raise ValueError(
                    'entries of bipartite_extractors must be lists/tuples, '
                    'but element {} is {}'.format(i, bs))
            if len(bs) != i + 1:
                raise ValueError(
                    'entry {} of bipartite_extractors should have length {} '
                    'but has length {}'.format(i, i + 1, len(bs)))
            for j, b in enumerate(bs):
                if not (b is None or isinstance(b, BiPartiteFeatureExtractor)):
                    raise ValueError(
                        'All entries of bipartite_extractors must be a '
                        'BiPartiteFeatureExtractor or None, but element '
                        '({}, {}) is {}'.format(i, j, b))

        # make immutable
        self.bipartite_extractors = tuple(
            tuple(e) for e in bipartite_extractors)

        if global_extractors is not None:
            if not isinstance(global_extractors, (list, tuple)):
                raise ValueError(
                    'global_extractors must be a list/tuple, got {}'.format(
                        global_extractors))
            for i, g in enumerate(global_extractors):
                if not (g is None or
                        isinstance(g, GlobalBipartiteFeatureExtractor)):
                    raise ValueError(
                        'All entries for global_extractors must be '
                        '`GlobalBipartiteFeatureExtractor`s, but entry {} is {}'
                        .format(i, g))

            if len(global_extractors) != len(bipartite_extractors):
                raise ValueError('Inconsistent lengths: {} vs {}'.format(
                    len(global_extractors), len(bipartite_extractors)))

            # make immutable
            global_extractors = tuple(global_extractors)
        self.global_extractors = global_extractors

    def __call__(
            self,
            node_features,
            edge_features,
            global_features,
            global_edge_features=None,
    ):
        # input validation
        for i, efs in enumerate(edge_features):
            if not isinstance(efs, (list, tuple)):
                raise ValueError(
                    'entries of edge_features must be lists/tuples, '
                    'but element {} is {}'.format(i, efs))
            if len(efs) != i + 1:
                raise ValueError(
                    'entry {} of bipartite_extractors should have length {} '
                    'but has length {}'.format(i, i + 1, len(efs)))
            for j, ef in enumerate(efs):
                assert_flat_tensor('edge_features[{}][{}]'.format(i, j), ef, 2,
                                   FLOAT_TYPES)
        extractors = self.bipartite_extractors
        K = len(extractors)
        if node_features is None:
            node_features = [None] * K
        elif len(node_features) != K:
            raise ValueError('Expected {} node features, got {}'.format(
                K, len(node_features)))
        for i, nf in enumerate(node_features):
            if nf is not None:
                assert_flat_tensor('node_features[{}]'.format(i), nf, 2,
                                   FLOAT_TYPES)
        # -------------------------
        # finished validating inputs
        # -------------------------
        all_out_edge_features = utils.lower_triangular(K)
        all_out_node_features = [[None] * K for _ in range(K)]
        for i in range(K):
            for j in range(i):
                if extractors[i][j] is None:
                    continue
                af, bf, ef = extractors[i][j](node_features[j],
                                              node_features[i],
                                              edge_features[i][j])
                all_out_node_features[i][j] = bf
                all_out_node_features[j][i] = af
                all_out_edge_features[i][j] = ef
            # symmetric cloud
            af, bf, ef = extractors[i][i](node_features[i],
                                          None,
                                          edge_features[i][i],
                                          symmetric=True)
            assert (bf is None)
            all_out_node_features[i][i] = af
            all_out_edge_features[i][i] = ef

        # add global features
        if self.global_extractors is not None:
            for i, extractor in enumerate(self.global_extractors):
                if extractor is not None:
                    ef = (None if global_edge_features is None else
                          global_edge_features[i])
                    all_out_node_features[i].append(
                        extractor(global_features, node_features[i], ef))

        # concatenate node features
        for i in range(K):
            all_out_node_features[i] = tf.concat(all_out_node_features[i],
                                                 axis=-1)
        return all_out_node_features, all_out_edge_features


@gin.configurable
def get_base_extractor(sizes,
                       flat_node_indices,
                       row_splits,
                       batch_row_splits_or_k,
                       edge_reduction_fn=edge.reduce_max,
                       units_scale=4,
                       unit_expansion_factor=SQRT_2):
    """
    Args:
        sizes: [K] list of int scalars used in edge_reduction_fn, size of all
            elements in the depth across the entire batch.
        flat_node_indices:
        row_splits:
        batch_row_splits_or_k:
        edge_reduction_fn: see `deep_cloud.ops.edge`.
        units_scale: number of dense units is proportional to this.
        unit_expansion_factor: rate at which the number of units increase.

    Returns:
        Sensible `KPartiteFeatureExtractor`.
    """
    local_extractors = []
    global_extractors = []
    depth = len(flat_node_indices)
    for i in range(depth):
        extractors = []
        local_extractors.append(extractors)
        for j in range(i + 1):
            # double units in one sample dimension
            # the other sample dimension increases the receptive field
            # number of ops is constant for a sample rate of 0.25
            units = int(np.round(units_scale * unit_expansion_factor**j))
            extractors.append(
                BiPartiteFeatureExtractor(flat_node_indices[i][j],
                                          row_splits[i][j],
                                          sizes[i],
                                          initial_units=units,
                                          edge_network_fn=mlp([units]),
                                          edge_reduction_fn=edge_reduction_fn,
                                          dense_factory=Dense))
        # global extractors work per-node
        # for sample rate of 0.25, doubling units per layer keeps ops constant
        units = int(np.round(2 * units_scale * unit_expansion_factor**i))
        global_extractors.append(
            GlobalBipartiteFeatureExtractor(batch_row_splits_or_k[i], units,
                                            mlp([units])))

    return KPartiteFeatureExtractor(
        local_extractors,
        global_extractors,
    )
