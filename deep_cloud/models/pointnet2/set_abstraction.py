"""
Reimplementation of:

https://github.com/charlesq34/pointnet2/blob/master/utils/pointnet_util.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from deep_cloud.models.weighpoint import core
from deep_cloud import neigh
from more_keras.meta_models import builder as b
from more_keras.layers import utils as layer_utils
from more_keras.ragged.layers import ragged_lambda
from deep_cloud.layers import sample
layers = tf.keras.layers


def pointnet2_mlp(features=(64, 64, 128)):
    raise NotImplementedError


def set_abstraction(
        features,
        neighborhood,
        edge_mlp,
        reduction=tf.reduce_max,
        node_mlp=None,
        coords_as_features=True,
):
    """
    Reimplementation of original `pointnet_sa_module` function.

    Args:
        features: [b, n_i?, filters_in] float32 tensor of flattend batched point
            features.
        neighborhood: `deepcloud.neigh.Neighborhood` instance with `n_i` inputs
            and `n_o` output points.
        edge_mlp: callable acting on each edge features.
        reduction: operation to reduce neighborhoods to point features.
        node_mpl: callacble acting on each point after reduction.
        coords_as_features: if True, use relative coords in neighborhood as well
            as features in edges.

    Returns:
        features: [b, n_o?, filters_o] float32 array, where filters_o is the
          number of output features of `edge_mlp` if `node_mlp` is None else the
          number of output features of `node_mlp`.
    """

    def flat_rel_coords():
        return b.as_batched_model_input(
            neighborhood.rel_coords.flat_values).flat_values

    if features is None:
        features = flat_rel_coords()
    else:
        features = layer_utils.flatten_leading_dims(features)
        offset_batched_neighbors = neighborhood.offset_batched_neighbors
        if coords_as_features:
            features = tf.gather(features, offset_batched_neighbors.flat_values)
            features = layers.Lambda(tf.concat, arguments=dict(axis=-1))(
                [features, flat_rel_coords()])
        else:
            # more efficient than original implementation
            features = edge_mlp(features)
            features = tf.gather(features, offset_batched_neighbors.flat_values)

    # features is not flat, [B, f]
    features = tf.RaggedTensor.from_nested_row_splits(
        features, offset_batched_neighbors.nested_row_splits)
    # features is now [b, n_o?, k?, E]
    features = ragged_lambda(reduction, arguments=dict(axis=-2))(features)
    # features is now [b, n_o?, E]
    if node_mlp is not None:
        if isinstance(features, tf.RaggedTensor):
            features = tf.ragged.map_flat_values(node_mlp, features)
        else:
            features = node_mlp(features)

    return features


SetAbstractionSpec = collections.namedtuple('SetAbstractionSpec',
                                            ['radius', 'n_out', 'mlp'])


def pointnet2_head(coords,
                   normals=None,
                   specs=None,
                   global_mlp=None,
                   coords_as_features=True):
    if specs is None:
        specs = (
            SetAbstractionSpec(0.2, 512, pointnet2_mlp((64, 64, 128))),
            SetAbstractionSpec(0.4, 128, pointnet2_mlp((128, 128, 256))),
        )
    if global_mlp is None:
        global_mlp = pointnet2_mlp((256, 512, 1024))
    features = None if normals is None else b.as_batched_model_input(normals)
    for radius, n_out, mlp in specs:
        neighbors, sample_rate = core.query_pairs(coords, radius)
        neighborhood = neigh.InPlaceNeighborhood(coords, neighbors)

        sample_indices = sample.sample(sample_rate, n_out)
        neighborhood = neigh.SampledNeighborhood(neighbors, sample_indices)
        features = set_abstraction(features,
                                   neighborhood,
                                   mlp,
                                   coords_as_features=coords_as_features)
        coords = neighborhood.out_coords
    # global mlp
