from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gin
import numpy as np
import tensorflow as tf
from more_keras.layers import Dense
from more_keras.layers import BatchNormalization
from more_keras.layers import Dropout
from more_keras.layers import bias
from more_keras.layers import utils
from more_keras.layers import source
from deep_cloud.layers import query as _q


@gin.configurable
def optimizer(factory=tf.keras.optimizers.Adam, lr=1e-3, decay=0.0):
    return factory(lr=lr, decay=decay)


@gin.configurable
def model_dir(model_id, base_directory='/tmp'):
    return os.path.join(base_directory, model_id)


@gin.configurable(blacklist=['flat_features', 'units'])
def mlp_layer(flat_features,
              units,
              activation='relu',
              use_batch_normalization=False,
              dropout_rate=None):
    """
    Basic multi-layer perceptron layer + call.

    Args:
        flat_features: [N, units_in] float32 non-ragged float features
        units: number of output features
        activation: used in Dense
        dropout_rate: rate used in dropout (no dropout if this is None)
        use_batch_normalization: applied after dropout if True

    Returns:
        [N, units] float32 output features
    """
    flat_features = Dense(units,
                          activation=activation,
                          use_bias=not use_batch_normalization)(flat_features)
    if dropout_rate is not None:
        flat_features = Dropout(rate=dropout_rate)(flat_features)
    if use_batch_normalization:
        flat_features = BatchNormalization()(flat_features)
    return flat_features


@gin.configurable
def generalized_activation(features,
                           activation='relu',
                           add_bias=False,
                           use_batch_normalization=False,
                           dropout_rate=None):
    """
    Generalized activation to be performed (presumably) after a learned layer.

    Can include (in order)
        * bias addition;
        * standard activation;
        * batch normalization; and/or
        * dropout.
    """
    if add_bias and not use_batch_normalization:
        features = bias.add_bias(features)
    features = tf.keras.layers.Lambda(
        tf.keras.activations.get(activation))(features)
    if use_batch_normalization:
        features = BatchNormalization()(features)
    if dropout_rate:
        features = Dropout(dropout_rate)(features)
    return features


@gin.configurable(whitelist=['max_neighbors', 'sample_rate_reciprocal_offset'])
def query_pairs(coords,
                radius,
                name=None,
                max_neighbors=None,
                sample_rate_reciprocal_offset=0):
    """
    Get neighbors and inverse density sample rates from the provided coords.

    sample_rate is given by
    sample_rate = 1. / (sum(num_neighbors) - sample_rate_reciprocal_offset)

    Args:
        coords: [n, num_dims] float32 point cloud coordinates
        radius: float scalar ball search radius
        name: used in layers
        max_neighbors: if not None, neighborhoods are cropped to this size.
        sample_rate_reciprocal_offset: affects sample rate. Values close to 1
            result in sparse neighborhoods being selected more frequently

    Returns:
        neighbors: [n, ?] int32 ragged tensor of neighboring indices
        sample_rate: sampling rate based roughly on inverse density.
    """
    neighbors = _q.query_pairs(coords,
                               radius,
                               name=name,
                               max_neighbors=max_neighbors)
    sample_rate_reciprocal = utils.cast(utils.row_lengths(neighbors),
                                        tf.float32)
    if sample_rate_reciprocal_offset != 0:
        sample_rate_reciprocal = utils.lambda_call(
            tf.math.subtract, sample_rate_reciprocal,
            sample_rate_reciprocal_offset)
    sample_rate = utils.reciprocal(sample_rate_reciprocal)
    return neighbors, sample_rate


@gin.configurable(whitelist=['k'])
def query_knn(coords, radius=None, name=None, k=8):
    """
    Find the k nearest neighbors of the given point cloud coordinates.

    Args:
        coords: [n, num_dims] coordinates of point cloud
        radius: not used - just here to make interface same as query_pairs
        name: used for debugging
        k: number of neighbors to find

    Returns:
        neighbors: [n, k] int64 array of neighbors
        sample_rate: inverse density sample rate defined as the sum of
            distances of k nearest neighbors.
    """
    neighbors, dists = _q.query_knn(coords, k, name=name)
    sample_rate = utils.reduce_sum(dists, axis=-1)
    return neighbors, sample_rate


@gin.configurable(blacklist=['num_res'])
def constant_radii(num_res):
    """Constant unscaled squared radii."""
    return np.power(4.0, np.arange(num_res)).astype(np.float32)


def _cost(radii2, density):
    return tf.reduce_sum(radii2 * density)


def _rescale_for_budget(radii2, density, budget):
    return radii2 * budget / _cost(radii2, density)


@gin.configurable(blacklist=['num_res'])
def fixed_budget_radii(num_res):
    """
    Learnable unscaled squared radii based on a fixed computational budget.

    The budget assumes point cloud density quarters at each resolution, and
    aims to keep the sum across all resolutions of the sum of all neighborhood
    sizes roughly constante. Note there are generally more filters in later
    resolutions, so the number of floating point operations may increase if the
    model learns to expand final layer radii at the expense of lower
    resolutions.

    Args:
        num_res: number of resolutions.

    Returns:
        [num_res] float32 tensor of squared unscaled radii values, initially
        the same as `constant_radii(num_res)`.
    """
    r20 = constant_radii(num_res)
    radii = source.variable(shape=(num_res,),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(np.sqrt(r20)))
    radii2 = tf.keras.layers.Lambda(tf.square)(radii)
    density = np.power(4.0, -np.arange(num_res)).astype(np.float32)
    budget = utils.lambda_call(_cost, r20, density)
    return utils.lambda_call(_rescale_for_budget, radii2, density, budget)


def convolve(features,
             radius2,
             filters,
             neighborhood,
             coords_transform,
             weights_transform,
             convolver_fn,
             global_features=None):
    """
    Tie together components making up a convolution.

    N_i: sum over batch examples of number of points in input cloud
    N_o: sum over batch examples of number of points in output cloud
    B: batch_size

    Args:
        features: [N_i, filters_in] float32 tensor of flattened batched
            point features.
        radius2: float32 scalar squared radius value.
        filters: python int, number of filters out.
        neighborhood: `deepcloud.neigh.Neighborhood.Neighborhood` instance.
        coords_transform: function mapping relative coordinates to coord
            features.
        weights_transform: function mapping relative coordinates to weights.
        convolver_fn: One of `weighpoint.model.convolvers.Convolver` instance
            methods.
        global_features: [B, filters] float32 or None. If not None, these are
            added after the convolution.

    Returns:
        features: [N_o, filters] float32 array.
        nested_row_splits: row splits of implicit ragged features.
        i.e. the ragged features can be created using
            `tf.RaggedTensor.from_row_splits(features, nested_row_splits)`
    """
    rel_coords = neighborhood.rel_coords.flat_values
    offset_batched_neighbors = neighborhood.offset_batched_neighbors

    coord_features = coords_transform(rel_coords, radius2)
    weights = weights_transform(rel_coords, radius2)

    features = convolver_fn(features, coord_features, offset_batched_neighbors,
                            filters, weights)
    if global_features is not None:
        features = tf.RaggedTensor.from_row_splits(
            features, offset_batched_neighbors.row_splits)
        features = add_local_global(features, global_features)
        features = features.flat_values
    return features, offset_batched_neighbors.nested_row_splits


def _add_local_global(args):
    local_features, global_features = args
    assert (local_features.shape.ndims == 3)
    assert (global_features.shape.ndims == 2)
    assert (local_features.shape[-1] == global_features.shape[-1])
    return local_features + tf.expand_dims(global_features, axis=-2)


def add_local_global(local_features, global_features):
    """
    Add local and global features.

    Args:
        local_features: [B, n?, f] float32 possible ragged tensor of point
            features
        global_features: None or [B, f] float32 tensor of global features.

    Returns:
        [B, n?, f] possibly ragged tensor of modified local features. Does
        no modification if `global_features is None`.
    """
    if global_features is None:
        return local_features
    assert (global_features.shape[-1] == local_features.shape[-1])
    return utils.ragged_lambda(_add_local_global)(
        [local_features, global_features])
