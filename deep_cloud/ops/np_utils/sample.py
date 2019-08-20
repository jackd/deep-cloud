"""Numpy implementations for sampling."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def gaussian(x, axis=-1, keepdims=False):
    return np.exp(-np.sum(np.square(x), axis=axis, keepdims=keepdims))


def get_total_weight(neighborhood_size,
                     indices=None,
                     coords=None,
                     scale_factor=0.05,
                     weight_fn=gaussian):
    if weight_fn is None:
        weight = neighborhood_size.astype(np.float64)
    else:
        rel_coords = np.repeat(coords, neighborhood_size,
                               axis=0) - coords[indices]
        if scale_factor != 1:
            rel_coords /= scale_factor
        weight = weight_fn(rel_coords)
        start_indices = np.concatenate([[0], np.cumsum(neighborhood_size[:-1])])
        weight = np.add.reduceat(weight, start_indices)
    return weight


def inverse_density_sample(sample_size,
                           neighborhood_size,
                           indices=None,
                           coords=None,
                           scale_factor=0.05,
                           weight_fn=None,
                           replace=False):
    """Sample `sample_size` indices from [0, neighborhood_size.shape[0])."""
    weight = get_total_weight(neighborhood_size, indices, coords, scale_factor,
                              weight_fn)
    probs = 1 / weight
    probs /= np.sum(probs)
    return np.random.choice(neighborhood_size.shape[0],
                            sample_size,
                            replace=replace,
                            p=probs)


def inverse_density_mask(neighborhood_size,
                         prob_scale_factor=None,
                         mean_keep_rate=None,
                         indices=None,
                         coords=None,
                         scale_factor=0.05,
                         weight_fn=None):
    """Get a binary mask of shape `neighborhood_size.shape[0]`."""
    if (prob_scale_factor is None) == (mean_keep_rate is None):
        raise ValueError(
            'Exactly 1 of `prob_scale_factor` and `mean_keep_rate` should be '
            '`None`')
    weight = get_total_weight(neighborhood_size, indices, coords, scale_factor,
                              weight_fn)
    if mean_keep_rate is None:
        keep_rate = prob_scale_factor / weight
    else:
        keep_rate = 1.0 / weight
        keep_rate *= mean_keep_rate / np.mean(keep_rate)
    return np.random.uniform(size=keep_rate.size) < keep_rate
