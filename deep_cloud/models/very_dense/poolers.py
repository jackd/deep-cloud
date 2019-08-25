from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from deep_cloud.ops.np_utils import sample
from deep_cloud.models.very_dense import utils


class Pooler(object):

    @abc.abstractmethod
    def __call__(self, coords, normals, node_indices):
        """Apply pooling to coords and node_indices."""
        raise NotImplementedError

    def output_size(self, input_size):
        """Get the output size for a given input size."""
        return None


class SlicePooler(Pooler):

    def __init__(self, survival_rate=0.5):
        self._survival_rate = survival_rate

    def __call__(self, coords, normals, node_indices):
        size = self.output_size(coords.shape[0])
        if normals is not None:
            normals = normals[:size]
        return coords[:size], normals, node_indices[:size]

    def output_size(self, input_size):
        if input_size is None:
            return None
        return int(input_size * self._survival_rate)


class InverseDensitySamplePooler(Pooler):

    def __init__(self, survival_rate=0.5):
        self._survival_rate = survival_rate

    def __call__(self, coords, normals, node_indices):
        sample_indices = sample.inverse_density_sample(
            self.output_size(coords.shape[0]), node_indices.row_lengths)
        if normals is not None:
            normals = normals[sample_indices]
        return (
            coords[sample_indices],
            normals,
            node_indices[sample_indices],
        )

    def output_size(self, input_size):
        if input_size is None:
            return None
        return int(input_size * self._survival_rate)
