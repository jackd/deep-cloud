from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gin
import tensorflow_datasets as tfds
import six
from shape_tfds.shape.modelnet import pointnet as _pn
from shape_tfds.shape.modelnet import pointnet2 as _pn2


@gin.configurable
class UniformDensityConfig(tfds.core.BuilderConfig):

    def __init__(self, builder, num_points=1024, k=10, r0=0.1):
        if isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        self.base = builder
        self.num_points = num_points
        self.r0 = r0
        self.k = k
        builder_name = builder.name
        config = builder.builder_config
        builder_name = builder.name if config is None else '{}-{}'.format(
            builder.name, config.name)

        name = "uniform-density-{}-{}-{}-{}".format(builder_name, num_points, k,
                                                    r0)

        super(UniformDensityConfig, self).__init__(
            name=name,
            version="0.0.1",
            description="rescaled data used {}".format(builder_name))

    @property
    def up_dim(self):
        return self.base.up_dim

    def rescale(self, positions, normals, **kwargs):
        from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
        positions = positions[:self.num_points]
        normals = normals[:self.num_points]
        tree = cKDTree(positions)
        dists, indices = tree.query(tree.data, self.k)
        del indices
        new_scale_factor = self.r0 / np.mean(dists[:, -1])
        positions *= new_scale_factor
        out = dict(positions=positions, normals=normals)
        for k, v in kwargs.items():
            out[k] = v[:self.num_points]
        return out


class UniformDensityCloud(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []

    @property
    def up_dim(self):
        return self.builder_config.up_dim

    def _info(self):
        num_points = self.builder_config.num_points
        features = dict(**self.builder_config.base.info.features)
        cloud = dict(**features['cloud'])
        features['cloud'] = cloud
        for k, v in cloud.items():
            feature = tfds.core.features.Tensor(shape=(num_points,) +
                                                v.shape[1:],
                                                dtype=v.dtype)
            cloud[k] = feature

        return tfds.core.DatasetInfo(
            builder=self,
            description='Data used in the original pointnet paper',
            features=tfds.features.FeaturesDict(features),
            citation=_pn.CITATION,
            urls=_pn.URLS,
            supervised_keys=('cloud', 'label'))

    def _split_generators(self, dl_manager):
        builder = self.builder_config.base
        builder.download_and_prepare()
        gens = []
        for split, num_shards in ((tfds.Split.TRAIN, 8), (tfds.Split.TEST, 2)):
            gen = tfds.core.SplitGenerator(name=split,
                                           num_shards=num_shards,
                                           gen_kwargs=dict(split=split,
                                                           builder=builder))
            gen.split_info.statistics.num_examples = (
                builder.info.splits[split].num_examples)
            gens.append(gen)
        return gens

    def _generate_examples(self, builder, split):
        config = self.builder_config
        dataset = builder.as_dataset(split=split, shuffle_files=False)
        for i, example in enumerate(tfds.as_numpy(dataset)):
            cloud = example['cloud']
            cloud = config.rescale(**cloud)
            example['cloud'] = cloud
            yield i, example


@gin.configurable
def pointnet_builder(pointnet_version=1, uniform_density=False):
    # separate fn so gin doesn't interfering with tfds by mangling names
    if pointnet_version == 1:
        builder = _pn.Pointnet()
    elif pointnet_version == 2:
        builder = _pn2.Pointnet2(config=_pn2.CONFIG40)
    else:
        raise ValueError('Invalid pointnet_version {}: must be 1 or 2'.format(
            pointnet_version))
    if uniform_density:
        config = UniformDensityConfig(builder, num_points=1024, k=10, r0=0.1)
        builder = UniformDensityCloud(config=config)
    return builder


if __name__ == '__main__':
    for v in (1, 2):
        builder = pointnet_builder(v, uniform_density=True)
        builder.download_and_prepare()

    from mayavi import mlab
    for example in tfds.as_numpy(builder.as_dataset(split='train')):
        cloud = example['cloud']
        positions = cloud['positions']
        normals = cloud['normals']
        u, v, w = normals.T
        x, y, z = positions.T
        mlab.quiver3d(x, y, z, u, v, w)
        mlab.show()
