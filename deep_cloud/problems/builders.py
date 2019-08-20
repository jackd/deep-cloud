from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gin
import tensorflow_datasets as tfds
from shape_tfds.shape.modelnet import pointnet as _pn


@gin.configurable
class UniformDensityConfig(tfds.core.BuilderConfig):

    def __init__(self, num_points=1024, k=10, r0=0.1, roll_dims=1):
        self.num_points = num_points
        self.r0 = r0
        self.k = k
        self.roll_dims = 1
        name = "uniform-density-{}-{}-{}-{}".format(num_points, k, r0,
                                                    roll_dims)
        super(UniformDensityConfig, self).__init__(
            name=name,
            version="0.0.1",
            description="rescaled data used in original pointnet paper")

    @property
    def up_dim(self):
        return (1 + self.roll_dims) % 3

    def rescale(self, positions, normals, face_ids):
        from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
        positions = positions[:self.num_points]
        normals = normals[:self.num_points]
        face_ids = face_ids[:self.num_points]
        positions = np.roll(positions, self.roll_dims, axis=-1)
        normals = np.roll(normals, self.roll_dims, axis=-1)
        tree = cKDTree(positions)
        dists, indices = tree.query(tree.data, self.k)
        del indices
        new_scale_factor = self.r0 / np.mean(dists[:, -1])
        positions *= new_scale_factor
        return dict(positions=positions, normals=normals, face_ids=face_ids)


class UniformDensityPointnet(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [UniformDensityConfig(num_points=1024, k=10, r0=0.1)]

    @property
    def up_dim(self):
        return self.builder_config.up_dim

    def _info(self):
        num_points = self.builder_config.num_points
        cloud = tfds.core.features.FeaturesDict(
            dict(positions=tfds.core.features.Tensor(shape=(num_points, 3),
                                                     dtype=tf.float32),
                 normals=tfds.core.features.Tensor(shape=(num_points, 3),
                                                   dtype=tf.float32),
                 face_ids=tfds.core.features.Tensor(shape=(num_points,),
                                                    dtype=tf.int64)))
        return tfds.core.DatasetInfo(
            builder=self,
            description='Data used in the original pointnet paper',
            features=tfds.core.features.FeaturesDict(
                dict(cloud=cloud,
                     label=tfds.core.features.ClassLabel(
                         num_classes=_pn.NUM_CLASSES))),
            citation=_pn.CITATION,
            urls=_pn.URLS,
            supervised_keys=('cloud', 'label'))

    def _split_generators(self, dl_manager):
        builder = _pn.Pointnet()
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
def uniform_density_pointnet():
    # separate fn so gin doesn't interfering with tfds by mangling names
    return UniformDensityPointnet()


if __name__ == '__main__':
    builder = UniformDensityPointnet()
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
