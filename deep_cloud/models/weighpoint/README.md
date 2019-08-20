# WeighPoint: Weighted Point Cloud Convolutions

Point cloud convolutions in deep learning have seen a lot of interest lately. Broadly speaking, these involve grouping points according to local proximity using some data structure like a KDTree. While results on classification and segmentation tasks are promising, most publicly available implementations suffer from a number of factors including:

1. `k`-nearest-neighbors search to ensure a fixed number of neighbors, rather than a fixed neighborhood size (as is suggested should be the case for convolutions in integral form);
2. discontinuity in space: `k`-nearest neighbors is discontinous as the `k`th and `k+1`th neighbors switch order; and
3. custom kernels which require additional setup and maintenance.

To address these, we implement:

1. neighborhoods defined by ball-searches, implemented using `tf.RaggedTensor`s; and
2. a _weighted convolution_ operation that ensures the integrated function trails to zero at the ball-search radius and is invariant to point-density.

We make extensive use of the meta-model building process from [`more_keras`](https://github.com/jackd/more-keras/tree/master/more_keras/meta_models) that allows per-layer preprocessing operations to be built in conjunction with the learnable operations before being split into separate preprocessing and learned `tf.keras.Model`s.

The resulting architecture makes efficient use of CPUs for preprocessing without the need for custom kernels. As a bonus, the radii over which the ball searches occur can also be learned.

The project is under heavy development. Currently we are able to achieve competitive results on modelnet40 classification task (~90%) and are working towards semantic segmentation and point cloud generation implementations.

## Usage

See [examples/weighpoint](../../../examples/weighpoint). In particular, note we make extensive use of [gin-config](https://github.com/google/gin-config).

## Theory

See [this paper](https://drive.google.com/open?id=1VxAnRMcPhovMwqpY1Z9iYsOS8pjvdLFw) for a basic overview of the theory associated with the operations.

## Python/Tensorflow Compatibility

While we provide no guarantees, best efforts have been made to make this code compatible with python 2/3 and tensorflow versions `>=1.14` and `2.0.0-alpha`.

## Neighborhood Implementations

We use [`KDTree.query_pairs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) (along with random sampling and/or masking) to calculate neighborhoods with a variable number of neighbors. While no tensorflow implementation exists, we find performance is acceptable using `tf.py_function` during preprocessing (i.e. inside a `tf.data.Dataset.map`).

We store the calculated neighborhoods in [`tf.RaggedTensor`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/RaggedTensor?hl=en)s where possible and make extensive use of `tf.ragged` operations.

## Data Pipeline

The data pipeline developed in this project is critical to the timely training of the networks without introducing custom operations. It is made up of:

- [tensorflow_dataset](github.com/jackd/tensorflow/datasets) (`tfds`) base implementations that manage downloading and serialization of the raw point cloud data;
- [shape-tfds](https://github.com/jackd/shape-tfds): `tfds` implementations for 3D shape datasets.
- [deep_cloud.augment](../../augment) for model-independent preprocessing;
- [more_keras.meta_models](https://github.com/jackd/more-keras/tree/master/more_keras/meta_models) for tools to write per-layer preprocessing operations (like KDTrees) along with learned components. The result is a learnable `tf.keras.Model` along with a model-dependent preprocessing and batching functions with a `tf.data.Dataset` pipeline.

## Classification

TODO

## Segmentation

Work ongoing.

## Point Cloud Generation

Work ongoing

## Issues

- Saving in tf 2.0

## Components

- [convolvers]
- [transformers]
