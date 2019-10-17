from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
from tqdm import tqdm
import tensorflow_datasets as tfds

from deep_cloud.problems.partnet import PartnetProblem
from deep_cloud.ops.np_utils.tree_utils import pykd

problem = PartnetProblem()
warm_up = 5
benchmark = 10
total = warm_up + benchmark

tree_impl = pykd.KDTree
all_coords = []
for coords, _ in tqdm(tfds.as_numpy(
        problem.get_base_dataset('validation').take(total)),
                      total=total,
                      desc='getting base data...'):
    tree = tree_impl(coords)
    dists, indices = tree.query(tree.data, 2, return_distance=True)
    del indices
    scale = np.mean(dists[:, 1])
    coords *= 2 / scale
    all_coords.append(coords)


def run_fn(f, data, name):
    for i in tqdm(range(warm_up), desc='warming up {}'.format(name)):
        f(data[i])

    t = time()
    for i in tqdm(range(warm_up, total), desc='benchmarking {}'.format(name)):
        f(data[i])
    dt = time() - t
    print('{} runs took {} ms, {} ms / run'.format(benchmark, dt * 1000,
                                                   dt * 1000 / benchmark))


trees = [tree_impl(c) for c in all_coords]


def query_tree(tree):
    tree.query_ball_point(tree.data, 4, approx_neighbors=16)


run_fn(tree_impl, all_coords, 'just tree')
run_fn(query_tree, trees, 'just query')
run_fn(lambda c: query_tree(tree_impl(c)), all_coords, 'compute both')
