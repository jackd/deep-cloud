from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import spatial as sp
from sklearn import neighbors as ne
import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools
from deep_cloud.ops.np_utils.tree_utils import spatial
from deep_cloud.ops.np_utils.tree_utils import pykd
from deep_cloud.ops.np_utils.tree_utils import skkd


def _plot(data, times, names, ax=None, title=None):
    times = np.asanyarray(times)
    if ax is None:
        ax = plt.gca()
    x = [d.shape[0] for d in data]
    ax.plot(x, times.T)
    ax.legend(names)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if title is not None:
        ax.set_title(title)


def profile_construction(data,
                         constructors,
                         names,
                         num_runs=5,
                         warm_up=2,
                         ax=None,
                         title=None):
    assert (len(constructors) == len(names))
    times = [[] for _ in constructors]
    for d in tqdm(data, desc='Construction...'):
        for time, constructor in zip(times, constructors):
            for _ in range(warm_up):
                constructor(d)
            t = timeit(lambda: constructor(d), number=num_runs)
            time.append(t)
    # print(len(data), np.array(times).shape)
    _plot(data, times, names, ax, title=title)


def profile_fn(data,
               constructors,
               names,
               fn,
               num_runs=2,
               warm_up=2,
               ax=None,
               title=None,
               desc=None,
               include_construction=False):
    assert (len(constructors) == len(names))
    times = [[] for _ in constructors]
    for d in tqdm(data, desc=desc):
        for time, constructor in zip(times, constructors):
            for _ in range(warm_up):
                c = constructor(d)
                fn(c, d)

            if include_construction:
                t = timeit(lambda: fn(constructor(d), d), number=num_runs)
            else:
                t = timeit(lambda: fn(c, d), number=num_runs)
            time.append(t)
    _plot(data, times, names, ax=ax, title=title)


r = np.random.RandomState()  # pylint: disable=no-member

num_dims = 3
num_points = np.power(10, np.linspace(2, 4, 11)).astype(np.int64)
k = 8
dtype = np.float32
data = [r.uniform(size=(n, num_dims)).astype(dtype) for n in num_points]

fig, ax = plt.subplots(2, 2)
ax0, ax1, ax2, ax3 = np.reshape(ax, (-1,))

constructors = [
    functools.partial(spatial.KDTree, balanced_tree=False),
    functools.partial(spatial.KDTree, balanced_tree=True),
    skkd.KDTree,
    functools.partial(pykd.KDTree, leafsize=16),
    functools.partial(pykd.KDTree, leafsize=8),
    functools.partial(pykd.KDTree, leafsize=4),
]
names = [
    # 'scipy.KDTree',
    'spatial',
    'spatial-balanced',
    'skkd',
    'pykd-16',
    'pykd-8',
    'pykd-4',
]


def expected_radius(k, n):
    if num_dims == 2:
        return np.sqrt(k / (np.pi * n))
    elif num_dims == 3:
        return (3 / (4 * np.pi * n))**(1 / 3)
    else:
        raise NotImplementedError()


profile_construction(data, constructors, names, ax=ax0, title='construction')

profile_fn(data,
           constructors,
           names,
           fn=lambda tree, x: tree.query(x, k=8),
           ax=ax1,
           title='query',
           desc='Querying...')
profile_fn(data,
           constructors,
           names,
           fn=lambda tree, x: tree.query_ball_point(
               x, r=expected_radius(k, x.shape[0]), max_neighbors=2 * k),
           ax=ax2,
           title='query_ball_point',
           desc='Querying radius...')
profile_fn(data,
           constructors,
           names,
           fn=lambda tree, x: tree.query_ball_point(
               x, r=expected_radius(k, x.shape[0]), max_neighbors=8 * k),
           ax=ax3,
           include_construction=True,
           title='query_ball_point conservative',
           desc='Querying conservative...')
# profile_fn(data,
#            constructors,
#            names,
#            fn=lambda tree, x: tree.query(
#                x, k=k, distance_upper_bound=expected_radius(2 * k, x.shape[0])),
#            ax=ax3,
#            title='query_clipped',
#            desc='Querying clipped...')

plt.show()
