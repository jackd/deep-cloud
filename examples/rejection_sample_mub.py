"""Minimum upper bound validation after rejection sampling."""
import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

num_dims = 2
num_points = 1000000
r_max = 20

packing_fracs = [
    None,
    1.,
    np.pi * np.sqrt(3) / 6,  # https://en.wikipedia.org/wiki/Circle_packing
    np.pi / (3 * np.sqrt(2)
            )  # https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
]

x = np.random.uniform(size=(num_points, num_dims)) * r_max
x -= r_max / 2
tree = cKDTree(x)

out = []
consumed = np.zeros((num_points,), dtype=np.bool)
for i in tqdm(range(num_points)):
    if not consumed[i]:
        out.append(i)
        neighbors = tree.query_ball_point(x[i], 2.)
        consumed[neighbors] = True

xs = x[out]

# ax = plt.gca()
# ax.scatter(*(xs.T))
# ax.scatter(x[:, 0], x[:, 1], 0.1, color='r', alpha=0.3)
# collection = PatchCollection([Circle(xy, 2, fc=None) for xy in xs],
#                              alpha=0.1,
#                              ec=(0, 1, 0))
# ax.add_collection(collection)
# collection = PatchCollection([Circle(xy, 1, fc=None) for xy in xs],
#                              alpha=0.2,
#                              ec=(1, 0, 0))
# ax.add_collection(collection)
# ax.axis('square')
# plt.show()

dists = np.linalg.norm(xs, axis=-1)
dists = np.sort(dists)
dists = dists[dists < r_max // 2]
counts = np.arange(len(dists))

start = len(dists) // 2
end = int(len(dists) * 0.9)

dists = dists[start:end]
counts = counts[start:end]

ax = plt.gca()
ax.plot(dists, counts, color='b')
ax.plot(dists,
        np.ceil(packing_fracs[num_dims] * (dists + 1)**num_dims),
        color='k')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

# b = np.log(counts)
# A = np.stack([np.log(dists), np.ones_like(dists)], axis=-1)
# out = np.linalg.lstsq(A, b)
# print(out)
# # a = np.exp(c)
# print(a, m)
# print(np.polyfit(dists[start:end], counts[start:end], num_dims))
