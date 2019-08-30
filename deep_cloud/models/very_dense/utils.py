"""
Utilities for working with lower triangular packings.

i.e. lists of lists, where the nth entry has length n.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools


def lower_triangular(K, fill=None):
    """Return a list with space for lower triangular values."""
    return [[fill for _ in range(i + 1)] for i in range(K)]


def lower_indices(K):
    for i in range(K):
        for j in range(i + 1):
            yield (i, j)


def pack_lower_triangle(values):
    """Pack flat list into lower triangular list of lists."""
    N = len(values)
    consumed = 0
    out = []
    i = 1
    while consumed < N:
        out.append(values[consumed:consumed + i])
        consumed += i
        i += 1
    assert (consumed == N)
    return out


def flatten_lower_triangle(values):
    """Pack lower triangular list of lists into flat list."""
    return list(itertools.chain(*values))


def ttuple(tri_values):
    """Convert to tuple of tuples."""
    return tuple(tuple(t) for t in tri_values)


def llist(tri_values):
    """Convert to list of lists."""
    return list(list(t) for t in tri_values)


def is_triangle(values):
    for i, v in enumerate(values):
        if len(v) != i:
            return False
    return True


def map_triangle(fn, values):
    # TODO: generalize to *args, **kwargs. See how tf.nest.map_structure works?
    return pack_lower_triangle([fn(v) for v in flatten_lower_triangle(values)])
