from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools


def lower_triangular(K, fill=None):
    """Return a list with space for lower triangular values."""
    return [[fill for _ in range(i + 1)] for i in range(K)]


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
