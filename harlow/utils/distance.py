"""Fast and numba compatible scipy.distance like functions.
Inspired by and based on: https://github.com/talboger/fastdist"""

import numpy as np
from numba import jit, prange

nopython = True
fastmath = True
parallel = True
cache = True


@jit(nopython=nopython, fastmath=fastmath, cache=cache)
def euclidean_distance(u_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    dist = 0
    for ii in range(len(u_vec)):
        dist += (u_vec[ii] - v_vec[ii]) ** 2
    return np.sqrt(dist)


def fractional_distance(u_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """Fractional distance between two vectors."""
    dist = 0
    ndim = 2
    for ii in range(len(u_vec)):
        dist += (u_vec[ii] - v_vec[ii]) ** ndim
    return np.pow(1.0 / ndim)


@jit(nopython=nopython, fastmath=fastmath, parallel=parallel, cache=cache)
def pdist_condensed(x_mx: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distances between n-dimensional vectors. Similar to
    `scipy.spatial.distance.pdist` but faster for a large number of vectors.

    TODO: maybe it could be faster if there was just one loop, kk -> ii, jj.

    Args:
        x_mx:
            An `m` by `n` matrix of `m` vectors in an `n`-dimensional space.

    Returns:
        Condensed distance matrix.
    """
    n_point = x_mx.shape[0]
    n_elem = int((n_point**2 - n_point) / 2)
    dist = np.empty(n_elem)
    for ii in prange(n_point):
        for jj in prange(ii + 1, n_point):
            kk = int(ii * n_point - (ii**2 + 3 * ii) / 2 + jj - 1)
            dist[kk] = euclidean_distance(x_mx[ii], x_mx[jj])
    return dist


@jit(nopython=nopython, fastmath=fastmath, parallel=parallel, cache=cache)
def pdist_full_matrix(x_mx: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distances between n-dimensional vectors. Similar to
    `scipy.spatial.distance.pdist` and `scipy.spatial.distance.squareform` but faster
    for a large number of vectors.

    Args:
        x_mx:
            An `m` by `n` matrix of `m` vectors in an `n`-dimensional space.

    Returns:
        A full (`m` by `m`) distance matrix.
    """
    n_point = x_mx.shape[0]
    dist = np.zeros((n_point, n_point))
    for ii in prange(n_point):
        for jj in prange(ii + 1, n_point):
            dist[ii, jj] = euclidean_distance(x_mx[ii], x_mx[jj])
    # make it symmetric
    return dist + dist.T


@jit(nopython=nopython, fastmath=fastmath, parallel=parallel, cache=cache)
def fractional_pdist_full_matrix(x_mx: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distances between n-dimensional vectors. Similar to
    `scipy.spatial.distance.pdist` and `scipy.spatial.distance.squareform` but faster
    for a large number of vectors.

    Args:
        x_mx:
            An `m` by `n` matrix of `m` vectors in an `n`-dimensional space.

    Returns:
        A full (`m` by `m`) distance matrix.
    """

    n_point = x_mx.shape[0]
    dist = np.zeros((n_point, n_point))
    for ii in prange(n_point):
        for jj in prange(ii + 1, n_point):
            dist[ii, jj] = fractional_distance(x_mx[ii], x_mx[jj])
    # make it symmetric
    return dist + dist.T
