import numba as nb
import numpy as np

nopython = True
fastmath = True
cache = True


@nb.jit(nopython=nopython, fastmath=fastmath, cache=cache)
def np_apply_along_axis(func1d, axis, arr):
    """Until `axis` is supported by `numba`.
    Source: https://github.com/numba/numba/issues/1269#issuecomment-472574352"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for ii in range(len(result)):
            result[ii] = func1d(arr[:, ii])
    else:
        result = np.empty(arr.shape[0])
        for ii in range(len(result)):
            result[ii] = func1d(arr[ii, :])
    return result


@nb.jit(nopython=nopython, fastmath=fastmath, cache=cache)
def np_min(array, axis):
    return np_apply_along_axis(np.min, axis, array)


@nb.jit(nopython=nopython, fastmath=fastmath, cache=cache)
def np_all(array, axis):
    return np_apply_along_axis(np.all, axis, array)


@nb.jit(nopython=nopython, fastmath=fastmath, cache=cache)
def np_argmax(array, axis):
    return np_apply_along_axis(np.argmax, axis, array).astype(nb.int_)


@nb.jit(nopython=nopython, fastmath=fastmath, cache=cache)
def np_any(array, axis):
    return np_apply_along_axis(np.any, axis, array).astype(nb.bool_)


@nb.jit(nopython=nopython, fastmath=fastmath, cache=cache)
def euclidean_distance(X):
    arr_1 = np.expand_dims(X, axis=1)
    arr_2 = np.expand_dims(X, axis=0)
    Nd = arr_1.shape[-1]
    Np = arr_1.shape[0]
    arr = np.zeros((Np, Np, Nd))
    for i in range(Nd):
        arr[:, :, i] = (arr_1[:, :, i] - arr_2[:, :, i]) ** 2
    X_dist = np.sum(arr, axis=-1)
    return np.sqrt(X_dist)
