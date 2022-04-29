import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from harlow.utils.distance import pdist_condensed, pdist_full_matrix


def test_pdist_condensed():
    n_points = [10, 100, 1000]
    n_repeats = [100, 10, 5]
    n_dim = 4

    plot_fig = True
    compare_runtime = False

    t_harlow = np.empty(len(n_points))
    t_scipy = np.empty(len(n_points))
    for ii, (n_point, n_repeat) in enumerate(zip(n_points, n_repeats)):
        x_mx = np.random.rand(n_point, n_dim)

        dist = pdist_condensed(x_mx)
        dist_expected = distance.pdist(x_mx)

        # check if the correct values are returned
        np.testing.assert_array_almost_equal(dist, dist_expected)

        # runtime measurement
        if compare_runtime:
            t0 = time.time()
            for _ in range(n_repeat):
                pdist_condensed(x_mx)
            t_harlow[ii] = (time.time() - t0) / n_repeat

            t0 = time.time()
            for _ in range(n_repeat):
                distance.pdist(x_mx)
            t_scipy[ii] = (time.time() - t0) / n_repeat

    if compare_runtime and plot_fig:
        fig, ax = plt.subplots()
        ax.plot(n_points, t_scipy / t_harlow, "o-")
        ax.set_xscale("log")
        ax.set_xlabel("Number of vectors (points)")
        ax.set_ylabel("$t_\\mathrm{scipy}/t_\\mathrm{harlow}$")
        ax.set_title("`pdist` runtime with condensed output")
        ax.grid()


def test_pdist_full_matrix():
    n_points = [10, 100, 1000]
    n_repeats = [100, 10, 5]
    n_dim = 4

    plot_fig = True
    compare_runtime = False

    t_harlow = np.empty(len(n_points))
    t_scipy = np.empty(len(n_points))
    for ii, (n_point, n_repeat) in enumerate(zip(n_points, n_repeats)):
        x_mx = np.random.rand(n_point, n_dim)

        dist = pdist_full_matrix(x_mx)
        dist_expected = distance.squareform(distance.pdist(x_mx))

        # check if the correct values are returned
        np.testing.assert_array_almost_equal(dist, dist_expected)

        # runtime measurement
        if compare_runtime:
            t0 = time.time()
            for _ in range(n_repeat):
                pdist_full_matrix(x_mx)
            t_harlow[ii] = (time.time() - t0) / n_repeat

            t0 = time.time()
            for _ in range(n_repeat):
                distance.squareform(distance.pdist(x_mx))
            t_scipy[ii] = (time.time() - t0) / n_repeat

    if compare_runtime and plot_fig:
        fig, ax = plt.subplots()
        ax.plot(n_points, t_scipy / t_harlow, "o-")
        ax.set_xscale("log")
        ax.set_xlabel("Number of vectors (points)")
        ax.set_ylabel("$t_\\mathrm{scipy}/t_\\mathrm{harlow}$")
        ax.set_title("`pdist` runtime with full mx output")
        ax.grid()
