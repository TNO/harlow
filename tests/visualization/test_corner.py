"""Weak tests: does the code run without an error? the figures should be checked by
a human."""

import numpy as np

from harlow.visualization.corner import corner
from tests.test_functions import ackley_nd


def func_nd(x: np.ndarray):
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def test_1d():
    support_range = np.array([[0, 2]]).T
    corner(func=func_nd, support_range=support_range)


def test_2d():
    support_range = np.array([[0, 2], [-1, 1]]).T
    corner(func=func_nd, support_range=support_range)


def test_3d():
    support_range = np.array([[-2, 2], [-1, 1], [-np.pi, np.pi]]).T
    dim_labels = ["$\\theta$", "$A$", "$\\alpha_\\mathrm{x}$"]

    def func(x: np.ndarray):
        x = np.atleast_2d(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        return x1 + x2 ** 2 + np.sin(x3)

    corner(func=func, support_range=support_range)
    corner(func=func, support_range=support_range, dim_labels=dim_labels, iso_value=0)


def test_3d_rp33():
    # https://rprepo.readthedocs.io/en/latest/reliability_problems.html#sec-rp-33
    support_range = np.array([[-4, 4], [-4, 4], [-4, 4]]).T

    def func(x: np.ndarray):
        x = np.atleast_2d(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        g1 = -x1 - x2 - x3 + 3 * np.sqrt(3)
        g2 = -x3 + 3
        return np.amin(np.stack((g1, g2)), 0)

    corner(func=func, support_range=support_range, iso_value=0)


def test_ackley_nd():
    n_dim = 3
    support_range = np.tile(np.array([[-4], [4]]), reps=(1, n_dim))
    corner(func=ackley_nd, support_range=support_range, n_discr=100)
