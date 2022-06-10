"""
Helper functions for the adaptive sampling strategies.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from skopt.sampler import Lhs
from skopt.space import Space

tfd = tfp.distributions


def NLL(y, distr):
    return -distr.log_prob(y)


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=1))


def normal_sp(params):
    return tfd.Normal(
        loc=params[:, 0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:, 1:2])
    )  # both parameters are learnable


def latin_hypercube_sampling(
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_sample: int,
    method="maximin",
):
    domain = np.vstack((domain_lower_bound, domain_upper_bound)).astype(float).T
    space = Space(list(map(tuple, domain)))
    lhs = Lhs(criterion=method, iterations=5000)
    samples = lhs.generate(space.dimensions, n_sample)

    return np.array(samples, dtype=float)

def peaks_2d(x: np.ndarray) -> np.ndarray:
    # https://nl.mathworks.com/help/matlab/ref/peaks.html
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    return (
        3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-(x1 ** 2) - x2 ** 2)
        - 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2 ** 2)
    )

def hartmann(X):
    n = X.shape[0]
    results = []
    outer = 0.0
    for i in range(n):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = X[i, jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2

            new = alpha[ii] * np.exp(-inner)
            outer = outer + new
        results.append(-(2.58 + outer) / 1.94)

    return np.asarray(results)

def stybtang(X):
    #https://www.sfu.ca/~ssurjano/stybtang.html
    n = X.shape[0]
    d = X.shape[1]
    res = np.zeros(n)
    for i in range(n):
        _sum = 0.0
        for j in range(d):
            _sum = (np.power(X[i,j], 4) - 16.0 * np.power(X[i,j], 2) + 5.0 * X[i,j])
        res[i] = 0.5 * _sum

    return res