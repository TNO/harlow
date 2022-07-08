"""
Helper functions for the adaptive sampling strategies.
"""

import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt.sampler import Lhs
from skopt.space import Space

tfd = tfp.distributions


# TODO RRSE code should be checked for correctness
def _rrse(x, y):
    dims = x.shape
    x_bar = np.mean(y)
    x_bar_arr = np.zeros(dims)
    x_bar_arr.fill(x_bar)

    return math.sqrt(mean_squared_error(x, y) / mean_squared_error(x, x_bar_arr))


def evaluate(metric, model, test_points_X, test_points_y):
    """
    Evaluate user specified metric for the current iteration

    Returns:
    """
    score_mtrx = np.zeros((len(metric), len(model)))
    count_metric = 0
    count_model = 0
    if metric is None or test_points_X is None:
        score_mtrx[count_metric, count_model] = 0.0
    else:
        for m in model:
            count_metric = 0
            for metric_func in metric:
                score_mtrx[count_metric, count_model] = metric_func(
                    m.predict(test_points_X), test_points_y[:, count_model]
                )
                count_metric += 1
            count_model += 1

    return score_mtrx


def normalized_response(model: object, X: np.ndarray):
    preds = model.predict(X)
    return (model.predict(X) - np.min(preds)) / (np.max(preds) - np.min(preds))


def nrmse(model: object, test_points_X: np.ndarray, test_points_y: np.ndarray):
    """normalized root mean square error"""
    preds = model.predict(test_points_X)

    return np.sqrt(
        1
        / len(test_points_y)
        * np.sum(
            np.square(
                (test_points_y - preds)
                / (np.max(test_points_y) - np.min(test_points_y))
            )
        )
    )


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """Root Relative Squared Error"""
    return np.sqrt(
        np.sum(np.square(actual - predicted))
        / np.sum(np.square(actual - np.mean(actual)))
    )


def NLL(y, distr):
    return -distr.log_prob(y)


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def mae(x, y):
    return mean_absolute_error(x, y)


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
