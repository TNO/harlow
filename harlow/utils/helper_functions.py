"""
Helper functions for the adaptive sampling strategies.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from skopt.sampler import Lhs
from skopt.space import Space

tfd = tfp.distributions


def evaluate_modellist(metric, model, test_points_X, test_points_y):
    """
    Evaluate user specified metric for the current iteration

    Returns:
    """
    count_model = 0
    metric_dict = {}
    if not isinstance(metric, list):
        metric = [metric]
    if metric is None or test_points_X is None:
        raise ValueError
    else:
        for metric_fun in metric:
            scores = []
            for m in model:
                scores.append(
                    metric_fun(m.predict(test_points_X), test_points_y[:, count_model])
                )
            metric_dict[metric_fun.__name__] = scores

    return metric_dict


def evaluate(metric, true_y, predicted_y):
    """
    Evaluate user specified metric for the current iteration

    Returns:
    """
    metric_dict = {}
    if not isinstance(metric, list):
        metric = [metric]
    if metric is None or true_y is None:
        raise ValueError
    else:
        for metric_fun in metric:
            scores = []
            for m in range(predicted_y.shape[1]):
                scores.append(metric_fun(true_y[:, m], predicted_y[:, m]))
            metric_dict[metric_fun.__name__] = scores

    return metric_dict


def normalized_response(model: object, X: np.ndarray):
    preds = model.predict(X)
    return (model.predict(X) - np.min(preds)) / (np.max(preds) - np.min(preds))


def NLL(y, distr):
    return -distr.log_prob(y)


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
