"""
Helper functions for the adaptive sampling strategies.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from skopt.sampler import Lhs
from skopt.space import Space
from harlow.utils.transforms import TensorTransform

tfd = tfp.distributions
output_transform = TensorTransform()

def evaluate_modellist(metric, model, test_points_X, test_points_y):
    """
    Evaluate user specified metric for the current iteration

    Returns:
    """
    print('Eval modelist TestX shapes', test_points_X.shape)
    metric_dict = {}
    if not isinstance(metric, list):
        metric = [metric]
    if metric is None or test_points_X is None:
        raise ValueError
    else:
        # print('Eval shapes X, Y', test_points_X.shape, test_points_y.shape)
        for metric_fun in metric:
            count_model = 0
            scores = []
            for m in model:
                pred = output_transform.reverse(m.predict(test_points_X))
                # print('Eval prediction shaes', pred.shape)
                scores.append(
                    metric_fun(pred, test_points_y[:, count_model])
                )
                count_model +=1
            metric_dict[metric_fun.__name__] = scores
    print(metric_dict)
    return metric_dict


def evaluate(metric, true_y, predicted_y):
    """
    Evaluate user specified metric for the current iteration

    Returns:
    """
    count_model = 0
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
            print(metric_dict)
    return metric_dict


def normalized_response(model: object, X: np.ndarray):
    preds = output_transform.reverse(model.predict(X))
    return (preds - np.min(preds)) / (np.max(preds) - np.min(preds))


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
