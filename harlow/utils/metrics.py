import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from harlow.utils.transforms import TensorTransform

output_transform = TensorTransform()

# TODO RRSE code should be checked for correctness
def _rrse(x, y):
    dims = x.shape
    x_bar = np.mean(y)
    x_bar_arr = np.zeros(dims)
    x_bar_arr.fill(x_bar)

    return math.sqrt(mean_squared_error(x, y) / mean_squared_error(x, x_bar_arr))


def nrmse(model: object, test_points_X: np.ndarray, test_points_y: np.ndarray):
    """normalized root mean square error"""
    preds = output_transform.reverse(model.predict(test_points_X))

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


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def mae(x, y):
    return mean_absolute_error(x, y)
