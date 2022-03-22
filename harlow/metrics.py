import math

from sklearn.metrics import mean_squared_error, r2_score


def rmse(true, prediction):
    return math.sqrt(mean_squared_error(true, prediction))


def mse(true, prediction):
    return mean_squared_error(true, prediction)


def r2(true, prediction):
    r2_score(true, prediction)
