"""
Comparison of Lola-Voronoi and Fuzzy Lola-Voronoi for different dimensions.
The different methods are compared for a given random seed to determine:

* The surrogating wall clock time.
* The loss of efficiency caused by the heuristic, if any.
* That the optimizations are implemented correctly.
"""

import math

import numpy as np
from sklearn.metrics import mean_squared_error

from harlow.helper_functions import latin_hypercube_sampling
from tests.integration_tests.test_functions import peaks_2d

domains_lower_bound = np.array([-8, -8])
domains_upper_bound = np.array([8, 8])
n_initial_point = 15
n_new_points_per_iteration = 1
rmse_criterium = 0.001
np.random.seed(123)
n_iter_sampling = 30
n_iter_runs = 100


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def create_test_set_2D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))

    return test_X, test_y


def create_test_set_3D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))

    return test_X, test_y
