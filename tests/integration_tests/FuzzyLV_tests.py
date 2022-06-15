"""
Comparison of different methods for reducing the optimzal neighbourhood
calculation time. The different methods are compared for a given random
seed to determine:

* The surrogating wall clock time.
* The loss of efficiency caused by the heuristic, if any.
* That the optimizations are implemented correctly.
"""

import math

import numpy as np
from sklearn.metrics import mean_squared_error

from harlow.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.helper_functions import latin_hypercube_sampling
from harlow.surrogate_model import VanillaGaussianProcess
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


def create_test_set(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))

    return test_X, test_y


def run_2D_test():
    fuzzy_sampling_results = []
    test_X, test_y = create_test_set(domains_lower_bound, domains_upper_bound, 500)

    start_points_X = latin_hypercube_sampling(
        domains_lower_bound, domains_upper_bound, n_initial_point
    )
    start_points_y = peaks_2d(start_points_X).reshape((-1, 1))

    fuzzy_sampling_results.append(
        test_2D_fuzzy_sampling(
            start_points_X,
            start_points_y,
            domains_lower_bound,
            domains_upper_bound,
            test_X,
            test_y,
            n_iter_sampling,
        )
    )


def test_2D_fuzzy_sampling(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    surrogate_model = VanillaGaussianProcess()

    # ............................
    # Surrogating
    # ............................
    lv = FuzzyLolaVoronoi(
        target_function=peaks_2d,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
        fit_points_x=start_points_X,
        fit_points_y=start_points_y,
        test_points_x=test_X,
        test_points_y=test_y,
        evaluation_metric=rmse,
    )
    # main_start = time.time()
    lv.sample(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
        ignore_old_neighborhoods=False,
        ignore_far_neighborhoods=False,
    )

    return {
        "step_x": lv.step_x,
        "step_y": lv.step_y,
        "score": lv.step_score,
        "iteration": lv.step_iter,
        "gen_time": lv.step_gen_time,
        "fit_time": lv.step_fit_time,
    }


if __name__ == "__main__":
    run_2D_test()
