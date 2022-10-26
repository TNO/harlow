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

from harlow.sampling.cv_voronoi import CVVoronoi
from harlow.sampling.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.surrogating.surrogate_model import (
    ModelListGaussianProcess,
    VanillaGaussianProcess,
)
from harlow.utils.helper_functions import latin_hypercube_sampling

# I201
from harlow.utils.test_functions import peaks_2d_multivariate

domain_lower_bound = np.array([-8, -8])
domain_upper_bound = np.array([8, 8])
n_initial_point = 25
n_new_points_per_iteration = 1
rmse_criterium = 0.05
np.random.seed(123)
n_iter_sampling = 500
n_iter_runs = 100


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def create_test_set(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(domain_lower_bound, domain_upper_bound, n)
    test_y = peaks_2d_multivariate(test_X)

    return test_X, test_y


def run_2D_test():
    fuzzy_sampling_results = []
    test_X, test_y = create_test_set(domain_lower_bound, domain_upper_bound, 500)

    start_points_X = latin_hypercube_sampling(
        domain_lower_bound, domain_upper_bound, n_initial_point
    )
    start_points_y = peaks_2d_multivariate(start_points_X)

    fuzzy_sampling_results.append(
        test_2D_cvVoronoi_sampling(
            start_points_X,
            start_points_y,
            domain_lower_bound,
            domain_upper_bound,
            test_X,
            test_y,
            n_iter_sampling,
        )
    )


def run_2D_FLV_test():
    fuzzy_sampling_results = []
    test_X, test_y = create_test_set(domain_lower_bound, domain_upper_bound, 500)

    start_points_X = latin_hypercube_sampling(
        domain_lower_bound, domain_upper_bound, n_initial_point
    )
    start_points_y = peaks_2d_multivariate(start_points_X)

    fuzzy_sampling_results.append(
        test_2D_FLV_sampling(
            start_points_X,
            start_points_y,
            domain_lower_bound,
            domain_upper_bound,
            test_X,
            test_y,
            n_iter_sampling,
        )
    )


def test_2D_cvVoronoi_sampling(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    surrogate_model = VanillaGaussianProcess

    # ............................
    # Surrogating
    # ............................
    lv = CVVoronoi(
        target_function=peaks_2d_multivariate,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
        fit_points_x=start_points_X,
        fit_points_y=start_points_y,
        test_points_x=test_X,
        test_points_y=test_y,
        run_name=None,
    )
    # main_start = time.time()
    lv.sample(
        n_initial_points=n_initial_point,
        n_new_points_per_iteration=n_new_points_per_iteration,
        max_n_iterations=200,
    )

    return {
        "step_x": lv.step_x,
        "step_y": lv.step_y,
        "score": lv.step_score,
        "iteration": lv.step_iter,
        "gen_time": lv.step_gen_time,
        "fit_time": lv.step_fit_time,
    }


def test_2D_FLV_sampling(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    start_points_X = np.tile(start_points_X, (2, 1, 1))
    test_X = np.tile(test_X, (2, 1, 1))
    surrogate_model = ModelListGaussianProcess(
        ["m1", "m2"], list_params=[[0, 1], [0, 1]]
    )

    # ............................
    # Surrogating
    # ............................
    lv = FuzzyLolaVoronoi(
        target_function=peaks_2d_multivariate,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
        fit_points_x=start_points_X,
        fit_points_y=start_points_y,
        test_points_x=test_X,
        test_points_y=test_y,
        run_name=None,
    )
    # main_start = time.time()
    lv.sample(
        n_initial_points=n_initial_point,
        n_new_points_per_iteration=n_new_points_per_iteration,
        max_n_iterations=20,
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
    # run_2D_FLV_test()
