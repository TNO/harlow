"""Benchmarktest from Crombecq et al. (2011). Benchmarks LV and
Probabilistic sampling vs random sampling. """
import json
import math
import time

import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error

from harlow.helper_functions import latin_hypercube_sampling
from harlow.lola_voronoi import LolaVoronoi
from harlow.probabilistic_sampling import Probabilistic_sampler
from harlow.surrogate_model import VanillaGaussianProcess
from tests.integration_tests.test_functions import peaks_2d

domains_lower_bound = [np.array([-3, -3]), np.array([-5, -5]), np.array([-8, -8])]
domains_upper_bound = [np.array([3, 3]), np.array([5, 5]), np.array([8, 8])]
n_initial_point = 10
n_new_points_per_iteration = 1
rmse_criterium = 0.05


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def create_test_set(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))

    return test_X, test_y


def run_benchmark():
    lv_results = []
    ps_results = []
    random_results = []

    for i in range(0, len(domains_upper_bound)):
        test_X, test_y = create_test_set(
            domains_lower_bound[i], domains_upper_bound[i], 500
        )

        start_points_X = latin_hypercube_sampling(
            domains_lower_bound[i], domains_upper_bound[i], n_initial_point
        )
        start_points_y = peaks_2d(start_points_X).reshape((-1, 1))

        lv_results.append(
            test_LV_sampling(
                start_points_X,
                start_points_y,
                domains_lower_bound[i],
                domains_upper_bound[i],
                test_X,
                test_y,
            )
        )

        ps_results.append(
            test_probabilistic_sampling(
                start_points_X,
                start_points_y,
                domains_lower_bound[i],
                domains_upper_bound[i],
                test_X,
                test_y,
            )
        )

        random_results.append(
            test_random_sampling(
                start_points_X,
                start_points_y,
                domains_lower_bound[i],
                domains_upper_bound[i],
                test_X,
                test_y,
            )
        )

    with open("LV_results", "w") as fout:
        json.dump(lv_results, fout)

    with open("PS_results", "w") as fout:
        json.dump(ps_results, fout)

    with open("RAND_results", "w") as fout:
        json.dump(random_results, fout)


def test_LV_sampling(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
):

    surrogate_model = VanillaGaussianProcess()

    # ............................
    # Surrogating
    # ............................
    lv = LolaVoronoi(
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
    main_start = time.time()
    lv.sample(
        n_iter=None,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
        stopping_criterium=rmse_criterium,
    )

    return {
        "domain_lower_bound": domain_lower_bound.tolist(),
        "domain_upper_bound": domain_upper_bound.tolist(),
        "iterations": lv.iterations,
        "score": lv.score,
        "elapsed_time": time.time() - main_start,
    }


def test_probabilistic_sampling(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
):

    surrogate_model = VanillaGaussianProcess()

    # ............................
    # Surrogating
    # ............................
    ps = Probabilistic_sampler(
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
    main_start = time.time()
    ps.sample(
        n_iter=None,
        n_initial_point=n_initial_point,
        stopping_criterium=rmse_criterium,
    )

    return {
        "domain_lower_bound": domain_lower_bound.tolist(),
        "domain_upper_bound": domain_upper_bound.tolist(),
        "iterations": ps.iterations,
        "score": ps.score,
        "elapsed_time": time.time() - main_start,
    }


def test_random_sampling(
    points_x, points_y, domain_lower_bound, domain_upper_bound, test_X, test_y
):

    main_start = time.time()

    start_time = time.time()
    surrogate_model = VanillaGaussianProcess()
    surrogate_model.fit(points_x, points_y)
    logger.info(f"Fitted a new surrogate model in {time.time() - start_time} sec.")

    score = rmse(surrogate_model.predict(test_X), test_y)
    iteration = 1
    while score > rmse_criterium:

        X_new = latin_hypercube_sampling(
            n_sample=1,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
        )
        y_new = peaks_2d(X_new).reshape((-1, 1))
        points_x = np.concatenate((points_x, X_new))
        points_y = np.concatenate((points_y, y_new))

        start_time = time.time()
        surrogate_model.fit(points_x, points_y)
        logger.info(f"Fitted a surrogate model in {time.time() - start_time} sec.")
        iteration += 1
        score = rmse(surrogate_model.predict(test_X), test_y)
        logger.info(f"Score {score}")

    logger.info(f"Algorithm converged in {iteration} iterations")
    logger.info(f"Algorithm converged with score {score}")

    return {
        "domain_lower_bound": domain_lower_bound.tolist(),
        "domain_upper_bound": domain_upper_bound.tolist(),
        "iterations": iteration,
        "score": score,
        "elapsed_time": time.time() - main_start,
    }
