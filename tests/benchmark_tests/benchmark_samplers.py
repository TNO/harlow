"""
Comparison of Lola-Voronoi and Fuzzy Lola-Voronoi for different dimensions.
The different methods are compared for a given random seed to determine:

* The surrogating wall clock time.
* The loss of efficiency caused by the heuristic, if any.
* That the optimizations are implemented correctly.
"""

import argparse
import json
import math

import numpy as np
from sklearn.metrics import mean_squared_error

from harlow.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.helper_functions import latin_hypercube_sampling
from harlow.lola_voronoi import LolaVoronoi
from harlow.probabilistic_sampling import Probabilistic_sampler
from harlow.random_sampling import Latin_hypercube_sampler
from harlow.surrogate_model import VanillaGaussianProcess
from tests.integration_tests.test_functions import hartmann, peaks_2d

domains_lower_bound = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
domains_upper_bound = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# domains_lower_bound = np.array([-8., -8.])
# domains_upper_bound = np.array([8., 8.])
n_initial_point = 15
n_new_points_per_iteration = 1
rmse_criterium = 0.001
np.random.seed(0)
n_iter_sampling = 10
n_iter_runs = 2
compare = False
metric = "rmse"
stop_thresh = 0.01  # For RMSE or 0.005 - 0.0025


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def rrse(x, y):
    # print('Original', x)
    dims = x.shape
    x_bar = np.mean(y)
    x_bar_arr = np.zeros(dims)
    x_bar_arr.fill(x_bar)
    # test = (x - x_bar)
    # print('After minus', test)
    return math.sqrt(mean_squared_error(x, y) / mean_squared_error(x, x_bar_arr))


# def rrse(actual: np.ndarray, predicted: np.ndarray):
#     """ Root Relative Squared Error """
#     return np.sqrt(np.sum(np.square(actual - predicted)) /
#     np.sum(np.square(actual - np.mean(actual))))


def create_test_set_2D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))

    return test_X, test_y


def create_test_set_6D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = hartmann(test_X).reshape((-1, 1))

    return test_X, test_y


def dict_to_array(list_of_dicts, key):
    arr_0 = list_of_dicts[0][key][-1]
    for _, lst in enumerate(list_of_dicts[1::]):
        lst = lst[key]
        arr_0 = np.vstack((arr_0, lst[-1]))
    return arr_0


def steps_to_array(list_of_dicts, key):
    arr_0 = list_of_dicts[0][key]
    for _, lst in enumerate(list_of_dicts[1::]):
        lst = lst[key]
        arr_0 = np.vstack((arr_0, lst))
    return arr_0


def run_benchmark(name, crt, method, problem, test_size):
    sampling_res_list = []

    if problem == 2:
        test_X, test_y = create_test_set_2D(
            domains_lower_bound, domains_upper_bound, test_size
        )
        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = peaks_2d(start_points_X).reshape((-1, 1))
        target_func = peaks_2d

    elif problem == 6:
        test_X, test_y = create_test_set_6D(
            domains_lower_bound, domains_upper_bound, test_size
        )

        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = hartmann(start_points_X).reshape((-1, 1))
        target_func = hartmann

    elif problem == 8:
        pass

    sampling_run_res = test_sampler(
        start_points_X,
        start_points_y,
        domains_lower_bound,
        domains_upper_bound,
        test_X,
        test_y,
        crt,
        method,
        target_func,
        name,
    )

    print(sampling_run_res)
    sampling_res_list.append(sampling_run_res)

    # Save to json for plotting etc (Doesnt work for FLOLA)!!
    with open("{}_results".format(name), "w") as fout:
        json.dump(sampling_run_res, fout)
        # plot_results.append(res)

        # res_plot = create_flv_figures(plot_results)
        # writer.add_figure('Iter {}'.format(_i), res_plot, global_step=_i)

        # writer.add_scalar("RMSE", res.get('score')[-1],
        # res.get('iteration')[-1])
        # writer.add_scalar("Gen time", res.get('gen_time')[-1],
        # res.get('iteration')[-1])
        # writer.add_scalar("Fit time", res.get('fit_time')[-1],
        # res.get('iteration')[-1])

    return (sampling_res_list,)


def test_sampler(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    metric,
    sampler,
    target_f,
    name,
):

    surrogate_model = VanillaGaussianProcess()
    # ............................
    # Surrogating
    # ............................
    if sampler == "FLOLA":
        lv = FuzzyLolaVoronoi(
            target_function=target_f,
            surrogate_model=surrogate_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=metric,
            run_name=name,
        )
    elif sampler == "LOLA":
        lv = LolaVoronoi(
            target_function=target_f,
            surrogate_model=surrogate_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=metric,
            run_name=name,
        )
    elif sampler == "Prob":
        lv = Probabilistic_sampler(
            target_function=target_f,
            surrogate_model=surrogate_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=metric,
            run_name=name,
        )
    elif sampler == "Random":
        lv = Latin_hypercube_sampler(
            target_function=target_f,
            surrogate_model=surrogate_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=metric,
            run_name=name,
        )

    lv.sample(
        n_initial_point=n_initial_point,
        n_iter=n_iter_sampling,
        n_new_point_per_iteration=n_new_points_per_iteration,
        stopping_criterium=stop_thresh,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--sampler", default="FLOLA", type=str, help="Name of the sampler"
    )
    parser.add_argument(
        "-p",
        "--problem",
        default=6,
        type=int,
        help="Dimensionality of the problem to solve",
    )
    args = parser.parse_args()

    TEST_SIZE = 500
    if metric == "rmse":
        criterion = rmse
    elif metric == "rrse":
        criterion = rrse

    run_name = "Bench_{}_sampler".format(args.sampler)
    print(run_name)
    flv_sampling_out = run_benchmark(
        run_name, criterion, args.sampler, args.problem, TEST_SIZE
    )
