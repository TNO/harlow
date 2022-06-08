"""
Comparison of Lola-Voronoi and Fuzzy Lola-Voronoi for different dimensions.
The different methods are compared for a given random seed to determine:

* The surrogating wall clock time.
* The loss of efficiency caused by the heuristic, if any.
* That the optimizations are implemented correctly.
"""

import math
import argparse
import time
from loguru import logger
import json

import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error

from harlow.helper_functions import latin_hypercube_sampling
from harlow.random_sampling import Latin_hypercube_sampler
from harlow.lola_voronoi import LolaVoronoi
from harlow.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.probabilistic_sampling import Probabilistic_sampler
from harlow.surrogate_model import VanillaGaussianProcess, Vanilla_NN
# from tests.integration_tests.test_functions import peaks_2d, hartmann

domains_lower_bound = np.array([0., 0., 0., 0., 0., 0.])
domains_upper_bound = np.array([1., 1., 1., 1., 1., 1.])
# domains_lower_bound = np.array([-8., -8.])
# domains_upper_bound = np.array([8., 8.])
n_initial_point = 15
n_new_points_per_iteration = 1
rmse_criterium = 0.001
np.random.seed(0)
n_iter_sampling = 10
n_iter_runs = 2
compare = False
metric = 'rmse'
stop_thresh = 0.01 # For RMSE or 0.005 - 0.0025


def peaks_2d(x: np.ndarray) -> np.ndarray:
    # https://nl.mathworks.com/help/matlab/ref/peaks.html
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    return (
        3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-(x1 ** 2) - x2 ** 2)
        - 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2 ** 2)
    )

def hartmann(X):
    n = X.shape[0]
    results = []
    outer = 0.0
    for i in range(n):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = X[i, jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2

            new = alpha[ii] * np.exp(-inner)
            outer = outer + new
        results.append(-(2.58 + outer) / 1.94)

    return np.asarray(results)


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))

def rrse(x,y):
    # print('Original', x)
    dims = x.shape
    x_bar = np.mean(y)
    x_bar_arr = np.zeros(dims)
    x_bar_arr.fill(x_bar)
    # test = (x - x_bar)
    # print('After minus', test)
    return math.sqrt(mean_squared_error(x,y) / mean_squared_error(x, x_bar_arr))

# def rrse(actual: np.ndarray, predicted: np.ndarray):
#     """ Root Relative Squared Error """
#     return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


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
    writer = SummaryWriter(comment='-' + name)
    frame_idx = 0
    sampling_res_list = []

    plot_results = []

    if problem == 2:
        test_X, test_y = create_test_set_2D(domains_lower_bound,
                                    domains_upper_bound, test_size)
        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = peaks_2d(start_points_X).reshape((-1, 1))
        target_func = peaks_2d

    elif problem == 6:
        test_X, test_y = create_test_set_6D(domains_lower_bound,
                                         domains_upper_bound, test_size)

        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = hartmann(start_points_X).reshape((-1, 1))
        target_func = hartmann
    
    elif problem == 8:
        pass

    sampling_run_res =  test_sampler(
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

    #Save to json for plotting etc (Doesnt work for FLOLA)!!
    with open("{}_results".format(name), "w") as fout:
        json.dump(sampling_run_res, fout)
        # plot_results.append(res)

        # res_plot = create_flv_figures(plot_results)
        # writer.add_figure('Iter {}'.format(_i), res_plot, global_step=_i)

        # writer.add_scalar("RMSE", res.get('score')[-1], res.get('iteration')[-1])
        # writer.add_scalar("Gen time", res.get('gen_time')[-1], res.get('iteration')[-1])
        # writer.add_scalar("Fit time", res.get('fit_time')[-1], res.get('iteration')[-1])

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
    if sampler == 'FLOLA':
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
            run_name=name
        )
    elif sampler == 'LOLA':
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
            run_name=name
        )
    elif sampler == 'Prob':
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
    elif sampler == 'Random':
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



if __name__== "__main__":\

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampler", default='FLOLA', type=str, help='Name of the sampler')
    parser.add_argument("-p", "--problem", default=6, type=int, help='Dimensionality of the problem to solve')
    args = parser.parse_args()

    TEST_SIZE = 500
    if metric == 'rmse':
        criterion = rmse
    elif metric == 'rrse':
        criterion = rrse

    run_name = 'Bench_{}_sampler'.format(args.sampler)
    print(run_name)
    flv_sampling_out = run_benchmark(run_name, criterion, args.sampler, args.problem, TEST_SIZE)

