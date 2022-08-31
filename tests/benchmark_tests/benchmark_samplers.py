"""
Comparison of Lola-Voronoi and Fuzzy Lola-Voronoi for different dimensions.
The different methods are compared for a given random seed to determine:

* The surrogating wall clock time.
* The loss of efficiency caused by the heuristic, if any.
* That the optimizations are implemented correctly.
"""

import argparse
import json
import os

import numpy as np

from harlow.sampling import (
    FuzzyLolaVoronoi,
    LatinHypercube,
    LolaVoronoi,
    ProbabilisticSampler,
)
from harlow.surrogating import VanillaGaussianProcess
from harlow.utils.helper_functions import latin_hypercube_sampling, mae, rmse, rrse
from harlow.utils.test_functions import hartmann, peaks_2d, stybtang

np.random.seed(0)
stop_thresh = 0.01  # For RMSE or 0.005 - 0.0025
stop_thresh = None


def create_test_set_2D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))

    return test_X, test_y


def create_test_set_6D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = hartmann(test_X).reshape((-1, 1))

    return test_X, test_y


def create_test_set_8D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = stybtang(test_X).reshape((-1, 1))

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


def run_benchmark(
    name, crt, method, adapt_steps, n_new_points, problem, n_initial_point, test_size
):
    sampling_res_list = []
    save_path = os.path.join("saves", name)
    os.makedirs(save_path, exist_ok=True)

    if problem == 2:
        domains_lower_bound = np.array([-8.0, -8.0])
        domains_upper_bound = np.array([8.0, 8.0])
        test_X, test_y = create_test_set_2D(
            domains_lower_bound, domains_upper_bound, test_size
        )
        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = peaks_2d(start_points_X).reshape((-1, 1))
        target_func = peaks_2d

    elif problem == 6:
        domains_lower_bound = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        domains_upper_bound = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        test_X, test_y = create_test_set_6D(
            domains_lower_bound, domains_upper_bound, test_size
        )

        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = hartmann(start_points_X).reshape((-1, 1))
        target_func = hartmann

    elif problem == 8:
        domains_lower_bound = np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        domains_upper_bound = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        test_X, test_y = create_test_set_8D(
            domains_lower_bound, domains_upper_bound, test_size
        )
        # print('TEST_X, TEST_y', test_X, test_X.shape, test_y, test_y.shape)
        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = stybtang(start_points_X).reshape((-1, 1))
        # print('START_X, START_y', start_points_X, start_points_X.shape,
        # start_points_y, start_points_y.shape)
        target_func = stybtang

    sampling_run_res = test_sampler(
        start_points_X,
        start_points_y,
        domains_lower_bound,
        domains_upper_bound,
        test_X,
        test_y,
        crt,
        method,
        adapt_steps,
        n_initial_point,
        n_new_points,
        target_func,
        name,
        save_path,
    )

    sampling_res_list.append(sampling_run_res)

    # Save to json for plotting etc !!
    json_name = "{}_results_with_score_{}".format(
        name, sampling_run_res.get("score")[-1]
    )
    json_save_path = os.path.join("json_saves", json_name)
    os.makedirs(json_save_path, exist_ok=True)
    with open(json_name, "w") as fout:
        sampling_run_res["step_x"] = [i.tolist() for i in sampling_run_res["step_x"]]
        sampling_run_res["step_y"] = [i.tolist() for i in sampling_run_res["step_y"]]
        sampling_run_res["score"] = [i.tolist() for i in sampling_run_res["score"]]
        json.dump(sampling_run_res, fout)

    return sampling_res_list


def test_sampler(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    metric,
    sampler,
    n_iter_sampling,
    n_initial_point,
    n_new_points_per_iteration,
    target_f,
    name,
    save_path,
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
            save_dir=save_path,
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
            save_dir=save_path,
        )
    elif sampler == "Prob":
        lv = ProbabilisticSampler(
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
        lv = LatinHypercube(
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
    parser.add_argument(
        "-i",
        "--init_p",
        default=15,
        type=int,
        help="Number of initial points to start sampling",
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="rmse",
        type=str,
        help="Define the meassure function",
    )
    parser.add_argument(
        "-st",
        "--steps",
        default=3000,
        type=int,
        help="Number of iterative adaptive sampling steps",
    )
    parser.add_argument(
        "-n",
        "--n_points_iter",
        default=1,
        type=int,
        help="Number of points we add at every adaptive sampling steps",
    )
    args = parser.parse_args()

    TEST_SIZE = 500
    if args.metric == "all":
        metric = [rmse, rrse, mae]
    else:
        metric = [rmse]

    run_name = "Bench_{}_with_{}_initial_points_on_{}D_problem".format(
        args.sampler, args.init_p, args.problem
    )
    print(run_name)
    flv_sampling_out = run_benchmark(
        run_name,
        metric,
        args.sampler,
        args.steps,
        args.n_points_iter,
        args.problem,
        args.init_p,
        TEST_SIZE,
    )
