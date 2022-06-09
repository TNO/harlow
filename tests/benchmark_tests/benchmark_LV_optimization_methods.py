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
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from harlow.sampling.lola_voronoi import LolaVoronoi
from harlow.surrogating.surrogate_model import GaussianProcess
from harlow.utils.helper_functions import latin_hypercube_sampling
from tests.integration_tests.test_functions import peaks_2d

domains_lower_bound = np.array([-8, -8])
domains_upper_bound = np.array([8, 8])
n_initial_point = 5
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


def run_benchmark():
    full_sampling_results = []
    median_sampling_results = []
    new_sampling_results = []
    both_sampling_results = []

    for _i in range(n_iter_runs):
        test_X, test_y = create_test_set(domains_lower_bound, domains_upper_bound, 500)

        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )
        start_points_y = peaks_2d(start_points_X).reshape((-1, 1))

        full_sampling_results.append(
            test_sampling_full(
                start_points_X,
                start_points_y,
                domains_lower_bound,
                domains_upper_bound,
                test_X,
                test_y,
                n_iter_sampling,
            )
        )

        median_sampling_results.append(
            test_sampling_median(
                start_points_X,
                start_points_y,
                domains_lower_bound,
                domains_upper_bound,
                test_X,
                test_y,
                n_iter_sampling,
            )
        )

        new_sampling_results.append(
            test_sampling_new(
                start_points_X,
                start_points_y,
                domains_lower_bound,
                domains_upper_bound,
                test_X,
                test_y,
                n_iter_sampling,
            )
        )

        both_sampling_results.append(
            test_sampling_both(
                start_points_X,
                start_points_y,
                domains_lower_bound,
                domains_upper_bound,
                test_X,
                test_y,
                n_iter_sampling,
            )
        )

    return (
        full_sampling_results,
        median_sampling_results,
        new_sampling_results,
        both_sampling_results,
    )


def test_sampling_full(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    surrogate_model = GaussianProcess()

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


def test_sampling_median(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    surrogate_model = GaussianProcess()

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
    # main_start = time.time()
    lv.sample(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
        ignore_old_neighborhoods=False,
        ignore_far_neighborhoods=True,
    )

    return {
        "step_x": lv.step_x,
        "step_y": lv.step_y,
        "score": lv.step_score,
        "iteration": lv.step_iter,
        "gen_time": lv.step_gen_time,
        "fit_time": lv.step_fit_time,
    }


def test_sampling_new(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    surrogate_model = GaussianProcess()

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
    # main_start = time.time()
    lv.sample(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
        ignore_old_neighborhoods=True,
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


def test_sampling_both(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
):

    surrogate_model = GaussianProcess()

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
    # main_start = time.time()
    lv.sample(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
        ignore_old_neighborhoods=True,
        ignore_far_neighborhoods=True,
    )

    return {
        "step_x": lv.step_x,
        "step_y": lv.step_y,
        "score": lv.step_score,
        "iteration": lv.step_iter,
        "gen_time": lv.step_gen_time,
        "fit_time": lv.step_fit_time,
    }


(
    full_sampling_out,
    median_sampling_out,
    new_sampling_out,
    both_sampling_out,
) = run_benchmark()


# =======================================================================
# PLOTTING
# =======================================================================

# Meshgrid of function solution
npts_grid = 100
x_min = -8.0
x_max = 8.0
x = np.linspace(x_min, x_max, npts_grid)
y = np.linspace(x_min, x_max, npts_grid)
Z = np.zeros((npts_grid, npts_grid))
full_samples_list = []
median_sampling_list = []
new_sampling_list = []
both_sampling_list = []

for idx_x, xi in enumerate(x):
    for idx_y, yi in enumerate(y):
        Z[idx_x, idx_y] = peaks_2d([xi, yi])


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


iterations = steps_to_array(full_sampling_out, "iteration")
ql = 0.1
qh = 1.0 - ql
fig, ax = plt.subplots(3, 1, figsize=(12, 9))

full_score = steps_to_array(full_sampling_out, "score")
median_score = steps_to_array(median_sampling_out, "score")
new_score = steps_to_array(new_sampling_out, "score")
both_score = steps_to_array(both_sampling_out, "score")
full_time = steps_to_array(full_sampling_out, "gen_time")
median_time = steps_to_array(median_sampling_out, "gen_time")
new_time = steps_to_array(new_sampling_out, "gen_time")
both_time = steps_to_array(both_sampling_out, "gen_time")

# Mean score
ax[0].plot(np.mean(full_score, axis=0), label="Full")
ax[0].plot(np.mean(median_score, axis=0), label="Median", linestyle="dashed")
ax[0].plot(np.mean(new_score, axis=0), label="New", linestyle="dashed")
ax[0].plot(np.mean(both_score, axis=0), label="Both", linestyle="dashed")
ax[0].set_ylabel("RMSE")

# Score quantiles
ax[0].fill_between(
    iterations[0],
    np.quantile(full_score, ql, axis=0),
    np.quantile(full_score, qh, axis=0),
    alpha=0.5,
)
ax[0].fill_between(
    iterations[0],
    np.quantile(median_score, ql, axis=0),
    np.quantile(median_score, qh, axis=0),
    linestyle="dashed",
    alpha=0.5,
)
ax[0].fill_between(
    iterations[0],
    np.quantile(new_score, ql, axis=0),
    np.quantile(new_score, qh, axis=0),
    linestyle="dashed",
    alpha=0.5,
)
ax[0].fill_between(
    iterations[0],
    np.quantile(both_score, ql, axis=0),
    np.quantile(both_score, qh, axis=0),
    linestyle="dashed",
    alpha=0.5,
)
ax[0].set_ylabel("RMSE")

# Mean generation time
ax[1].plot(np.mean(full_time, axis=0), label="Full")
ax[1].plot(np.mean(median_time, axis=0), label="Median")
ax[1].plot(np.mean(new_time, axis=0), label="New")
ax[1].plot(np.mean(both_time, axis=0), label="Both")
ax[1].set_xlabel("Iteration no.")
ax[1].set_ylabel("Gen. time [s]")

# Generation time quantiles
ax[1].fill_between(
    iterations[0],
    np.quantile(full_time, ql, axis=0),
    np.quantile(full_time, qh, axis=0),
    alpha=0.5,
)
ax[1].fill_between(
    iterations[0],
    np.quantile(median_time, ql, axis=0),
    np.quantile(median_time, qh, axis=0),
    alpha=0.5,
)
ax[1].fill_between(
    iterations[0],
    np.quantile(new_time, ql, axis=0),
    np.quantile(new_time, qh, axis=0),
    alpha=0.5,
)
ax[1].fill_between(
    iterations[0],
    np.quantile(both_time, ql, axis=0),
    np.quantile(both_time, qh, axis=0),
    alpha=0.5,
)
ax[1].set_xlabel("Iteration no.")
ax[1].set_ylabel("Gen. time [s]")

# Test points scatter plot for all iterations
ax[2].contourf(x, y, Z)
ax[2].scatter(
    dict_to_array(full_sampling_out, "step_x")[:, 0],
    dict_to_array(full_sampling_out, "step_x")[:, 1],
    label="Full",
    alpha=0.5,
)
ax[2].scatter(
    dict_to_array(median_sampling_out, "step_x")[:, 0],
    dict_to_array(median_sampling_out, "step_x")[:, 1],
    label="Median",
    alpha=0.5,
)
ax[2].scatter(
    dict_to_array(new_sampling_out, "step_x")[:, 0],
    dict_to_array(new_sampling_out, "step_x")[:, 1],
    label="New",
    alpha=0.5,
)
ax[2].scatter(
    dict_to_array(both_sampling_out, "step_x")[:, 0],
    dict_to_array(both_sampling_out, "step_x")[:, 1],
    label="Both",
    alpha=0.5,
)


ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
ax[2].legend()
plt.show()
