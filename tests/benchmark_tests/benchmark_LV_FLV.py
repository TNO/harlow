"""
Comparison of Lola-Voronoi and Fuzzy Lola-Voronoi for different dimensions.
The different methods are compared for a given random seed to determine:

* The surrogating wall clock time.
* The loss of efficiency caused by the heuristic, if any.
* That the optimizations are implemented correctly.
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error

from harlow.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.helper_functions import latin_hypercube_sampling
from harlow.lola_voronoi import LolaVoronoi
from harlow.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.surrogate_model import VanillaGaussianProcess, Vanilla_NN
# from tests.integration_tests.test_functions import peaks_2d, hartmann

domains_lower_bound = np.array([0., 0., 0., 0., 0., 0.])
domains_upper_bound = np.array([1., 1., 1., 1., 1., 1.])
# domains_lower_bound = np.array([-8., -8.])
# domains_upper_bound = np.array([8., 8.])
n_initial_point = 15
n_new_points_per_iteration = 1
rmse_criterium = 0.001
np.random.seed(123)
n_iter_sampling = 30
n_iter_runs = 5


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))

# def rrse(x,y):
#     # print('Original', x)
#     dims = x.shape
#     x_bar = np.mean(y)
#     x_bar_arr = np.zeros(dims)
#     x_bar_arr.fill(x_bar)
#     # test = (x - x_bar)
#     # print('After minus', test)
#     return math.sqrt(mean_squared_error(x,y) / mean_squared_error(x, x_bar_arr))

def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


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


def create_flv_figures(flv_results):
    # Meshgrid of function solution
    npts_grid = 100
    x_min = -8.0
    x_max = 8.0
    x = np.linspace(x_min, x_max, npts_grid)
    y = np.linspace(x_min, x_max, npts_grid)
    Z = np.zeros((npts_grid, npts_grid))

    for idx_x, xi in enumerate(x):
        for idx_y, yi in enumerate(y):
            Z[idx_x, idx_y] = peaks_2d([xi, yi])

    iterations = steps_to_array(flv_results, "iteration")
    ql = 0.1
    qh = 1.0 - ql
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    flv_score = steps_to_array(flv_results, "score")
    flv_time = steps_to_array(flv_results, "gen_time")
    # print(flv_score, flv_time)
    # Mean score
    ax[0].plot(np.mean(flv_score, axis=0), label="FLV", linestyle="dashed")
    ax[0].set_ylabel("RMSE")
    # Score quantiles
    ax[0].fill_between(
        iterations,
        np.quantile(flv_score, ql, axis=0),
        np.quantile(flv_score, qh, axis=0),
        linestyle="dashed",
        alpha=0.5,
    )
    # Mean generation time
    ax[1].plot(np.mean(flv_time, axis=0), label="FLV")
    ax[1].set_xlabel("Iteration no.")
    ax[1].set_ylabel("Gen. time [s]")
    # Generation time quantiles
    ax[1].fill_between(
        iterations,
        np.quantile(flv_time, ql, axis=0),
        np.quantile(flv_time, qh, axis=0),
        alpha=0.5,
    )
    ax[1].set_xlabel("Iteration no.")
    ax[1].set_ylabel("Gen. time [s]")
    # Test points scatter plot for all iterations
    ax[2].contourf(x, y, Z)
    ax[2].scatter(
        dict_to_array(flv_results, "step_x")[:, 0],
        dict_to_array(flv_results, "step_x")[:, 1],
        label="FLV",
        alpha=0.5,
    )

    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()
    ax[2].legend()
    ax[2].grid()

    return fig

def run_benchmark(name, crt):
    writer = SummaryWriter(comment='-' + name)
    frame_idx = 0
    LV_sampling_results = []
    FLV_sampling_results = []
    plot_results = []

    fig, ax = corner.corner(
        hartmann, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]).T
    )
    fig.show()

    plotting.plot_function(peaks_2d, np.array([[-4, 4], [-4, 4]]), show=True)
    for _i in range(n_iter_runs):
        plot_results.clear()
        test_X, test_y = create_test_set_6D(domains_lower_bound,
                                         domains_upper_bound, 500)
        # test_X, test_y = create_test_set_2D(domains_lower_bound,
        #                                  domains_upper_bound, 500)
        test_X, test_y = create_test_set_2D(domains_lower_bound,
                                         domains_upper_bound, 500)

        start_points_X = latin_hypercube_sampling(
            domains_lower_bound, domains_upper_bound, n_initial_point
        )

        #start_points_y = hartmann(start_points_X).reshape((-1, 1))
        start_points_y = peaks_2d(start_points_X).reshape((-1, 1))

        # LV_sampling_results.append(
        #     test_sampling_LV(
        #         start_points_X,
        #         start_points_y,
        #         domains_lower_bound,
        #         domains_upper_bound,
        #         test_X,
        #         test_y,
        #         n_iter_sampling,
        #     )
        # )
        res = test_sampling_FLV(
                start_points_X,
                start_points_y,
                domains_lower_bound,
                domains_upper_bound,
                test_X,
                test_y,
                n_iter_sampling,
                crt,)
        FLV_sampling_results.append(res)
        plot_results.append(res)

        res_plot = create_flv_figures(plot_results)
        writer.add_figure('Iter {}'.format(_i), res_plot, global_step=_i)

        # writer.add_scalar("RMSE", res.get('score')[-1], res.get('iteration')[-1])
        # writer.add_scalar("Gen time", res.get('gen_time')[-1], res.get('iteration')[-1])
        # writer.add_scalar("Fit time", res.get('fit_time')[-1], res.get('iteration')[-1])

    #     FLV_sampling_results.append(
    #         test_sampling_FLV(
    #             start_points_X,
    #             start_points_y,
    #             domains_lower_bound,
    #             domains_upper_bound,
    #             test_X,
    #             test_y,
    #             n_iter_sampling,
    #         )
    #     )

    return (LV_sampling_results, FLV_sampling_results)



def test_sampling_FLV(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    n_iter,
    metric,
):

    # surrogate_model = Vanilla_NN()
    # surrogate_model.create_model(input_dim=(6,), activation="relu",
    #                              learning_rate=0.01)
    surrogate_model = VanillaGaussianProcess()

    # ............................
    # Surrogating
    # ............................
    lv = FuzzyLolaVoronoi(
        target_function=hartmann,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
        fit_points_x=start_points_X,
        fit_points_y=start_points_y,
        test_points_x=test_X,
        test_points_y=test_y,
        evaluation_metric=metric,
    )
    # main_start = time.time()
    lv.sample(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
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

    if metric == 'rmse':
        criterion = rmse
    elif metric == 'rrse':
        criterion = rrse

    run_name = 't1_fig'
    (lv_sampling_out, flv_sampling_out,) = run_benchmark(run_name, criterion)


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

    iterations = steps_to_array(flv_sampling_out, "iteration")
    ql = 0.1
    qh = 1.0 - ql
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))

    if compare == True:
        lv_score = steps_to_array(lv_sampling_out, "score")
        flv_score = steps_to_array(flv_sampling_out, "score")
        lv_time = steps_to_array(lv_sampling_out, "gen_time")
        flv_time = steps_to_array(flv_sampling_out, "gen_time")
        # Mean score
        ax[0].plot(np.mean(lv_score, axis=0), label="LV")
        ax[0].plot(np.mean(flv_score, axis=0), label="FLV", linestyle="dashed")
        ax[0].set_ylabel("RMSE")
        # Score quantiles
        ax[0].fill_between(
            iterations[0],
            np.quantile(lv_score, ql, axis=0),
            np.quantile(lv_score, qh, axis=0),
            alpha=0.5,
        )
        ax[0].fill_between(
            iterations[0],
            np.quantile(flv_score, ql, axis=0),
            np.quantile(flv_score, qh, axis=0),
            linestyle="dashed",
            alpha=0.5,
        )
        # Mean generation time
        ax[1].plot(np.mean(lv_time, axis=0), label="LV")
        ax[1].plot(np.mean(flv_time, axis=0), label="FLV")
        ax[1].set_xlabel("Iteration no.")
        ax[1].set_ylabel("Gen. time [s]")
        # Generation time quantiles
        ax[1].fill_between(
            iterations[0],
            np.quantile(lv_time, ql, axis=0),
            np.quantile(lv_time, qh, axis=0),
            alpha=0.5,
        )
        ax[1].fill_between(
            iterations[0],
            np.quantile(flv_time, ql, axis=0),
            np.quantile(flv_time, qh, axis=0),
            alpha=0.5,
        )
        ax[1].set_xlabel("Iteration no.")
        ax[1].set_ylabel("Gen. time [s]")
        # Test points scatter plot for all iterations
        ax[2].contourf(x, y, Z)
        ax[2].scatter(
            dict_to_array(lv_sampling_out, "step_x")[:, 0],
            dict_to_array(lv_sampling_out, "step_x")[:, 1],
            label="LV",
            alpha=0.5,
        )
        ax[2].scatter(
            dict_to_array(flv_sampling_out, "step_x")[:, 0],
            dict_to_array(flv_sampling_out, "step_x")[:, 1],
            label="FLV",
            alpha=0.5,
        )

    else:
        flv_score = steps_to_array(flv_sampling_out, "score")
        flv_time = steps_to_array(flv_sampling_out, "gen_time")
        # Mean score
        ax[0].plot(np.mean(flv_score, axis=0), label="FLV", linestyle="dashed")
        ax[0].set_ylabel("RMSE")
        # Score quantiles
        ax[0].fill_between(
            iterations[0],
            np.quantile(flv_score, ql, axis=0),
            np.quantile(flv_score, qh, axis=0),
            linestyle="dashed",
            alpha=0.5,
        )
        # Mean generation time
        ax[1].plot(np.mean(flv_time, axis=0), label="FLV")
        ax[1].set_xlabel("Iteration no.")
        ax[1].set_ylabel("Gen. time [s]")
        # Generation time quantiles
        ax[1].fill_between(
            iterations[0],
            np.quantile(flv_time, ql, axis=0),
            np.quantile(flv_time, qh, axis=0),
            alpha=0.5,
        )
        ax[1].set_xlabel("Iteration no.")
        ax[1].set_ylabel("Gen. time [s]")
        # Test points scatter plot for all iterations
        ax[2].contourf(x, y, Z)
        ax[2].scatter(
            dict_to_array(flv_sampling_out, "step_x")[:, 0],
            dict_to_array(flv_sampling_out, "step_x")[:, 1],
            label="FLV",
            alpha=0.5,
        )

ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
ax[2].legend()
plt.show()