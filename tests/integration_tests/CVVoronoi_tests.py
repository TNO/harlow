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
import torch
from sklearn.metrics import mean_squared_error

from harlow.sampling.cv_voronoi import CVVoronoi
from harlow.sampling.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.surrogating.surrogate_model import (
    ModelListGaussianProcess,
    VanillaGaussianProcess,
)
from harlow.utils.helper_functions import latin_hypercube_sampling
from harlow.model.model_twin_girder_betti import IJssel_bridge_model  # noqa:
# I201
from harlow.utils.test_functions import peaks_2d_multivariate

domains_lower_bound = np.array([-8, -8])
domains_upper_bound = np.array([8, 8])
n_initial_point = 15
n_new_points_per_iteration = 1
rmse_criterium = 0.05
np.random.seed(123)
n_iter_sampling = 500
n_iter_runs = 100


def get_param_idx(params_dict):
    return {key: idx_key for idx_key, key in enumerate(params_dict)}


# ====================================================================
# INITIALIZE MODEL
# ====================================================================
# Set model parameters
meas_type = "fugro"
E = 210e6
max_elem_length = 2.0 * 1e3

# Sensor names and positions
sensor_names = ["H1_S", "H2_S", "H3_S", "H4_S", "H5_S", "H7_S", "H8_S", "H9_S", "H10_S"]
sensor_positions = [
    20.42,
    34.82,
    47.700,
    61.970,
    68.600,
    96.800,
    113.9,
    123.900,
    147.500,
]

# All parameters. This is the order that parameters will be expcted in
# within the various functions in this scrípt (e.g. model response function)
params_all = ["Kr1", "Kr2", "Kr3", "Kr4", "Kv"]

N_tasks = len(sensor_positions)
N_features = len(params_all)

# Parameters that are shared between models
params_common = [
    "Kv",
]

# Parameters that are not shared by all models
params_model = {
    "H1_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H2_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H3_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H4_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H5_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H7_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H8_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H9_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
    "H10_S": ["Kr1", "Kr2", "Kr3", "Kr4"],
}

# Define FE models and append to list
models_dict = {
    idx_model: IJssel_bridge_model(
        sname, E, max_elem_length=max_elem_length, truck_load=meas_type
    )
    for idx_model, sname in enumerate(sensor_names)
}
node_xs = models_dict[0].node_xs

# # ====================================================================
# # DOMAIN BOUNDS
# # ====================================================================

# Prior for first support rotational stiffness Kr1
Kr_low = 4.0
Kr_high = 10.0

# Ground truth and prior for K-brace spring vertical stiffness
Kv_low = 0.0
Kv_high = 8.0

c = -0.1754

# Bounds of the time domain
t_low = np.min(node_xs)
t_high = np.max(node_xs)

params_priors = {
    "Kv": {"dist": "uniform", "low": Kv_low, "high": Kv_high},
    "Kr1": {"dist": "uniform", "low": Kr_low, "high": Kr_high},
    "Kr2": {"dist": "uniform", "low": Kr_low, "high": Kr_high},
    "Kr3": {"dist": "uniform", "low": Kr_low, "high": Kr_high},
    "Kr4": {"dist": "uniform", "low": Kr_low, "high": Kr_high},
}

# Create domain bounds
domain_lower_bound = np.array([params_priors[param]["low"] for param in params_all])
domain_upper_bound = np.array([params_priors[param]["high"] for param in params_all])

# Bounds as tensor
bounds = torch.tensor(np.vstack([domain_lower_bound, domain_upper_bound]))

# Indices of all params
param_idx = get_param_idx(params_all)

# Build list of parameter indices per model
list_params = [
    [param_idx[param] for param in params_model[key] + params_common]
    for key in params_model.keys()
]

# # ====================================================================
# # MODEL FUNCTION
# # ====================================================================
def response(X, pts):

    # Initialize
    X = np.atleast_2d(X)
    N_x = X.shape[0]
    N_y = len(pts)
    res = np.zeros((N_x, N_y))

    print(f"Evaluating response function at {N_x} points for {N_y} outputs")

    # Iterate over parameter array
    for idx_x, x in enumerate(X):
        Kr1 = x[param_idx["Kr1"]]
        Kr2 = x[param_idx["Kr2"]]
        Kr3 = x[param_idx["Kr3"]]
        Kr4 = x[param_idx["Kr4"]]
        Kv = x[param_idx["Kv"]]

        # Rotational stiffness input array
        arr_Kr = np.repeat(np.array([Kr1, Kr2, Kr3, Kr4]), 2)
        arr_Kr = np.append(arr_Kr, np.zeros(4))

        # Iterate over models and evaluate
        for idx_t, model_key in enumerate(models_dict.keys()):

            # Get model and prediction t
            t = pts[idx_t]
            model = models_dict[model_key]

            # Evaluate each model in list
            res[idx_x, idx_t] = np.interp(
                t,
                model.node_xs,
                model.il_stress_truckload(c, lane="left", Kr=10 ** arr_Kr, Kv=10 ** Kv),
            )
    return res


# Model function taking only X as input, to be passed to the sampler.
def func_model(X):
    return response(X, sensor_positions)


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def create_test_set(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(domain_lower_bound, domain_upper_bound, n)
    test_y = func_model(test_X)

    return test_X, test_y


def run_2D_test():
    fuzzy_sampling_results = []
    test_X, test_y = create_test_set(domain_lower_bound, domain_upper_bound, 500)

    start_points_X = latin_hypercube_sampling(
        domain_lower_bound, domain_upper_bound, n_initial_point
    )
    start_points_y = func_model(start_points_X)

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
    test_X, test_y = create_test_set(domains_lower_bound, domains_upper_bound, 500)

    start_points_X = latin_hypercube_sampling(
        domains_lower_bound, domains_upper_bound, n_initial_point
    )
    start_points_y = peaks_2d_multivariate(start_points_X)

    fuzzy_sampling_results.append(
        test_2D_FLV_sampling(
            start_points_X,
            start_points_y,
            domains_lower_bound,
            domains_upper_bound,
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
        target_function=func_model,
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
