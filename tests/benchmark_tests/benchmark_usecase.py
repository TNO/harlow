"""
Comparison of different Multi-task GPR approaches & different samplers for 
the use case of Ijssel bridge: 
-- Currently implementation for with Multitask GP and FLOLA sampler

* ModelList GPs: A list of independent Gaussian process surrogates
* BatchIndependent GPs: For independent outputs using the same covariance and likelihood
* MultiTask GPs: For learning similarities between outcomes
"""

from timeit import default_timer as timer
import argparse
import os
import json
from typing import List

import numpy as np
import torch

# from botorch.models.transforms import Normalize, Standardize
from sklearn.metrics import mean_squared_error
from harlow.utils.metrics import mae, rmse, rrse

from harlow.surrogating.surrogate_model import (
    BatchIndependentGaussianProcess,
    DeepKernelMultiTaskGaussianProcess,
    ModelListGaussianProcess,
    MultiTaskGaussianProcess,
)
from harlow.utils.helper_functions import latin_hypercube_sampling
from harlow.utils.transforms import ExpandDims, TensorTransform
from model.model_twin_girder_betti import IJssel_bridge_model  # noqa: I201

#Samplers
from harlow.sampling.cv_voronoi import CVVoronoi
from harlow.sampling.fuzzy_lolavoronoi import FuzzyLolaVoronoi
from harlow.sampling.lola_voronoi import LolaVoronoi
from harlow.sampling.probabilistic_sampling import ProbabilisticSampler
from harlow.sampling.random_sampling import LatinHypercube

from harlow.surrogating.surrogate_model import VanillaGaussianProcess, GaussianProcessRegression
# from tests.integration_tests.test_functions import peaks_2d_multivariate

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================


def create_test_set(min_domain, max_domain, n):

    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = response(test_X, sensor_positions)

    return test_X, test_y


def rmse(x, y):
    return mean_squared_error(x, y, squared=False)


def get_param_idx(params_dict):
    return {key: idx_key for idx_key, key in enumerate(params_dict)}


# ====================================================================
# SURROGATING PARAMETERS
# ====================================================================
N_train = 20
N_update = 50
N_iter = 100
rmse_criterium = 0.1
silence_warnings = True

np.random.seed(0)

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
# # SAMPLER SETUP
# # ====================================================================
def test_sampler(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
    surrogate_model,
    evaluation_metric,
    logging_metrics,
    sampler,
    n_iter_sampling,
    n_initial_point,
    n_new_points_per_iteration,
    target_f,
    name,
    save_path,
):
    #TODO Test all Multi-output GPs for all Samplers
    meta_model = surrogate_model[0]
    # ............................
    # Surrogating
    # ............................
    if sampler == "FLOLA":
        lv = FuzzyLolaVoronoi(
            target_function=target_f,
            surrogate_model=meta_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=evaluation_metric,
            logging_metrics=logging_metrics,
            run_name=name,
            save_dir=save_path,
        )
    elif sampler == "LOLA":
        lv = LolaVoronoi(
            target_function=target_f,
            surrogate_model=meta_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=evaluation_metric,
            logging_metrics=logging_metrics,
            run_name=name,
            save_dir=save_path,
        )
    elif sampler == "Prob":
        lv = ProbabilisticSampler(
            target_function=target_f,
            surrogate_model=meta_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=evaluation_metric,
            logging_metrics=logging_metrics,
            run_name=name,
        )
    elif sampler == "Random":
        lv = LatinHypercube(
            target_function=target_f,
            surrogate_model=meta_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=evaluation_metric,
            logging_metrics=logging_metrics,
            run_name=name,
        )

    elif sampler == "CV":
        lv = CVVoronoi(
            target_function=target_f,
            surrogate_model=meta_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=start_points_X,
            fit_points_y=start_points_y,
            test_points_x=test_X,
            test_points_y=test_y,
            evaluation_metric=evaluation_metric,
            logging_metrics=logging_metrics,
            run_name=name,
            save_dir=save_path,
        )
    lv.sample(
        n_initial_points=n_initial_point,
        n_new_points_per_iteration=n_new_points_per_iteration,
        stopping_criterium=rmse_criterium,
        max_n_iterations=n_iter_sampling,
    )

    return {
        "step_x": lv.step_x,
        "step_y": lv.step_y,
        "score": lv.step_score,
        "iteration": lv.step_iter,
        "gen_time": lv.step_gen_time,
        "fit_time": lv.step_fit_time,
    }

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


# ====================================================================
# TRANSFORMS
# ====================================================================

expand_transform = ExpandDims
input_transform = TensorTransform
output_transform = TensorTransform

# ====================================================================
# GENERATE TEST AND TRAIN DATA
# ====================================================================
# Each column of train_Y corresponds to one GP
print(f"Create training set N = {N_train}:")
train_X, train_y = create_test_set(domain_lower_bound, domain_upper_bound, N_train)
# train_X = expand_transform().forward(train_X)

# test_X, test_y = create_test_set(domain_lower_bound, domain_upper_bound, N_test)

# To check surrogate updating
print(f"Create update set N = {N_update}:")
update_X, update_y = create_test_set(domain_lower_bound, domain_upper_bound, N_update)
# update_X = expand_transform().forward(update_X)

# Check if machine is GPU compatible and assign the device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print("Training on Device: {}".format(device))

# gpu=False

# ====================================================================
# DEFINE SURROGATE
# ====================================================================
N_tasks = train_y.shape[1]
N_features = train_X.shape[1]

surrogate_MLGP = ModelListGaussianProcess(
    model_names=sensor_names,
    list_params=list_params,
    training_max_iter=N_iter,
    input_transform=input_transform,
    output_transform=output_transform,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_BIGP = BatchIndependentGaussianProcess(
    num_tasks=N_tasks,
    training_max_iter=N_iter,
    input_transform=input_transform,
    output_transform=output_transform,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_MTGP = MultiTaskGaussianProcess(
    num_tasks=N_tasks,
    training_max_iter=N_iter,
    input_transform=input_transform,
    output_transform=output_transform,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_DKLGP = DeepKernelMultiTaskGaussianProcess(
    num_tasks=N_tasks,
    num_features=N_features,
    training_max_iter=N_iter,
    input_transform=input_transform,
    output_transform=output_transform,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_GPR = GaussianProcessRegression(
    training_max_iter=N_iter,
    input_transform=input_transform,
    output_transform=output_transform,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_VGP = VanillaGaussianProcess()

# Create a list of surrogates
list_surrogates = [surrogate_MTGP, surrogate_GPR, surrogate_VGP]
list_GP_type = [
    "MultiTask GP",
]

def run_benchmark(
    name,
    evaluation_metric,
    logging_metrics,
    method,
    adapt_steps,
    n_new_points,
    problem,
    n_initial_point,
    test_size,
):
    sampling_res_list = []
    save_path = os.path.join("saves", name)
    os.makedirs(save_path, exist_ok=True)

    sampling_run_res = test_sampler(
        train_X,
        train_y,
        domain_lower_bound,
        domain_upper_bound,
        update_X,
        update_y,
        list_surrogates,
        evaluation_metric,
        logging_metrics,
        method,
        adapt_steps,
        n_initial_point,
        n_new_points,
        func_model,
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
        "-em",
        "--evaluation_metric",
        default="rmse",
        type=str,
        help="Define the evaluation metric function",
    )
    parser.add_argument(
        "-lm",
        "--logging_metrics",
        default="all",
        type=str,
        help="Define the logging metrics functions as list",
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
    if args.logging_metrics == "all":
        evaluation_metric = rmse
        logging_metrics = [rrse, mae, rmse]
    else:
        evaluation_metric = rmse
        logging_metrics = [mae]

    run_name = "Bench_{}_with_{}_init_pts_on_{}X{}_dim_problem_w_train_size_{}".format(
        args.sampler, args.init_p,train_y.shape[1], train_X.shape[1], N_train,
    )
    print(run_name)
    flv_sampling_out = run_benchmark(
        run_name,
        evaluation_metric,
        logging_metrics,
        args.sampler,
        args.steps,
        args.n_points_iter,
        args.problem,
        args.init_p,
        TEST_SIZE,
    )