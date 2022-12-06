import argparse
import os
import pickle
import numpy as np
import torch

from harlow.sampling import CVVoronoi, LatinHypercube, ProbabilisticSampler, FuzzyLolaVoronoi, LolaVoronoi
from harlow.surrogating.surrogate_model import (
    BatchIndependentGaussianProcess,
    VanillaGaussianProcess,
)
from harlow.utils.examples.model_twin_girder_betti import IJssel_bridge_model
from harlow.utils.helper_functions import latin_hypercube_sampling
from harlow.utils.metrics import logrmse, mae, nrmse, rmse, rrse
from harlow.utils.test_functions import F_3_4_6, F_3_6, F_4_5_6, hartmann, peaks_2d
from harlow.utils.transforms import ExpandDims, TensorTransform

# from system_identification_detailed_FEM import DIR_FIGS, DIR_DATA, DIR_MODELS
# from system_identification_detailed_FEM import PARAM_BOUNDS, ALL_SENSOR_LIST

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
np.random.seed(0)

# OPTIONS
plot_params_sensititivy = True
load_type = "BLW"
# # ====================================================================
# # HELPER FUNCTIONS
# # ====================================================================


def create_test_set(min_domain, max_domain, n):

    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = response(test_X, sensor_positions)
    return test_X, test_y


def create_test_set_2D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = peaks_2d(test_X).reshape((-1, 1))
    return test_X, test_y


def create_test_set_6D(min_domain, max_domain, n):
    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = hartmann(test_X).reshape((-1, 1))
    return test_X, test_y


def get_param_idx(params_dict):
    return {key: idx_key for idx_key, key in enumerate(params_dict)}



# ========================================================================
# PROBLEM DEFINITION
# ========================================================================
# Get outputs
# sensor_list = ALL_SENSOR_LIST
# Ns = len(sensor_list)
#
# # Parameter in this list must be in the same order as in the fem model evaluation function
# params_list = list(PARAM_BOUNDS.keys())
#
# # Load target values from pickle
# file = open(DIR_DATA / f"train_y_{load_type}.pickle", "rb")
# arr_y_full = pickle.load(file)
# file.close()
#
# # Load training X points from pickle
# file = open(DIR_DATA / f"train_X_{load_type}.pickle", "rb")
# arr_X_full = pickle.load(file)
# file.close()
#
# # Loop over the list of dicts and arange the data in a 2d numpy array
# Nt = len(arr_y_full[0]["sensor1"]) - 1
# Nx = len(arr_y_full)
# vec_t = np.linspace(0.0, 101.6, Nt)
#
# y_strains = np.zeros((Nx, Nt * Ns))
# y_batch_strains = np.zeros((Ns, Nx, Nt))
#
# # Remove first column (self weight loadcase) and cast to numpy array
# for i, y_i in enumerate(arr_y_full):
#     # Horizontally concatenate all outputs. Output size is (n_pts, n_sensors*n_t)
#     y_strains[i, :] = np.array([y_i[f"sensor{j + 1}"][1:] for j in range(Ns)]).ravel()  # Discards first point
#
#     # Arrange output in one batch for each sensor. Output size is (n_sensors, n_pts, n_t)
#     for j in range(Ns):
#         y_batch_strains[j, i, :] = y_i[f"sensor{j + 1}"][1:]
#
# print(y_batch_strains.shape)


# # ====================================================================
# # PROBLEM SETUP & GENERATE TEST AND TRAIN DATA
# # ====================================================================
def create_train_test_sets(problem, ijs_low, ijs_up):
    if problem == 2:
        domain_lower_bound = np.array([-8.0, -8.0])
        domain_upper_bound = np.array([8.0, 8.0])
        print(f"Create training set N = {N_train}:")
        train_X, train_y = create_test_set_2D(
            domain_lower_bound, domain_upper_bound, N_train
        )
        print(f"Create update set N = {N_update}:")
        update_X, update_y = create_test_set_2D(
            domain_lower_bound, domain_upper_bound, N_update
        )
        target_func = peaks_2d
    elif problem == 6:
        domain_lower_bound = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        domain_upper_bound = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        print(f"Create training set N = {N_train}:")
        train_X, train_y = create_test_set_6D(
            domain_lower_bound, domain_upper_bound, N_train
        )
        print(f"Create update set N = {N_update}:")
        update_X, update_y = create_test_set_6D(
            domain_lower_bound, domain_upper_bound, N_update
        )
        target_func = hartmann
    elif problem == 'ijssel':
        # # Each column of train_Y corresponds to one GP
        domain_lower_bound = ijs_low
        domain_upper_bound = ijs_up
        print(f"Create training set N = {N_train}:")
        train_X, train_y = create_test_set(
            domain_lower_bound, domain_upper_bound, N_train
        )
        # To check surrogate updating
        print(f"Create update set N = {N_update}:")
        update_X, update_y = create_test_set(
            domain_lower_bound, domain_upper_bound, N_update
        )
        target_func = func_model
    elif problem == 'moordeijk':
        pass

    return train_X, train_y, update_X, update_y, target_func, domain_lower_bound, domain_upper_bound

# # ====================================================================
# # SURROGATING PARAMETERS
# # ====================================================================
N_train = 10
N_update = 50
N_iter = 100
rmse_criterium = 0.01
silence_warnings = True
# # ====================================================================
# # INITIALIZE MODEL
# # ====================================================================
# # Set model parameters
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
ijssel_lower_bound = np.array([params_priors[param]["low"] for param in params_all])
ijssel_upper_bound = np.array([params_priors[param]["high"] for param in params_all])

# Bounds as tensor
bounds = torch.tensor(np.vstack([ijssel_lower_bound, ijssel_upper_bound]))

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
                model.il_stress_truckload(c, lane="left", Kr=10**arr_Kr, Kv=10**Kv),
            )
    return res


# Model function taking only X as input, to be passed to the sampler.
def func_model(X):
    return response(X, sensor_positions)


# Check if machine is GPU compatible and assign the device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Training on Device: {}".format(device))

# # ====================================================================
# # TRANSFORMS
# # ====================================================================

expand_transform = ExpandDims
input_transform = TensorTransform
output_transform = TensorTransform

# # ====================================================================
# # DEFINE SURROGATE
# # ====================================================================

surrogate_GPR = BatchIndependentGaussianProcess(
    training_max_iter=N_iter,
    input_transform=input_transform,
    output_transform=output_transform,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_VGP = VanillaGaussianProcess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folds",
        default=5,
        type=int,
        help="Number of Folds in K-Fold Cross Validation",
    )
    parser.add_argument(
        "-p",
        "--problem",
        default=2,
        type=int,
        help="Dimensionality of the problem to solve",
    )
    parser.add_argument(
        "-n",
        "--new_points",
        default=1,
        type=int,
        help="Number of new points per iteration",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=2,
        type=int,
        help="Number of iterations for adaptive sampling",
    )
    parser.add_argument(
        "-s",
        "--sampler",
        default='RANDOM',
        type=str,
        help="Short name of the sampler to use. Options: LOLA, FLOLA, CV, PROB, RANDOM",
    )
    args = parser.parse_args()

    evaluation_metric = nrmse
    logging_metrics = [rrse, mae, rmse, nrmse, logrmse]

    # # ====================================================================
    # # SAMPLER SETUP
    # # ====================================================================
    if args.sampler == 'LOLA':
        test_sampler = LolaVoronoi
    elif args.sampler == 'FLOLA':
        test_sampler = FuzzyLolaVoronoi
    elif args.sampler == 'CV':
        test_sampler = CVVoronoi
    elif args.sampler == 'PROB':
        test_sampler = ProbabilisticSampler
    elif args.sampler == 'RANDOM':
        test_sampler = LatinHypercube

    # Problems to solver
    problems = [2, 6, 'ijssel']
    # problems = ['ijssel']
    for p in problems:
        # Create train and test data for the problem at hand
        train_X, train_y, update_X, update_y, target_func, domain_lower_bound, domain_upper_bound = \
            create_train_test_sets(p, ijssel_lower_bound, ijssel_upper_bound)
        run_name = (
            "Bench_{}_with_{}_init_pts_on_{}X{}_dim_problem_{}_w_"
            "test_Size{}_K_fold={}".format(
                args.sampler,
                N_train,
                train_X.shape[1],
                train_y.shape[1],
                p,
                N_update,
                args.folds,
            )
        )
        print(run_name)
        print('Train X, Y shapes {}, {} | Test X, Y shapes {}, {}'.format(train_X.shape, train_y.shape, update_X.shape, update_y.shape))
        save_path = os.path.join("saves", run_name)
        os.makedirs(save_path, exist_ok=True)

        sampler = test_sampler(
            target_function=target_func,
            surrogate_model_constructor=surrogate_VGP,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            fit_points_x=train_X,
            fit_points_y=train_y,
            test_points_x=update_X,
            test_points_y=update_y,
            evaluation_metric=evaluation_metric,
            logging_metrics=logging_metrics,
            run_name=run_name,
            save_dir=save_path,
            # n_fold=args.folds,
        )

        sampler.set_initial_set(train_X, train_y)
        sampler.construct_surrogate()

        if args.sampler in ["FLOLA", "LOLA"]:
            sampler.surrogate_loop(train_y.shape[1], args.iterations)
        else:
            sampler.surrogate_loop(args.new_points, args.iterations)
