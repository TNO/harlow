import os
import numpy as np
import torch
from harlow.sampling import CVVoronoi

from harlow.surrogating.surrogate_model import VanillaGaussianProcess, BatchIndependentGaussianProcess
from harlow.utils.helper_functions import latin_hypercube_sampling
from harlow.utils.metrics import mae, rmse, rrse
from harlow.utils.examples.model_twin_girder_betti import IJssel_bridge_model
from harlow.utils.transforms import ExpandDims, TensorTransform
np.random.seed(0)
stop_thresh = 0.01  # For RMSE or 0.005 - 0.0025

# # ====================================================================
# # HELPER FUNCTIONS
# # ====================================================================


def create_test_set(min_domain, max_domain, n):

    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = response(test_X, sensor_positions)

    return test_X, test_y


def get_param_idx(params_dict):
    return {key: idx_key for idx_key, key in enumerate(params_dict)}


# # ====================================================================
# # SURROGATING PARAMETERS
# # ====================================================================
N_train = 20
N_update = 50
N_iter = 100
rmse_criterium = 0.1
silence_warnings = True

np.random.seed(0)

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


# # ====================================================================
# # GENERATE TEST AND TRAIN DATA
# # ====================================================================

# # Each column of train_Y corresponds to one GP
print(f"Create training set N = {N_train}:")
train_X, train_y = create_test_set(domain_lower_bound, domain_upper_bound, N_train)

# To check surrogate updating
print(f"Create update set N = {N_update}:")
update_X, update_y = create_test_set(domain_lower_bound, domain_upper_bound, N_update)

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
    evaluation_metric = rmse
    logging_metrics = [rrse, mae, rmse]

    run_name = "Bench_{}_with_{}_init_pts_on_{}X{}_dim_problem_w_train_size_{}".format(
        'CVVoronoi', 15, train_y.shape[1], train_X.shape[1], N_train,
    )

    save_path = os.path.join("saves", run_name)
    os.makedirs(save_path, exist_ok=True)

    # # ====================================================================
    # # SAMPLER SETUP
    # # ====================================================================

    cv = CVVoronoi(
        target_function=func_model,
        surrogate_model=surrogate_VGP,
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
    )

    cv.sample(
        n_initial_points=15,
        n_new_points_per_iteration=1,
        stopping_criterium=rmse_criterium,
        max_n_iterations=1000,
    )