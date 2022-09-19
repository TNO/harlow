"""
Comparison of different Multi-task GPR approaches:

* ModelList GPs: A list of independent Gaussian process surrogates
* BatchIndependent GPs: For independent outputs using the same covariance and likelihood
* MultiTask GPs: For learning similarities between outcomes
"""

from timeit import default_timer as timer

import numpy as np
import torch

# from botorch.models.transforms import Normalize, Standardize
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from harlow.surrogating import (
    BatchIndependentGaussianProcess,
    DeepKernelMultiTaskGaussianProcess,
    ModelListGaussianProcess,
    MultiTaskGaussianProcess,
)
from harlow.utils.helper_functions import latin_hypercube_sampling
from model.model_twin_girder_betti import IJssel_bridge_model  # noqa: I201

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================


def create_test_set(min_domain, max_domain, n):

    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = response(test_X, sensor_positions)

    return torch.FloatTensor(test_X), torch.FloatTensor(test_y)


def rmse(x, y):
    return mean_squared_error(x, y, squared=False)


def get_param_idx(params_dict):
    return {key: idx_key for idx_key, key in enumerate(params_dict)}


# ====================================================================
# SURROGATING PARAMETERS
# ====================================================================
N_train = 200
N_update = 100
N_test = 100
N_pred = 100
N_iter = 1000
rmse_criterium = 0.1
silence_warnings = True

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
                model.il_stress_truckload(c, lane="left", Kr=10**arr_Kr, Kv=10**Kv),
            )

    return res


# Model function taking only X as input, to be passed to the sampler.
def func_model(X):
    return response(X, sensor_positions)


# ====================================================================
# STANDARDIZING AND NORMALIZING
# ====================================================================
def input_transform(X):
    return X  # Normalize(d=N_features, bounds=bounds)(X)


def output_transform(y):
    return y  # Standardize(m=N_tasks)(y)[0]


# ====================================================================
# GENERATE TEST AND TRAIN DATA
# ====================================================================
# Each column of train_Y corresponds to one GP
print(f"Create training set N = {N_train}:")
train_X, train_y = create_test_set(domain_lower_bound, domain_upper_bound, N_train)
# test_X, test_y = create_test_set(domain_lower_bound, domain_upper_bound, N_test)

# print(f"Create update set N = {N_update}:")
# # To check surrogate updating
# update_X, update_y = create_test_set(domain_lower_bound, domain_upper_bound, N_update)

# Check if machine is GPU compatible and assign the device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print("Training on Device: {}".format(device))

# gpu=False
# ====================================================================
# DEFINE SURROGATE
# ====================================================================
N_tasks = train_y.shape[1]

surrogate_MLGP = ModelListGaussianProcess(
    input_transform(train_X),
    output_transform(train_y),
    model_names=sensor_names,
    list_params=list_params,
    training_max_iter=N_iter,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_BIGP = BatchIndependentGaussianProcess(
    input_transform(train_X),
    output_transform(train_y),
    num_tasks=N_tasks,
    training_max_iter=N_iter,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_MTGP = MultiTaskGaussianProcess(
    input_transform(train_X),
    output_transform(train_y),
    num_tasks=N_tasks,
    training_max_iter=N_iter,
    silence_warnings=silence_warnings,
    dev=device,
)

surrogate_DKLGP = DeepKernelMultiTaskGaussianProcess(
    input_transform(train_X),
    output_transform(train_y),
    num_tasks=N_tasks,
    training_max_iter=N_iter,
    silence_warnings=silence_warnings,
    dev=device,
)


# Create a list of surrogates
list_surrogates = [surrogate_MLGP, surrogate_BIGP, surrogate_MTGP, surrogate_DKLGP]
list_GP_type = [
    "ModelList GP",
    "BatchIndependent GP",
    "MultiTask GP",
    "DeepKernelMultiTask GP",
]

# ====================================================================
# FIT
# ====================================================================
print("============= Initial fit =====================")
for i, surrogate_i in enumerate(list_surrogates):
    t1 = timer()
    print("Fitting: " + list_GP_type[i])
    surrogate_i.fit(train_X, train_y)
    t2 = timer()
    print("Fitted " + list_GP_type[i] + f" in {t2 - t1} sec.")
    print("--------------------------------------------------")

# ====================================================================
# UPDATE
# ====================================================================
# print("============= Update =====================")
# for i, surrogate_i in enumerate(list_surrogates):
#     print("Updating: " + list_GP_type[i])
#     surrogate_i.update(update_X, update_y)

# ====================================================================
# SURROGATE PREDICT
# ====================================================================

# Tensor of prediction points
vec_Kv = np.linspace(Kv_low, Kv_high, N_pred)
pred_X = np.tile(np.array([7.0, 7.0, 7.0, 7.0]), (N_pred, 1))
pred_X = np.hstack((pred_X, vec_Kv.reshape(-1, 1)))
pred_X = torch.FloatTensor(pred_X)

# Physical model prediction
print(f"Create test set: N = {N_pred}")
true_y = response(pred_X, sensor_positions)

print("============= Predict =====================")
# Surrogate model prediction
y_pred = []
for i, surrogate_i in enumerate(list_surrogates):
    print("Predicting: " + list_GP_type[i])
    y_pred.append(surrogate_i.predict(input_transform(pred_X), return_std=False))
    print("--------------------------------------------------")

# Initialize plots
nrows = 3
ncols = int(np.ceil(N_tasks / 3))

for i, surrogate_i in enumerate(list_surrogates):
    print("Plotting: " + list_GP_type[i])
    print("--------------------------------------------------")

    # Initialize plot
    f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    # Get surrogate output
    mean_i = surrogate_i.mean
    upper_i = surrogate_i.cr_u
    lower_i = surrogate_i.cr_l
    train_X_i = surrogate_i.train_X.cpu().numpy()
    train_y_i = surrogate_i.train_y.cpu().numpy()
    for j, ax_i in enumerate(axes.ravel()):
        grid_idx = np.unravel_index(j, (nrows, ncols))

        # Plot training data as black stars
        ax_i.plot(
            input_transform(train_X)[:, -1],
            output_transform(train_y)[:, j],
            "k*",
            label="Observations",
        )

        # Predictive mean as blue line
        ax_i.plot(
            input_transform(pred_X)[:, -1].numpy(),
            output_transform(mean_i)[:, j].cpu().numpy(),
            "b",
            label="Mean",
        )

        # Shade in confidence
        ax_i.fill_between(
            input_transform(pred_X)[:, -1].numpy(),
            output_transform(lower_i)[:, j].cpu().detach().numpy(),
            output_transform(upper_i)[:, j].cpu().detach().numpy(),
            alpha=0.5,
            label="Confidence",
        )
        ax_i.plot(
            input_transform(pred_X)[:, -1].numpy(),
            output_transform(torch.from_numpy(true_y))[:, j],
            color="red",
            linestyle="dashed",
            label="Model",
        )
        ax_i.set_title(f"Sensor: {sensor_names[j]}")

    axes[0, 0].legend()
    plt.suptitle("Model: " + list_GP_type[i])
    plt.show()


plt.figure()
for i, surrogate_i in enumerate(list_surrogates):
    plt.plot(surrogate_i.vec_loss, label=list_GP_type[i])
    plt.xlabel("N_iter")
    plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
