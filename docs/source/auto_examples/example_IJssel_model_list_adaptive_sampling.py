"""
Also check:

https://www.sciencedirect.com/science/article/pii/S002199911630184X
https://github.com/PredictiveScienceLab/py-aspgp
"""

import numpy as np
import torch
from model.model_twin_girder_betti import IJssel_bridge_model  # noqa: I201
from sklearn.metrics import mean_squared_error

from harlow.helper_functions import latin_hypercube_sampling
from harlow.probabilistic_sampling import Probabilistic_sampler
from harlow.surrogate_model import ModelListGaussianProcess

# from matplotlib import pyplot as plt


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================


def create_test_set(min_domain, max_domain, n):

    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = response(test_X, sensor_positions)

    return torch.tensor(test_X).float(), torch.tensor(test_y).float()


def rmse(x, y):

    list_rmse = []
    for xi, yi in zip(x, y.T):
        list_rmse.append(mean_squared_error(xi, yi, squared=False))
    return np.max(list_rmse)


def get_param_idx(params_dict):
    return {key: idx_key for idx_key, key in enumerate(params_dict)}


# ====================================================================
# SURROGATING PARAMETERS
# ====================================================================
N_train = 10
N_test = 50
N_pred = 50
N_max_iter = 1000
N_update = 100
rmse_criterium = 0.1
min_loss_rate = 0.001

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
N_tasks = len(sensor_positions)

# All parameters. This is the order that parameters will be expcted in
# within the various functions in this scrípt (e.g. model response function)
params_all = ["Kr1", "Kr2", "Kr3", "Kr4", "Kv"]

# Parameters that are shared between models
params_common = [
    "Kv",
]

# Parameters that are not shared by all models
params_model = {
    "H1_S": ["Kr1", "Kr2"],
    "H2_S": ["Kr1", "Kr2"],
    "H3_S": ["Kr1", "Kr2"],
    "H4_S": ["Kr2", "Kr3"],
    "H5_S": ["Kr2", "Kr3"],
    "H7_S": ["Kr2", "Kr3"],
    "H8_S": ["Kr3", "Kr4"],
    "H9_S": ["Kr3", "Kr4"],
    "H10_S": ["Kr3", "Kr4"],
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


# ====================================================================
# GENERATE TEST AND TRAIN DATA
# ====================================================================
# Each column of train_Y corresponds to one GP
train_X, train_y = create_test_set(domain_lower_bound, domain_upper_bound, N_train)
test_X, test_y = create_test_set(domain_lower_bound, domain_upper_bound, N_test)

# To check surrogate updating
update_X, update_y = create_test_set(domain_lower_bound, domain_upper_bound, N_update)

# ====================================================================
# DEFINE SURROGATE
# ====================================================================

surrogate = ModelListGaussianProcess(
    # torch.cat([train_X, update_X], dim=0),
    # torch.cat([train_y, update_y], dim=0),
    train_X,
    train_y,
    model_names=sensor_names,
    list_params=list_params,
    training_max_iter=N_max_iter,
    min_loss_rate=min_loss_rate,
    show_progress=True,
    silence_warnings=True,
)

# ====================================================================
# DEFINE SAMPLER
# ====================================================================

ps = Probabilistic_sampler(
    target_function=func_model,
    surrogate_model=surrogate,
    domain_lower_bound=domain_lower_bound,
    domain_upper_bound=domain_upper_bound,
    fit_points_x=train_X,
    fit_points_y=train_y,
    test_points_x=test_X,
    test_points_y=test_y,
    evaluation_metric=rmse,
)

ps.sample(
    n_iter=None,
    n_initial_point=N_train,
    stopping_criterium=rmse_criterium,
)


#
# # ====================================================================
# # FIT
# # ====================================================================
# #surrogate.fit(train_X, train_y)
#
# # ====================================================================
# # UPDATE
# # ====================================================================
# #surrogate.update(update_X, update_y)
#
# # ====================================================================
# # SURROGATE PREDICT
# # ====================================================================
#
# # Tensor of prediction points
# vec_Kv = np.linspace(Kv_low, Kv_high, N_pred)
# pred_X = np.tile(np.array([7.0, 7.0, 7.0, 7.0]), (N_pred, 1))
# pred_X = np.hstack((pred_X, vec_Kv.reshape(-1, 1)))
# pred_X = torch.tensor(pred_X).float()
#
# # Physical model prediction
# true_y = response(pred_X, sensor_positions)
#
# # Surrogate model prediction
# pred_y = surrogate.predict(pred_X, return_std = False)
#
# # Initialize plots
# nrows = 3
# ncols = int(np.ceil(N_tasks/3))
# f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
#
# for idx, ax_i in enumerate(axes.ravel()):
#
#     mean_i = surrogate.mean[idx]
#     upper_i = surrogate.upper[idx]
#     lower_i = surrogate.lower[idx]
#
#     grid_idx = np.unravel_index(idx, (nrows, ncols))
#
#     train_X_i = surrogate.model.train_inputs[idx][0].detach().numpy()
#     train_y_i = surrogate.model.train_targets[idx].detach().numpy()
#
#     # Plot training data as black stars
#     ax_i.plot(train_X[:, -1], train_y[:, idx], 'k*', label = "Observations")
#
#     # Predictive mean as blue line
#     ax_i.plot(pred_X[:, -1].numpy(), mean_i.numpy(), 'b', label = "Mean")
#
#     # Shade in confidence
#     ax_i.fill_between(
#     pred_X[:, -1].numpy(),
#     lower_i.detach().numpy(),
#     upper_i.detach().numpy(),
#     alpha=0.5,
#     label = "Confidence"
#     )
#     ax_i.plot(
#     pred_X[:, -1].numpy(),
#     true_y[:, idx],
#     color="red",
#     linestyle="dashed",
#     label = "Model"
#     )
#     ax_i.set_title(f"Sensor: {sensor_names[idx]}")
#
# axes[0,0].legend()
