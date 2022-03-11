"""
Also check:

https://www.sciencedirect.com/science/article/pii/S002199911630184X
https://github.com/PredictiveScienceLab/py-aspgp
"""


import numpy as np
from sklearn.metrics import mean_squared_error

from harlow.helper_functions import latin_hypercube_sampling
from harlow.lola_voronoi import LolaVoronoi
from harlow.surrogate_model import GaussianProcess
from model.model_twin_girder_betti import IJssel_bridge_model  # noqa: I201

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================


def create_test_set(min_domain, max_domain, n):

    test_X = latin_hypercube_sampling(min_domain, max_domain, n)
    test_y = response(test_X)

    return test_X, test_y


def rmse(x, y):
    return mean_squared_error(x, y, squared=False)


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
# sensor_names = ["H1_S",
# "H2_S",
# "H3_S",
# "H4_S",
# "H5_S",
# "H7_S",
# "H8_S",
# "H9_S",
# "H10_S"
# ]

# sensor_positions = [
#     20.42,
#     34.82,
#     47.700,
#     61.970,
#     68.600,
#     96.800,
#     113.9,
#     123.900,
#     147.500,
# ]

sensor_names = ["H4_S"]
sensor_positions = [61.97]

# sensor_names = ["H1_S", "H10_S"]
# sensor_positions = [20.42, 147.500]
N_tasks = len(sensor_positions)

# All parameters. This is the order that parameters will be expcted in
# within the various functions in this scrípt (e.g. model response function)
params_all = ["Kr2", "Kr3", "Kv", "t"]

# Parameters that are shared between models
params_common = [
    "t" "Kv",
]

# Parameters that are not shared by all models
params_model = {
    "H4_S": ["Kr2", "Kr3"],
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

params_bounds = {
    "Kv": {"low": Kv_low, "high": Kv_high},
    "Kr1": {"low": Kr_low, "high": Kr_high},
    "Kr2": {"low": Kr_low, "high": Kr_high},
    "Kr3": {"low": Kr_low, "high": Kr_high},
    "Kr4": {"low": Kr_low, "high": Kr_high},
    "t": {"low": t_low, "high": t_high},
}


# Create domain bounds
domain_lower_bound = np.array([params_bounds[param]["low"] for param in params_all])
domain_upper_bound = np.array([params_bounds[param]["high"] for param in params_all])

# Indices of all params
param_idx = get_param_idx(params_all)


# ====================================================================
# MODEL FUNCTION
# ====================================================================
def response(X):

    # Initialize
    X = np.atleast_2d(X)
    N_x = X.shape[0]
    N_y = len(sensor_positions)
    res = np.zeros((N_x, N_y))

    print(f"Evaluating model for N = {N_x}")

    # Iterate over parameter array
    for idx_x, x in enumerate(X):
        Kr2 = x[param_idx["Kr2"]]
        Kr3 = x[param_idx["Kr3"]]
        Kv = x[param_idx["Kv"]]
        t = x[param_idx["t"]]

        # Rotational stiffness input array
        arr_Kr = np.repeat(np.array([0.0, Kr2, Kr3, 0.0]), 2)
        arr_Kr = np.append(arr_Kr, np.zeros(4))

        # Iterate over models and evaluate
        for idx_y, model_key in enumerate(models_dict.keys()):

            # Get model and prediction t
            model = models_dict[model_key]

            # Evaluate each model in list
            res[idx_x, idx_y] = np.interp(
                t,
                model.node_xs,
                model.il_stress_truckload(c, lane="left", Kr=10 ** arr_Kr, Kv=10 ** Kv),
            )

    return res


# ====================================================================
# SURROGATING PARAMETERS
# ====================================================================
N_test = 200
N_iter = 50
n_initial_point = 10
n_new_points_per_iteration = 1
rmse_criterium = 0.05

# Generate initial set
init_X, init_Y = create_test_set(
    domain_lower_bound, domain_upper_bound, n_initial_point
)

# Generate test set
test_X, test_Y = create_test_set(domain_lower_bound, domain_upper_bound, N_test)


# ====================================================================
# SURROGATING
# ====================================================================
def test_LV_sampling(
    start_points_X,
    start_points_y,
    domain_lower_bound,
    domain_upper_bound,
    test_X,
    test_y,
):

    surrogate_model = GaussianProcess()

    # ............................
    # Surrogating
    # ............................
    lv = LolaVoronoi(
        target_function=response,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
        fit_points_x=start_points_X,
        fit_points_y=start_points_y,
        test_points_x=test_X,
        test_points_y=test_y,
        evaluation_metric=rmse,
    )

    lv.sample(
        n_iter=N_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_points_per_iteration,
        stopping_criterium=rmse_criterium,
    )

    return {
        "domain_lower_bound": domain_lower_bound.tolist(),
        "domain_upper_bound": domain_upper_bound.tolist(),
        "iterations": lv.iterations,
        "score": lv.score,
    }


test_LV_sampling(init_X, init_Y, domain_lower_bound, domain_upper_bound, test_X, test_Y)
