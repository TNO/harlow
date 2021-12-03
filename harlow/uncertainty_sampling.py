"""
Adaptive sampling based on uncertainty heuristics.
"""
# import itertools
# import time
from typing import Callable

import numpy as np
from helper_functions import latin_hypercube_sampling
from scipy.optimize import differential_evolution


class UncertaintySampler:
    def __init__(
        self,
        target_function: Callable[[np.ndarray], np.ndarray],
        surrogate_model,  # TODO: should be a class from `surrogate_model.py`
        domain_lower_bound: np.ndarray,
        domain_upper_bound: np.ndarray,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        epsilon: float = 0.005,
        metric: str = "r2",
        verbose: bool = False,
    ):
        self.domain_lower_bound = domain_lower_bound
        self.domain_upper_bound = domain_upper_bound
        self.target_function = lambda x: target_function(x).reshape((-1, 1))
        self.surrogate_model = surrogate_model
        self.fit_points_x = fit_points_x
        self.fit_points_y = fit_points_y
        self.test_points_x = test_points_x
        self.test_points_y = test_points_y
        self.epsilon = epsilon
        self.metric = metric
        self.verbose = verbose

    def adaptive_surrogating(
        self, n_initial_point: int = None, n_new_point_per_iteration: int = 1
    ):
        # target_function = self.target_function
        domain_lower_bound = self.domain_lower_bound
        domain_upper_bound = self.domain_upper_bound
        n_dim = len(domain_lower_bound)
        convergence = False

        if n_initial_point is None:
            n_initial_point = 5 * n_dim

        if not self.surrogate_model.is_probabilistic:
            raise NotImplementedError(
                "Uncertainty based sampling only implemented for probabilistic \
                surrogate models."
            )

        if self.fit_points_x is None:
            points_x = latin_hypercube_sampling(
                n_sample=n_initial_point,
                domain_lower_bound=domain_lower_bound,
                domain_upper_bound=domain_upper_bound,
            )
            points_y = self.target_function(points_x)
        else:
            points_x = self.fit_points_x
            points_y = self.fit_points_y

        while convergence is False:
            y_range = np.max(points_y.flatten()) - np.min(points_y.flatten())
            self.surrogate_model = self.surrogate_model()
            self.surrogate_model.fit(points_x, points_y)

            bounds = [
                (domain_lower_bound[i], domain_upper_bound[i])
                for i in range(len(domain_lower_bound))
            ]

            diff_evolution_result = differential_evolution(
                self.prediction_std, bounds=bounds
            )
            std_max = diff_evolution_result.fun

            if (
                std_max - self.surrogate_model.observation_noise_variance_var
                <= self.epsilon * y_range
            ):
                convergence = True
            else:
                x_new = diff_evolution_result.x
                y_new = self.target_function(x_new)
                points_x = np.concatenate(points_x, x_new)
                points_y = np.concatenate(points_y, y_new)

    def prediction_std(self, x):

        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))
        # noise = self.surrogate_model.observation_noise_variance_var
        noise = 0
        std = self.surrogate_model.predict(x)[1] - noise
        return std
