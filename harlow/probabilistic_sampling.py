"""
Adaptive sampling based on uncertainty heuristics.
This algorithm is from:
    Xuzheng Chai (2019) Probabilistic system identification and reliability updating
    for hydraulic structures - Application to sheet pile walls

Adapted from implementation in Prob_Taralli:
    https://gitlab.com/tno-bim/taralli/-/blob/d82a5f42e918f4864d4d6f18f0dbdf8c1f2799c6/
    prob_taralli/surrogating/adaptive_infill_gpr.py

"""
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from harlow.helper_functions import latin_hypercube_sampling


class Probabilistic_sampler:
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
        n_initial_points: int = None,
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
        self.n_initial_points = n_initial_points
        self.epsilon = epsilon
        self.metric = metric
        self.verbose = verbose

        self.iterations = 0

    def adaptive_surrogating(self):
        domain_lower_bound = self.domain_lower_bound
        domain_upper_bound = self.domain_upper_bound
        n_dim = len(domain_lower_bound)
        convergence = False

        if self.n_initial_points is None:
            self.n_initial_point = 5 * n_dim

        if not self.surrogate_model.is_probabilistic:
            raise NotImplementedError(
                "Uncertainty based sampling only implemented for probabilistic \
                surrogate models."
            )

        if self.fit_points_x is None:
            points_x = latin_hypercube_sampling(
                n_sample=self.n_initial_point,
                domain_lower_bound=domain_lower_bound,
                domain_upper_bound=domain_upper_bound,
            )
            points_y = self.target_function(points_x)
        else:
            points_x = self.fit_points_x
            points_y = self.fit_points_y

        while convergence is False:

            bounds = [
                (domain_lower_bound[i], domain_upper_bound[i])
                for i in range(len(domain_lower_bound))
            ]

            self.surrogate_model.fit(points_x, points_y)

            diff_evolution_result = differential_evolution(
                self.prediction_std, bounds=bounds
            )
            std_max = -diff_evolution_result.fun

            if std_max <= self.epsilon:
                convergence = True
            else:
                x_new = diff_evolution_result.x
                y_new = self.target_function(x_new)
                points_x = np.concatenate((points_x, np.expand_dims(x_new, axis=0)))
                points_y = np.concatenate((points_y, y_new.flatten()))

            self.iterations += 1

        return points_x, points_y

    def prediction_std(self, x):
        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))

        std = -(
            self.surrogate_model.predict(x, return_std=True)[1]
            - self.surrogate_model.noise_std
        )

        return std
