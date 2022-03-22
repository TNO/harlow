"""
Adaptive sampling based on model uncertainty.
This algorithm is from:
    Xuzheng Chai (2019) Probabilistic system identification and reliability updating
    for hydraulic structures - Application to sheet pile walls

Adapted from implementation in Prob_Taralli:
    https://gitlab.com/tno-bim/taralli/-/blob/d82a5f42e918f4864d4d6f18f0dbdf8c1f2799c6/
    prob_taralli/surrogating/adaptive_infill_gpr.py

"""
import math
import time
from typing import Callable

import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

from harlow.helper_functions import latin_hypercube_sampling
from harlow.sampling_baseclass import Sampler
from harlow.surrogate_model import Surrogate


class Probabilistic_sampler(Sampler):
    def __init__(
        self,
        target_function: Callable[[np.ndarray], np.ndarray],
        surrogate_model: Surrogate,
        domain_lower_bound: np.ndarray,
        domain_upper_bound: np.ndarray,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        evaluation_metric: Callable = None,
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
        self.metric = evaluation_metric
        self.verbose = verbose

        self.iterations = 0

    def sample(
        self,
        n_initial_point: int = None,
        n_iter: int = 20,
        stopping_criterium: float = None,
        epsilon: float = 0.005,
    ):

        n_dim = len(self.domain_lower_bound)

        if n_initial_point is None:
            n_initial_point = 5 * n_dim

        if stopping_criterium:
            n_iter = 1000

        if stopping_criterium and not self.metric:
            self.metric = lambda x, y: math.sqrt(mean_squared_error(x, y))

        logger.info(f"Adaptive sampling iteration {self.iterations}.")

        if not self.surrogate_model.is_probabilistic:
            raise NotImplementedError(
                "Uncertainty based sampling only implemented for probabilistic \
                surrogate models."
            )

        if self.fit_points_x is None:
            points_x = latin_hypercube_sampling(
                n_sample=n_initial_point,
                domain_lower_bound=self.domain_lower_bound,
                domain_upper_bound=self.domain_upper_bound,
            )
            points_y = self.target_function(points_x)
        else:
            points_x = self.fit_points_x
            points_y = self.fit_points_y

        convergence = False

        while convergence is False:
            bounds = [
                (self.domain_lower_bound[i], self.domain_upper_bound[i])
                for i in range(n_dim)
            ]

            start_time = time.time()
            self.surrogate_model.fit(points_x, points_y)
            logger.info(
                f"Fitted a new surrogate model in {time.time() - start_time} sec."
            )

            start_time = time.time()
            diff_evolution_result = differential_evolution(
                self.prediction_std, bounds=bounds
            )
            logger.info(
                f"Finished differential evolution in {time.time() - start_time} sec."
            )
            std_max = -diff_evolution_result.fun

            if not stopping_criterium and (
                std_max <= epsilon or self.iterations > n_iter
            ):
                logger.info("std_max <= epsilon or max iterations reached")
                convergence = True
            elif stopping_criterium:
                score = self.metric(
                    self.surrogate_model.predict(self.test_points_x), self.test_points_y
                )
                logger.info(f"Evaluation metric score on provided testset: {score}")
                if score <= stopping_criterium:
                    self.number_of_iterations_at_convergence = self.iterations
                    logger.info(f"Algorithm converged in {self.iterations} iterations")
                    convergence = True

            x_new = diff_evolution_result.x
            y_new = self.target_function(x_new)
            points_x = np.concatenate((points_x, np.expand_dims(x_new, axis=0)))
            points_y = np.concatenate((points_y, y_new))

            self.score = score
            self.iterations += 1

        return points_x, points_y

    def result_as_dict(self):
        pass

    def prediction_std(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        std = -(
            self.surrogate_model.predict(x, return_std=True)[1]
            - self.surrogate_model.noise_std
        )

        return std
