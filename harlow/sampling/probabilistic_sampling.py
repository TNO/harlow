"""
Adaptive sampling based on uncertainty heuristics.
This algorithm is from:
    Xuzheng Chai (2019) Probabilistic system identification and reliability updating
    for hydraulic structures - Application to sheet pile walls

Adapted from implementation in Prob_Taralli:
    https://gitlab.com/tno-bim/taralli/-/blob/d82a5f42e918f4864d4d6f18f0dbdf8c1f2799c6/
    prob_taralli/surrogating/adaptive_infill_gpr.py

"""
import time
from typing import Callable

import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution

from harlow.sampling.sampling_baseclass import Sampler
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import evaluate, latin_hypercube_sampling
from harlow.utils.log_writer import write_scores, write_timer
from harlow.utils.metrics import rmse


class ProbabilisticSampler(Sampler):
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
        evaluation_metric: Callable = rmse,
        logging_metrics: list = None,
        verbose: bool = False,
        run_name: str = None,
        save_dir: str = "",
    ):

        super(Probabilistic_sampler, self).__init__(
            target_function,
            surrogate_model,
            domain_lower_bound,
            domain_upper_bound,
            fit_points_x,
            fit_points_y,
            test_points_x,
            test_points_y,
            evaluation_metric,
            logging_metrics,
            verbose,
            run_name,
            save_dir,
        )

        self.iterations = 0

    def sample(
        self,
        n_initial_points: int = 20,
        n_new_points_per_iteration: int = 1,
        stopping_criterium: float = None,
        max_n_iterations: int = 5000,
        epsilon: float = 0.005,
    ):

        n_dim = len(self.domain_lower_bound)

        if n_initial_points is None:
            n_initial_point = 5 * n_dim

        logger.info(f"Adaptive sampling iteration {self.iterations}.")

        if not self.surrogate_model.is_probabilistic:
            raise NotImplementedError(
                "Uncertainty based sampling only implemented for probabilistic \
                surrogate models."
            )

        # ..........................................
        # Initial sample of points
        # ..........................................
        gen_start_time = time.time()
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
        self.step_gen_time.append(time.time() - gen_start_time)

        start_time = time.time()
        self.surrogate_model.fit(points_x, points_y)
        logger.info(f"Fitted a new surrogate model in {time.time() - start_time} sec.")
        self.step_fit_time.append(time.time() - gen_start_time)

        convergence = False

        while convergence is False:
            bounds = [
                (self.domain_lower_bound[i], self.domain_upper_bound[i])
                for i in range(n_dim)
            ]

            def prediction_std(x):
                if x.ndim == 1:
                    x = np.expand_dims(x, axis=0)

                std = -(
                    self.surrogate_model.predict(x, return_std=True)[1]
                    # - self.surrogate_model.noise_std
                )

                return std

            start_time = time.time()
            diff_evolution_result = differential_evolution(
                prediction_std, bounds=bounds
            )
            logger.info(
                f"Finished differential evolution in {time.time() - start_time} sec."
            )
            std_max = -diff_evolution_result.fun
            self.step_gen_time.append(time.time() - start_time)

            x_new = np.expand_dims(diff_evolution_result.x, axis=0)
            y_new = self.target_function(x_new)

            start_time = time.time()
            self.surrogate_model.update(x_new, y_new)
            logger.info(
                f"Fitted a new surrogate model in {time.time() - start_time} sec."
            )
            self.step_fit_time.append(time.time() - gen_start_time)

            points_x = np.concatenate((points_x, x_new))
            points_y = np.concatenate((points_y, y_new))

            if std_max <= epsilon or self.iterations > max_n_iterations:

                logger.info("std_max <= epsilon or max iterations reached")
                convergence = True
            elif stopping_criterium:
                score = evaluate(
                    self.logging_metrics,
                    self.surrogate_model,
                    self.test_points_x,
                    self.test_points_y,
                )
                self.score = score[self.evaluation_metric.__name__]
                logger.info(f"Evaluation metric score on provided testset: {score}")
                if self.score <= stopping_criterium:
                    self.number_of_iterations_at_convergence = self.iterations
                    logger.info(f"Algorithm converged in {self.iterations} iterations")
                    convergence = True

            self.fit_points_x = points_x
            self.fit_points_y = points_y
            self.step_x.append(points_x)
            self.step_y.append(points_y)
            self.step_score.append(self.score)
            self.step_iter.append(self.iterations + 1)

            timing_dict = {
                "Gen time": self.step_gen_time[self.iterations + 1],
                "Fit time": self.step_fit_time[self.iterations + 1],
            }
            write_scores(self.writer, score, self.iterations + 1)
            write_timer(self.writer, timing_dict, self.iterations + 1)

            self.score = score
            self.iterations += 1

        self.writer.close()

        return self.fit_points_x, self.fit_points_y
