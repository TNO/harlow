import time
from typing import Callable

import numpy as np
from loguru import logger

from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import evaluate, latin_hypercube_sampling
from harlow.utils.log_writer import write_scores, write_timer
from harlow.utils.metrics import rmse


class Latin_hypercube_sampler:
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
        super(Latin_hypercube_sampler, self).__init__(
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

    def sample(
        self,
        n_initial_points: int = None,
        max_n_iterations: int = 20,
        n_new_points_per_iteration: int = 1,
        stopping_criterium: float = None,
    ):
        start_time = time.time()
        points_x = self.fit_points_x
        points_y = self.fit_points_y

        surrogate_model = self.surrogate_model
        surrogate_model.fit(points_x, points_y)
        logger.info(f"Fitted a new surrogate model in {time.time() - start_time} sec.")

        score = evaluate(
            self.logging_metrics,
            self.surrogate_model,
            self.test_points_x,
            self.test_points_y,
        )
        self.score = score[self.evaluation_metric.__name__]
        self.step_x.append(points_x)
        self.step_y.append(points_y)
        self.step_score.append(score)
        self.step_iter.append(0)
        self.step_fit_time.append(time.time() - start_time)

        iteration = 0
        while self.score > stopping_criterium:
            start_time = time.time()
            X_new = latin_hypercube_sampling(
                n_sample=1,
                domain_lower_bound=self.domain_lower_bound,
                domain_upper_bound=self.domain_upper_bound,
            )
            self.step_gen_time.append(time.time() - start_time)
            y_new = self.target_function(X_new).reshape((-1, 1))
            points_x = np.concatenate((points_x, X_new))
            points_y = np.concatenate((points_y, y_new))

            start_time = time.time()
            surrogate_model.fit(points_x, points_y)
            logger.info(f"Fitted a surrogate model in {time.time() - start_time} sec.")
            self.step_fit_time.append(time.time() - start_time)

            self.fit_points_x = points_x
            self.fit_points_y = points_y
            score = evaluate(
                self.logging_metrics,
                self.surrogate_model,
                self.test_points_x,
                self.test_points_y,
            )
            self.score = score[self.evaluation_metric.__name__]
            logger.info(f"Score {score}")
            self.step_x.append(points_x)
            self.step_y.append(points_y)
            self.step_score.append(score[0])
            self.step_iter.append(iteration + 1)
            timing_dict = {
                "Gen time": self.step_gen_time[iteration + 1],
                "Fit time": self.step_fit_time[iteration + 1],
            }
            write_scores(self.writer, score, iteration + 1)
            write_timer(self.writer, timing_dict, iteration + 1)

            iteration += 1

            if iteration >= max_n_iterations:
                break

        logger.info(f"Algorithm converged in {iteration} iterations")
        logger.info(f"Algorithm converged with score {score}")
        self.writer.close()
        return self.fit_points_x, self.fit_points_y
