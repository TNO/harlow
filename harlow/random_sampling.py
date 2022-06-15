import time
from typing import Callable

import numpy as np
from loguru import logger
from tensorboardX import SummaryWriter

from harlow.helper_functions import evaluate, latin_hypercube_sampling


class Latin_hypercube_sampler:
    def __init__(
        self,
        target_function,
        surrogate_model,
        domain_lower_bound: np.ndarray,
        domain_upper_bound: np.ndarray,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        evaluation_metric: Callable = None,
        run_name: str = None,
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
        # Internal storage for inspection
        self.step_x = []
        self.step_y = []
        self.step_score = []
        self.step_iter = []
        self.step_fit_time = []
        self.step_gen_time = []
        # Init writer for live web-based logging.
        self.writer = SummaryWriter(comment="-" + run_name)

    def sample(
        self,
        n_initial_point: int = None,
        n_iter: int = 20,
        n_new_point_per_iteration: int = 1,
        stopping_criterium: float = None,
    ):
        start_time = time.time()
        points_x = self.fit_points_x
        points_y = self.fit_points_y

        surrogate_model = self.surrogate_model
        surrogate_model.fit(points_x, points_y)
        logger.info(f"Fitted a new surrogate model in {time.time() - start_time} sec.")

        score = evaluate(
            self.metric, self.surrogate_model, self.test_points_x, self.test_points_y
        )
        self.score = score[0]
        self.step_x.append(points_x)
        self.step_y.append(points_y)
        self.step_score.append(score[0])
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
                self.metric,
                self.surrogate_model,
                self.test_points_x,
                self.test_points_y,
            )
            self.score = score[0]
            logger.info(f"Score {score}")
            self.step_x.append(points_x)
            self.step_y.append(points_y)
            self.step_score.append(score[0])
            self.step_iter.append(iteration + 1)
            # Writer log & various metric scores
            if len(score) == 1:
                self.writer.add_scalar("RMSE", score[0], iteration + 1)
            elif len(score) == 2:
                self.writer.add_scalar("RMSE", score[0], iteration + 1)
                self.writer.add_scalar("RRSE", score[1], iteration + 1)
            elif len(score) == 3:
                self.writer.add_scalar("RMSE", score[0], iteration + 1)
                self.writer.add_scalar("RRSE", score[1], iteration + 1)
                self.writer.add_scalar("MAE", score[2], iteration + 1)
            self.writer.add_scalar(
                "Gen time", self.step_gen_time[iteration], iteration + 1
            )
            self.writer.add_scalar(
                "Fit time", self.step_fit_time[iteration], iteration + 1
            )

            iteration += 1

        logger.info(f"Algorithm converged in {iteration} iterations")
        logger.info(f"Algorithm converged with score {score}")
        self.writer.close()
        return self.fit_points_x, self.fit_points_y
