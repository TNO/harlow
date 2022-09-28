import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import shortuuid
from loguru import logger
from tensorboardX import SummaryWriter

from harlow.sampling.step_info import StepInfo
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import evaluate
from harlow.utils.metrics import rmse


class Sampler(ABC):
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
        self.domain_lower_bound = domain_lower_bound
        self.domain_upper_bound = domain_upper_bound
        self.target_function = target_function
        self.surrogate_model = surrogate_model
        self.fit_points_x = fit_points_x
        self.fit_points_y = fit_points_y
        self.test_points_x = test_points_x
        self.test_points_y = test_points_y
        self.evaluation_metric = evaluation_metric
        self.logging_metrics = (
            [self.evaluation_metric]
            if not logging_metrics
            else list(set(logging_metrics.append(self.evaluation_metric)))
        )
        self.verbose = verbose
        self.run_name = run_name
        self.save_dir = save_dir

        self.step_x = []
        self.step_y = []
        self.step_score = []
        self.step_iter = []
        self.step_fit_time = []
        self.step_gen_time = []
        self.steps = []

        if not run_name:
            self.run_name = Sampler.generate_run_name()
        # Init writer for live web-based logging.
        self.writer = SummaryWriter(comment="-" + self.run_name)

    @abstractmethod
    def sample(
            self,
            n_initial_points: int = 20,
            n_new_points_per_iteration: int = 1,
            stopping_criterium: float = 0.05,
            max_n_iterations: int = 5000,
    ):
        pass

    def observer(self, X):
        """
        Wrapper for the user-specified `target_function` that checks input
        and output consistency.
        """

        # Check input points `X`
        if not isinstance(X, np.ndarray):
            raise ValueError(
                f"Parameters `X` must be of type {np.ndarray} but are"
                f" of type {type(X)} "
            )
        if X.ndim != 2:
            raise ValueError(
                f"Input array `X` must have shape `(n_points, n_features)`"
                f" but has shape {X.shape}."
            )

        # Call the target function
        y = self.target_function(X)

        # Check target `y` is a numpy array
        if not isinstance(y, np.ndarray):
            raise ValueError(
                f"Targets `y` must be of type {np.ndarray} but are of"
                f" type {type(y)}."
            )

        # Check shape of `y`
        if y.ndim != 2:
            raise ValueError(
                f"Target array `y` must have exactly 2 dimensions and shape "
                f"(n_points, n_outputs) but has {y.ndim} dimensions and shape "
                f"{y.shape} "
            )

        # Check consistency of input and output shapes
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"Size of input array `X` and output array `y` must match for "
                f"dimension 0 but are {X.shape[0]} and {y.shape[0]} respectively."
            )

        return y

    # Returns True if the loop should finish
    def stopping_criterium(self, iteration: int, max_iter: int) -> bool:
        return iteration > max_iter

    @abstractmethod
    def best_new_points(self, n) -> np.ndarray:
        pass

    @staticmethod
    def generate_run_name():
        return shortuuid.uuid()

    def loop_initialization(self):
        # initialize surrogate
        fit_start_time = time.time()
        self.surrogate_model.fit(self.fit_points_x, self.fit_points_y)
        # step_info.py 0 is initialization
        fit_time = time.time() - fit_start_time

        predicted_points_y = self.surrogate_model.predict(self.test_points_x)
        score = evaluate(self.logging_metrics, self.test_points_y, predicted_points_y)
        self.step_score.append(score)

        #TODO: might break because passing a
        self.steps.append(StepInfo(self.fit_points_x, self.fit_points_y, score, 0, 0, fit_time))

    def loop_iteration(self, iteration: int, n_new_points_per_interation: int):
        logger.info(
            f"Started adaptive iteration step: {iteration}"
        )
        gen_start_time = time.time()
        new_fit_points_x = self.best_new_points(n_new_points_per_interation)
        gen_time = time.time() - gen_start_time
        logger.info(
            f"Found the next best {n_new_points_per_interation} point(s) in "
            f"{gen_time} sec."
        )

        target_func_start_time = time.time()
        new_fit_points_y = self.observer(new_fit_points_x)
        target_func_time = time.time() - target_func_start_time
        logger.info(
            f"Executed target function on {n_new_points_per_interation} point(s) in "
            f"{target_func_time} sec."
        )

        fit_start_time = time.time()
        self.surrogate_model.update(new_fit_points_x, new_fit_points_y)
        fit_time = time.time() - fit_start_time
        logger.info(
            f"Fitted a new surrogate model in {fit_time} sec."
        )

        # Evaluate
        predicted_points_y = self.surrogate_model.predict(self.test_points_x)
        score = evaluate(self.logging_metrics, self.test_points_y, predicted_points_y)

        self.steps.append(StepInfo(new_fit_points_x, new_fit_points_y, score, gen_time, target_func_time, fit_time))

        self.fit_points_x = np.vstack([self.fit_points_x, new_fit_points_x])
        self.fit_points_y = np.vstack([self.fit_points_y, new_fit_points_y])

    def surrogate_loop(self, n_new_points_per_interation: int, max_iter: int):
        self.loop_initialization()

        iteration = 0
        while not self.stopping_criterium(iteration, max_iter):
            self.loop_iteration(iteration, n_new_points_per_interation)
            iteration += 1

