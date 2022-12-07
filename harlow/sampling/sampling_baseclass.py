import enum
import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Union

import json
import numpy as np
import shortuuid
from loguru import logger
from tensorboardX import SummaryWriter

from harlow.sampling.step_info import StepInfo
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import evaluate
from harlow.utils.metrics import rmse


class TargetFunctionEvaluationFailedException(Exception):
    pass


class FailureHandling(enum.Enum):
    fail = 0
    filter = 1
    retry = 2
    retry_new = 3


class Sampler(ABC):
    def __init__(
        self,
        target_function: Callable[[np.ndarray], np.ndarray],
        surrogate_model_constructor,
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
        save_dir: Union[str, Path] = 'output',
        stopping_score: float = None,
        failure_handling: FailureHandling = FailureHandling.fail,
    ):
        self.domain_lower_bound = domain_lower_bound
        self.domain_upper_bound = domain_upper_bound
        self.target_function = target_function
        self.surrogate_model_constructor = surrogate_model_constructor
        self.surrogate_models: List[Surrogate] = []
        self.fit_points_x = fit_points_x
        self.fit_points_y = fit_points_y
        self.dim_out = 0
        self.test_points_x = test_points_x
        self.test_points_y = test_points_y
        self.evaluation_metric = evaluation_metric
        self.logging_metrics = (
            [self.evaluation_metric] if not logging_metrics else logging_metrics
        )
        self.verbose = verbose
        self.run_name = run_name
        self.save_dir: Path = Path(save_dir)

        self.step_x = []
        self.step_y = []
        self.step_score = []
        self.step_iter = []
        self.step_fit_time = []
        self.step_gen_time = []
        self.steps = {}
        self.stopping_score = stopping_score
        self.failure_handling = failure_handling
        self.max_target_func_retries = 3

        if not run_name:
            self.run_name = self._generate_run_name()
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

    def _filter_failed(
        self, x: np.ndarray, y: np.ndarray, s: np.array
    ) -> (np.ndarray, np.ndarray):
        return x[s], y[s]

    def _retry_failed(
        self, x: np.ndarray, y: np.ndarray, s: np.array
    ) -> (np.ndarray, np.ndarray):
        # TODO: implement
        raise NotImplementedError()

        # if retry_attempt > self.max_target_func_retries:
        #     raise TargetFunctionEvaluationFailedException(
        #         f"after retrying {self.max_target_func_retries} times")

    def _target_function_failure_handling(
        self, x: np.ndarray, y: np.ndarray, s: np.array
    ) -> (np.ndarray, np.ndarray):
        if self.failure_handling == FailureHandling.fail:
            if not s.all():
                raise TargetFunctionEvaluationFailedException()
            return x, y
        if self.failure_handling == FailureHandling.filter:
            return self._filter_failed(x, y, s)
        if self.failure_handling == FailureHandling.retry:
            return self._retry_failed(x, y, s)

    def exec_target_function(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        res = self.observer(x)
        if len(res) == 2:
            y, s = res
            return self._target_function_failure_handling(x, y, s)
        else:
            y = res
            return x, y

    def observer(self, X) -> (np.ndarray, np.array):
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
        res = self.target_function(X)
        if len(res) == 2:
            y, _ = res
        else:
            y = res

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

        return res

    # Returns True if the loop should finish
    def _stopping_criterium(
        self, iteration: int, max_iter: int, last_score: float
    ) -> bool:
        if self.stopping_score is not None:
            logger.info(f"Evaluation metric score on provided testset: {last_score}")
            if last_score <= self.stopping_score:
                logger.info(f"Algorithm converged in {iteration} iterations")

        return iteration > max_iter

    @abstractmethod
    def _best_new_points(self, n) -> np.ndarray:
        pass

    def _generate_run_name(self):
        return shortuuid.uuid()+f'_{type(self).__name__}'

    def _fit_models(self):
        # Standard case assumes single model
        self.surrogate_models[0].fit(self.fit_points_x, self.fit_points_y)

    def _update_models(
        self, new_fit_points_x: np.ndarray, new_fit_points_y: np.ndarray
    ):
        # Standard case assumes single model
        self.surrogate_models[0].update(new_fit_points_x, new_fit_points_y)

    def predict(self, points_x: np.ndarray):
        # Standard case assumes single model
        return self.surrogate_models[0].predict(points_x)

    def _loop_initialization(self):
        fit_start_time = time.time()
        self._fit_models()
        fit_time = time.time() - fit_start_time
        self.predicted_points_y = self.predict(self.test_points_x)
        score = evaluate(self.logging_metrics, self.test_points_y,
                         self.predicted_points_y)
        self.step_score.append(score)
        self.steps['initialization'] = StepInfo(self.fit_points_x, self.fit_points_y, score, 0, 0,
                     fit_time).__dict__
        self._write_results(0)

    def _evaluate(self):
        return evaluate(self.logging_metrics, self.test_points_y,
                 self.predicted_points_y)

    def _loop_iteration(self, sample_iteration: int, n_new_points_per_interation: int):
        logger.info(f"Started adaptive iteration step: {sample_iteration}")
        gen_start_time = time.time()
        new_fit_points_x = self._best_new_points(n_new_points_per_interation)
        gen_time = time.time() - gen_start_time
        logger.info(
            f"Found the next best {n_new_points_per_interation} point(s) in "
            f"{gen_time} sec."
        )

        target_func_start_time = time.time()
        # This line overrides the new_fit_points_x
        new_fit_points_x, new_fit_points_y = self.exec_target_function(new_fit_points_x)
        target_func_time = time.time() - target_func_start_time
        logger.info(
            f"Executed target function on {n_new_points_per_interation} point(s) in "
            f"{target_func_time} sec."
        )

        fit_start_time = time.time()
        # surrogate_model.update(new_fit_points_x, new_fit_points_y)
        self._update_models(new_fit_points_x, new_fit_points_y)
        fit_time = time.time() - fit_start_time
        logger.info(f"Fitted a new surrogate model in {fit_time} sec.")

        # Evaluate
        self.predicted_points_y = self.predict(self.test_points_x)
        score = self._evaluate()
        self.steps[sample_iteration] = StepInfo(
                new_fit_points_x,
                new_fit_points_y,
                score,
                gen_time,
                target_func_time,
                fit_time,
            ).__dict__

        self.fit_points_x = np.vstack([self.fit_points_x, new_fit_points_x])
        self.fit_points_y = np.vstack([self.fit_points_y, new_fit_points_y])
        self._write_results(sample_iteration)
        return score

    def set_initial_set(self, points_x: np.ndarray, points_y: np.ndarray):
        self.fit_points_x = points_x
        self.fit_points_y = points_y
        self.dim_out = points_y.shape[1]

    def set_test_set(self, points_x: np.ndarray, points_y: np.ndarray):
        self.test_points_x = points_x
        self.test_points_y = points_y

    def _write_results(self, sample_iteration: int):
        destination = self.save_dir/self.run_name
        destination.mkdir(parents=True, exist_ok=True)
        with open(destination/f"{self.run_name}_steps.json",
                  'w') as f_out:
            json.dump(self.steps, f_out)
        self.save_surrogates(destination / 'surrogates', sample_iteration)

    def save_surrogates(self, iterations_folder: Path, sample_iteration: int):
        surrogates_folder = iterations_folder / 'surrogates_iter-{:04d}_points-{:06d}'.format(sample_iteration, len(self.fit_points_x))
        print(surrogates_folder)
        surrogates_folder.mkdir(parents=True, exist_ok=True)
        for i, surrogate in enumerate(self.surrogate_models):
            surrogate_name = 'surrogate_{:02d}_iter-{:04d}_points-{:06d}'.format(i, sample_iteration, len(self.fit_points_x))
            surrogate.save(surrogates_folder/surrogate_name)

    def load_surrogates(self, surrogates_folder: Path, dim_out):
        self.dim_out = dim_out
        self.surrogate_models = []
        # creating placeholders
        for _ in surrogates_folder.iterdir():
            self.surrogate_models.append(None)
        # Filling the placeholders with the appropriate models
        for surrogate_file in surrogates_folder.iterdir():
            model_index = int(str(surrogate_file.name).split('_')[1])
            print(model_index)
            self.surrogate_models[model_index] = (self.surrogate_model_constructor.load(surrogate_file))

    def surrogate_loop(self, n_new_points_per_interation: int, max_iter: int):
        self._loop_initialization()

        iteration = 1
        # TODO: check if infinity is the bad part of a score
        score = math.inf
        while not self._stopping_criterium(iteration, max_iter, score):
            score = self._loop_iteration(iteration, n_new_points_per_interation)
            iteration += 1
