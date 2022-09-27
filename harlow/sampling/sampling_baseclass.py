from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import shortuuid
from tensorboardX import SummaryWriter

from harlow.surrogating.surrogate_model import Surrogate
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
        self.target_function = lambda x: target_function(x).reshape((-1, 1))
        self.surrogate_model = surrogate_model
        self.fit_points_x = fit_points_x
        self.fit_points_y = fit_points_y
        self.test_points_x = test_points_x
        self.test_points_y = test_points_y
        self.evaluation_metric = evaluation_metric
        # self.logging_metrics = (
        #     [self.evaluation_metric]
        #     if not logging_metrics
        #     else list(set(logging_metrics.append(self.evaluation_metric)))
        # )
        self.logging_metrics = logging_metrics
        self.verbose = verbose
        self.run_name = run_name
        self.save_dir = save_dir

        self.step_x = []
        self.step_y = []
        self.step_score = []
        self.step_iter = []
        self.step_fit_time = []
        self.step_gen_time = []

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

    @staticmethod
    def generate_run_name():
        return shortuuid.uuid()
