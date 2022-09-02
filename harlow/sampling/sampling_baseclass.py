from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from harlow.surrogating.surrogate_model import Surrogate


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
        evaluation_metric: Callable = None,
        verbose: bool = False,
    ):
        self.domain_lower_bound = np.array(domain_lower_bound)
        self.domain_upper_bound = np.array(domain_upper_bound)
        self.target_function = lambda x: target_function(x).reshape((-1, 1))
        self.surrogate_model = surrogate_model
        self.fit_points_x = fit_points_x
        self.fit_points_y = fit_points_y
        self.test_points_x = test_points_x
        self.test_points_y = test_points_y
        self.metric = evaluation_metric
        self.verbose = verbose

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def result_as_dict(self):
        pass
