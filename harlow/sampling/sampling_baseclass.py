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
        run_name: str = None,
    ):
        self.domain_lower_bound = np.array(domain_lower_bound)
        self.domain_upper_bound = np.array(domain_upper_bound)
        self.target_function = target_function
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
        if y.ndim < 2:
            raise ValueError(
                f"Target array `y` must have at least 2 dimensions and shape "
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
