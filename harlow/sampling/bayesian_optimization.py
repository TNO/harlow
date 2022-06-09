"""
Active sampling test using BoTorch and the Predictive Variance acquisition
function.

The aim is to:

1. Use active learning to efficiently sample and surrogate an expensive
to evaluate model.
    * Relevant to the development of efficient surrogating methods in
    ERP DT WP.A.
    * Based on: https://botorch.org/tutorials/closed_loop_botorch_only

2. Extend to the case where we have a ModelListGP surrogate, i.e. a surrogate
of a multi-output model with each output being an independent task.
    * This is one of the surrogate models that would be interesting to
    test in the Moerdijk case of ERP DT WP.B.

Usefull:
    * Implementation in ax:
    https://github.com/facebook/Ax/issues/460

    * Issues with qNegIntegratedPosteriorVariance:
    https://github.com/pytorch/botorch/issues/573
"""

import math
import os
import time
from typing import Callable

import numpy as np
import torch
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.optim import optimize_acqf
from loguru import logger
from sklearn.metrics import mean_squared_error

from harlow.sampling.sampling_baseclass import Sampler
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import latin_hypercube_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class qNIPV_sampler(Sampler):
    """
    TODO: Can we use q-batches to parallelize the task of finding a new point for each
    surrogate in a list of surrogates?
    """

    def __init__(
        self,
        target_function: Callable[[np.ndarray], np.ndarray],
        surrogate_model: Surrogate,
        domain_lower_bound: np.ndarray,
        domain_upper_bound: np.ndarray,
        n_init_points: int = 10,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        n_max_iter: int = None,
        stopping_criterion: float = None,
        n_mc_points: int = 256,
        q: int = 1,
        num_restarts: int = 10,
        raw_samples: int = 512,
        optimizer_options: dict = None,
        surrogate_kwargs: dict = None,
        func_sample: Callable = latin_hypercube_sampling,
        evaluation_metric: Callable = None,
        verbose: bool = False,
    ):
        self.target_function = lambda x: torch.tensor(
            target_function(x).reshape((-1, 1))
        )
        self.surrogate_model_u = surrogate_model
        self.domain_lower_bound = domain_lower_bound
        self.domain_upper_bound = domain_upper_bound
        self.n_init_points = n_init_points
        self.n_max_iter = n_max_iter
        self.stopping_criterion = stopping_criterion
        self.n_mc_points = n_mc_points
        self.q = q
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.surrogate_kwargs = surrogate_kwargs
        self.func_sample = func_sample
        self.metric = evaluation_metric
        self.verbose = verbose

        # TODO: Replace `torch.tensor` with user-specified input/output transforms
        # To avoid torch errors for parameters set to None
        self.fit_points_x = (
            torch.tensor(fit_points_x) if fit_points_x is not None else None
        )
        self.fit_points_y = (
            torch.tensor(fit_points_y) if fit_points_y is not None else None
        )
        self.test_points_x = (
            torch.tensor(test_points_x) if test_points_x is not None else None
        )
        self.test_points_y = (
            torch.tensor(test_points_y) if test_points_y is not None else None
        )

        # Search space
        self.bounds = torch.tensor(np.vstack((domain_lower_bound, domain_upper_bound)))
        self.ndim = self.bounds.shape[-1]

        # Convergence
        self.iterations = 0
        self.score = None

        # Check if sampling function is provided, otherwise use default
        self.func_sample = lambda n: func_sample(
            n_sample=n,
            domain_lower_bound=self.domain_lower_bound,
            domain_upper_bound=self.domain_upper_bound,
        )

        # Optimizer kwargs
        self.optimizer_options = optimizer_options
        if not self.optimizer_options:
            self.optimizer_options = {"batch_limit": 5, "maxiter": 200}

        # Surrogate kwargs
        if not self.surrogate_kwargs:
            self.surrogate_kwargs = {}

        # Initial sampling
        if (self.n_init_points is None) and (
            (self.fit_points_x is None) or (self.fit_points_y is None)
        ):
            raise ValueError(
                "Either `n_init_points` or a pair of samples and target values"
                "`fit_points_x` and `fit_points_y` must be specified"
            )

        if self.n_init_points is None:
            self.n_init_points = self.fit_points_x.shape[0]

        # Draw mc samples
        self.mc_samples = self.func_sample(self.n_mc_points)

    def optimize_acqf_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a
        noisy observation.
        """
        # optimize
        candidates, acq_values = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,  # used for intialization heuristic
            options=self.optimizer_options,
        )
        # observe new values
        new_x = candidates.detach()
        new_y = self.target_function(new_x).unsqueeze(-1)  # add output dimension
        return new_x, new_y

    def get_model(self):
        """
        Initialize a GP surrogate model
        """
        self.surrogate_model = self.surrogate_model_u(
            self.fit_points_x, self.fit_points_y, **self.surrogate_kwargs
        )

    def sample(self):

        if self.stopping_criterion is not None:
            self.n_max_iter = 1000

        if self.stopping_criterion and not self.evaluation_metric:
            self.evaluation_metric = lambda x, y: math.sqrt(mean_squared_error(x, y))

        logger.info(
            f"qNIPV sampling for {self.n_max_iter} iterations with"
            f"{self.n_init_points} initial points and stopping"
            f"criterion = {self.stopping_criterion}."
        )

        # If no initial points are passed, use latin hypercube sampling.
        if self.fit_points_x is None:

            # Draw samples
            self.fit_points_x = self.func_sample(
                n=self.n_init_points,
            )

            # Evaluate
            self.fit_points_y = self.target_function(self.fit_points_x)
            self.fit_points_x = torch.tensor(self.fit_points_x)
            self.fit_points_y = torch.tensor(self.fit_points_y).squeeze()

        # Initialize model
        # TODO: The GPyTorch models must be initialized with data and
        # therefore behave differently than other models in harlow.
        # The GPyTorch models should be harmonized with the other
        # surrogates
        self.get_model()
        if not self.surrogate_model.is_probabilistic:
            raise NotImplementedError(
                "Uncertainty based sampling only implemented for probabilistic \
                surrogate models."
            )

        convergence = False
        while (convergence is False) and (self.iterations < self.n_max_iter):

            start_time = time.time()

            # Initial fit of surrogate model
            self.surrogate_model.fit(self.fit_points_x, self.fit_points_y)
            logger.info(
                f"Fitted a new surrogate model in {time.time() - start_time} sec."
            )

            # Define acquisition function
            # Objective and sampler set to None based on this issue:
            # https://github.com/pytorch/botorch/issues/573
            qNIPV = qNegIntegratedPosteriorVariance(
                model=self.surrogate_model.model,
                mc_points=torch.tensor(self.n_mc_points),
                # sampler=qmc_sampler,
                objective=None,
                sampler=None,
            )

            # Optimize acquisition function
            start_time = time.time()
            new_x, new_y = self.optimize_acqf_and_get_observation(qNIPV)
            logger.info(
                f"Finished optimizing acquisition function in "
                f"{time.time() - start_time} sec."
            )

            # Check user-specified stopping criterion
            self.score = self.metric(
                self.surrogate_model.predict(self.test_points_x), self.test_points_y
            )

            if self.stopping_criterion:
                if self.score <= self.stopping_criterion:
                    self.number_of_iterations_at_convergence = self.iterations
                    logger.info(f"Algorithm converged in {self.iterations} iterations")
                    convergence = True

            # Update training points
            self.fit_points_x = torch.cat([self.fit_points_x, new_x])
            self.fit_points_y = torch.cat(
                [
                    self.fit_points_y,
                    new_y.reshape(
                        -1,
                    ),
                ]
            )

            self.iterations += 1
            print(f"Sampling iteration = {self.iterations} / {self.n_max_iter}")

        return self.fit_points_x, self.fit_points_y

    def result_as_dict(self):
        pass

    def prediction_std(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        # Return minimum across all models
        std_model = np.atleast_1d(self.surrogate_model.predict(x, return_std=True)[1])

        # TODO: This part should be improved
        try:
            std_noise = np.atleast_1d(self.surrogate_model.noise_std)
        except:  # noqa: E722, B001
            std_noise = [
                std_i.detach().numpy() for std_i in self.surrogate_model.noise_std
            ]

        std = [
            -(std_model_i - std_noise[idx]) for idx, std_model_i in enumerate(std_model)
        ]

        return np.min(std)
