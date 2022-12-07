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

import os
import time
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.optim import optimize_acqf
from loguru import logger

from harlow.sampling.sampling_baseclass import Sampler
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import latin_hypercube_sampling
from harlow.utils.metrics import rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class NegativeIntegratedPosteriorVarianceSampler(Sampler):
    """
    NOTE: This sampler is experimental and currently only works with GPyTorch surrogates
    """

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
        save_dir: Union[str, Path] = 'output',
        n_mc_points: int = 256,
        q: int = 1,
        num_restarts: int = 10,
        raw_samples: int = 512,
        optimizer_options: dict = None,
        func_sample: Callable = latin_hypercube_sampling,
    ):

        super(NegativeIntegratedPosteriorVarianceSampler, self).__init__(
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

        # Sampler specific initialization
        self.n_mc_points = n_mc_points
        self.q = q
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.func_sample = func_sample
        self.metric = evaluation_metric
        self.verbose = verbose
        self.mc_samples = None

        # Search space
        self.bounds = torch.tensor(np.vstack((domain_lower_bound, domain_upper_bound)))
        self.ndim = self.bounds.shape[-1]

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

        # Check surrogate model
        if not self.surrogate_model.is_probabilistic:
            raise NotImplementedError(
                "Uncertainty based sampling only implemented for probabilistic \
                surrogate models."
            )

        if not self.surrogate_model.is_torch:
            raise AttributeError(
                "This BoTorch-based sampler is experimental and currently only "
                "supports GPyTorch GP surrogates"
            )

    def optimize_acqf_and_get_observation(self, acq_func):
        """
        Optimizes the acquisition function, and returns a new candidate and a
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
            return_best_only=True,
        )
        # observe new values
        new_x = candidates.detach().cpu().numpy()
        new_y = self.observer(new_x)
        return new_x, new_y

    def sample(
        self,
        n_initial_points: int = 20,
        n_new_points_per_iteration: int = 1,
        stopping_criterion: float = 0.05,
        max_n_iterations: int = 5000,
    ):

        # Initial sampling
        if (n_initial_points is None) and (
            (self.fit_points_x is None) or (self.fit_points_y is None)
        ):
            raise ValueError(
                "Either `n_initial_points` or a pair of samples and target values"
                "`fit_points_x` and `fit_points_y` must be specified"
            )

        if n_initial_points is None:
            n_initial_points = self.fit_points_x.shape[0]

        # Draw mc samples
        self.mc_samples = self.func_sample(self.n_mc_points)

        if stopping_criterion and not self.evaluation_metric:
            raise ValueError("Specified stopping criterion but no evaluation metric.")

        logger.info(
            f"qNIPV sampling for {max_n_iterations} max iterations with "
            f"{n_initial_points} initial points and stopping "
            f"criterion = {stopping_criterion}."
        )

        # If no initial points are passed, use latin hypercube sampling.
        if self.fit_points_x is None:

            # Draw samples
            self.fit_points_x = self.func_sample(
                n=n_initial_points,
            )

            # Evaluate
            self.fit_points_y = self.observer(self.fit_points_x)

        iteration = 0
        convergence = False
        while (convergence is False) and (iteration < max_n_iterations):

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
                mc_points=torch.tensor(self.mc_samples),
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
            score = self.metric(
                self.surrogate_model._predict(self.test_points_x), self.test_points_y
            )

            if stopping_criterion:
                if score <= stopping_criterion:
                    logger.info(f"Algorithm converged in {iteration} iterations")
                    convergence = True

            # Update training points
            self.fit_points_x = torch.cat([self.fit_points_x, new_x])
            self.fit_points_y = torch.cat(
                [
                    self.fit_points_y,
                    new_y.mreshape(
                        -1,
                    ),
                ]
            )

            iteration += 1

            if self.verbose:
                print(f"Sampling iteration = {iteration} / {max_n_iterations}")

        return self.fit_points_x, self.fit_points_y
