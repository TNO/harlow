"""Surrogate model (function) module for fitting (not adaptive) and prediction.

`f_surrogate(x) ~= f_target(x)` for `R^n -> R^1` functions.

The main requirements towards each surrogate model are that they:
* can be fitted to points from the target function.
* can make predictions at user selected points.

"""
import re
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import gpytorch
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from harlow.utils.helper_functions import NLL, normal_sp
from harlow.utils.transforms import Identity, TensorTransform, Transform

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

# TODO add retraining strategies


class Surrogate(ABC):
    """
    Abstract base class for the surrogate models. Each surrogate
    model must initialize the abstact base class using the following
    statement during initialization:

        super().__init__(
            input_transform=input_transform,
            output_transform=output_transform
        )

    All surrogates must also implement the following methods:
    * `create_model`
    * `_fit`
    * `_predict`
    * `_update`
    """

    def __init__(
        self,
        input_transform: Optional[Transform] = Identity,
        output_transform: Optional[Transform] = Identity,
        **kwargs,
    ):
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.n_batches = None
        self.n_features = None
        self.n_training = None

    @property
    @abstractmethod
    def is_probabilistic(self):
        pass

    @property
    @abstractmethod
    def is_multioutput(self):
        pass

    @property
    @abstractmethod
    def is_torch(self):
        pass

    @abstractmethod
    def _fit(self, X, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _predict(self, X, **kwargs) -> Union[np.ndarray, torch.tensor]:
        raise NotImplementedError

    @abstractmethod
    def _update(self, new_X, new_y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_model(self):
        raise NotImplementedError

    @staticmethod
    def check_inputs(X, y=None):

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

        if y is not None:
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

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Calls the `_fit()` method of the surrogate model instance after applying
        transforms and checking inputs. A user specified surrogate model only
        has to implement the `_fit()` method. Note that `fit()` is a user-facing
        method and should not be called from within the class.
        """
        self.check_inputs(X, y=y)
        X = self.input_transform().forward(X)
        y = self.output_transform().forward(y)
        self._fit(X, y, **kwargs)

    def update(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Calls the `_update()` method of the surrogate model instance after applying
        transforms and checking inputs. A user specified surrogate model only
        has to implement the `_update()` method. Note that `update()` is a user-facing
        method and should not be called from within the class.
        """
        self.check_inputs(X, y=y)
        X = self.input_transform().forward(X)
        y = self.output_transform().forward(y)
        self._update(X, y, **kwargs)

    def predict(
        self, X: np.ndarray, return_std: Optional = False, **kwargs
    ) -> Union[
        torch.tensor,
        np.ndarray,
        Tuple[torch.tensor, torch.tensor],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Calls the `_predict()` method of the surrogate model instance after applying
        transforms and checking inputs. A user specified surrogate model only
        has to implement the `_predict()` method. Note that `predict()` is a user-facing
        method and should not be called from within the class.
        """
        self.check_inputs(X)
        X = self.input_transform().forward(X)

        if return_std:
            samples, std = self._predict(X, return_std=return_std, **kwargs)
            return self.output_transform().reverse(samples), std
        else:
            samples = self._predict(X, return_std=return_std, **kwargs)
            return self.output_transform().reverse(samples)


class VanillaGaussianProcess(Surrogate):
    is_probabilistic = True
    is_multioutput = False
    is_torch = False
    kernel = 1.0 * RBF(1.0) + WhiteKernel(1.0, noise_level_bounds=(5e-5, 5e-2))

    def __init__(
        self,
        train_restarts: int = 10,
        kernel=kernel,
        noise_std=None,
        input_transform=Identity,
        output_transform=Identity,
        **kwargs,
    ):
        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.train_restarts = train_restarts
        self.noise_std = noise_std
        self.kernel = kernel

        self.create_model()

    def create_model(self):
        self.model = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=self.train_restarts, random_state=0
        )

    def _fit(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.model.fit(self.X, self.y)
        self.noise_std = self.get_noise()

    def get_noise(self):
        attrs = list(vars(VanillaGaussianProcess.kernel).keys())

        white_kernel_attr = []
        for attr in attrs:
            if re.match(pattern="^k[0-9]", string=attr):
                attr_val = getattr(VanillaGaussianProcess.kernel, attr)
                if re.match(pattern="^WhiteKernel", string=str(attr_val)):
                    white_kernel_attr.append(attr)

        if len(white_kernel_attr) == 0:
            raise ValueError(
                "The used kernel should have an additive WhiteKernel component but it was not \
                provided."
            )

        if len(white_kernel_attr) > 1:
            raise ValueError(
                f"The used kernel should have only one additive WhiteKernel component, \
                {len(white_kernel_attr)} components were provided."
            )

        return getattr(self.model.kernel, white_kernel_attr[0]).noise_level ** 0.5

    def _predict(self, X, return_std=False, **kwargs):
        samples, std = self.model.predict(X, return_std=True)

        # Sklearn reshapes the predictions into a 1d array if there is only one
        # output. Reshape to column.
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        if return_std:
            return samples, std
        else:
            return samples

    def _update(self, new_X, new_y, **kwargs):
        new_X = new_X
        if new_X.ndim > self.X.ndim:
            self.X = np.expand_dims(self.X, axis=0)
        self.X = np.concatenate([self.X, new_X], axis=0)

        if new_y.ndim > self.y.ndim:
            self.y = np.expand_dims(self.y, axis=0)
        self.y = np.concatenate([self.y, new_y], axis=0)

        # TODO check if this the best way to use for incremental
        #  learning/online learning
        self.kernel.set_params(**(self.model.kernel_.get_params()))
        self.create_model()

        self._fit(self.X, self.y)


class GaussianProcessTFP(Surrogate):
    is_probabilistic = True
    is_multioutput = False
    is_torch = False

    def __init__(
        self,
        train_iterations=50,
        input_transform=Identity,
        output_transform=Identity,
        **kwargs,
    ):

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.train_iterations = train_iterations

    def create_model(self):
        def _build_gp(amplitude, length_scale, observation_noise_variance):
            kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

            return tfd.GaussianProcess(
                kernel=kernel,
                index_points=self.observation_index_points,
                observation_noise_variance=observation_noise_variance,
            )

        self.gp_joint_model = tfd.JointDistributionNamed(
            {
                "amplitude": tfd.LogNormal(loc=0.0, scale=np.float64(1.0)),
                "length_scale": tfd.LogNormal(loc=0.0, scale=np.float64(1.0)),
                "observation_noise_variance": tfd.LogNormal(
                    loc=0.0, scale=np.float64(1.0)
                ),
                "observations": _build_gp,
            }
        )

    def optimize_parameters(self, verbose=0):
        constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

        self.amplitude_var = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=constrain_positive,
            name="amplitude",
            dtype=np.float64,
        )

        self.length_scale_var = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=constrain_positive,
            name="length_scale",
            dtype=np.float64,
        )

        self.observation_noise_variance_var = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=constrain_positive,
            name="observation_noise_variance_var",
            dtype=np.float64,
        )

        trainable_variables = [
            v.trainable_variables[0]
            for v in [
                self.amplitude_var,
                self.length_scale_var,
                self.observation_noise_variance_var,
            ]
        ]

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        @tf.function(autograph=False)
        def train_model():
            with tf.GradientTape() as tape:
                loss = -self.target_log_prob(
                    self.amplitude_var,
                    self.length_scale_var,
                    self.observation_noise_variance_var,
                )
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))

            return loss

        lls_ = np.zeros(self.train_iterations)
        for i in range(self.train_iterations):
            loss = train_model()
            lls_[i] = loss

        self.kernel = tfk.ExponentiatedQuadratic(
            self.amplitude_var, self.length_scale_var
        )
        if verbose == 1:
            print("Trained parameters:")
            print(f"amplitude: {self.amplitude_var._value().numpy()}")
            print(f"length_scale: {self.length_scale_var._value().numpy()}")
            print(
                "observation_noise_variance: "
                f"{self.observation_noise_variance_var._value().numpy()}"
            )

    def target_log_prob(self, amplitude, length_scale, observation_noise_variance):
        return self.gp_joint_model.log_prob(
            {
                "amplitude": amplitude,
                "length_scale": length_scale,
                "observation_noise_variance": observation_noise_variance,
                "observations": tf.squeeze(self.observations),
            }
        )

    def _fit(self, X, y, **kwargs):
        self.observation_index_points = X
        self.observations = y.flatten()
        self.create_model()
        self.optimize_parameters()

    def _predict(
        self, X, iterations=50, return_std=False, return_samples=False, **kwargs
    ):
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=X,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
            observation_noise_variance=self.observation_noise_variance_var,
            predictive_noise_variance=0.0,
        )

        samples = gprm.sample(iterations)

        if return_samples:
            if return_std:
                return (
                    np.mean(samples, axis=0).reshape(-1, 1),
                    np.std(samples, axis=0).reshape(-1, 1),
                    samples.numpy(),
                )
            else:
                return np.mean(samples, axis=0).reshape(-1, 1), samples.numpy()
        if return_std:
            return np.mean(samples, axis=0).reshape(-1, 1), np.std(
                samples, axis=0
            ).reshape(-1, 1)
        else:
            return np.mean(samples, axis=0).reshape(-1, 1)

    def _update(self, new_X, new_y, **kwargs):
        self.observation_index_points = np.concatenate(
            [self.observation_index_points, new_X]
        )

        if new_y.ndim > self.observations.ndim:
            new_y = new_y.flatten()
        self.observations = np.concatenate([self.observations, new_y])
        self.optimize_parameters(verbose=False)


class GaussianProcessRegression(Surrogate):
    """
    DEPRECATED

    Simple Gaussian process regression model using GPyTorch

    Notes:
        * This model must be initialized with data
        * The `.fit(X, y)` method replaces the current `train_X` and `train_y`
        with its arguments every time it is called.
        * The `.update(X, y)` method will append the new X and y training tensors
        to the existing `train_X`, `train_y` and perform the fitting.
        * Both `.fit()` and `.update()` will re-instantiate a new model. There
        is likely a better solution to this.

    TODO:
    * GpyTorch probably has existing functionality for updating and refitting
    models. This is likely the prefered approach and should replace the current
    approach where the model is redefined at each call to `fit()` or `.update()`.
    * Rewrite to use the existing Gaussian process surrogate class
    * Add type hinting
    * Improve docstrings
    """

    is_probabilistic = True
    is_multioutput = False
    is_torch = True

    def __init__(
        self,
        training_max_iter=100,
        learning_rate=0.1,
        input_transform=TensorTransform,
        output_transform=TensorTransform,
        min_loss_rate=None,
        optimizer=None,
        mean=None,
        covar=None,
        show_progress=True,
        silence_warnings=False,
        fast_pred_var=False,
        dev=None,
        **kwargs,
    ):

        raise DeprecationWarning(
            "This model is deprecated and will be removed as it does not comply to "
            "the input/output shape convention and is a subset of the "
            "`BatchIndependentGaussianProcess` and `VanillaGaussianProcess` models."
        )

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.training_max_iter = training_max_iter
        self.noise_std = None
        self.likelihood = None
        self.mll = None
        self.learning_rate = learning_rate
        self.min_loss_rate = min_loss_rate
        self.mean_module = mean
        self.covar_module = covar
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.predictions = None
        self.fast_pred_var = fast_pred_var
        self.device = dev

        if self.optimizer is None:
            warnings.warn("No optimizer specified, using default.", UserWarning)

        # Silence torch and numpy warnings (related to converting
        # between np.arrays and torch.tensors).
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def create_model(self):

        # Reset optimizer
        self.optimizer = None

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training"
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(
            self.train_X, self.train_y.squeeze(-1), self.likelihood
        ).to(self.device)

    def _fit(self, train_X, train_y, **kwargs):

        self.train_X = train_X.to(self.device)
        self.train_y = train_y.to(self.device)

        # Create model
        self.create_model()

        # Switch the model to train mode
        self.model.train()
        self.likelihood.train()

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, torch.ravel(self.train_y))
            loss.backward()
            self.vec_loss.append(loss.item())
            self.optimizer.step()

            # TODO: Will this work for negative losss? CHECK
            loss_ratio = None
            if self.min_loss_rate:
                loss_ratio = (loss_0 - loss.item()) - self.min_loss_rate * loss.item()

            # From https://stackoverflow.com/questions/5290994
            # /remove-and-replace-printed-items
            if self.show_progress:
                print(
                    f"Iter = {_i} / {self.training_max_iter},"
                    f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                    end="\r",
                    flush=True,
                )

            # Get noise value
            self.noise_std = self.get_noise()

            # Check criterion and break if true
            if self.min_loss_rate:
                if loss_ratio < 0.0:
                    break

            # Set previous iter loss to current
            loss_0 = loss.item()

        print(
            f"Iter = {self.training_max_iter},"
            f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}"
        )

    def _predict(self, X_pred, return_std=False, **kwargs):

        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var(self.fast_pred_var):
            self.prediction = self.likelihood(self.model(X_pred))

        # Get mean, variance and std. dev per model
        self.mean = self.prediction.mean
        self.var = self.prediction.variance
        self.std = self.prediction.variance.sqrt()

        # Get confidence intervals per model
        self.cr_l, self.cr_u = self.prediction.confidence_region()

        if return_std:
            return self.prediction.sample(), self.std
        else:
            return self.prediction.sample()

    def sample_posterior(self, n_samples=1):
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return self.prediction.n_sample(n_samples)

    def _update(self, new_X, new_y, **kwargs):

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)

        self.optimizer = None
        self._fit(self.train_X, self.train_y)

    def get_noise(self):
        return self.model.likelihood.noise.sqrt()


class ModelListGaussianProcess(Surrogate):
    """
    Utility class to generate a surrogate composed of multiple independent
    Gaussian processes. Currently uses GPyTorch.

    It is assumed that the training inputs are common between the N_task GPs,
    but not all features are used in each GP.

    Notes:
        * This model must be initialized with data
        * The `.fit(X, y)` method replaces the current `train_X` and `train_y`
        with its arguments every time it is called.
        * The `.update(X, y)` method will append the new X and y training tensors
        to the existing `train_X`, `train_y` and perform the fitting.
        * Both `.fit()` and `.update()` will re-instantiate a new model. There
        is likely a better solution to this.

    TODO:
    * GpyTorch probably has existing functionality for updating and refitting
    models. This is likely the prefered approach and should replace the current
    approach where the model is redefined at each call to `fit()` or `.update()`.
    * Rewrite to use the existing Gaussian process surrogate class
    * Add type hinting
    * Improve docstrings
    """

    is_probabilistic = True
    is_multioutput = True
    is_torch = True

    def __init__(
        self,
        model_names=None,
        training_max_iter=100,
        learning_rate=0.1,
        input_transform=TensorTransform,
        output_transform=TensorTransform,
        min_loss_rate=None,
        optimizer=None,
        mean=None,
        covar=None,
        list_params=None,
        show_progress=True,
        silence_warnings=False,
        fast_pred_var=False,
        dev=None,
    ):

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model_names = model_names
        self.model = None
        self.training_max_iter = training_max_iter
        self.noise_std = None
        self.likelihood = None
        self.mll = None
        self.learning_rate = learning_rate
        self.min_loss_rate = min_loss_rate
        self.mean_module = mean
        self.covar_module = covar
        self.list_params = list_params
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.prediction = None
        self.fast_pred_var = fast_pred_var
        self.device = dev

        if self.optimizer is None:
            warnings.warn("No optimizer specified, using default.", UserWarning)

        # Silence torch and numpy warnings (related to converting
        # between np.arrays and torch.tensors).
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def create_model(self):

        # Reset optimizer
        self.optimizer = None

        if self.list_params is None:
            self.list_params = [
                [idx for idx in range(self.train_X.shape[1])]
            ] * self.train_y.shape[1]

        if self.model_names is None:
            self.model_names = [f"model_{idx}" for idx in range(self.train_y.shape[1])]

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training "
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        if self.train_y.shape[1] != len(self.model_names):
            raise ValueError(
                f"Dim 1 of `train_y` must be equal to the number of models"
                f" but is {self.train_y.shape[1]} != {len(self.model_names)}"
            )

        if self.train_y.shape[1] != len(self.list_params):
            raise ValueError(
                f"Dim 1 of `train_y` must be equal to the length of the list of "
                f"parameters but is {self.train_y.shape[1]} != {len(self.list_params)}"
            )

        # Assemble the models and likelihoods
        list_likelihoods = []
        list_surrogates = []
        for i, _name in enumerate(self.model_names):

            # Initialize list of GP surrogates
            list_likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())
            list_surrogates.append(
                ExactGPModel(
                    self.train_X[:, self.list_params[i]],
                    self.train_y[:, i],
                    list_likelihoods[i],
                )
            )

            # Collect the independent GPs in ModelList and LikelihoodList objects
            self.model = gpytorch.models.IndependentModelList(*list_surrogates).to(
                self.device
            )
            self.likelihood = gpytorch.likelihoods.LikelihoodList(*list_likelihoods).to(
                self.device
            )

    def _fit(self, train_X, train_y, **kwargs):

        self.train_X = train_X.to(self.device)
        self.train_y = train_y.to(self.device)

        # Create model
        self.create_model()

        # Switch the model to train mode
        self.model.train()
        self.likelihood.train()

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = SumMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            self.vec_loss.append(loss.item())
            self.optimizer.step()

            # TODO: Will this work for negative losss? CHECK
            loss_ratio = None
            if self.min_loss_rate:
                loss_ratio = (loss_0 - loss.item()) - self.min_loss_rate * loss.item()

            # From https://stackoverflow.com/questions/5290994
            # /remove-and-replace-printed-items
            if self.show_progress:
                print(
                    f"Iter = {_i} / {self.training_max_iter},"
                    f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                    end="\r",
                    flush=True,
                )

            # Get noise value
            self.noise_std = self.get_noise()

            # Check criterion and break if true
            if self.min_loss_rate:
                if loss_ratio < 0.0:
                    break

            # Set previous iter loss to current
            loss_0 = loss.item()

        print(
            f"Iter = {self.training_max_iter},"
            f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}"
        )

    def _predict(self, X_pred, return_std=False, as_array=False, **kwargs):

        X_pred = X_pred.to(self.device)

        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Get input features per model
        X_list = [X_pred[:, prm_list] for prm_list in self.list_params]

        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var(self.fast_pred_var):
            self.prediction = self.likelihood(*self.model(*X_list))

        # Generate output for each model
        self.mean = torch.zeros(X_pred.shape[0], len(self.model_names))
        self.cr_l = torch.zeros(X_pred.shape[0], len(self.model_names))
        self.cr_u = torch.zeros(X_pred.shape[0], len(self.model_names))
        self.std = torch.zeros(X_pred.shape[0], len(self.model_names))
        self.var = torch.zeros(X_pred.shape[0], len(self.model_names))
        sample = torch.zeros(X_pred.shape[0], len(self.model_names))

        for j, (_submodel, _prediction) in enumerate(
            zip(self.model.models, self.prediction)
        ):

            # Get mean, variance and std. dev per model
            self.mean[:, j] = _prediction.mean
            self.var[:, j] = _prediction.variance
            self.std[:, j] = _prediction.variance.sqrt()

            # Get posterior predictive samples per model
            sample[:, j] = _prediction.sample()

            self.cr_l[:, j], self.cr_u[:, j] = _prediction.confidence_region()

        if return_std:
            return (
                sample.numpy() if as_array else sample,
                self.std.numpy() if as_array else self.std,
            )
        else:
            return sample.numpy() if as_array else sample

    def sample_posterior(self, n_samples=1):

        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return [prediction.n_sample(n_samples) for prediction in self.prediction]

    def _update(self, new_X, new_y, **kwargs):

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)

        self.optimizer = None
        self._fit(self.train_X, self.train_y)

    def get_noise(self):
        return [
            likelihood.noise.sqrt() for likelihood in self.model.likelihood.likelihoods
        ]


class BatchIndependentGaussianProcess(Surrogate):
    """
    Utility class to generate a surrogate composed of multiple independent
    Gaussian processes with the same covariance and likelihood:
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html

    Notes:
        * This model must be initialized with data
        * The `.fit(X, y)` method replaces the current `train_X` and `train_y`
        with its arguments every time it is called.
        * The `.update(X, y)` method will append the new X and y training tensors
        to the existing `train_X`, `train_y` and perform the fitting.
        * Both `.fit()` and `.update()` will re-instantiate a new model. There
        is likely a better solution to this.

    TODO:
    * GpyTorch probably has existing functionality for updating and refitting
    models. This is likely the prefered approach and should replace the current
    approach where the model is redefined at each call to `fit()` or `.update()`.
    * Rewrite to use the existing Gaussian process surrogate class
    * Add type hinting
    * Improve docstrings
    """

    is_probabilistic = True
    is_multioutput = True
    is_torch = True

    def __init__(
        self,
        training_max_iter=100,
        learning_rate=0.1,
        input_transform=TensorTransform,
        output_transform=TensorTransform,
        min_loss_rate=None,
        optimizer=None,
        mean=None,
        covar=None,
        show_progress=True,
        silence_warnings=False,
        fast_pred_var=False,
        dev=None,
    ):

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.training_max_iter = training_max_iter
        self.noise_std = None
        self.likelihood = None
        self.mll = None
        self.learning_rate = learning_rate
        self.min_loss_rate = min_loss_rate
        self.mean_module = mean
        self.covar_module = covar
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.predictions = None
        self.fast_pred_var = fast_pred_var
        self.device = dev

        if self.optimizer is None:
            warnings.warn("No optimizer specified, using default.", UserWarning)

        # Silence torch and numpy warnings (related to converting
        # between np.arrays and torch.tensors).
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def create_model(self):

        # Reset optimizer
        self.optimizer = None
        self.num_tasks = self.train_y.shape[1]

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training"
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)

        self.model = BatchIndependentMultitaskGPModel(
            self.train_X, self.train_y, self.likelihood, self.num_tasks
        ).to(self.device)

    def _fit(self, train_X, train_y, **kwargs):

        self.train_X = train_X.to(self.device)
        self.train_y = train_y.to(self.device)

        # Create model
        self.create_model()

        # Switch the model to train mode
        self.model.train()
        self.likelihood.train()

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.vec_loss.append(loss.item())
            self.optimizer.step()

            # TODO: Will this work for negative losss? CHECK
            loss_ratio = None
            if self.min_loss_rate:
                loss_ratio = (loss_0 - loss.item()) - self.min_loss_rate * loss.item()

            # From https://stackoverflow.com/questions/5290994
            # /remove-and-replace-printed-items
            if self.show_progress:
                print(
                    f"Iter = {_i} / {self.training_max_iter},"
                    f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                    end="\r",
                    flush=True,
                )

            # Get noise value
            self.noise_std = self.get_noise()

            # Check criterion and break if true
            if self.min_loss_rate:
                if loss_ratio < 0.0:
                    break

            # Set previous iter loss to current
            loss_0 = loss.item()

        print(
            f"Iter = {self.training_max_iter},"
            f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}"
        )

    def _predict(self, X_pred, return_std=False, **kwargs):

        X_pred = X_pred.to(self.device)

        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var(self.fast_pred_var):
            self.prediction = self.likelihood(self.model(X_pred))

        # Get mean, variance and std. dev per model
        self.mean = self.prediction.mean
        self.var = self.prediction.variance
        self.std = self.prediction.variance.sqrt()

        # Get confidence intervals per model
        self.cr_l, self.cr_u = self.prediction.confidence_region()

        if return_std:
            return self.prediction.sample(), self.std
        else:
            return self.prediction.sample()

    def sample_posterior(self, n_samples=1):
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return self.prediction.n_sample(n_samples)

    def _update(self, new_X, new_y, **kwargs):

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)

        self.optimizer = None
        self._fit(self.train_X, self.train_y)

    def get_noise(self):
        return self.model.likelihood.noise.sqrt()


class MultiTaskGaussianProcess(Surrogate):
    """
    !!!!!!!!!!!! IN PROGRESS !!!!!!!!!!!!!!!

    Utility class to generate a surrogate composed of multiple correlated
    Gaussian processes:
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html

    Notes:
        * This model must be initialized with data
        * The `.fit(X, y)` method replaces the current `train_X` and `train_y`
        with its arguments every time it is called.
        * The `.update(X, y)` method will append the new X and y training tensors
        to the existing `train_X`, `train_y` and perform the fitting.
        * Both `.fit()` and `.update()` will re-instantiate a new model. There
        is likely a better solution to this.

    TODO:
    * GpyTorch probably has existing functionality for updating and refitting
    models. This is likely the prefered approach and should replace the current
    approach where the model is redefined at each call to `fit()` or `.update()`.
    * Rewrite to use the existing Gaussian process surrogate class
    * Add type hinting
    * Improve docstrings
    """

    is_probabilistic = True
    is_multioutput = True
    is_torch = True

    def __init__(
        self,
        num_tasks=None,
        training_max_iter=100,
        learning_rate=0.1,
        input_transform=TensorTransform,
        output_transform=TensorTransform,
        min_loss_rate=None,
        optimizer=None,
        mean=None,
        covar=None,
        show_progress=True,
        silence_warnings=False,
        fast_pred_var=False,
        dev=None,
    ):

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.num_tasks = num_tasks
        self.training_max_iter = training_max_iter
        self.noise_std = None
        self.likelihood = None
        self.mll = None
        self.learning_rate = learning_rate
        self.min_loss_rate = min_loss_rate
        self.mean_module = mean
        self.covar_module = covar
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.predictions = None
        self.fast_pred_var = fast_pred_var
        self.device = dev

        if self.optimizer is None:
            warnings.warn("No optimizer specified, using default.", UserWarning)

        # Silence torch and numpy warnings (related to converting
        # between np.arrays and torch.tensors).
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def create_model(self):

        if self.num_tasks is None:
            self.num_tasks = self.train_y.shape[1]

        # Reset optimizer
        self.optimizer = None

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training"
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        if self.train_y.shape[1] != self.num_tasks:
            raise ValueError(
                f"Dim 1 of `train_y` must be equal to the number of tasks"
                f"but is {self.train_y.shape[1]} != {self.num_tasks}"
            )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)
        self.model = MultitaskGPModel(
            self.train_X, self.train_y, self.likelihood, self.num_tasks
        ).to(self.device)

    def _fit(self, train_X, train_y, **kwargs):

        self.train_X = train_X.to(self.device)
        self.train_y = train_y.to(self.device)

        # Create model
        self.create_model()

        # Switch the model to train mode
        self.model.train()
        self.likelihood.train()

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.vec_loss.append(loss.item())
            self.optimizer.step()

            # TODO: Will this work for negative losss? CHECK
            loss_ratio = None
            if self.min_loss_rate:
                loss_ratio = (loss_0 - loss.item()) - self.min_loss_rate * loss.item()

            # From https://stackoverflow.com/questions/5290994
            # /remove-and-replace-printed-items
            if self.show_progress:
                print(
                    f"Iter = {_i} / {self.training_max_iter},"
                    f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                    end="\r",
                    flush=True,
                )

            # Get noise value
            self.noise_std = self.get_noise()

            # Check criterion and break if true
            if self.min_loss_rate:
                if loss_ratio < 0.0:
                    break

            # Set previous iter loss to current
            loss_0 = loss.item()

        print(
            f"Iter = {self.training_max_iter},"
            f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}"
        )

    def _predict(self, X_pred, return_std=False, **kwargs):

        X_pred = X_pred.to(self.device)

        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var(self.fast_pred_var):
            self.prediction = self.likelihood(self.model(X_pred))

        # Get mean, variance and std. dev per model
        self.mean = self.prediction.mean
        self.var = self.prediction.variance
        self.std = self.prediction.variance.sqrt()

        # Get confidence intervals per model
        self.cr_l, self.cr_u = self.prediction.confidence_region()

        if return_std:
            return self.prediction.sample(), self.std
        else:
            return self.prediction.sample()

    def sample_posterior(self, n_samples=1):
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return self.prediction.n_sample(n_samples)

    def _update(self, new_X, new_y, **kwargs):

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)

        self.optimizer = None
        self._fit(self.train_X, self.train_y)

    def get_noise(self):
        return self.model.likelihood.noise.sqrt()


class DeepKernelMultiTaskGaussianProcess(Surrogate):
    """
    !!!!!!!!!!!! IN PROGRESS !!!!!!!!!!!!!!!

    MultiTask Deep kernel learning Gaussian process, based on this single-output
    example:
    https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/
    Simple_GP_Regression_With_LOVE_Fast_Variances_and_Sampling.html

    Notes:
        * This model must be initialized with data
        * The `.fit(X, y)` method replaces the current `train_X` and `train_y`
        with its arguments every time it is called.
        * The `.update(X, y)` method will append the new X and y training tensors
        to the existing `train_X`, `train_y` and perform the fitting.
        * Both `.fit()` and `.update()` will re-instantiate a new model. There
        is likely a better solution to this.

    TODO:
    * GpyTorch probably has existing functionality for updating and refitting
    models. This is likely the prefered approach and should replace the current
    approach where the model is redefined at each call to `fit()` or `.update()`.
    * Rewrite to use the existing Gaussian process surrogate class
    * Add type hinting
    * Improve docstrings
    """

    is_probabilistic = True
    is_multioutput = True
    is_torch = True

    def __init__(
        self,
        input_transform=TensorTransform,
        output_transform=TensorTransform,
        training_max_iter=100,
        learning_rate=0.1,
        min_loss_rate=None,
        optimizer=None,
        mean=None,
        covar=None,
        show_progress=True,
        silence_warnings=False,
        fast_pred_var=False,
        dev=None,
    ):

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.training_max_iter = training_max_iter
        self.noise_std = None
        self.likelihood = None
        self.mll = None
        self.learning_rate = learning_rate
        self.min_loss_rate = min_loss_rate
        self.mean_module = mean
        self.covar_module = covar
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.predictions = None
        self.fast_pred_var = fast_pred_var
        self.device = dev
        self.num_tasks = None
        self.num_features = None

        if self.optimizer is None:
            warnings.warn("No optimizer specified, using default.", UserWarning)

        # Silence torch and numpy warnings (related to converting
        # between np.arrays and torch.tensors).
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def create_model(self):

        self.num_tasks = self.train_y.shape[1]
        self.num_features = self.train_X.shape[1]

        # Reset optimizer
        self.optimizer = None

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training"
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        if self.train_y.shape[1] != self.num_tasks:
            raise ValueError(
                f"Dim 1 of `train_y` must be equal to the number of tasks"
                f"but is {self.train_y.shape[1]} != {self.num_tasks}"
            )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)
        self.model = DeepKernelLearningGPModel(
            self.train_X,
            self.train_y,
            self.likelihood,
            self.num_tasks,
            self.num_features,
        ).to(self.device)

    def _fit(self, train_X, train_y, **kwargs):

        self.train_X = train_X.to(self.device)
        self.train_y = train_y.to(self.device)

        # Create model
        self.create_model()

        # Switch the model to train mode
        self.model.train()
        self.likelihood.train()

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.vec_loss.append(loss.item())
            self.optimizer.step()

            # TODO: Will this work for negative losss? CHECK
            loss_ratio = None
            if self.min_loss_rate:
                loss_ratio = (loss_0 - loss.item()) - self.min_loss_rate * loss.item()

            # From https://stackoverflow.com/questions/5290994
            # /remove-and-replace-printed-items
            if self.show_progress:
                print(
                    f"Iter = {_i} / {self.training_max_iter},"
                    f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                    end="\r",
                    flush=True,
                )

            # Get noise value
            self.noise_std = self.get_noise()

            # Check criterion and break if true
            if self.min_loss_rate:
                if loss_ratio < 0.0:
                    break

            # Set previous iter loss to current
            loss_0 = loss.item()

        print(
            f"Iter = {self.training_max_iter},"
            f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}"
        )

    def _predict(self, X_pred, return_std=False, **kwargs):

        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var(self.fast_pred_var):
            self.prediction = self.likelihood(self.model(X_pred))

        # Get mean, variance and std. dev per model
        self.mean = self.prediction.mean
        self.var = self.prediction.variance
        self.std = self.prediction.variance.sqrt()

        # Get confidence intervals per model
        self.cr_l, self.cr_u = self.prediction.confidence_region()

        if return_std:
            return self.prediction.sample(), self.std
        else:
            return self.prediction.sample()

    def sample_posterior(self, n_samples=1):
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return self.prediction.n_sample(n_samples)

    def _update(self, new_X, new_y, **kwargs):

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)

        self.optimizer = None
        self._fit(self.train_X, self.train_y)

    def get_noise(self):
        return self.model.likelihood.noise.sqrt()


class NeuralNetwork(Surrogate):
    """
    Class for Neural Networks.
    The class takes an uncompiled tensorflow Model, e.g.


    """

    learning_rate_update = 0.001
    is_probabilistic = False
    is_multioutput = True
    is_torch = False

    def __init__(
        self,
        epochs=10,
        batch_size=32,
        loss="mse",
        input_transform=Identity,
        output_transform=Identity,
        **kwargs,
    ):

        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )
        self.model = None

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.n_output_dim = None
        self.n_features = None

    def create_model(
        self, input_dim=(2,), output_dim=1, activation="relu", learning_rate=0.01
    ):
        inputs = Input(shape=input_dim)
        hidden = Dense(64, activation=activation)(inputs)
        hidden = Dense(32, activation=activation)(hidden)
        hidden = Dense(16, activation=activation)(hidden)
        out = Dense(output_dim)(hidden)

        self.model = Model(inputs=inputs, outputs=out)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    def _fit(self, X, y, **kwargs):

        self.X = X
        self.y = y
        self.n_features = self.X.shape[1]
        self.n_output_dim = self.y.shape[1]
        self.create_model(input_dim=(self.n_features,), output_dim=self.n_output_dim)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def _update(self, X_new, y_new, **kwargs):
        optimizer = Adam(learning_rate=self.learning_rate_update)
        self.model.compile(optimizer=optimizer, loss=self.loss)

        self.X = np.vstack([self.X, X_new])
        self.y = np.vstack([self.y, y_new])

        self.model.fit(
            self.X,
            self.y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=False,
        )

    def _predict(self, X, **kwargs):
        if self.model:
            if len(X.shape) == 1:
                X = np.expand_dims(X, axis=0)
            self.preds = self.model.predict(X)

            return self.preds


class BayesianNeuralNetwork(Surrogate):
    learning_rate_initial = 0.01
    learning_rate_update = 0.001
    is_probabilistic = True
    is_multioutput = False
    is_torch = False

    def __init__(
        self,
        epochs=10,
        batch_size=32,
        input_transform=Identity,
        output_transform=Identity,
        **kwargs,
    ):
        super().__init__(
            input_transform=input_transform, output_transform=output_transform
        )

        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size

    def kernel_divergence_fn(self, q, p, _):
        return tfp.distributions.kl_divergence(q, p) / (self.X.shape[0] * 1.0)

    def bias_divergence_fn(self, q, p, _):
        return tfp.distributions.kl_divergence(q, p) / (self.X.shape[0] * 1.0)

    def create_model(self, input_dim=(2,), activation="relu", learning_rate=0.01):
        inputs = Input(shape=input_dim)

        hidden = tfp.layers.DenseFlipout(
            128,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel_divergence_fn,
            bias_divergence_fn=self.bias_divergence_fn,
            activation=activation,
        )(inputs)
        hidden = tfp.layers.DenseFlipout(
            64,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel_divergence_fn,
            bias_divergence_fn=self.bias_divergence_fn,
            activation=activation,
        )(hidden)
        hidden = tfp.layers.DenseFlipout(
            32,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel_divergence_fn,
            bias_divergence_fn=self.bias_divergence_fn,
            activation=activation,
        )(hidden)
        hidden = tfp.layers.DenseFlipout(
            16,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel_divergence_fn,
            bias_divergence_fn=self.bias_divergence_fn,
            activation=activation,
        )(hidden)
        params = tfp.layers.DenseFlipout(
            2,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel_divergence_fn,
            bias_divergence_fn=self.bias_divergence_fn,
        )(hidden)
        dist = tfp.layers.DistributionLambda(normal_sp)(params)

        self.model = Model(inputs=inputs, outputs=dist)
        self.model.compile(Adam(learning_rate=self.learning_rate_initial), loss=NLL)

    def _fit(self, X, y, **kwargs):
        self.X = X
        self.y = y
        n_features = self.X.shape[1]

        self.create_model(input_dim=(n_features,))
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=True
        )

    def _update(self, X_new, y_new, **kwargs):
        self.model.compile(Adam(learning_rate=self.learning_rate_update), loss=NLL)
        self.X = np.vstack([self.X, X_new])
        self.y = np.vstack([self.y, y_new])
        self.model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch_size)

    def _predict(self, X, return_std=False, iterations=50, **kwargs):
        if self.model:
            preds = np.zeros(shape=(X.shape[0], iterations))

            for i in range(iterations):
                y_ = self.model.predict(X)
                y__ = np.reshape(y_, (X.shape[0]))
                preds[:, i] = y__

            mean = np.mean(preds, axis=1).reshape(-1, 1)
            self.predictions = preds

            if return_std:
                stdv = np.std(preds, axis=1).reshape(-1, 1)
                return mean, stdv
            else:
                return mean

    def get_predictions(self):
        return self.predictions


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    """
    From: https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.html # noqa: E501
    """

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    """
    From: https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html # noqa: E501
    """

    def __init__(self, train_x, train_y, likelihood, N_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=N_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=N_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    """
    From: https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html  # noqa: E501
    """

    def __init__(self, train_x, train_y, likelihood, N_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([N_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([N_tasks])),
            batch_shape=torch.Size([N_tasks]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class LargeFeatureExtractor(torch.nn.Sequential):
    """
    From: https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/
    Simple_GP_Regression_With_LOVE_Fast_Variances_and_Sampling.html
    """

    def __init__(self, input_dim, n_features):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module("linear1", torch.nn.Linear(input_dim, 100))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(100, 50))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(50, 5))
        self.add_module("relu3", torch.nn.ReLU())
        self.add_module("linear4", torch.nn.Linear(5, n_features))


class DeepKernelLearningGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, N_tasks, n_features):
        super(DeepKernelLearningGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=N_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=4, ard_num_dims=n_features
            ),
            num_tasks=N_tasks,
            rank=1,
        )

        # Also add the deep net
        self.feature_extractor = LargeFeatureExtractor(
            input_dim=train_x.size(-1), n_features=n_features
        )

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
