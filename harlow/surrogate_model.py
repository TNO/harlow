"""Surrogate model (function) module for fitting (not adaptive) and prediction.

`f_surrogate(x) ~= f_target(x)` for `R^n -> R^1` functions.

The main requirements towards each surrogate model are that they:
* can be fitted to points from the target function.
* can make predictions at user selected points.

"""
import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

# TODO make architectures configurable through API
# TODO chech if class structure is appropriate for all sampling techniques
# TODO add retrainingn strategies


def NLL(y, distr):
    return -distr.log_prob(y)


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=1))


def normal_sp(params):
    return tfd.Normal(
        loc=params[:, 0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:, 1:2])
    )  # both parameters are learnable


# TODO add superclass (abstract) check enforcing methods at subclassing
class Surrogate(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def update(self, new_X, new_y):
        pass


class GaussianProcess(Surrogate):
    is_probabilistic = True
    kernel = 1.0 * RBF(1.0) + WhiteKernel(1.0, noise_level_bounds=(5e-5, 5e-2))

    def __init__(self, train_restarts=10, kernel=kernel, **kwargs):
        self.model = None
        self.train_restarts = train_restarts
        self.noise_std = None
        self.kernel = kernel

        self.create_model()

    def create_model(self):
        self.model = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=self.train_restarts
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.noise_std = self.get_noise()

    def get_noise(self):
        attrs = list(vars(GaussianProcess.kernel).keys())

        white_kernel_attr = []
        for attr in attrs:
            if re.match(pattern="^k[0-9]", string=attr):
                attr_val = getattr(GaussianProcess.kernel, attr)
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

    def predict(self, X):

        samples, std = self.model.predict(X, return_std=True)

        return samples, std

    def update(self, new_X, new_y):
        X = np.concatenate([self.observation_index_points, new_X])

        if new_y.ndim > self.observations.ndim:
            new_y = new_y.flatten()
        y = np.concatenate([self.observations, new_y])

        self.fit(X, y)


class GaussianProcessTFP(Surrogate):
    is_probabilistic = True

    def __init__(self, train_iterations=50, **kwargs):
        self.model = None
        self.train_iterations = train_iterations

        self.create_model()

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
                "observations": self.observations,
            }
        )

    def fit(self, X, y):
        self.observation_index_points = X
        self.observations = y
        self.optimize_parameters()

    def predict(self, X, iterations=10, return_samples=False):
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
            return np.mean(samples, axis=0), np.std(samples, axis=0), samples
        return np.mean(samples, axis=0), np.std(samples, axis=0)

    def update(self, new_X, new_y):
        self.observation_index_points = np.concatenate(
            [self.observation_index_points, new_X]
        )

        if new_y.ndim > self.observations.ndim:
            new_y = new_y.flatten()
        self.observations = np.concatenate([self.observations, new_y])
        self.optimize_parameters(verbose=True)


class NN(Surrogate):
    learning_rate_initial = 0.01
    learning_rate_update = 0.001
    is_probabilistic = False

    def __init__(self, **kwargs):
        self.model = None

        self.create_model()

    def create_model(self, input_dim=(2,)):
        inputs = Input(shape=input_dim)
        hidden = Dense(64, activation="relu")(inputs)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dense(16, activation="relu")(hidden)
        out = Dense(1)(hidden)
        optimizer = Adam(learning_rate=self.learning_rate_initial)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.compile(optimizer=optimizer, loss="mse")

    def fit(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs, batch_size=32)

    def update(self, X_new, y_new):
        optimizer = Adam(learning_rate=self.learning_rate_update)
        self.model.compile(optimizer=optimizer, loss=root_mean_squared_error)
        self.model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=False)

    def predict(self, X):
        if self.model:
            if len(X.shape) == 1:
                X = np.expand_dims(X, axis=0)
            preds = self.model.predict(X)

            return preds

    def get_model_parameters(self):
        if self.model is not None:
            return self.model.get_weights()

    def get_fmin(self):
        return self.model.predict(self.X).min()


class Prob_NN(Surrogate):
    learning_rate_initial = 0.01
    learning_rate_update = 0.001
    is_probabilistic = True

    def __init__(self, **kwargs):
        self.model = None

        self.create_model()

    def create_model(self):
        inputs = Input(shape=(2,))
        hidden = Dense(20, activation="relu")(inputs)
        hidden = Dense(50, activation="relu")(hidden)
        hidden = Dense(20, activation="relu")(hidden)
        params = Dense(2)(hidden)
        dist = tfp.layers.DistributionLambda(normal_sp)(params)
        optimizer = Adam(learning_rate=self.learning_rate_initial)
        self.model = Model(inputs=inputs, outputs=dist)
        self.model.compile(optimizer=optimizer, loss=NLL)

    def fit(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs, batch_size=32)

    def update(self, X_new, y_new):
        optimizer = Adam(learning_rate=self.learning_rate_update)
        self.model.compile(optimizer=optimizer, loss=NLL)
        self.model.fit(X_new, y_new, epochs=10, batch_size=32)

    def predict(self, X, iterations=50, return_samples=False):
        if self.model:
            preds = np.zeros(shape=(X.shape[0], iterations))

            for i in range(iterations):
                y_ = self.model.predict(X)
                y__ = np.reshape(y_, (X.shape[0]))
                preds[:, i] = y__

            mean = np.mean(preds, axis=1)
            stdv = np.std(preds, axis=1)

            if return_samples:
                mean, stdv, preds

            return mean, stdv

    def get_model_parameters(self):
        if self.model is not None:
            return self.model.get_weights()

    def get_fmin(self):
        return self.model.predict(self.X).min()


class Bayesian_NN(Surrogate):
    learning_rate_initial = 0.01
    learning_rate_update = 0.001
    is_probabilistic = True

    def __init__(self, X, **kwargs):
        self.model = None
        self.is_probabilistic = True

        self.create_model(X)

    def create_model(self, x):
        def kernel_divergence_fn(q, p, _):
            return tfp.distributions.kl_divergence(q, p) / (x.shape[0] * 1.0)

        def bias_divergence_fn(q, p, _):
            return tfp.distributions.kl_divergence(q, p) / (x.shape[0] * 1.0)

        inputs = Input(shape=(2,))

        hidden = tfp.layers.DenseFlipout(
            128,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
        )(inputs)
        hidden = tfp.layers.DenseFlipout(
            64,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
        )(hidden)
        hidden = tfp.layers.DenseFlipout(
            32,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
        )(hidden)
        hidden = tfp.layers.DenseFlipout(
            16,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
        )(hidden)
        params = tfp.layers.DenseFlipout(
            2,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
        )(hidden)
        dist = tfp.layers.DistributionLambda(normal_sp)(params)

        self.model = Model(inputs=inputs, outputs=dist)
        self.model.compile(Adam(learning_rate=self.learning_rate_initial), loss=NLL)

    def fit(self, X, y, epochs=25, verbose=0):
        self.X = X
        self.y = y
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=verbose)

    def update(self, X_new, y_new, epochs=25, verbose=0):
        if self.model is None:
            self._create_model()
        else:
            self.model.compile(Adam(learning_rate=self.learning_rate_update), loss=NLL)
            self.model.fit(X_new, y_new, epochs=epochs, batch_size=32, verbose=verbose)

    def predict(self, X, iterations=50, return_samples=False):
        if self.model:
            preds = np.zeros(shape=(X.shape[0], iterations))

            for i in range(iterations):
                y_ = self.model.predict(X)
                y__ = np.reshape(y_, (X.shape[0]))
                preds[:, i] = y__

            mean = np.mean(preds, axis=1)
            stdv = np.std(preds, axis=1)

            if return_samples:
                mean, stdv, preds

            return mean, stdv

    def get_model_parameters(self):
        if self.model is not None:
            return self.model.get_weights()

    def get_fmin(self):
        return self.model.predict(self.X).min()
