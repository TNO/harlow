import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow as tf

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

# TODO make architectures configurable through API
# TODO chech if class structure is appropriate for all sampling techniques
# TODO add retrainingn strategies


def NLL(y, distr):
    return -distr.log_prob(y)


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred-y_true), axis=1))


def normal_sp(params):
    return tfd.Normal(
        loc=params[:, 0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:, 1:2])
    )  # both parameters are learnable

## TODO add superclass (abstract) check enforcing methods at subclassing

class GaussianProcess:
    def __init__(self, normalize_Y=False, train_iterations = 1000,  **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None
        self.train_iterations = train_iterations

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
            print(f"observation_noise_variance: {self.observation_noise_variance_var._value().numpy()}")

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


    def predict(self, X, num_samples=50, return_samples = False):
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=X,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
            observation_noise_variance=self.observation_noise_variance_var,
            predictive_noise_variance=0.0,
        )

        samples = gprm.sample(num_samples)

        if return_samples:
            return np.mean(samples, axis=0), samples
        return np.mean(samples, axis=0)


    def predict_uncertainty(self, X, num_samples=50):
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=X,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
            observation_noise_variance=self.observation_noise_variance_var,
            predictive_noise_variance=0.0,
        )

        samples = gprm.sample(num_samples)

        return np.std(samples, axis=0)


    def update(self, new_X, new_y):
        self.observation_index_points = np.concatenate([self.observation_index_points, new_X])

        if new_y.ndim > self.observations.ndim:
            new_y = new_y.flatten()
        self.observations = np.concatenate([self.observations, new_y])
        self.optimize_parameters(verbose=True)


class NN:
    learning_rate_initial = 0.01
    learning_rate_update = 0.001

    def __init__(self, normalize_Y=False, **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None

    def create_model(self, input_dim = (2,)):
        inputs = Input(shape=input_dim)
        hidden = Dense(64, activation="relu")(inputs)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dense(16, activation="relu")(hidden)
        out = Dense(1)(hidden)
        optimizer = Adam(learning_rate=self.learning_rate_initial)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.compile(optimizer=optimizer, loss='mse')


    def fit(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs, batch_size=32)


    def update(self, X_new, y_new):

        if self.normalize_Y:
            y_new = (y_new - y_new.mean()) / (y_new.std())

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


class Prob_NN:
    learning_rate_initial = 0.01
    learning_rate_update = 0.001

    def __init__(self, normalize_Y=False, **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None

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

        if self.normalize_Y:
            y_new = (y_new - y_new.mean()) / (y_new.std())

        optimizer = Adam(learning_rate=self.learning_rate_update)
        self.model.compile(optimizer=optimizer, loss=NLL)
        self.model.fit(X_new, y_new, epochs=10, batch_size=32)


    def predict(self, X, its=10):
        if self.model:
            preds = np.zeros(shape=(X.shape[0], its))

            for i in range(its):
                y_ = self.model.predict(X)
                y__ = np.reshape(y_, (X.shape[0]))
                preds[:, i] = y__

            mean = np.mean(preds, axis=1)
            stdv = np.std(preds, axis=1)

            return mean, stdv

    def get_model_parameters(self):
        if self.model is not None:
            return self.model.get_weights()

    def get_fmin(self):
        return self.model.predict(self.X).min()


class Bayesian_NN:
    learning_rate_initial = 0.01
    learning_rate_update = 0.001

    def __init__(self, normalize_Y=False, **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None

    def create_model(self, x):

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (
            x.shape[0] * 1.0
        )
        bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (
            x.shape[0] * 1.0
        )

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
        if self.normalize_Y:
            y_new = (y_new - y_new.mean()) / (y_new.std())

        if self.model is None:
            self._create_model()
        else:
            self.model.compile(Adam(learning_rate=self.learning_rate_update), loss=NLL)
            self.model.fit(X_new, y_new, epochs=epochs, batch_size=32, verbose=verbose)


    def predict(self, X, its=10):
        if self.model:
            preds = np.zeros(shape=(X.shape[0], its))

            for i in range(its):
                y_ = self.model.predict(X)
                y__ = np.reshape(y_, (X.shape[0]))
                preds[:, i] = y__

            mean = np.mean(preds, axis=1)
            stdv = np.std(preds, axis=1)

            return mean, stdv

    def get_model_parameters(self):
        if self.model is not None:
            return self.model.get_weights()

    def get_fmin(self):
        return self.model.predict(self.X).min()
