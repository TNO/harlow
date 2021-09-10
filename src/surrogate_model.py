import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

#TODO make architectures configurable through API
#TODO chech if class structure is appropriate for all sampling techniques


def NLL(y, distr):
  return -distr.log_prob(y)

def normal_sp(params):
  return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable


class GaussianProcess():

    def __init__(self, normalize_Y = False, **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None


    def create_model(self):
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()



    def fit(self, X,y):
        gprm = tfd.GaussianProcessRegressionModel(self.kernel, index_points = None, observation_index_points=X, observations=y)


    #TODO finish model class

class Prob_NN():

    def __init__(self, normalize_Y = False, **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None

    def create_model(self):

        inputs = Input(shape=(2,))
        hidden = Dense(20, activation="relu")(inputs)
        hidden = Dense(50, activation="relu")(hidden)
        hidden = Dense(20, activation="relu")(hidden)
        params = Dense(2)(hidden)
        dist = tfp.layers.DistributionLambda(normal_sp)(params)

        self.model = Model(inputs=inputs, outputs=dist)
        self.model.compile(Adam(), loss=NLL)


    def fit(self, X,y, epochs = 10):
        self.model.fit(X, y, epochs=epochs, batch_size=32)


    def updateModel(self, X_all, Y_all, X_new, Y_new):

        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean()) / (Y_all.std())

        if self.model is None:
            self._create_model()
        else:
            self.model.fit(X_all, Y_all, epochs=10, batch_size=32)


    def predict(self, X, its = 10):
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


class Bayesian_NN():
    learning_rate_initial = 0.01
    learning_rate_update = 0.001

    def __init__(self, normalize_Y = False, **kwargs):
        self.normalize_Y = normalize_Y
        self.model = None

    def create_model(self, x):

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (x.shape[0] * 1.0)
        bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (x.shape[0] * 1.0)

        inputs = Input(shape=(2,))

        hidden = tfp.layers.DenseFlipout(128, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn, activation="relu")(inputs)
        hidden = tfp.layers.DenseFlipout(64, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn, activation="relu")(hidden)
        hidden = tfp.layers.DenseFlipout(32, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn, activation="relu")(hidden)
        hidden = tfp.layers.DenseFlipout(16, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn, activation="relu")(hidden)
        params = tfp.layers.DenseFlipout(2, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn)(hidden)
        dist = tfp.layers.DistributionLambda(normal_sp)(params)

        self.model = Model(inputs=inputs, outputs=dist)
        self.model.compile(Adam(learning_rate=self.learning_rate_initial), loss=NLL)


    def fit(self, X,y, epochs = 25, verbose = 0):
        self.X = X
        self.y = y
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=verbose)


    def updateModel(self, X_new, y_new, epochs=25, verbose=0):

        #self.X = np.concatenate((self.X, X_new))
        #self.y = np.concatenate((self.y, y_new))

        if self.normalize_Y:
            Y_all = (self.y - self.y.mean()) / (self.y.std())

        if self.model is None:
            self._create_model()
        else:
            self.model.compile(Adam(learning_rate=self.learning_rate_update), loss=NLL)
            self.model.fit(X_new, y_new, epochs=epochs, batch_size=32, verbose = verbose)


    def predict(self, X, its = 10):
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
