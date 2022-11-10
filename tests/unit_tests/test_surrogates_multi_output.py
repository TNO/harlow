"""
Surrogate model unit tests, to ensure that all surrogates:

    * Correctly inherit from the `Surrogate` abstract class
    * Can be initialized with the same set of keyword arguments
    * Have consistent input and output shapes
    * Behave consistently when calling fit, update, predict
"""

import numpy as np

from harlow.surrogating.surrogate_model import (  # noqa F401; DeepKernelMultiTaskGaussianProcess,; DeepKernelMultiTaskGaussianProcess,
    BatchIndependentGaussianProcess,
    BayesianNeuralNetwork,
    GaussianProcessTFP,
    ModelListGaussianProcess,
    MultiTaskGaussianProcess,
    NeuralNetwork,
    VanillaGaussianProcess,
)

surrogate_list = [
    BatchIndependentGaussianProcess(),
    BayesianNeuralNetwork(),
    # DeepKernelMultiTaskGaussianProcess(),
    GaussianProcessTFP(),
    ModelListGaussianProcess(),
    MultiTaskGaussianProcess(),
    NeuralNetwork(),
    VanillaGaussianProcess(),
    BayesianNeuralNetwork(),
]

N_train = 20
N_update = 10
N_pred = 50
N_features = 10
N_outputs = 15


train_X = np.random.rand(N_train, N_features)
train_y = np.random.rand(N_train, N_outputs)
update_X = np.random.rand(N_update, N_features)
update_y = np.random.rand(N_update, N_outputs)
pred_X = np.random.rand(N_pred, N_features)


def test_surrogate_multi_output():

    for surrogate in surrogate_list:

        if surrogate.is_multioutput is True:

            # Training step
            try:
                surrogate.fit(train_X, train_y)
            except Exception as e:
                raise Exception(
                    f"Surrogate fit failed for {surrogate} with exception: {e}"
                )

            # TODO: Check number of samples internally for each surrogate model to make
            #   sure that `update_X` and `update_y` are added to the existing samples.

            # Training step
            try:
                surrogate.update(update_X, update_y)
            except Exception as e:
                raise Exception(
                    f"Surrogate update failed for {surrogate} with exception: {e}"
                )

            # Prediction step
            try:
                pred_y = surrogate.predict(pred_X)
            except Exception as e:
                raise Exception(
                    f"Surrogate prediction failed for {surrogate} with exception: {e}"
                )

            print(f"Assertions for surrogate {surrogate}")
            assert isinstance(pred_y, np.ndarray)
            assert pred_y.ndim == 2
            assert pred_y.shape[0] == N_pred
            assert pred_y.shape[1] == N_outputs


if __name__ == "__main__":
    test_surrogate_multi_output()
    print("Finished")
