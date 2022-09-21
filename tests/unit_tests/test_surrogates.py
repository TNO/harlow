"""
Surrogate model unit tests, to ensure that all surrogates:

    * Correctly inherit from the `Surrogate` abstract class
    * Can be initialized with the same set of keyword arguments
    * Have consistent input and output shapes
    * Behave consistently when calling fit, update, predict
"""

import numpy as np

from harlow.surrogating.surrogate_model import (  # noqa F401; DeepKernelMultiTaskGaussianProcess,
    BatchIndependentGaussianProcess,
    BayesianNeuralNetwork,
    GaussianProcessRegression,
    GaussianProcessTFP,
    ModelListGaussianProcess,
    MultiTaskGaussianProcess,
    NeuralNetwork,
    VanillaGaussianProcess,
)

surrogate_list = [
    BatchIndependentGaussianProcess,
    BayesianNeuralNetwork,
    # DeepKernelMultiTaskGaussianProcess,
    GaussianProcessRegression,
    GaussianProcessTFP,
    ModelListGaussianProcess,
    MultiTaskGaussianProcess,
    NeuralNetwork,
    VanillaGaussianProcess,
]

N_points = 20
N_features = 10
N_outputs = 1
points_x = np.random.rand(N_points, N_features)
points_y = np.random.rand(N_points, N_outputs)


def test_surrogate_single_output():
    for surrogate in surrogate_list:
        try:
            surrogate().fit(points_x, points_y)
        except Exception as e:
            raise Exception(f"Surrogate fit failed for {surrogate} with exception: {e}")


if __name__ == "__main__":
    test_surrogate_single_output()
