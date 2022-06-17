"""
Forward and reverse transforms typically used in surrogating and optimization.

Notes:
    * It may be usefull to define a convention for in/output shapes to ensure that
    the different surrogating, sampling, transformation and vizualization tools adhere
    to a common structure for the input and output data.
    * For now we assume that features `X` have size (n_batch x n_points x n_features)
    and outputs `y` have size (n_batch x n_points x n_outputs).

TODO:
    * More thorough testing
    * Add docstrings
    * Add type hints
"""
from abc import ABC, abstractmethod

import numpy as np
import torch


class Transform(ABC):
    """
    Abstract base class for the input and output transforms. Implements the forward
    and reverse transform methods.
    """

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def reverse(self, X):
        raise NotImplementedError


class ChainTransform(Transform):
    """
    Class used to chain together a series of transforms
    """

    def __init__(self):
        pass

    def forward(self, X):
        pass

    def reverse(self, X):
        pass


class Identity(Transform):
    """
    Defines the identity transform that can be used as a placeholder
    if no transforms are specified.
    """

    def __init__(self):
        pass

    def forward(self, X):
        return X

    def reverse(self, X):
        return X


class TensorTransform(Transform):
    """
    Transforms an input `numpy.ndarray` to `torch.tensor` and vice-versa
    """

    def __init__(self, target_type=torch.float32):
        self.target_type = target_type

    def forward(self, X):
        return torch.from_numpy(X).type(self.target_type)

    def reverse(self, X):
        return X.detach().cpu().numpy()


class Standardize(Transform):
    """
    Scales features to zero mean and unit standard deviation.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def forward(self, X):
        self.mean = np.expand_dims(X.mean(axis=-2), axis=-2)
        self.std = np.expand_dims(X.std(axis=-2), axis=-2)

        return (X - self.mean) / self.std

    def reverse(self, X, mean=None, std=None):

        if mean is not None:
            self.mean = mean

        if std is not None:
            self.std = std

        if (self.mean is None) or (self.std is None):
            raise RuntimeError(
                "The mean or standard deviation are `None`. These"
                "must either be initialized by calling the forward"
                "transform or provided as keyword arguments."
            )

        return self.mean + self.std * X


class Normalize(Transform):
    """
    Applies min-max scaling to given features.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def forward(self, X):
        self.min = np.expand_dims(X.min(axis=-2), axis=-2)
        self.max = np.expand_dims(X.max(axis=-2), axis=-2)

        return (X - self.min) / (self.max - self.min)

    def reverse(self, X, xmin=None, xmax=None):

        if xmin is not None:
            self.min = xmin

        if xmax is not None:
            self.max = xmax

        if (self.min is None) or (self.max is None):
            raise RuntimeError(
                "The min or max are `None`. These"
                "must either be initialized by calling the forward"
                "transform or provided as keyword arguments."
            )

        return X * (self.max - self.min) + self.min
