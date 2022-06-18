import numpy as np
import torch

from harlow.utils.transforms import (
    ChainTransform,
    Identity,
    Normalize,
    Standardize,
    TensorTransform,
)

N_features = 5
N_outputs = 3
N_points = 10
N_batch = 2

test_X = np.random.rand(N_batch, N_points, N_features) * 1000
test_y = np.random.rand(N_batch, N_points, N_outputs) * 1000


def test_identity_transform():
    transform = Identity()
    transformed_X = transform.forward(test_X)
    untransformed_X = transform.reverse(transformed_X)
    assert np.allclose(transformed_X, test_X)
    assert np.allclose(untransformed_X, test_X)


def test_standardize_transform():
    transform = Standardize()
    transformed_X = transform.forward(test_X)
    untransformed_X = transform.reverse(transformed_X)
    assert np.allclose(transformed_X.mean(axis=-2), 0.0)
    assert np.allclose(transformed_X.std(axis=-2), 1.0)
    assert np.allclose(untransformed_X.mean(axis=-2), test_X.mean(axis=-2))
    assert np.allclose(untransformed_X.std(axis=-2), test_X.std(axis=-2))


def test_normalize_transform():
    transform = Normalize()
    transformed_X = transform.forward(test_X)
    untransformed_X = transform.reverse(transformed_X)
    assert np.allclose(transformed_X.min(axis=-2), 0.0)
    assert np.allclose(transformed_X.max(axis=-2), 1.0)
    assert np.allclose(untransformed_X.min(axis=-2), test_X.min(axis=-2))
    assert np.allclose(untransformed_X.max(axis=-2), test_X.max(axis=-2))


def test_tensor_transform():
    transform = TensorTransform()
    transformed_X = transform.forward(test_X)
    untransformed_X = transform.reverse(transformed_X)
    assert torch.is_tensor(transformed_X)
    assert isinstance(untransformed_X, np.ndarray)


def test_chain_transform():
    transform = ChainTransform(
        Identity(), Normalize(), Standardize(), TensorTransform()
    )
    transformed_X = transform.forward(test_X)
    untransformed_X = transform.reverse(transformed_X)
    assert torch.is_tensor(transformed_X)
    assert isinstance(untransformed_X, np.ndarray)
