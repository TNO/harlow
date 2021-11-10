"""Target functions for testing."""

import numpy as np


def ackley_nd(
    x_mx: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> np.ndarray:
    """
    n-dimensional Ackley function based on https://www.sfu.ca/~ssurjano/ackley.html.

    Args:
        x_mx:
            Independent variable values, [x1, x2, ..., xd], vectorized input is
            supported (provide them along the row dimension).
        a:
            Constant/parameter in the function.
        b:
            Constant/parameter in the function.
        c:
            Constant/parameter in the function.

    Returns:
        Dependent variable values.
    """
    x_mx = np.atleast_2d(x_mx)
    t1 = -a * np.exp(-b * np.sqrt(np.mean(x_mx ** 2, axis=1)))
    t2 = -np.exp(np.mean(np.cos(c * x_mx), axis=1))

    return t1 + t2 + a + np.exp(1)


def six_hump_camel_2D(X):
    x = X[:, 0]
    y = X[:, 1]

    x2 = np.power(x, 2)
    x4 = np.power(x, 4)
    y2 = np.power(y, 2)

    return ((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2) + (x * y) + ((-4.0 + 4.0 * y2) * y2)


def six_hump_camel_2D_2input(x, y):
    x2 = np.power(x, 2)
    x4 = np.power(x, 4)
    y2 = np.power(y, 2)

    return ((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2) + (x * y) + ((-4.0 + 4.0 * y2) * y2)


def forresterEtAl(X):
    term_a = 6 * X - 2
    term_b = 12 * X - 4
    return np.power(term_a, 2) * np.sin(term_b)


def bohachevsky_2D(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    t1 = np.power(x1, 2)
    t2 = 2 * np.power(x2, 2)
    t3 = -0.3 * np.cos(3 * np.pi * x1)
    t4 = -0.4 * np.cos(4 * np.pi * x2)

    return t1 + t2 + t3 + t4 + 0.7
