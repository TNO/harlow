"""Target functions for testing."""

import numpy as np


def lldeh_1d(x: np.ndarray, a: float) -> np.ndarray:
    """
    An interesting 1D non-linear function.

    Source: https://www.reddit.com/r/math/comments/5mwr9g/what_are_your_favourite_or_most_interesting/  # noqa E301
    Args:
        x:
        a: parameter to control the shape, interesting variants: a=0, a=2.1

    Returns:

    """
    g = np.tan(0.5 * a)
    return 2 * x**2 / (x**2 + g * x + 1)


def peaks_2d(x: np.ndarray) -> np.ndarray:
    # https://nl.mathworks.com/help/matlab/ref/peaks.html
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    return (
        3 * (1 - x1) ** 2 * np.exp(-(x1**2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1**3 - x2**5) * np.exp(-(x1**2) - x2**2)
        - 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2**2)
    )


def peaks_2d_multivariate(x: np.ndarray) -> np.ndarray:
    # https://nl.mathworks.com/help/matlab/ref/peaks.html
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    result = (
        3 * (1 - x1) ** 2 * np.exp(-(x1**2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1**3 - x2**5) * np.exp(-(x1**2) - x2**2)
        - 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2**2)
    )
    return np.stack((result, result**2), -1)


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
    t1 = -a * np.exp(-b * np.sqrt(np.mean(x_mx**2, axis=1)))
    t2 = -np.exp(np.mean(np.cos(c * x_mx), axis=1))

    return t1 + t2 + a + np.exp(1)


def six_hump_camel_2D(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]

    x2 = np.power(x, 2)
    x4 = np.power(x, 4)
    y2 = np.power(y, 2)

    return ((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2) + (x * y) + ((-4.0 + 4.0 * y2) * y2)


def six_hump_camel_2D_2input(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.power(x, 2)
    x4 = np.power(x, 4)
    y2 = np.power(y, 2)

    return ((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2) + (x * y) + ((-4.0 + 4.0 * y2) * y2)


def forrester_1d(x: np.ndarray) -> np.ndarray:
    term_a = 6 * x - 2
    term_b = 12 * x - 4
    return np.power(term_a, 2) * np.sin(term_b)


def bohachevsky_2D(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    x2 = X[:, 1]

    t1 = np.power(x1, 2)
    t2 = 2 * np.power(x2, 2)
    t3 = -0.3 * np.cos(3 * np.pi * x1)
    t4 = -0.4 * np.cos(4 * np.pi * x2)

    return t1 + t2 + t3 + t4 + 0.7


def shekel(X: np.ndarray) -> np.ndarray:
    xx = [X[:, 0], X[:, 1], X[:, 2], X[:, 3]]
    m = 10
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    C = np.array(
        [
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        ]
    )

    outer = 0
    for i in range(0, m):
        b_i = b[i]
        inner = 0
        for j in range(0, 4):
            x_j = xx[j]
            C_ji = C[j, i]
            inner = inner + np.power((x_j - C_ji), 2)
        outer = outer + 1 / (inner + b_i)

    return -1.0 * outer


def hartmann(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    results = []
    for i in range(n):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = X[i, jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2

            new = alpha[ii] * np.exp(-inner)
            outer = outer + new
        results.append(-(2.58 + outer) / 1.94)

    return np.asarray(results)


def stybtang(X: np.ndarray) -> np.ndarray:
    # https://www.sfu.ca/~ssurjano/stybtang.html
    n = X.shape[0]
    d = X.shape[1]
    res = np.zeros(n)
    for i in range(n):
        _sum = 0.0
        for j in range(d):
            _sum = np.power(X[i, j], 4) - 16.0 * np.power(X[i, j], 2) + 5.0 * X[i, j]
        res[i] = 0.5 * _sum

    return res
