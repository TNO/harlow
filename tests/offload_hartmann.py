import random

import numpy as np


def failing_hartman(X: np.ndarray) -> (np.ndarray, np.array):
    y = hartmann(X)
    s = np.array([random.random() >= 0.2 for x in range(y.shape[0])])
    # if random.random() < 0.8:
    #     print('Deliberetly failing')
    #     raise TargetFunctionEvaluationFailedException('I am going down!')
    return y, s


def succeeding_hartman(X: np.ndarray) -> (np.ndarray, np.array):
    y = hartmann(X)
    s = np.array([True for x in range(y.shape[0])])
    # if random.random() < 0.8:
    #     print('Deliberetly failing')
    #     raise TargetFunctionEvaluationFailedException('I am going down!')
    return y, s


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
    y = np.stack((result, result**2), -1)
    s = np.array([True for x in range(y.shape[0])])

    return y, s


# Maybe just upload the original hartman
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

    return np.asarray(results).reshape(-1, 1)


def main():
    x = np.load("x.npy")
    print(x)
    y = hartmann(x)
    np.save("y.npy", y)


if __name__ == "__main__":
    main()
