import numpy as np


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
