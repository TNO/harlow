import numpy as np

from tests.test_functions import ackley_nd


def test_ackley_nd():
    # test up to this dimension (a few is enough to check the generalization)
    n_max_dim = 6
    a = 20
    b = 0.2
    c = 2 * np.pi

    # the global minimum is known for all versions (dimensions)
    for ii in range(n_max_dim):
        x_mx = np.zeros(ii + 1)
        y_expected = 0.0
        y = ackley_nd(x_mx=x_mx, a=a, b=b, c=c)

        np.testing.assert_almost_equal(y_expected, y)

    # test a few points against this Matlab implementation:
    # https://www.sfu.ca/~ssurjano/Code/ackleym.html
    x_mx = np.array([[1, 2], [3, -4]])
    y_expected = np.array([5.422131717799510, 10.138626172095204])
    y = ackley_nd(x_mx=x_mx, a=a, b=b, c=c)
    np.testing.assert_almost_equal(y_expected, y)

    # points that has the same value irrespective of the dimension, against the same
    # Matlab implementation
    y_expected = np.array([3.625384938440363, 6.593599079287213, 11.013420717655567])
    for ii in range(n_max_dim):
        x_mx = np.tile(np.array([[1], [-2], [4]]), reps=(1, ii + 1))

        y = ackley_nd(x_mx=x_mx, a=a, b=b, c=c)

        np.testing.assert_almost_equal(y_expected, y)
