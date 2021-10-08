from test_functions import bohachevsky_2D, forresterEtAl
from sklearn.gaussian_process import GaussianProcessRegressor
from plotting import plot_function_custom, add_samples_to_plot
from lolaVoronoi import LolaVoronoi
from sklearn.metrics import r2_score
import numpy as np
from surrogate_model import GaussianProcess


def test_2D():
    domain = [[-100.0, 100.0], [-100.0, 100.0]]
    n_points = 20
    n_iters = 10
    n_per_iters = 10

    X1 = np.random.uniform(domain[0][0], domain[0][1], n_points)
    X2 = np.random.uniform(domain[1][0], domain[1][1], n_points)
    X = np.stack([X1, X2], -1)

    indices = np.random.permutation(X1.shape[0])
    train_idx, test_idx = (
        indices[: round(len(indices) * 0.8)],
        indices[round(len(indices) * 0.8) :],
    )
    train_X = X[train_idx, :]
    test_X = X[test_idx, :]
    train_y = bohachevsky_2D(train_X)
    test_y = bohachevsky_2D(test_X)

    gp = GaussianProcessRegressor()
    gp.fit(train_X, train_y)

    p = gp.predict(test_X)
    print(f"test R2: {r2_score(test_y, p)}")

    plot = plot_function_custom(
        bohachevsky_2D,
        train_X,
        y=gp.predict(train_X),
        plot_sample_locations=True,
        show=False,
    )

    lv = LolaVoronoi(
        gp,
        train_X,
        train_y,
        test_X,
        test_y,
        [[domain[0], domain[1]]],
        bohachevsky_2D,
        n_iteration=n_iters,
        n_per_iteration=n_per_iters,
    )
    lv.run_sequential_design()

    plot = add_samples_to_plot(
        plot,
        lv.train_X[-n_iters * n_per_iters :],
        bohachevsky_2D(lv.train_X[-n_iters * n_per_iters :]),
        "g",
    )
    plot.show()

    plot2 = plot_function_custom(
        bohachevsky_2D,
        lv.train_X,
        lv.model.predict(lv.train_X),
        plot_sample_locations=True,
        show=True,
    )
    plot2.show()


def test_1D():
    domain = [0.0, 1.0]
    n_points = 10
    n_iters = 5
    n_per_iters = 3

    X_range = np.linspace(0, 1, 1000)
    y_range = forresterEtAl(X_range)
    X = np.random.uniform(domain[0], domain[1], n_points)

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = (
        indices[: round(len(indices) * 0.8)],
        indices[round(len(indices) * 0.8) :],
    )
    train_X = np.sort(X[train_idx])
    test_X = np.sort(X[test_idx])
    train_y = forresterEtAl(train_X)
    test_y = forresterEtAl(test_X)

    gp = GaussianProcessRegressor()
    gp.fit(train_X.reshape(-1, 1), train_y)

    p = gp.predict(test_X.reshape(-1, 1))
    print(f"test R2: {r2_score(test_y, p)}")

    plot = plot_function_custom(
        forresterEtAl,
        train_X,
        y=gp.predict(train_X.reshape(-1, 1)),
        plot_sample_locations=True,
        show=False,
    )
    plot.plot(X_range, y_range, "r")

    lv = LolaVoronoi(
        gp,
        train_X.reshape(-1, 1),
        train_y.reshape(-1, 1),
        test_X.reshape(-1, 1),
        test_y.reshape(-1, 1),
        [[domain[0], domain[1]]],
        forresterEtAl,
        n_iteration=n_iters,
        n_per_iteration=n_per_iters,
    )
    lv.run_sequential_design()

    plot = add_samples_to_plot(
        plot,
        lv.train_X[-n_iters * n_per_iters :],
        forresterEtAl(lv.train_X[-n_iters * n_per_iters :]),
        "g",
    )
    plot.show()


def test_tfdGP():
    import matplotlib.pyplot as plt
    import tensorflow_probability as tfp

    num_training_points = 100
    index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
    index_points_ = index_points_.astype(np.float64)
    observations_ = ((np.sin(3 * np.pi * index_points_[..., 0])) +
                     np.random.normal(loc=0,
                                      scale=np.sqrt(0.1),
                                      size=(num_training_points)))

    gp = GaussianProcess()
    gp.fit(index_points_, observations_)

    predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
    predictive_index_points_ = predictive_index_points_[..., np.newaxis]
    mean, dev, samples = gp.predict(predictive_index_points_)

    plt.figure(figsize=(12, 4))
    plt.plot(predictive_index_points_,  (np.sin(3 * np.pi * predictive_index_points_[..., 0])), label='True fn')
    plt.scatter(index_points_[:, 0], observations_,
                label='Observations')
    for i in range(50):
        plt.plot(predictive_index_points_, samples[i, :], c='r', alpha=.1,
                 label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()


def test_fun(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.sin(x1 - 3) * np.cos(x2 / 4)


if __name__ == "__main__":
    test_tfdGP()
