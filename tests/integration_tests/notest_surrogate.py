"""NOTE: this file is not yet harmonized with the general changes in the packages so
it is excluded from the testing (see the file name)."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from harlow.lola_voronoi import LolaVoronoi
from harlow.probabilistic_sampling import Probabilistic_sampler
from harlow.surrogate_model import GaussianProcessTFP, VanillaGaussianProcess
from harlow.visualization.plotting import add_samples_to_plot, plot_function_custom
from tests.integration_tests.test_functions import bohachevsky_2D, forrester_1d, shekel


def test_2D():
    domain = np.array([[-100.0, 100.0], [-100.0, 100.0]])
    n_points = 40
    n_iters = 10
    n_per_iters = 4
    show_plot = False

    X1 = np.random.uniform(domain[0, 0], domain[0, 1], n_points)
    X2 = np.random.uniform(domain[1, 0], domain[1, 1], n_points)
    X = np.stack([X1, X2], -1)

    indices = np.random.permutation(X1.shape[0])
    train_idx, test_idx = (
        indices[: round(len(indices) * 0.5)],
        indices[round(len(indices) * 0.5) :],
    )
    train_X = X[train_idx, :]
    test_X = X[test_idx, :]
    train_y = bohachevsky_2D(train_X)
    test_y = bohachevsky_2D(test_X)

    gp = GaussianProcessTFP()
    gp.fit(train_X, train_y)
    gp_copy = GaussianProcessTFP()
    gp_copy.fit(train_X, train_y)

    p = gp.predict(test_X)
    print(f"test R2: {r2_score(test_y, p)}")
    print(f"test2 R2: {r2_score(test_y, gp_copy.predict(test_X))}")

    plot = plot_function_custom(
        bohachevsky_2D,
        train_X,
        y_vec=gp.predict(train_X),
        plot_sample_locations=True,
        show=show_plot,
    )

    lv = LolaVoronoi(
        gp,
        train_X,
        train_y,
        test_X,
        test_y,
        domain,
        bohachevsky_2D,
        n_init=1,
        n_iteration=n_iters,
        n_new_point_per_iteration=n_per_iters,
    )
    lv.run_sequential_design()

    plt.plot(np.arange(0, n_iters + 1), lv.score)

    add_samples_to_plot(
        plot,
        lv.train_X[-n_iters * n_per_iters :],
        bohachevsky_2D(lv.train_X[-n_iters * n_per_iters :]),
        "g",
    )

    plot_function_custom(
        bohachevsky_2D,
        lv.train_X,
        lv.surrogate_model.predict(lv.train_X),
        plot_sample_locations=True,
        show=show_plot,
    )

    random_scores = [r2_score(test_y, p)]
    for _ in range(n_iters):
        X1_random_test = np.random.uniform(domain[0, 0], domain[0, 1], n_per_iters)
        X2_random_test = np.random.uniform(domain[1, 0], domain[1, 1], n_per_iters)
        X_random_test = np.stack([X1_random_test, X2_random_test], -1)
        train_X = np.concatenate([train_X, X_random_test])
        train_y = np.concatenate([train_y, bohachevsky_2D(X_random_test)])

        gp_copy.update(X_random_test, bohachevsky_2D(X_random_test))
        rand_y = gp_copy.predict(test_X)
        random_scores.append(r2_score(test_y, rand_y))

    plt.plot(np.arange(0, n_iters + 1), random_scores)

    plot_function_custom(
        bohachevsky_2D, train_X, train_y, plot_sample_locations=True, show=show_plot
    )


def test_1D():
    domain = np.array([0.0, 1.0])
    n_points = 10
    n_iters = 5
    n_per_iters = 3

    X_range = np.linspace(0, 1, 1000)
    y_range = forrester_1d(X_range)
    X = np.random.uniform(domain[0], domain[1], n_points)

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = (
        indices[: round(len(indices) * 0.5)],
        indices[round(len(indices) * 0.5) :],
    )
    train_X = np.sort(X[train_idx])
    test_X = np.sort(X[test_idx])
    train_y = forrester_1d(train_X)
    test_y = forrester_1d(test_X)

    gp = GaussianProcessTFP()
    gp.fit(train_X.reshape(-1, 1), train_y)

    p = gp.predict(test_X.reshape(-1, 1))
    print(f"test R2: {r2_score(test_y, p)}")

    plot = plot_function_custom(
        forrester_1d,
        train_X.reshape(-1, 1),
        y_vec=gp.predict(train_X.reshape(-1, 1)),
        plot_sample_locations=True,
        show=False,
    )
    plot.plot(X_range, y_range, "r")

    lv = LolaVoronoi(
        gp,
        train_X.reshape(-1, 1),
        train_y,
        test_X.reshape(-1, 1),
        test_y,
        [domain],
        forrester_1d,
        n_iteration=n_iters,
        n_new_point_per_iteration=n_per_iters,
        evaluation_metric="rmse",
    )
    lv.run_sequential_design()

    add_samples_to_plot(
        plot,
        lv.train_X[-n_iters * n_per_iters :],
        forrester_1d(lv.train_X[-n_iters * n_per_iters :]),
        "g",
    )


def test_tfdGP():
    num_training_points = 100
    index_points_ = np.random.uniform(-1.0, 1.0, (num_training_points, 1))
    index_points_ = index_points_.astype(np.float64)
    observations_ = (np.sin(3 * np.pi * index_points_[..., 0])) + np.random.normal(
        loc=0, scale=np.sqrt(0.1), size=(num_training_points)
    )

    gp = GaussianProcessTFP()
    gp.fit(index_points_, observations_)

    predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
    predictive_index_points_ = predictive_index_points_[..., np.newaxis]
    mean, samples = gp.predict(predictive_index_points_, return_samples=True)

    plt.figure(figsize=(12, 4))
    plt.plot(
        predictive_index_points_,
        (np.sin(3 * np.pi * predictive_index_points_[..., 0])),
        label="True fn",
    )
    plt.scatter(index_points_[:, 0], observations_, label="Observations")
    for i in range(50):
        plt.plot(
            predictive_index_points_,
            samples[i, :],
            c="r",
            alpha=0.1,
            label="Posterior Sample" if i == 0 else None,
        )
    leg = plt.legend(loc="upper right")
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()


def test_shekel():
    domain = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    n_points = 40

    X1 = np.random.uniform(domain[0, 0], domain[0, 1], n_points)
    X2 = np.random.uniform(domain[1, 0], domain[1, 1], n_points)
    X3 = np.random.uniform(domain[2, 0], domain[2, 1], n_points)
    X4 = np.random.uniform(domain[3, 0], domain[3, 1], n_points)

    X1 = np.append(X1, 4.0)
    X2 = np.append(X2, 4.0)
    X3 = np.append(X3, 4.0)
    X4 = np.append(X4, 4.0)

    X = np.stack([X1, X2, X3, X4], -1)

    y = shekel(X)

    print(y)


def visual_test_probSampling_1D():
    domain = np.array([0.0, 1.0])
    n_points = 10
    X_range = np.linspace(0, 1, 1000)
    y_range = forrester_1d(X_range)
    X = np.random.uniform(domain[0], domain[1], n_points)

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = (
        indices[: round(len(indices) * 0.5)],
        indices[round(len(indices) * 0.5) :],
    )
    train_X = np.sort(X[train_idx])
    test_X = np.sort(X[test_idx])
    train_y = forrester_1d(train_X).reshape((-1, 1))
    test_y = forrester_1d(test_X)

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X.reshape(-1, 1))
    test_X = scaler.transform(test_X.reshape(-1, 1))

    gp = VanillaGaussianProcess()
    gp.fit(train_X, train_y)

    p = gp.predict(test_X)
    print(f"test R2: {r2_score(test_y, p)}")

    plot = plot_function_custom(
        forrester_1d,
        train_X.reshape(-1, 1),
        y_vec=gp.predict(train_X.reshape(-1, 1)),
        plot_sample_locations=True,
        show=False,
    )
    plot.plot(X_range, y_range, "r")

    gpr = VanillaGaussianProcess()
    lv = Probabilistic_sampler(
        target_function=forrester_1d,
        surrogate_model=gpr,
        domain_lower_bound=np.array([0.0]),
        domain_upper_bound=np.array([1.0]),
        fit_points_x=train_X.reshape(-1, 1),
        fit_points_y=train_y,
        test_points_x=test_X.reshape(-1, 1),
        test_points_y=test_y,
    )

    points_x, points_y = lv.sample()

    add_samples_to_plot(
        plot,
        points_x[0 : len(train_X)],
        forrester_1d(points_x[0 : len(train_X)]),
        "r",
    )
    add_samples_to_plot(
        plot,
        points_x[-(lv.iterations * 1) :],
        forrester_1d(points_x[-(lv.iterations * 1) :]),
        "g",
    )

    plt.show()
