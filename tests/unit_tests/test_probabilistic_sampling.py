import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from surrogate_model import GaussianProcess, GaussianProcessTFP

from harlow.plotting import add_samples_to_plot, plot_function_custom
from harlow.probabilistic_sampling import Probabilistic_sampler
from tests.test_functions import forresterEtAl


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


def visual_test_1D():
    domain = np.array([0.0, 1.0])
    n_points = 10
    X_range = np.linspace(0, 1, 1000)
    y_range = forresterEtAl(X_range)
    X = np.random.uniform(domain[0], domain[1], n_points)

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = (
        indices[: round(len(indices) * 0.5)],
        indices[round(len(indices) * 0.5) :],
    )
    train_X = np.sort(X[train_idx])
    test_X = np.sort(X[test_idx])
    train_y = forresterEtAl(train_X)
    test_y = forresterEtAl(test_X)

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X.reshape(-1, 1))
    test_X = scaler.transform(test_X.reshape(-1, 1))

    gp = GaussianProcessTFP()
    gp.fit(train_X, train_y)

    p = gp.predict(test_X)
    print(f"test R2: {r2_score(test_y, p[0])}")

    plot = plot_function_custom(
        forresterEtAl,
        train_X.reshape(-1, 1),
        y=gp.predict(train_X.reshape(-1, 1)),
        plot_sample_locations=True,
        show=False,
    )
    plot.plot(X_range, y_range, "r")

    gpr = GaussianProcess()
    lv = Probabilistic_sampler(
        target_function=forresterEtAl,
        surrogate_model=gpr,
        domain_lower_bound=np.array([0.0]),
        domain_upper_bound=np.array([1.0]),
        fit_points_x=train_X.reshape(-1, 1),
        fit_points_y=train_y,
        test_points_x=test_X.reshape(-1, 1),
        test_points_y=test_y,
    )

    points_x, points_y = lv.adaptive_surrogating()

    add_samples_to_plot(
        plot,
        points_x[0 : -(lv.iterations * 1)],
        forresterEtAl(points_x[0 : -(lv.iterations * 1)]),
        "r",
    )
    add_samples_to_plot(
        plot,
        points_x[-(lv.iterations * 1) :],
        forresterEtAl(points_x[-(lv.iterations * 1) :]),
        "g",
    )

    plt.show()


if __name__ == "__main__":
    visual_test_1D()
