import matplotlib.pyplot as plt
import numpy as np

from harlow.lola_voronoi import LolaVoronoi
from harlow.surrogate_model import GaussianProcess
from tests.integration_tests.test_functions import forrester_1d, peaks_2d


def plot_1d_lola_voronoi(
    lv: LolaVoronoi, n_initial_point: int, n_new_point_per_iteration: int
):
    """Utility function for visualizing the results of surrogating 1D functions with
    LolaVoronoi."""
    n_iter = int((len(lv.fit_points_y) - n_initial_point) / n_new_point_per_iteration)
    xx = np.linspace(lv.domain_lower_bound, lv.domain_upper_bound, 100)
    yy_tf = lv.target_function(xx).ravel()
    yy_sm = lv.surrogate_model.predict(xx)

    adaptive_points_x = lv.fit_points_x[n_initial_point:]
    adaptive_points_y = lv.fit_points_y[n_initial_point:]

    fig, ax = plt.subplots()
    ax.plot(xx, yy_tf, label="target function")
    ax.plot(xx, yy_sm, "--", label="surrogate function")
    ax.scatter(
        lv.fit_points_x[:n_initial_point],
        lv.fit_points_y[:n_initial_point],
        label="initial points",
        alpha=0.5,
    )
    ax.scatter(adaptive_points_x, adaptive_points_y, label="adaptive points", alpha=0.5)
    for ii in range(n_iter):
        idx_start = ii * n_new_point_per_iteration
        idx_end = (ii + 1) * n_new_point_per_iteration
        xs = adaptive_points_x[idx_start:idx_end]
        ys = adaptive_points_y[idx_start:idx_end]

        for jj, (x, y) in enumerate(zip(xs, ys)):
            if len(xs) == 1:
                ax.text(x, y, f"${ii + 1}$")
            else:
                ax.text(x, y, f"${ii + 1}^{jj + 1}$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig, ax


def test_sine_1d():
    n_new_points_per_iteration = [1, 2]
    n_initial_point = 4

    domain_lower_bound = np.array([-3])
    domain_upper_bound = np.array([3])
    plot_fig = True

    def target_function(x: np.ndarray):
        return np.sin(x)

    for n_new_point_per_iteration in n_new_points_per_iteration:

        n_iter = int(10 / n_new_point_per_iteration)

        surrogate_model = GaussianProcess()

        # ............................
        # Surrogating
        # ............................
        lv = LolaVoronoi(
            target_function=target_function,
            surrogate_model=surrogate_model,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
        )
        lv.adaptive_surrogating(
            n_iter=n_iter,
            n_initial_point=n_initial_point,
            n_new_point_per_iteration=n_new_point_per_iteration,
        )

        # ............................
        # Check accuracy
        # ............................
        xx = np.linspace(domain_lower_bound, domain_upper_bound, 100)
        yy_tf = target_function(xx).ravel()
        yy_sm = lv.surrogate_model.predict(xx)

        np.testing.assert_allclose(yy_tf, yy_sm, atol=1e-1)

        # ............................
        # Plot
        # ............................
        if plot_fig:
            plot_1d_lola_voronoi(
                lv,
                n_initial_point=n_initial_point,
                n_new_point_per_iteration=n_new_point_per_iteration,
            )


def test_forrester_1d():
    n_new_point_per_iteration = 2
    n_initial_point = 5
    n_iter = 10

    domain_lower_bound = np.array([0])
    domain_upper_bound = np.array([1])
    plot_fig = True

    def target_function(x: np.ndarray):
        return forrester_1d(x).ravel()

    surrogate_model = GaussianProcess()

    # ............................
    # Surrogating
    # ............................
    lv = LolaVoronoi(
        target_function=target_function,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
    )
    lv.adaptive_surrogating(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_point_per_iteration,
    )

    # ............................
    # Check accuracy
    # ............................
    xx = np.linspace(domain_lower_bound, domain_upper_bound, 100)
    yy_tf = target_function(xx).ravel()
    yy_sm = lv.surrogate_model.predict(xx)

    np.testing.assert_allclose(yy_tf, yy_sm, atol=1)

    # ............................
    # Plot
    # ............................
    if plot_fig:
        plot_1d_lola_voronoi(
            lv,
            n_initial_point=n_initial_point,
            n_new_point_per_iteration=n_new_point_per_iteration,
        )


def test_peaks_2d():
    """Work in progress."""
    # section 6.2.3 of [1]
    n_iter = 6
    n_new_point_per_iteration = 2
    n_initial_point = 10

    domain_lower_bound = np.array([-5, -5])
    domain_upper_bound = np.array([5, 5])
    plot_fig = True

    def target_function(x: np.ndarray):
        x = np.atleast_2d(x)
        return peaks_2d(x)

    surrogate_model = GaussianProcess()

    # ............................
    # Surrogating
    # ............................
    lv = LolaVoronoi(
        target_function=target_function,
        surrogate_model=surrogate_model,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
    )
    lv.adaptive_surrogating(
        n_iter=n_iter,
        n_initial_point=n_initial_point,
        n_new_point_per_iteration=n_new_point_per_iteration,
    )

    # ............................
    # Check accuracy
    # ............................
    n_grid = 50
    x1_vec = np.linspace(domain_lower_bound[0], domain_upper_bound[0], n_grid)
    x2_vec = np.linspace(domain_lower_bound[1], domain_upper_bound[1], n_grid)
    x1_mx, x2_mx = np.meshgrid(x1_vec, x2_vec)
    x12_vec = np.vstack((x1_mx.ravel(), x2_mx.ravel())).T

    yy_tf = target_function(x12_vec).reshape((n_grid, n_grid))
    yy_sm = lv.surrogate_model.predict(x12_vec).reshape((n_grid, n_grid))

    # np.testing.assert_allclose(yy_tf, yy_sm, atol=1e-1)

    # ............................
    # Plot
    # ............................
    if plot_fig:
        fig, axs = plt.subplots(1, 2, sharex="all", sharey="all")
        ax1 = axs[0]
        ax2 = axs[1]

        ax1.contourf(x1_mx, x2_mx, yy_tf)
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax1.set_title("Target function")

        cs = ax2.contourf(x1_mx, x2_mx, yy_sm)
        ax2.scatter(
            lv.fit_points_x[:, 0], lv.fit_points_x[:, 1], color="red", alpha=0.5
        )
        ax2.set_xlabel("$x_1$")
        ax2.set_ylabel("$x_2$")
        ax2.set_title("Surrogate function")
        # TODO: apply the same color range to both contourf plots
        plt.colorbar(cs, ax=axs)
