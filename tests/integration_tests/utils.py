import numpy as np
from matplotlib import pyplot as plt

from harlow.lola_voronoi import LolaVoronoi


def plot_1d_lola_voronoi(
    lv: LolaVoronoi,
    n_initial_point: int,
    n_new_point_per_iteration: int,
    n_grid_point: int = 100,
):
    """Utility function for visualizing the results of surrogating 1D functions with
    LolaVoronoi."""
    n_iter = int((len(lv.fit_points_y) - n_initial_point) / n_new_point_per_iteration)
    xx = np.linspace(lv.domain_lower_bound, lv.domain_upper_bound, n_grid_point)
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
