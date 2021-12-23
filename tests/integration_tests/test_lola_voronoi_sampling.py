"""Test whether the samples match (or close to) the Matlab SUMO samples.
To reduce runtime a dummy surrogate model class is used: `EmptySurrogate`."""

import numpy as np

from harlow.lola_voronoi import LolaVoronoi
from tests.integration_tests.test_functions import forrester_1d, lldeh_1d
from tests.integration_tests.utils import plot_1d_lola_voronoi


class EmptySurrogate:
    def __init__(self):
        pass

    def update(self, x, y):
        pass

    def fit(self, x, y):
        return self.update(x, y)

    @staticmethod
    def predict(x):
        return np.zeros(x.shape[0])


def test_forrester_1d_against_sumo():
    n_new_point_per_iteration = 1
    n_iter = 15

    start_points_x = np.array([[0.0, 0.25, 0.50, 0.75, 1.0]]).T
    domain_lower_bound = np.array([0])
    domain_upper_bound = np.array([1])
    plot_fig = True

    def target_function(x: np.ndarray):
        return forrester_1d(x).ravel()

    start_points_y = target_function(start_points_x).reshape((-1, 1))
    surrogate_model = EmptySurrogate()

    # ............................
    # Surrogating
    # ............................
    lv = LolaVoronoi(
        target_function=target_function,
        surrogate_model=surrogate_model,
        fit_points_x=start_points_x,
        fit_points_y=start_points_y,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
    )
    lv.sample(
        n_iter=n_iter,
        n_new_point_per_iteration=n_new_point_per_iteration,
    )

    print(np.sort(lv.fit_points_x[len(start_points_x) :, :], axis=0))
    # ............................
    # Check accuracy
    # ............................
    # TODO: compare with SUMO results

    # ............................
    # Plot
    # ............................
    if plot_fig:
        plot_1d_lola_voronoi(
            lv,
            n_initial_point=len(start_points_x),
            n_new_point_per_iteration=n_new_point_per_iteration,
        )


def test_lldeh_1d_against_sumo():
    n_new_point_per_iteration = 1
    n_iter = 15

    domain_lower_bound = np.array([-20])
    domain_upper_bound = np.array([20])
    start_points_x = np.linspace(domain_lower_bound, domain_upper_bound, 5)

    plot_fig = True

    def target_function(x: np.ndarray):
        return lldeh_1d(x, a=2.1)

    start_points_y = target_function(start_points_x)
    surrogate_model = EmptySurrogate()

    # ............................
    # Surrogating
    # ............................
    lv = LolaVoronoi(
        target_function=target_function,
        surrogate_model=surrogate_model,
        fit_points_x=start_points_x,
        fit_points_y=start_points_y,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
    )
    lv.sample(
        n_iter=n_iter,
        n_new_point_per_iteration=n_new_point_per_iteration,
    )

    print(np.sort(lv.fit_points_x[len(start_points_x) :, :], axis=0))
    # ............................
    # Check accuracy
    # ............................
    # TODO: compare with SUMO results

    # ............................
    # Plot
    # ............................
    if plot_fig:
        _, ax = plot_1d_lola_voronoi(
            lv,
            n_initial_point=len(start_points_x),
            n_new_point_per_iteration=n_new_point_per_iteration,
            n_grid_point=500,
        )
        ax.set_aspect("equal", "box")
        ax.legend(bbox_to_anchor=(0.5, 1.05))
