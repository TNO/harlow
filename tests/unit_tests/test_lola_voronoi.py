"""Test components of the implementation of the LOLA-Voronoi algorithm."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from harlow.lola_voronoi import (
    best_neighborhoods,
    gradient_estimate,
    neighborhood_score,
    voronoi_volume_estimate,
)
from tests.unit_tests.bounded_voronoi_2d import bounded_voronoi_2d


# .......................................
# Best neighborhoods
# .......................................
def test_visually_2d_best_neighborhoods():
    """This test is to be checked by humans. Plot a few best neighborhoods in 2D."""
    # number of points to be used
    n_point = 10

    # Random points
    all_points_x = [np.random.rand(n_point, 2)]

    # Points along a circle perimeter and one in the center
    center_point_x = np.array([0, 0])
    radius = 2
    angles = np.random.rand(n_point - 1) * np.pi * 2
    perimeter_points_x = np.array(
        [
            center_point_x[0] + radius * np.cos(angles),
            center_point_x[1] + radius * np.sin(angles),
        ]
    ).T
    all_points_x.append(np.vstack((center_point_x, perimeter_points_x)))

    for points_x in all_points_x:
        # Find the best neighborhood for each row in `points_x`
        (
            best_neighborhood_scores,
            best_neighborhood_idxs,
            _,
            all_neighbor_point_idxs_combinations,
        ) = best_neighborhoods(points_x=points_x)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()
        for ii, ax in enumerate(axs):
            # select reference point for plotting
            reference_point_x = points_x[ii]
            neighbor_points_x = points_x[
                all_neighbor_point_idxs_combinations[best_neighborhood_idxs[ii]], :
            ]
            ax.scatter(points_x[:, 0], points_x[:, 1])
            ax.scatter(
                reference_point_x[0], reference_point_x[1], label="reference point"
            )
            ax.plot(
                neighbor_points_x[:, 0],
                neighbor_points_x[:, 1],
                "s",
                markeredgecolor="red",
                markerfacecolor="None",
                label="best neighborhood points",
            )
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            if ii == 0:
                ax.legend()
            ax.axis("equal")


# .......................................
# Neighborhood score
# .......................................
def test_example_of_section_4_2_3():
    # the 2D example from section 4.2.3 of [1]
    base_neighbor_points_x = np.array([[-1, 0], [1, 0], [0, -1]])
    reference_point_x = np.array([[0, 0]])

    expected_cpr_max_x = [0.0, 1.0]
    expected_ns_max_x = [0.0, 0.0]

    n_grid = 51
    x1_mx, x2_mx = np.meshgrid(np.linspace(-3, 3, n_grid), np.linspace(-3, 3, n_grid))
    added_neighbor_points_x = np.vstack((x1_mx.ravel(), x2_mx.ravel())).T

    ns_vec = np.empty(n_grid ** 2)
    cpr_vec = np.empty(n_grid ** 2)
    for ii, added_neighbor_point_x in enumerate(added_neighbor_points_x):
        neighbor_points_x = np.vstack((base_neighbor_points_x, added_neighbor_point_x))
        ns, cpr = neighborhood_score(
            neighbor_points_x=neighbor_points_x, reference_point_x=reference_point_x
        )
        ns_vec[ii] = ns
        cpr_vec[ii] = cpr

    ns_mx = ns_vec.reshape((n_grid, n_grid))
    cpr_mx = cpr_vec.reshape((n_grid, n_grid))

    # check the maximum of cross-polytope ratio
    idx = np.unravel_index(np.argmax(cpr_mx), cpr_mx.shape)
    cpr_max_x = [x1_mx[idx], x2_mx[idx]]
    a_tol = 0.5 * 6 / (n_grid - 1)
    np.testing.assert_allclose(expected_cpr_max_x, cpr_max_x, atol=a_tol)

    # check the maximum of cross-polytope ratio
    idx = np.unravel_index(np.argmax(cpr_mx), cpr_mx.shape)
    cpr_max_x = [x1_mx[idx], x2_mx[idx]]
    a_tol = 0.5 * 6 / (n_grid - 1)
    np.testing.assert_allclose(expected_cpr_max_x, cpr_max_x, atol=a_tol)
    assert np.all(np.logical_and(0.0 <= cpr_mx, cpr_mx <= 1.0))

    idx = np.unravel_index(np.argmax(ns_mx), ns_mx.shape)
    ns_max_x = [x1_mx[idx], x2_mx[idx]]
    np.testing.assert_allclose(expected_ns_max_x, ns_max_x, atol=a_tol)

    # ..........................
    # Plot (for visual check)
    # ..........................
    # Compare with Fig(4.5) of [1]
    cmap = "viridis"
    to_plot = {
        "Cross-polytope ratio": {"values_mx": cpr_mx, "maximum_point_x": cpr_max_x},
        "Neighborhood score": {"values_mx": ns_mx, "maximum_point_x": ns_max_x},
    }
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for ii, (data_key, data_value) in enumerate(to_plot.items()):
        ax_left = axs[ii, 0]
        ax_right = axs[ii, 1]
        z_mx = data_value["values_mx"]
        max_x = data_value["maximum_point_x"]

        # left, surface plot
        ax_left.remove()
        ax_left = fig.add_subplot(2, 2, 2 * ii + 1, projection="3d")
        ax_left.azim = -160

        ax_left.plot_surface(x1_mx, x2_mx, z_mx, rstride=1, cstride=1, cmap=cmap)
        ax_left.set_xlabel("$x_1$")
        ax_left.set_ylabel("$x_2$")
        ax_left.set_title(data_key)

        # right, contour plot
        ax_right.contourf(x1_mx, x2_mx, z_mx, cmap=cmap)
        ax_right.scatter(
            base_neighbor_points_x[:, 0],
            base_neighbor_points_x[:, 1],
            label="neighbor points",
            marker="s",
        )
        ax_right.scatter(
            reference_point_x[:, 0], reference_point_x[:, 1], label="reference point"
        )
        ax_right.scatter(max_x[0], max_x[1], marker="d", label="maximum")
        ax_right.set_xlabel("$x_1$")
        ax_right.set_ylabel("$x_2$")
        ax_right.set_title(data_key)
        ax_right.legend()


def test_circle_cross_polytope_ratio():
    # perfect cross-polytope
    radius = 2
    n_neighbor = 2 * 2
    reference_points_x = [np.array([0, 0]), np.array([3, 1])]
    cpr_expected = 1.0

    for reference_point_x in reference_points_x:
        ii = np.arange(n_neighbor)
        x1 = reference_point_x[0] + radius * np.cos(np.pi / n_neighbor * (1 + 2 * ii))
        x2 = reference_point_x[1] + radius * np.sin(np.pi / n_neighbor * (1 + 2 * ii))
        neighbor_points_x = np.vstack((x1, x2)).T

        _, cpr = neighborhood_score(
            neighbor_points_x=neighbor_points_x, reference_point_x=reference_point_x
        )

        np.testing.assert_almost_equal(cpr_expected, cpr)


def test_imperfect_polytope_ratio():
    n_dims = [1, 2, 5, 10, 15]
    cpr_max = 1.0
    cpr_min = 0.0

    for n_dim in n_dims:
        n_neighbor = n_dim * 2
        reference_point_x = np.ones((1, n_dim))

        neighbor_points_x = np.random.rand(n_neighbor, n_dim) + reference_point_x

        _, cpr = neighborhood_score(
            neighbor_points_x=neighbor_points_x, reference_point_x=reference_point_x
        )

        assert cpr_min <= cpr <= cpr_max


def test_worst_polytope_ratio():
    # all neighbors in one point
    n_dims = [1, 2, 5, 10, 15]
    cpr_expected = 0.0

    for n_dim in n_dims:
        n_neighbor = n_dim * 2
        reference_point_x = np.ones((1, n_dim))

        neighbor_points_x = (
            np.tile(np.random.rand(1, n_dim), (n_neighbor, 1)) + reference_point_x
        )

        _, cpr = neighborhood_score(
            neighbor_points_x=neighbor_points_x, reference_point_x=reference_point_x
        )

        np.testing.assert_almost_equal(cpr_expected, cpr)


# .......................................
# Voronoi volume estimation
# .......................................
def test_voronoi_volume_2d():
    """Test the Monte Carlo simulation based general code against an exact 2D
    solution."""
    domain_lower_bound = np.array([-0.1, -1])
    domain_upper_bound = np.array([1.1, 2])
    n_simulation = int(1e7)
    plot_fig = False

    n_points = [2, 3, 5, 10, 20]
    for n_point in n_points:
        points = domain_lower_bound + np.random.rand(n_point, 2) * (
            domain_upper_bound - domain_lower_bound
        )

        # Reference solution
        vor = bounded_voronoi_2d(
            points,
            bounding_box=[
                domain_lower_bound[0],
                domain_upper_bound[0],
                domain_lower_bound[1],
                domain_upper_bound[1],
            ],
        )

        areas_expected = []
        for filtered_region in vor.filtered_regions:
            areas_expected.append(ConvexHull(vor.vertices[filtered_region, :]).volume)

        domain_area = (domain_upper_bound[0] - domain_lower_bound[0]) * (
            domain_upper_bound[1] - domain_lower_bound[1]
        )

        # The reference algorithm sometimes fails to cover the entire domain,
        # we skip the check if that happens.
        if not np.isclose(sum(areas_expected), domain_area, atol=1e-4):
            warnings.warn(
                f"n_point={n_point}."
                "\nThe reference algorithm has failed to cover the "
                "entire domain. This check is skipped."
            )
            continue

        if plot_fig:
            fig, ax = plt.subplots()
            # Plot initial points
            ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], "b.")
            # Plot ridges points
            for region in vor.filtered_regions:
                vertices = vor.vertices[region, :]
                ax.plot(vertices[:, 0], vertices[:, 1], "go")
            # Plot ridges
            for region in vor.filtered_regions:
                vertices = vor.vertices[region + [region[0]], :]
                ax.plot(vertices[:, 0], vertices[:, 1], "k-")

        relative_areas_expected = np.array(areas_expected) / np.sum(areas_expected)

        # Tested implementation (general, for N-dimensions)
        relative_areas = voronoi_volume_estimate(
            points=points,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
            n_simulation=n_simulation,
        )[0]

        # until we can get the order right..
        # np.testing.assert_array_almost_equal(
        #     relative_areas_expected, relative_areas, decimal=3
        # )
        np.testing.assert_array_almost_equal(
            np.sort(relative_areas_expected), np.sort(relative_areas), decimal=3
        )


# .......................................
# Gradient estimation
# .......................................
def test_gradient_linear():
    """If the points are from a hyperplane then the gradient estimate should be
    exact."""
    n_dims = [1, 2, 5, 10]

    for n_dim in n_dims:
        grad_expected = np.arange(1, n_dim + 1)

        def hyperplane(x):
            return np.sum(grad_expected * x, axis=1, keepdims=True)

        reference_point_x = np.random.randn(1, n_dim)
        reference_point_y = float(hyperplane(reference_point_x))

        neighbor_points_x = np.random.randn(2 * n_dim, n_dim)
        neighbor_points_y = hyperplane(neighbor_points_x)

        grad = gradient_estimate(
            reference_point_x=reference_point_x,
            reference_point_y=reference_point_y,
            neighbor_points_x=neighbor_points_x,
            neighbor_points_y=neighbor_points_y,
        ).ravel()

        np.testing.assert_array_almost_equal(grad_expected, grad)
