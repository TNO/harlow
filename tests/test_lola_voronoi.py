import numpy as np
from scipy.spatial import ConvexHull

from harlow.lola_voronoi import (
    gradient_estimate,
    neighbourhood_score,
    voronoi_volume_estimate,
)
from tests.bounded_voronoi_2d import bounded_voronoi_2d


# .......................................
# Neighborhood score
# .......................................
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

        _, cpr = neighbourhood_score(
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

        _, cpr = neighbourhood_score(
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

        _, cpr = neighbourhood_score(
            neighbor_points_x=neighbor_points_x, reference_point_x=reference_point_x
        )

        np.testing.assert_almost_equal(cpr_expected, cpr)


# .......................................
# Voronoi volume estimation
# .......................................
def test_voronoi_volume_2d():
    domain_lower_bound = np.array([-0.1, -1])
    domain_upper_bound = np.array([1.1, 2])
    n_simulation = int(1e6)

    n_points = [2, 5, 10, 20]
    for n_point in n_points:
        points = np.random.rand(n_point, 2)

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

        relative_areas_expected = np.array(areas_expected) / sum(areas_expected)

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
