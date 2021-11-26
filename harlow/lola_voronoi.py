"""Lola-Vornoi adaptive design strategy for global surrogate modelling.

The algorithm is proposed and described in this paper:
Crombecq, Karel, et al. (2011) A novel hybrid sequential design strategy for global
surrogate modeling of computer experiments. SIAM Journal on Scientific Computing 33.4
(2011): 1948-1974.

The implementation is based on and inspired by:
* adapted from https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling/blob/master/src/adaptive_techniques/LOLA_function.m  # noqa E501
* gitlab.com/energyincities/besos/-/blob/master/besos/
"""

from math import sqrt
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import mean_squared_error, r2_score
from skopt.sampler import Lhs
from skopt.space import Space

# TODO add logging


"""
    class LolaVoronoi creates a LV object.

    :param model: The surrogate model
    :param train_X: training data (inputs)
    :param train_y: training data (output)
    :param test_X: testing data (inputs)
    :param test_y: testing data (output)
    :param domain: the domain to use for sampling. numpy ndarray
    :param f: the evaluation function
    :param n_init: number of initial samples
    :param n_iterations: number of iterations
    :param n_per_iteration: number of samples to draw per iteration of the sequential
        algorithm
    :param metric: the evaluation metric to use
"""


class LolaVoronoi:
    def __init__(
        self,
        surrogate_model,
        train_X,
        train_y,
        test_X,
        test_y,
        domain,
        target_function,
        n_init=20,
        n_iteration=10,
        n_new_point_per_iteration=5,
        metric="r2",
        verbose=False,
    ):
        self.surrogate_model = surrogate_model
        self.dimension = len(train_X[0, :])
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.domain = domain
        self.n_init = n_init
        self.n_iteration = n_iteration
        self.n_new_point_per_iteration = n_new_point_per_iteration
        self.target_function = target_function
        self.score = np.empty((self.n_iteration + 1))
        self.verbose = verbose

        # new names
        self.reference_points_x = None
        self.reference_points_y = None
        self.all_neighbor_points_x = None
        self.all_neighbor_points_y = None
        self.domain_lower_bound = None
        self.domain_upper_bound = None

        if metric == "r2":
            self.metric = r2_score
        elif metric == "mse":
            self.metric = mean_squared_error
        elif metric == "rmse":
            self.metric = lambda x, y: sqrt(mean_squared_error(x, y))

        if np.ndim(self.test_X) == 1:
            self.score[0] = self.metric(
                self.test_y, self.surrogate_model.predict(self.test_X.reshape(-1, 1))
            )
        else:
            self.score[0] = self.metric(
                self.test_y, self.surrogate_model.predict(self.test_X)
            )

    def update_surrogate_model(self):
        """updates the surrogate model weights with newly sampled data
        TODO: is there a point of having this function?
        """
        self.surrogate_model.update(self.new_data, self.new_data_y)

    def retrain_model(self):
        """retrains the surrogate model on all current training data
        TODO: is there a point of having this function?
        """
        self.surrogate_model.fit(self.train_X, self.train_y)

    def run_sequential_design(self):
        """entry point to start the sequential algorithm"""
        self.N, self.S = initialize_samples(self.train_X)
        self.sample()
        self.update_surrogate_model()

        for i in range(self.n_iteration):
            for train_new in self.new_data:
                self.N, self.S = update_neighbourhood(
                    self.N, self.train_X, self.S, train_new
                )
                N_new, S_new = initialize_samples(self.train_X, train_new)
                self.N = np.append(self.N, N_new, axis=2)
                self.S = np.append(self.S, S_new, axis=0)

            self.reference_points_x.append(self.new_reference_points_x)
            self.reference_points_y.append(self.new_reference_points_y)

            self.sample()
            self.update_surrogate_model()
            self.score[i + 1] = self.metric(
                self.test_y, self.surrogate_model.predict(self.test_X)
            )

    def sample(self):
        """Find the next reference point(s) where the target function will be
        evaluated."""
        reference_points_x = self.reference_points_x
        reference_points_y = self.reference_points_y
        all_neighbor_points_x = self.all_neighbor_points_x
        all_neighbor_points_y = self.all_neighbor_points_y
        domain_lower_bound = self.domain_lower_bound
        domain_upper_bound = self.domain_upper_bound
        n_new_point_per_iteration = self.n_new_point_per_iteration
        n_dim = reference_points_x.shape[1]

        nonlinearity_measures = lola_score(
            all_neighbor_points_x=all_neighbor_points_x,
            all_neighbor_points_y=all_neighbor_points_y,
            reference_points_x=reference_points_x,
            reference_points_y=reference_points_y,
        )
        (
            relative_volumes,
            random_points,
            distance_mx,
            closest_indicator_mx,
        ) = voronoi_volume_estimate(
            points=reference_points_x,
            domain_lower_bound=domain_lower_bound,
            domain_upper_bound=domain_upper_bound,
        )
        hybrid_score = lola_voronoi_score(nonlinearity_measures, relative_volumes)

        # TODO: check np.partition as an alternative
        idxs_new_neighbor = np.argsort(hybrid_score)[:n_new_point_per_iteration]

        new_reference_points_x = np.empty((n_new_point_per_iteration, n_dim))
        for ii, idx_new_neighbor in enumerate(idxs_new_neighbor):
            # all the distances to reference point whose neighborhood will get a new
            # point where the target function is evaluated
            distances_in_neighbor = distance_mx[:, idx_new_neighbor]

            # consider only those points that are within the voronoi cell of the
            # reference point
            idx_mask_in_neighbor = closest_indicator_mx[:, idx_new_neighbor]

            # to avoid selecting points that are not in the neighborhood
            distances_in_neighbor[~idx_mask_in_neighbor] = -1

            # largest distance within the same voronoi cell
            idx_max_distance = np.argmax(distances_in_neighbor)
            new_reference_point_x = random_points[idx_max_distance]
            new_reference_points_x[ii, :] = new_reference_point_x

        # evaluate the target function
        new_reference_points_y = self.target_function(new_reference_points_x)

        # collect results
        self.new_reference_points_x = new_reference_points_x
        self.new_reference_points_y = new_reference_points_y


def initialize_samples(train_X, train_new=None):
    m = 2 * len(train_X[0, :])
    d = train_X.shape[1]

    if train_new is None:
        n = len(train_X)
        neighbourhood_points = np.empty((m, d, n))
        score_points = np.empty(n)
        train_ref = train_X * 1.0
    else:
        if np.ndim(train_new) == 1:
            train_ref = np.expand_dims(train_new, axis=0)
        else:
            train_ref = train_new * 1.0
        n_new = len(train_ref)
        neighbourhood_points = np.empty([m, d, n_new])
        score_points = np.empty([n_new])

    for i in range(len(train_ref)):
        mask = train_X != train_ref[i, :]
        train_norefpoint = train_X[mask[:, 0], :]
        neighbourhood_points[:, :, i] = train_norefpoint[0:m, :]
        score_points[i] = neighbourhood_score(
            neighbourhood_points[:, :, i], train_ref[i, :], train_ref.shape[1]
        )

    ind = 0

    for cand in train_X:
        ind += 1

        neighbourhood_points, score_points = update_neighbourhood(
            neighbourhood_points, train_ref, score_points, cand
        )

    return neighbourhood_points, score_points


def lola_voronoi_score(
    nonlinearity_measures: np.ndarray, relative_volumes: np.ndarray
) -> np.ndarray:
    """Eq.(5.1) of [1]."""
    return relative_volumes + nonlinearity_measures / np.sum(nonlinearity_measures)


def update_neighbourhood(neighbours, train_X, scores, candidates):
    if np.ndim(train_X) == 1:
        train_X = np.expand_dims(train_X, axis=0)
        neighbours = np.reshape(
            neighbours, (neighbours.shape[1], neighbours.shape[1], 1)
        )
        scores = np.expand_dims(scores, axis=0)

    if np.ndim(candidates) == 1:
        candidates = np.expand_dims(candidates, axis=0)

    m = 2 * len(train_X[0, :])
    ind = 0

    for candidate in candidates:
        for p in train_X:
            if sum(p == candidate) < len(train_X[0, :]):
                neighbours_temp = np.dstack([neighbours[:, :, ind]] * m)
                scores_temp = np.zeros((m))

                for i in range(m):
                    if sum(sum(neighbours_temp[:, :, i] == candidate)) < len(
                        train_X[0, :]
                    ):
                        neighbours_temp[i, :, i] = candidate
                    else:
                        pass
                    scores_temp[i] = neighbourhood_score(
                        neighbours_temp[:, :, i], p, np.ndim(candidates)
                    )
                min_ind = np.argmin(scores_temp)

                neighbours[:, :, ind] = neighbours_temp[:, :, min_ind]
                scores[ind] = scores_temp[min_ind]
                ind += 1
            else:
                pass

    return neighbours, scores


def neighbourhood_score(
    neighbor_points_x: np.ndarray, reference_point_x: np.ndarray
) -> Tuple[float, float]:
    """

    Args:
        neighbor_points_x:
            n_point x n_dim.
        reference_point_x:
            1 x n_dim

    Returns:
        Neighborhood score.
    """
    # shape the input if not in the right shape, TODO: is this really needed?
    reference_point_x = reference_point_x.reshape((1, -1))
    n_dim = reference_point_x.shape[1]
    neighbor_points_x = neighbor_points_x.reshape((-1, n_dim))

    # cohesion, Eq(4.3) of [1]
    cohesion = np.mean(
        np.sqrt(np.sum((neighbor_points_x - reference_point_x) ** 2, axis=1))
    )

    # cross-polytope resemblance
    if n_dim == 1:
        pr1 = neighbor_points_x[0, :]
        pr2 = neighbor_points_x[1, :]
        # Eq(4.6) of [1]
        cross_polytope_ratio = 1 - np.abs(pr1 + pr2) / (
            np.abs(pr1) + np.abs(pr2) + np.abs(pr1 - pr2)
        )
    else:
        # Eq(4.4) of [1]
        # smallest neighbor distance for each neighbor
        neighbor_distances = squareform(pdist(neighbor_points_x, metric="euclidean"))
        np.fill_diagonal(neighbor_distances, np.inf)
        min_neighbor_distances = np.min(neighbor_distances, axis=1)
        adhesion = np.mean(min_neighbor_distances)
        # Eq(4.5) of [1]
        cross_polytope_ratio = adhesion / (np.sqrt(2) * cohesion)

    # Eq(4.7) of [1]
    return cross_polytope_ratio / cohesion, cross_polytope_ratio


def lola_score(
    all_neighbor_points_x: np.ndarray,
    all_neighbor_points_y: np.ndarray,
    reference_points_x: np.ndarray,
    reference_points_y: np.ndarray,
) -> np.ndarray:
    # non-linearity measure for each point
    n_reference_point = len(reference_points_y)
    es = np.empty(n_reference_point)

    for ii, (
        neighbor_points_x,
        neighbor_points_y,
        reference_point_x,
        reference_point_y,
    ) in enumerate(
        zip(
            all_neighbor_points_x,
            all_neighbor_points_y,
            reference_points_x,
            reference_points_y,
        )
    ):

        reference_point_gradient = gradient_estimate(
            reference_point_x=reference_point_x,
            reference_point_y=reference_point_y,
            neighbor_points_x=neighbor_points_x,
            neighbor_points_y=neighbor_points_y,
        )[0]
        es[ii] = nonlinearity_measure(
            reference_point_x=reference_point_x,
            reference_point_y=reference_point_y,
            reference_point_gradient=reference_point_gradient,
            neighbor_points_x=neighbor_points_x,
            neighbor_points_y=neighbor_points_y,
        )

    return es


def nonlinearity_measure(
    reference_point_x: np.ndarray,
    reference_point_y: float,
    reference_point_gradient: np.ndarray,
    neighbor_points_x: np.ndarray,
    neighbor_points_y: np.ndarray,
) -> float:
    # Eq.(4.9) of [1]
    e = np.sum(
        np.abs(
            neighbor_points_y
            - (
                reference_point_y
                + reference_point_gradient * (neighbor_points_x - reference_point_x)
            )
        )
    )[0]
    return e


def voronoi_volume_estimate(
    points: np.ndarray,
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_simulation: int = None,
    random_points: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the relative volume of the Voronoi tessellation of `points` bounded by
    `domain_lower_bound` and `domain_upper_bound`.

    The algorithm is described in section 3 of [1].

    Args:
        points:
            n_point x n_dim.
        domain_lower_bound:
            n_dim.
        domain_upper_bound:
            n_dim
        n_simulation:
            Number of random points used to estimate the relative volumes. If
            `random_points` is provided then this argument is ignored.
        random_points:

    Returns:
        Relative volumes in the same order as `points`.
    """
    # dimensions are not checked
    if n_simulation is None:
        n_simulation = 100 * points.shape[0]

    if random_points is None:
        n_dim = len(domain_lower_bound)
        random_points = domain_lower_bound + np.random.rand(n_simulation, n_dim) * (
            domain_upper_bound - domain_lower_bound
        )

    # all relevant distances, n_simulation x n_point
    distance_mx = cdist(random_points, points, metric="euclidean")

    # index of the closest `point` to each `random_point`
    col_idx_min = np.argmin(distance_mx, axis=1)
    row_idx_min = np.arange(distance_mx.shape[0])

    # indicator matrix to count the number of `random_points` closest to each `points`
    closest_indicator_mx = np.zeros(distance_mx.shape)
    # place one in the indicator matrix if the element (distance) is a closest distance
    closest_indicator_mx[row_idx_min, col_idx_min] = 1

    # count the closest `random_points` for each `points`
    closest_counts = np.sum(closest_indicator_mx, axis=0)

    # relative volume estimate
    relative_volumes = closest_counts / n_simulation

    return relative_volumes, random_points, distance_mx, closest_indicator_mx


def hypercube_sampling(domain, n_samples, method="maximin"):
    space = Space(list(map(tuple, domain)))
    lhs = Lhs(criterion=method, iterations=5000)
    samples = lhs.generate(space.dimensions, n_samples)

    return samples


def gradient_estimate(
    reference_point_x: np.ndarray,
    reference_point_y: float,
    neighbor_points_x: np.ndarray,
    neighbor_points_y: np.ndarray,
) -> np.ndarray:
    """
    Estimate the gradient at `reference_point` by fitting a hyperplane to
    `neighbor_points` in a least-square sense. A hyperplane that goes exactly
    through the `reference_point`.

    We think that the there is a mistake in Eq.(4.8) of [1], the right hand side
    should be `f(p_neighbor) - f(p_reference)`, or on the left hand side the
    `-p_reference` should be dropped if the formulation is in line with this:
    "Without loss of generality, we assume that pr lies in the origin."
    section 4.2.1 of [1].

    Args:
        reference_point_x:
            1 x n_dim.
        reference_point_y:
        neighbor_points_x:
            n_neighbor x n_dim.
        neighbor_points_y:
            n_neighbor x 1.

    Returns:
        Gradient estimate, 1 x n_dim.
    """
    # shape the input if not in the right shape, TODO: is this really needed?
    reference_point_x = reference_point_x.reshape((1, -1))
    n_dim = reference_point_x.shape[1]
    neighbor_points_x = neighbor_points_x.reshape((-1, n_dim))
    neighbor_points_y = neighbor_points_y.reshape((-1, 1))

    # to ensure that we hyperplane goes through `reference_point`
    neighbor_points_x_diff = neighbor_points_x - reference_point_x
    neighbor_points_y_diff = neighbor_points_y - reference_point_y

    # least-square fit of the hyperplane, the gradient is the hyperplane coefficients
    gradient = np.linalg.lstsq(
        neighbor_points_x_diff, neighbor_points_y_diff, rcond=None
    )[0].reshape((1, n_dim))

    return gradient
