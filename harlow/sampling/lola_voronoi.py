import itertools
import os
import pickle
import time
from typing import Callable, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist

from harlow.sampling.sampling_baseclass import Sampler
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import evaluate, latin_hypercube_sampling
from harlow.utils.log_writer import write_scores, write_timer
from harlow.utils.metrics import rmse
from harlow.utils.numba_utils import euclidean_distance, np_all, np_any, np_min

# from harlow.distance import pdist_full_matrix

# TODO
#  * improve logging
#  * pretty timedelta: https://gist.github.com/thatalextaylor/7408395

nopython = True
fastmath = True


# -----------------------------------------------------
# USER FACING API (class)
# -----------------------------------------------------
# TODO: is this class really needed or just an unnecessary complication? it has a
#  single method
class LolaVoronoi(Sampler):
    """Lola-Vornoi adaptive design strategy for global surrogate modelling.

    The algorithm is proposed and described in this paper:
    [1] Crombecq, Karel, et al. (2011) A novel hybrid sequential design strategy for global
    surrogate modeling of computer experiments. SIAM Journal on Scientific Computing 33.4
    (2011): 1948-1974.

    The implementation is influenced by:
    * gitlab.com/energyincities/besos/-/blob/master/besos/
    * https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling/blob/master/src/adaptive_techniques/LOLA_function.m  # noqa E501

    """

    def __init__(
        self,
        target_function: Callable[[np.ndarray], np.ndarray],
        surrogate_model: Surrogate,
        domain_lower_bound: np.ndarray,
        domain_upper_bound: np.ndarray,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        evaluation_metric: Callable = rmse,
        logging_metrics: list = None,
        verbose: bool = False,
        run_name: str = None,
        save_dir: str = "",
    ):

        super(LolaVoronoi, self).__init__(
            target_function,
            surrogate_model,
            domain_lower_bound,
            domain_upper_bound,
            fit_points_x,
            fit_points_y,
            test_points_x,
            test_points_y,
            evaluation_metric,
            logging_metrics,
            verbose,
            run_name,
            save_dir,
        )

    def sample(
        self,
        n_initial_points: int = None,
        max_n_iterations: int = 20,
        n_new_points_per_iteration: int = 1,
        stopping_criterium: float = None,
        ignore_far_neighborhoods: Optional[bool] = True,
        ignore_old_neighborhoods: Optional[bool] = True,
    ):
        # ..........................................
        # Initialize
        # ..........................................
        target_function = self.target_function
        domain_lower_bound = self.domain_lower_bound
        domain_upper_bound = self.domain_upper_bound
        n_dim = len(domain_lower_bound)

        if n_initial_points is None:
            n_initial_points = 5 * n_dim

        # ..........................................
        # Initial sample of points
        # ..........................................
        gen_start_time = time.time()
        if self.fit_points_x is None:
            # latin hypercube sampling to get the initial sample of points
            points_x = latin_hypercube_sampling(
                n_sample=n_initial_points,
                domain_lower_bound=domain_lower_bound,
                domain_upper_bound=domain_upper_bound,
            )
            # evaluate the target function
            points_y = target_function(points_x)

            self.fit_points_x = points_x
            self.fit_points_y = points_y
        else:
            points_x = self.fit_points_x
            points_y = self.fit_points_y

        self.step_gen_time.append(time.time() - gen_start_time)

        # fit the surrogate model
        start_time = time.time()
        self.surrogate_model.fit(points_x, points_y.ravel())
        logger.info(
            f"Fitted the first surrogate model in {time.time() - start_time} sec."
        )

        score = evaluate(
            self.logging_metrics,
            self.test_points_x,
            self.test_points_y,
        )
        self.step_x.append(points_x)
        self.step_y.append(points_y)
        self.step_score.append(score)
        self.step_iter.append(0)
        self.step_fit_time.append(time.time() - start_time)
        # ..........................................
        # Iterative improvement (adaptive stage)
        # ..........................................
        n_point_last_iter = 0
        for ii in range(max_n_iterations):
            logger.info(
                f"Started adaptive iteration step: {ii+1} (max steps:"
                f" {max_n_iterations})."
            )

            start_time = time.time()
            new_points_x = best_new_points(
                points_x=points_x,
                points_y=points_y,
                domain_lower_bound=domain_lower_bound,
                domain_upper_bound=domain_upper_bound,
                n_point_last_iter=n_point_last_iter,
                n_new_point=n_new_points_per_iteration,
                ignore_far_neighborhoods=ignore_far_neighborhoods,
                ignore_old_neighborhoods=ignore_old_neighborhoods,
            )
            self.step_gen_time.append(time.time() - start_time)
            n_point_last_iter = points_x.shape[0]
            logger.info(
                f"Found the next best {n_new_points_per_iteration} point(s) in "
                f"{time.time() - start_time} sec."
            )

            # evaluate the target function
            new_points_y = target_function(new_points_x)

            # add the new points to the old ones
            points_x = np.vstack((points_x, new_points_x))
            points_y = np.vstack((points_y, new_points_y))

            # refit the surrogate
            start_time = time.time()
            self.surrogate_model.update(new_points_x, new_points_y.ravel())
            self.step_fit_time = time.time() - start_time
            logger.info(
                f"Fitted a new surrogate model in {time.time() - start_time} sec."
            )

            #
            self.fit_points_x = points_x
            self.fit_points_y = points_y
            score = evaluate(
                self.logging_metrics,
                self.test_points_x,
                self.test_points_y,
            )
            self.step_x.append(points_x)
            self.step_y.append(points_y)
            self.step_score.append(score)
            self.step_iter.append(ii + 1)
            timing_dict = {
                "Gen time": self.step_gen_time[ii + 1],
                "Fit time": self.step_fit_time[ii + 1],
            }
            write_scores(self.writer, score, ii + 1)
            write_timer(self.writer, timing_dict, ii + 1)

            logger.info(f"Evaluation metric score on provided testset: {score}")
            self.score = score[self.evaluation_metric.__name__]
            self.iterations = ii
            # Save model every 5 iterations
            if self.iterations % 5 == 0:
                self.save_model()

            if self.score <= stopping_criterium or self.score <= stopping_criterium:
                logger.info(f"Sampler ended in {ii} iterations")
                # Save model if converged
                self.save_model()
                break

        self.writer.close()
        return self.fit_points_x, self.fit_points_y

    def save_model(self):
        save_name = self.run_name + "_{}_iters.pkl".format(self.iterations)
        save_path = os.path.join(self.save_dir, save_name)

        with open(save_path, "wb") as file:
            pickle.dump(self.surrogate_model, file)


# -----------------------------------------------------
# SUPPORTING FUNCTIONS
# -----------------------------------------------------
def best_new_points(
    points_x: np.ndarray,
    points_y: np.ndarray,
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_point_last_iter: int,
    n_new_point: int = 1,
    ignore_far_neighborhoods: Optional[bool] = True,
    ignore_old_neighborhoods: Optional[bool] = True,
):
    # shape the input if not in the right shape
    n_dim = len(domain_lower_bound)
    points_x = points_x.reshape((-1, n_dim))
    points_y = points_y.reshape((-1, 1))

    # Find the best neighborhoods for each row of `points_x`
    (
        best_neighborhood_scores,
        best_neighborhood_idxs,
        all_neighborhood_scores,
        all_neighbor_point_idxs_combinations,
    ) = best_neighborhoods(
        points_x,
        n_point_last_iter,
        ignore_far_neighborhoods=ignore_far_neighborhoods,
        ignore_old_neighborhoods=ignore_old_neighborhoods,
    )

    # Find the `n_new_point_per_step` best next/new point(s)
    idx = all_neighbor_point_idxs_combinations[best_neighborhood_idxs]
    all_best_neighbor_points_x = points_x[idx, :]
    all_best_neighbor_points_y = points_y[idx, :]

    new_points_x = best_new_points_with_neighbors(
        reference_points_x=points_x,
        reference_points_y=points_y,
        all_neighbor_points_x=all_best_neighbor_points_x,
        all_neighbor_points_y=all_best_neighbor_points_y,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
        n_next_point=n_new_point,
    )

    return new_points_x


def best_new_points_with_neighbors(
    reference_points_x: np.ndarray,
    reference_points_y: np.ndarray,
    all_neighbor_points_x: np.ndarray,
    all_neighbor_points_y: np.ndarray,
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_next_point: int,
):
    """Find the new/next reference point(s) where the target function will be evaluated.

    TODO: reuse the random point used here for other calls of this function (
        `best_next_points`)
    """
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
    idxs_new_neighbor = np.argsort(-hybrid_score)[:n_next_point]

    new_reference_points_x = np.empty((n_next_point, n_dim))
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

    return new_reference_points_x


def best_neighborhoods(
    points_x: np.ndarray,
    n_point_last_iter: int,
    ignore_far_neighborhoods: Optional[bool] = True,
    ignore_old_neighborhoods: Optional[bool] = True,
):
    """Find the best neighborhood for each row of `points_x`. This function is expected
    to be used with the initial sample of `points_x`, when no best neighborhoods yet
    available from preceding iteration steps. This function exists to separate numba
    compatible code from non-compatible one."""

    # n-choose-k, all possible neighbor combinations, the elements of the matrix are
    # `points_x` indices
    n_point, n_dim = points_x.shape
    n_neighbor = 2 * n_dim

    all_neighbor_point_idxs_combinations = np.array(
        list(itertools.combinations(np.arange(n_point), n_neighbor))
    )
    (
        best_neighborhood_scores,
        best_neighborhood_idxs,
        all_neighborhood_scores,
    ) = best_neighborhoods_numba(
        points_x,
        all_neighbor_point_idxs_combinations,
        n_point_last_iter,
        ignore_far_neighborhoods=ignore_far_neighborhoods,
        ignore_old_neighborhoods=ignore_old_neighborhoods,
    )
    return (
        best_neighborhood_scores,
        best_neighborhood_idxs,
        all_neighborhood_scores,
        all_neighbor_point_idxs_combinations,
    )


def best_neighborhoods_numba(
    points_x: np.ndarray,
    all_neighbor_point_idxs_combinations: np.ndarray,
    n_point_last_iter: int,
    ignore_far_neighborhoods: Optional[bool] = True,
    ignore_old_neighborhoods: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the best neighborhood for each row of `points_x`. This function is expected
    to be used with the initial sample of `points_x`, when no best neighborhoods yet
    available from preceding iteration steps.

    Args:
        points_x: n_point x n_dim.
        all_neighbor_point_idxs_combinations:
        n_point_last_iter:
        ignore_far_neighborhoods:
        ignore_old_neighborhoods:
    Returns:

    """

    # TODO: It should be possible to perform the calculation of
    # all neighbourhoods after discarding the points that are
    # too far away. This will reduce storage requirements
    # and the computational cost of obtaining all combinations.
    n_point, n_dim = points_x.shape
    n_neighborhood = all_neighbor_point_idxs_combinations.shape[0]

    # Number of new points
    n_new_points = points_x.shape[0] - n_point_last_iter

    # this matrix contains neighborhoods that contain the reference point as well, these
    # will be assigned -1 and the rest will get a neighborhood score
    # each column belong to one point, same order as points_x (reference_point)
    all_neighborhood_scores = -np.ones((n_neighborhood, n_point))

    # TODO: both loops are completely independent hence parallelizable
    # for ii in nb.prange(n_point):
    # This loop is executed for each point in the domain
    for ii in range(n_point):
        reference_point_x = np.expand_dims(points_x[ii], 0)
        # consider only the valid neighborhoods: the ones that do not contain
        # `reference_point_x`
        idx_valid_neighborhoods = np.where(
            np_all(all_neighbor_point_idxs_combinations != ii, axis=1)
        )[0]

        if ignore_old_neighborhoods:
            # Find index of old points
            arr_valid_neighbourhoods = all_neighbor_point_idxs_combinations[
                idx_valid_neighborhoods
            ]
            arr_new_point_no = np.arange(
                n_point_last_iter, n_point_last_iter + n_new_points
            )

            # Remove entries from valid neighborhoods that
            # do not contain a new point
            arr_valid_neighbourhoods_mask = np.repeat(
                False, len(arr_valid_neighbourhoods)
            )
            for _idx_pt, pt in enumerate(arr_new_point_no):

                # Mask valid neighbourhoods (i.e. neighbourhoods that contain
                # the current new point). Terminate early if all elements
                # in mask array are true.
                arr_valid_neighbourhoods_mask = np.bitwise_or(
                    arr_valid_neighbourhoods_mask,
                    np_any(arr_valid_neighbourhoods == pt, axis=1),
                )
                if np.all(arr_valid_neighbourhoods_mask):
                    break
            arr_valid_neighbourhoods = arr_valid_neighbourhoods[
                arr_valid_neighbourhoods_mask
            ]
            idx_valid_neighborhoods = np.where(arr_valid_neighbourhoods_mask)[0]

        # consider only the new neighborhoods that are not too far away.
        # neighborhoods that contain a point farther away the median Euclidean distance
        # are dismissed.
        # section [5] of the paper
        if np.where(idx_valid_neighborhoods)[0].size:
            if ignore_far_neighborhoods:
                valid_pts_combinations = all_neighbor_point_idxs_combinations[
                    idx_valid_neighborhoods
                ]
                all_neighbor_points_x = points_x[valid_pts_combinations, :]
                dists = np.linalg.norm(
                    all_neighbor_points_x - reference_point_x, axis=-1
                )
                med = np.median(dists)
                idx_valid_neighborhoods = idx_valid_neighborhoods[
                    np.all(dists <= med, axis=-1)
                ]

        # compute the neighbourhood_score (`ns`) for each neighborhood
        #
        if np.where(idx_valid_neighborhoods)[0].size:
            for jj in range(len(idx_valid_neighborhoods)):
                idx_valid_neighborhood = idx_valid_neighborhoods[jj]
                neighbor_points_x = points_x[
                    all_neighbor_point_idxs_combinations[idx_valid_neighborhood], :
                ]
                # this is a time consuming function
                ns = neighborhood_score(
                    neighbor_points_x=neighbor_points_x,
                    reference_point_x=reference_point_x,
                )[0]
                all_neighborhood_scores[idx_valid_neighborhood, ii] = ns

    # best neighborhoods, the order is the same as in `points_x`
    # best_neighborhood_idxs = np_argmax(all_neighborhood_scores, axis=0)
    best_neighborhood_idxs = np.argmax(all_neighborhood_scores, axis=0)
    flat_idxs = n_point * best_neighborhood_idxs + np.arange(n_point)
    # best_neighborhood_scores = all_neighborhood_scores.ravel()[
    #     flat_idxs.astype(nb.int_)
    # ]
    best_neighborhood_scores = all_neighborhood_scores.ravel()[flat_idxs.astype(np.int)]
    return (
        best_neighborhood_scores,
        best_neighborhood_idxs,
        all_neighborhood_scores,
    )


def lola_voronoi_score(
    nonlinearity_measures: np.ndarray, relative_volumes: np.ndarray
) -> np.ndarray:
    """Eq.(5.1) of [1]."""
    return relative_volumes + nonlinearity_measures / np.sum(nonlinearity_measures)


def neighborhood_score(
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
    # shapes are not checked!
    # shape the input if not in the right shape, TODO: is this really needed?
    reference_point_x = reference_point_x.reshape((1, -1))
    n_dim = reference_point_x.shape[1]
    # neighbor_points_x = neighbor_points_x.reshape((-1, n_dim))

    # cohesion, Eq(4.3) of [1]
    cohesion = np.mean(
        np.sqrt(np.sum((neighbor_points_x - reference_point_x) ** 2, axis=1))
    )

    # cross-polytope ratio
    if n_dim == 1:
        pr1 = neighbor_points_x[0, 0]
        pr2 = neighbor_points_x[1, 0]
        # Eq(4.6) of [1]
        cross_polytope_ratio = 1 - np.abs(pr1 + pr2) / (
            np.abs(pr1) + np.abs(pr2) + np.abs(pr1 - pr2)
        )
    else:
        # Eq(4.4) of [1]
        # smallest neighbor distance for each neighbor
        # neighbor_distances = squareform(pdist(neighbor_points_x, metric="euclidean"))
        # np.fill_diagonal(neighbor_distances, np.inf)
        # min_neighbor_distances = np.min(neighbor_distances, axis=1)
        #  neighbor_distances = pdist_full_matrix(neighbor_points_x)

        # This should be an equivalent alternative to squareform + pdist
        # that also works with numba. Based on this stackoverflow answer:
        # https://stackoverflow.com/questions/60839615/
        # efficiently-calculating-a-euclidean-dist-matrix-in-numpy
        # TODO: Check
        # TODO: would be good to avoid np.max
        # neighbor_distances = squareform(pdist(neighbor_points_x, metric="euclidean"))

        neighbor_distances = euclidean_distance(neighbor_points_x)
        np.fill_diagonal(neighbor_distances, np.max(neighbor_distances))
        # min_neighbor_distances = np_min(neighbor_distances, axis=1)
        min_neighbor_distances = np_min(neighbor_distances, axis=1)
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
    """Non-linearity measure for each point and its neighbor. Measures how much a
    neighborhood deviates from a hyperplane."""
    n_reference_point = len(reference_points_y)
    es = np.empty(n_reference_point)

    for (
        ii,
        (
            neighbor_points_x,
            neighbor_points_y,
            reference_point_x,
            reference_point_y,
        ),
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
    )
    return float(e)


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
        n_simulation = 1000 * points.shape[0]

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
    closest_indicator_mx = np.zeros(distance_mx.shape, dtype=bool)
    # place a one (True) in the indicator matrix if the element (distance) is a closest
    # distance
    closest_indicator_mx[row_idx_min, col_idx_min] = True

    # count the closest `random_points` for each `points`
    closest_counts = np.sum(closest_indicator_mx, axis=0)

    # relative volume estimate
    relative_volumes = closest_counts / n_simulation

    return relative_volumes, random_points, distance_mx, closest_indicator_mx


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
