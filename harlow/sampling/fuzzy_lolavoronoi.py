"""Fuzzy Lola-Vornoi adaptive design strategy for global surrogate modelling.

The algorithm is proposed and described in this paper:
[1] Crombecq, Karel, et al. (2011) A novel hybrid sequential design strategy for global
surrogate modeling of computer experiments. SIAM Journal on Scientific Computing 33.4
(2011): 1948-1974.

[2] van der Herten, J., Couckuyt, I., Deschrijver, D., & Dhaene, T. (2015).
A fuzzy hybrid sequential design strategy for global surrogate modeling of
 high-dimensional computer experiments.
SIAM Journal on Scientific Computing, 37(2), A1020-A1039.
"""
import os
import pickle
import time
from typing import Callable, Tuple

import numpy as np
import skfuzzy as fuzz
from loguru import logger
from scipy.spatial.distance import cdist, pdist, squareform
from skfuzzy import control as ctrl

from harlow.sampling.sampling_baseclass import Sampler
from harlow.utils.helper_functions import evaluate, latin_hypercube_sampling
from harlow.utils.log_writer import write_scores, write_timer
from harlow.utils.metrics import rmse


# -----------------------------------------------------
# USER FACING API (class)
# -----------------------------------------------------
class FuzzyLolaVoronoi(Sampler):
    def __init__(
        self,
        target_function: Callable[[np.ndarray], np.ndarray],
        surrogate_model_constructor,
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

        super(FuzzyLolaVoronoi, self).__init__(
            target_function,
            surrogate_model_constructor,
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

        self.dim_in = len(domain_lower_bound)
        self.dim_out = None if self.fit_points_y is None else self.fit_points_y.shape[1]

        # self.surrogate_models.append(self.surrogate_model_constructor())

        # TODO add this below again
        # if self.dim_out > 1:
        #     self.multiresponse_sampling = True
        #     if not self.surrogate_model.is_multioutput:
        #         raise ValueError(
        #             "Multiresponse target requires \
        #                          multiresponse surrogate"
        #         )

    def set_initial_set(self, points_x: np.ndarray, points_y: np.ndarray):
        super().set_initial_set(points_x, points_y)
        # Also create the output surrogates
        for _i in range(self.dim_out):
            self.surrogate_models.append(self.surrogate_model_constructor())

    def _fit_models(self):
        # Standard case assumes single model
        for i, dim_surrogate_model in enumerate(self.surrogate_models):
            dim_surrogate_model.fit(
                self.fit_points_x, np.expand_dims(self.fit_points_y[:, i], axis=1)
            )

    def _update_models(
        self, new_fit_points_x: np.ndarray, new_fit_points_y: np.ndarray
    ):
        # Standard case assumes single model
        for i, dim_surrogate_model in enumerate(self.surrogate_models):
            dim_surrogate_model.update(
                new_fit_points_x, np.expand_dims(new_fit_points_y[:, i], axis=1)
            )

    def _predict(self):
        # Standard case assumes single model
        y = np.zeros((self.test_points_x.shape[0], self.dim_out))

        for i, dim_surrogate_model in enumerate(self.surrogate_models):
            a = dim_surrogate_model.predict(self.test_points_x)
            y[:, i] = a[0]
        return y

    def _best_new_points(self, n):
        best_new_points = np.zeros((n, self.dim_out))

        for i, dim_surrogate_model in enumerate(self.surrogate_models):
            new_points = _best_new_points(
                points_x=self.fit_points_x,
                points_y=self.fit_points_y,
                domain_lower_bound=self.domain_lower_bound,
                domain_upper_bound=self.domain_upper_bound,
                n_new_point=n,
                dim_in=self.dim_in,
            )
            best_new_points[:, i] = new_points

        return best_new_points

    def sample(
        self,
        n_initial_points: int = 20,
        n_new_points_per_iteration: int = 1,
        stopping_criterium: float = None,
        max_n_iterations: int = 5000,
    ):
        """TODO: allow for providing starting points"""
        # ..........................................
        # Initialize
        # ..........................................
        target_function = self.target_function
        domain_lower_bound = self.domain_lower_bound
        domain_upper_bound = self.domain_upper_bound

        if n_initial_points is None:
            n_initial_points = 5 * self.dim_in

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
            self.dim_out = self.fit_points_x.shape[0]
        else:
            points_x = self.fit_points_x
            points_y = self.fit_points_y

        self.step_gen_time.append(time.time() - gen_start_time)

        # fit the surrogate model
        start_time = time.time()

        self.surrogate_models[0].fit(points_x, points_y)
        logger.info(
            f"Fitted the first surrogate model in {time.time() - start_time} sec."
        )

        predicted_y = self.surrogate_models[0].predict(
            self.test_points_x, as_array=True
        )

        score = evaluate(self.logging_metrics, self.test_points_y, predicted_y)
        self.step_x.append(points_x)
        self.step_y.append(points_y)
        self.step_score.append(score[self.evaluation_metric.__name__])
        self.step_iter.append(0)
        self.step_fit_time.append(time.time() - start_time)

        # ..........................................
        # Iterative improvement (adaptive stage)
        # ..........................................

        for ii in range(max_n_iterations):
            logger.info(
                f"Started adaptive iteration step: {ii+1} (max steps:"
                f" {max_n_iterations})."
            )

            start_time = time.time()
            new_points_x = _best_new_points(
                points_x=points_x,
                points_y=points_y,
                domain_lower_bound=domain_lower_bound,
                domain_upper_bound=domain_upper_bound,
                n_new_point=n_new_points_per_iteration,
                dim_in=self.dim_in,
            )
            self.step_gen_time.append(time.time() - start_time)
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
            self.surrogate_models[0].update(new_points_x, new_points_y.ravel())
            self.step_fit_time.append(time.time() - start_time)
            logger.info(
                f"Fitted a new surrogate model in {time.time() - start_time} sec."
            )
            #
            self.fit_points_x = points_x
            self.fit_points_y = points_y

            # Re-evaluate the surrogate model.
            predicted_y = self.surrogate_models[0].predict(
                self.test_points_x, as_array=True
            )
            score = evaluate(self.logging_metrics, self.test_points_y, predicted_y)

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

            # Currently use RMSE for convergence
            self.score = score[self.evaluation_metric.__name__]
            self.iterations = ii
            # Save model every 200 iterations
            if self.iterations % 200 == 0:
                save_name = self.run_name + "_{}_iters.pkl".format(self.iterations)
                save_path = os.path.join(self.save_dir, save_name)
                with open(save_path, "wb") as file:
                    pickle.dump(self.surrogate_models[0], file)

            if self.score <= stopping_criterium or ii >= max_n_iterations:
                logger.info(f"Algorithm converged in {ii} iterations")
                # Save model if converged
                save_name = self.run_name + "_{}_iters.pkl".format(self.iterations)
                save_path = os.path.join(self.save_dir, save_name)
                with open(save_path, "wb") as file:
                    pickle.dump(self.surrogate_models[0], file)
                break

        self.writer.close()
        return self.fit_points_x, self.fit_points_y


# -----------------------------------------------------
# SUPPORTING FUNCTIONS
# -----------------------------------------------------
def _best_new_points(
    points_x: np.ndarray,
    points_y: np.ndarray,
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_new_point: int,
    dim_in: int,
) -> np.ndarray:
    """
    Hybrid Sequential Strategy - Alg. (2) [2].
    Combines an exploitation (FLOLA) and an exploration (Voronoi) score
    and selects a new candindate sample in the neighborhood of the N_new
    highest ranked samples

    :param points_x: Nxd-dimensional input data points
    :param points_y: Nxd-dimensional target data points
    :param domain_lower_bound: The lower bound of the function's space
    :param domain_upper_bound: The upper bound of the function's space
    :param n_new_point: The number of new points we wish to sample
    :return new_reference_point_x: The new highly-ranked samples in the neighborhood
    """

    # shape the input if not in the right shape
    points_x = points_x.reshape((-1, dim_in))
    points_y = points_y.reshape((-1, 1))
    # Calculate distance matrix P
    distance_matrix = calculate_distance_matrix(points_x, dim_in)
    # Calculate the V for every Pr
    (
        relative_volumes,
        random_points,
        distance_mx,
        closest_indicator_mx,
    ) = voronoi_volume_estimate(
        points=points_x,
        domain_lower_bound=domain_lower_bound,
        domain_upper_bound=domain_upper_bound,
    )

    neighborhoods_scores = best_neighbourhoods(
        points_x, points_y, distance_matrix, relative_volumes
    )

    # TODO: check np.partition as an alternative
    idxs_new_neighbor = np.argsort(-neighborhoods_scores)[:n_new_point]

    new_reference_points_x = np.empty((n_new_point, dim_in))
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


def best_neighbourhoods(
    points_x: np.ndarray,
    points_y: np.ndarray,
    distance_matrix: np.ndarray,
    volume_estimate: np.ndarray,
) -> np.ndarray:
    """
    Exploitation Aglorithm - Alg. (1) [2].
    Computes a score fir all points p in P_r, indicating the nonlinearity
    of the region surrounding p. New samples are chosen in the neighborhood
    of the N_new highest ranked samples.

    :param points_x: Nxd-dimensional input data points
    :param points_y: Nxd-dimensional target data points
    :param distance_matrix: NxN distance matrix
    :volume_estimate: N-dimensional Voronoi volume estimate
    :returns: H_fuzz: N-dimensional Hybrid score
    """

    n_point, n_dim = points_x.shape
    # print('X shapes', points_x.shape)
    # print('Y shapes', points_y.shape)
    nonlinear_score, H_fuzzy = np.empty(n_point), np.empty(n_point)
    # Init FIS S
    # It should be done before the for loop BUT
    # requires adhesion MAX value, that is computed inside the LOOP
    # TODO CHeck it later.
    # FIS = init_FIS()
    K = 4 * n_dim
    if K >= n_point - 1:
        K = n_point - 1

    for ii in range(n_point):
        alpha = calculate_alpha(ii, K, distance_matrix)
        # gets the neighbours of Pr as indices
        neighbors_idx = get_neighbourhood(ii, distance_matrix, alpha, n_dim)
        # print("Neighbors index", neighbors_idx, '\n', neighbors_idx.shape)
        neighbors_coords = points_x[neighbors_idx, :]
        # print("Neighbor coords ", neighbors_coords, '\n', neighbors_coords.shape)
        adhesion, cohesion = get_adhesion_cohesion(ii, neighbors_idx, distance_matrix)
        # print("ADH & COH ", adhesion, adhesion.shape, cohesion, cohesion.shape)
        FIS = init_FIS(neighbors_coords, adhesion)
        w = assign_weights(neighbors_coords, cohesion, adhesion, FIS)
        # print('Weights', w)
        grad = flola_gradient_estimate(
            points_x[ii, :],
            points_y[ii],
            points_x[neighbors_idx, :],
            points_y[neighbors_idx, :],
            w,
        )
        # print('grad', grad, grad.shape)
        # E_fuzzy(P_r) Eq. (3.2) [2]
        nonlinear_score[ii] = nonlinearity_measure(
            points_x[ii, :],
            points_y[ii, :],
            grad,
            points_x[neighbors_idx, :],
            points_y[neighbors_idx, :],
        )
        # print('Nonlinear score', nonlinear_score[ii])

    # H_fuzzy(P_r) Eq.(5.1) [2]
    H_fuzzy = flola_voronoi_score(nonlinear_score, volume_estimate)
    # print(H_fuzzy, H_fuzzy.shape)
    return H_fuzzy


def init_FIS(data_points: np.ndarray, adhesion: np.ndarray):
    """
    Initialize the Fuzzy Inference System S; Section 4 [2].

    :param data_points: Nxd-dimensional input vector of N data points with d dimensions
    :param cohesion: The N-dimensional cohesion values of the neighbors of P_r
    :param adhesion: The N-dimensional adhesion values of the neighbors of P_r
    :return flola_sim: The Fuzzy Inference System S
    """

    a = np.sort(data_points, axis=None)
    # Different ways to define the function's ranges.
    # Sparse = Faster / Dense = Slower
    # a = np.arange(np.min(data_points), np.max(data_points), 0.1)
    # a_w = np.linspace(0, 1, 100, endpoint=True)
    # a_w = np.linspace(0, 1, 5, endpoint=True)
    a_w = np.arange(0, 1, 0.1)
    A_max = np.max(adhesion)
    coh = ctrl.Antecedent(a, "cohesion")
    adh = ctrl.Antecedent(a, "adhesion")
    wei = ctrl.Consequent(a_w, "weight")

    # Define the membership functions & populate the space
    coh["high"] = coh_high(coh.universe)
    adh["low"] = adh_low(adh.universe, A_max)
    adh["high"] = adh_high(adh.universe, A_max)

    # construct triangular output/weight member functions.
    wei["low"] = fuzz.trimf(wei.universe, [0, 0, 0.13])
    wei["average"] = fuzz.trimf(wei.universe, [0.16, 0.33, 0.5])
    wei["high"] = fuzz.trimf(wei.universe, [0.6, 1.01, 1.01])
    # Define the rules of the System
    rule1 = ctrl.Rule(coh["high"] & adh["low"], wei["high"])
    rule2 = ctrl.Rule(coh["high"] & adh["high"], wei["average"])
    rule3 = ctrl.Rule(~coh["high"] & adh["low"], wei["average"])
    rule4 = ctrl.Rule(~coh["high"] & adh["high"], wei["low"])
    # Setup the FIS
    FLOLA_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    flola_sim = ctrl.ControlSystemSimulation(FLOLA_ctrl)

    return flola_sim


# TODO time it if needed
def assign_weights(
    data_points: np.ndarray,
    cohesion: np.ndarray,
    adhesion: np.ndarray,
    flola_sim,
) -> np.ndarray:
    """
    Weight calculation of data_points based on cohesion & adhesion values between
    the data points; Section 4 [2].

    :param data_points: Nxd-dimensional input vector of N data points with d dimensions
    :param cohesion: The N-dimensional cohesion values of the neighbors of P_r
    :param adhesion: The N-dimensional adhesion values of the neighbors of P_r
    :param flola_sim: The Fuzzy Inference System S
    :return weight: The N-dimensional weight values returned by the evaluation of FIS S
    """

    S = flola_sim
    dims = data_points.shape
    weights = np.zeros(dims[0])

    # Get the weights for each neighbor
    for i in range(dims[0]):
        S.input["cohesion"] = cohesion[i]
        S.input["adhesion"] = adhesion[i]
        S.compute()
        weights[i] = S.output["weight"]

    return weights


def coh_high(x, s_c=0.3):
    """
    Cohesion High membership function
    """
    return 1 / (1 + np.exp(-s_c * x))


def adh_high(x, A_max, s_ah=0.3):
    """
    Adhesion High membership function
    """
    return np.exp(((-(x**2)) / 2 * ((A_max * s_ah) ** 2)))


def adh_low(x, A_max, s_al=0.27):
    """
    Adhesion Low membership function
    """
    res = (-((x - A_max) ** 2)) / 2 * ((A_max * s_al) ** 2)
    return np.exp(res)


def get_adhesion_cohesion(
    Pr_index: int, P_neigbors_idxs: np.ndarray, distance_matix: np.ndarray
) -> Tuple[np.array, np.array]:
    """
    Calculation of adhesion and cohesion values for the neighbors of P_r.
    According to Eq. (4.1) and (4.2) [2].

    :param Pr_index: Index of reference point
    :param P_neigbors_idxs: neighbour indexes for point Pr
    :param distance_matix: The precalculated distance matrix for all p in P
    :return: Adhesion and Cohesion arrays for N
    """

    C = distance_matix[Pr_index, P_neigbors_idxs]
    A = np.zeros((len(P_neigbors_idxs)))

    for i in range(len(P_neigbors_idxs)):
        neighbor_dists = distance_matix[P_neigbors_idxs[i], :]
        r = np.delete(neighbor_dists, P_neigbors_idxs[i])
        A[i] = np.min(r)

    return A, C


def get_neighbourhood(
    Pr_idx: int, distance_matrix: np.ndarray, alpha: float, n_dim: int
) -> np.array:
    """
    Calculate the neighbourhood for point Pr.
    Eq. (3.3) [2].

    :param Pr_idx: Index of reference point
    :param distance_matrix: The NxN-dimensional precalculated distance matrix
    for all p in P
    :param alpha: Distance parameter Eq. (3.4) [2]
    :param n_dim: Number of dimensions
    :return neighbors_idx: N-dimensional array with neighbour indexes for
    point Pr
    """

    distances_prIdx = distance_matrix[Pr_idx, :]
    neighbors_idx = np.where(distances_prIdx < alpha)[0]
    neighbors_idx = neighbors_idx[neighbors_idx != Pr_idx]

    # If number of neighbors is smaller than ndim Eq. (3.1) will be undertermined.
    # Then, take ndim nearest neighbors.
    if len(neighbors_idx) < n_dim:
        nearest_ndim_idx = np.argpartition(np.delete(distances_prIdx, Pr_idx), n_dim)
        neighbors_idx = distances_prIdx[nearest_ndim_idx[:n_dim]]

    return neighbors_idx


def calculate_alpha(Pr_idx: int, K: int, distance_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate distance parameter alpha.
    Eq. (3.4) [2].

    :param Pr_idx: Index of reference point
    :param K: Parameter K = 4d
    :param distance_matrix: The precalculated distance matrix for all p in P
    :return: alpha as float
    """

    distances_prIdx = distance_matrix[Pr_idx, :]
    distances_wo_prIdx = np.delete(distances_prIdx, Pr_idx)
    nearest_k_vals = np.sort(distances_wo_prIdx)[:K]

    return np.sum(nearest_k_vals) * (2 / K)


def calculate_distance_matrix(
    points_x: np.ndarray, n_dim: int, fractional: bool = False
) -> np.ndarray:
    """
    This function calculates the distance matrix for the points P.
    Instead of calculating the distance for alpha, A and
    C for Pr in the loop, this function is meant to be called once,
    prior to the main for loop iterating over all Pr.

    :param points_x: The set of point P
    :return: A distance matrix for P
    """

    if not fractional:
        return squareform(pdist(points_x, "euclidean"))
    else:
        return squareform(pdist(points_x, "minkowsky", p=n_dim))


def flola_voronoi_score(
    nonlinearity_measures: float, relative_volumes: np.ndarray
) -> np.ndarray:
    """Eq.(5.1) of [2]."""
    return relative_volumes + nonlinearity_measures / np.sum(nonlinearity_measures)


def nonlinearity_measure(
    reference_point_x: np.ndarray,
    reference_point_y: float,
    reference_point_gradient: np.ndarray,
    neighbor_points_x: np.ndarray,
    neighbor_points_y: np.ndarray,
) -> float:
    """Eq.(4.9) of [1]."""
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
    closest_indicator_mx = np.zeros(distance_mx.shape, dtype=bool)
    # place a one (True) in the indicator matrix if the element (distance) is a closest
    # distance
    closest_indicator_mx[row_idx_min, col_idx_min] = True

    # count the closest `random_points` for each `points`
    closest_counts = np.sum(closest_indicator_mx, axis=0)

    # relative volume estimate
    relative_volumes = closest_counts / n_simulation

    return relative_volumes, random_points, distance_mx, closest_indicator_mx


def flola_gradient_estimate(
    reference_point_x: np.ndarray,
    reference_point_y: np.ndarray,
    neighbor_points_x: np.ndarray,
    neighbor_points_y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Estimate the gradient at `reference_point` (P_r) by fitting a hyperplane to
    `neighbor_points` in a least-square sense. A hyperplane that goes exactly
    through the `reference_point`. Eq.(3.2) [2] wrt to weights for all neighbors of P_r
    obtained from solving FIS S.

    :param reference_point_x: The d-dimensional reference sample P_r
    :param reference_point y: -//-
    :param neigbor_points_x: The m x d-dimensional neighbors of P_r
    :param neigbor_points_y: -//-
    :param weights: The weights of every neighbor of P_r
    :return gradient: The d-dimensional gradient at P_r
    """

    reference_point_x = reference_point_x.reshape((1, -1))
    n_neighbors, n_dims = neighbor_points_x.shape
    # to ensure that we hyperplane goes through `reference_point`
    neighbor_points_x_diff = neighbor_points_x - reference_point_x
    neighbor_points_y_diff = neighbor_points_y - reference_point_y
    # Solve Weighted Least Squares
    # Least-Square fit of the hyperplane, the gradient is the hyperplane coefficient
    Aw = neighbor_points_x_diff * np.sqrt(weights[:, np.newaxis])
    Bw = neighbor_points_y_diff * np.sqrt(weights)
    gradient = np.linalg.lstsq(Aw, Bw, rcond=None)[0].reshape(n_neighbors, n_dims)

    return gradient
