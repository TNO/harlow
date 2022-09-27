"""Cross Validation Voronoi adaptive design strategy for global surrogate
modelling.

Note that this

The algorithm is proposed and described in this paper:
[1] Xu, Shengli, et al. A robust error-pursuing sequential sampling approach
 for global metamodeling based on voronoi diagram and cross validation.
 Journal of Mechanical Design 136.7 (2014): 071009

[2] Liu H, Xu S, Wang X, Yang S, Meng J. A (1996).
A multi-response adaptive sampling approach for global metamodeling
Proceedings of the Institution of Mechanical Engineers Part C Journal of
Mechanical Engineering Science 1989-1996 (vols 203-210)

[3] Kaminsky A.L., Wang Y, Pant K, Hashii W.N., Atachbarian A. (2020) An
efficient batch k-fold cross-validation voronoi adaptive sampling technique
for global surrogate modelling. Journal of Mechanical Design 143
 """

import os
import pickle
import time
from typing import Callable, Tuple

import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from harlow.sampling.sampling_baseclass import Sampler
from harlow.surrogating.surrogate_model import Surrogate
from harlow.utils.helper_functions import (
    evaluate_modellist,
    latin_hypercube_sampling,
    normalized_response,
)
from harlow.utils.log_writer import write_scores, write_timer
from harlow.utils.metrics import nrmse, rmse


# -----------------------------------------------------
# USER FACING API (class)
# -----------------------------------------------------
class CVVoronoi(Sampler):
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

        super(CVVoronoi, self).__init__(
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

        self.surrogates = []

    def sample(
        self,
        n_initial_points: int = 20,
        n_new_points_per_iteration: int = 1,
        stopping_criterium: float = 0.05,
        max_n_iterations: int = 5000,
    ):
        """TODO: allow for providing starting points"""
        # ..........................................
        # Initialize
        # ..........................................
        target_function = self.target_function
        domain_lower_bound = self.domain_lower_bound
        domain_upper_bound = self.domain_upper_bound
        n_dim = len(domain_lower_bound)

        """This is to support multi-output"""
        dim_out = self.fit_points_y.shape[1]

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
        print('Initial points X, Y shapes', points_x.shape, points_y.shape)
        for i in range(0, dim_out):
            s_i = self.surrogate_model
            s_i.fit(points_x, points_y[:, i])
            self.surrogates.append(s_i)
            logger.info(
                f"Fitted the first surrogate model {i+1} in"
                f" {time.time() - start_time} sec."
            )

        # Additional class objects to help keep track of the sampling. `gen_time`
        # denotes the time to generate new points. `fit_time` is the time
        # to fit the surrogate.
        score = evaluate_modellist(
            self.logging_metrics,
            self.surrogates,
            self.test_points_x,
            self.test_points_y,
        )
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

            """
                The following implements the algorithm from [1] [2] [3]
            """

            # Step 1. Calculate the Voronoi cells
            random_points, distance_mx, closest_indicator_mx = calculate_voronoi_cells(
                points_x,
                domain_lower_bound=domain_lower_bound,
                domain_upper_bound=domain_upper_bound,
            )

            # Step 2. Determine the Voronoi cell with the highest error
            most_sensitive_cell = identify_sensitive_voronoi_cell(
                self.surrogates,
                self.surrogate_model,
                points_x,
                points_y,
                dim_out,
                self.test_points_x,
                self.test_points_y,
                5,
            )

            # Step 3. Pick the point within the most sensitive cell, furthest
            # away from point i.
            new_points_x = pick_new_samples(
                most_sensitive_cell,
                random_points,
                distance_mx,
                closest_indicator_mx,
                n=1,
            )

            self.step_gen_time.append(time.time() - start_time)
            logger.info(
                f"Found the next best {n_new_points_per_iteration} point(s) in "
                f"{time.time() - start_time} sec."
            )

            # evaluate the target function
            new_points_y = target_function(new_points_x)
            # print('New points y shape', new_points_y.shape)
            # add the new points to the old ones 
            # We need to do that inside HERE NOT in the Surrogates because the 
            # Sampler ALSO needs the new points to measure the Voronoi cells
            points_x = np.vstack((points_x, new_points_x))
            print('Y and new Y shapes', points_y.shape, new_points_y.shape)
            points_y = np.vstack((points_y, np.swapaxes(new_points_y,0,1)))
            print('Points X, Y stacked shapes', points_x.shape, points_y.shape)
            # refit the surrogate
            start_time = time.time()
            for i, surrogate in enumerate(self.surrogates):
                # surrogate.update(new_points_x, new_points_y[i])
                surrogate.fit(points_x, points_y[:, i])
                self.step_fit_time.append(time.time() - start_time)
                logger.info(
                    f"Fitted a new surrogate model in {time.time() - start_time} sec."
                )

            self.fit_points_x = points_x
            self.fit_points_y = points_y

            score = evaluate_modellist(
                self.logging_metrics,
                self.surrogates,
                self.test_points_x,
                self.test_points_y,
            )
            logger.info(
                f"Fitted a new surrogate model in {time.time() - start_time} sec."
            )
            self.step_x.append(points_x)
            self.step_y.append(points_y)
            self.step_score.append(score[self.evaluation_metric.__name__])
            self.step_iter.append(ii + 1)
            timing_dict = {
                "Gen time": self.step_gen_time[ii + 1],
                "Fit time": self.step_fit_time[ii + 1],
            }
            write_scores(self.writer, score, ii + 1)
            write_timer(self.writer, timing_dict, ii + 1)

            self.score = score['rmse'][0]
            self.iterations = ii
            # Save model every 200 iterations
            if self.iterations % 200 == 0:
                for s_i, s in enumerate(self.surrogates):
                    save_name = (
                        f"{self.run_name}_" f"out{s_i}_{self.iterations}_iters.pkl"
                    )
                    save_path = os.path.join(self.save_dir, save_name)
                    with open(save_path, "wb") as file:
                        pickle.dump(s, file)

            if self.score <= stopping_criterium:
                logger.info(f"Algorithm converged in {ii} iterations")
                # Save model if converged
                for s_i, s in enumerate(self.surrogates):
                    save_name = (
                        f"{self.run_name}_" f"out{s_i}_{self.iterations}_converged.pkl"
                    )
                    save_path = os.path.join(self.save_dir, save_name)
                    with open(save_path, "wb") as file:
                        pickle.dump(s, file)
                break
        self.writer.close()

        return self.fit_points_x, self.fit_points_y


def pick_new_samples(
    most_sensitive_cell, random_points, distance_mx, closest_indicator_mx, n=1
):
    sens_cell = closest_indicator_mx[:, most_sensitive_cell]
    voronoi_i_distances = distance_mx[sens_cell, most_sensitive_cell]
    voronoi_i_points = random_points[sens_cell, :]

    max_dist_i = np.argpartition(voronoi_i_distances, -n)[-1:]

    return voronoi_i_points[max_dist_i, :][:]


def identify_sensitive_voronoi_cell(
    surrogates: list,
    surrogate_model: object,
    points_X: np.ndarray,
    points_y: np.ndarray,
    n_dim_out: int,
    test_points_X: np.ndarray,
    test_points_y: np.ndarray,
    k=5,
):
    normalized_responses = []
    surrogates_nrmse = []

    for idx, m in enumerate(surrogates):
        normalized_responses.append(normalized_response(m, points_X))
        # equation 4 from [2]
        surrogates_nrmse.append(
            nrmse(m, test_points_X, test_points_y[:, idx])
        )  # equation 5
        # from [2]

    total_error = sum(surrogates_nrmse)
    weights = np.asarray([i / total_error for i in surrogates_nrmse])  # eq.
    # 15 from
    # [2]

    # nrmse_sys = max(surrogates_nrmse)  # equation 8 from [2]. not used.
    kfold = KFold(n_splits=k, random_state=None, shuffle=False)
    kfold_results = np.zeros((k, len(surrogates)))
    split_indices = []
    i = 0
    # CV approach from [3] to avoid costly surrogate building for higher
    # number of points
    # print('Sensitive Voronoi cell Points X', points_X, points_X.shape)
    for train_index, test_index in kfold.split(points_X):
        split_indices.append(train_index)
        for s in range(0, n_dim_out):
            X_train, X_test = points_X[train_index], points_X[test_index]
            y_train, y_test = points_y[train_index], points_y[test_index]

            s_i = surrogate_model
            s_i.fit(X_train, y_train[:, s])
            # y_pred = s_i.predict(X_test).cpu().numpy()
            y_pred = s_i.predict(X_test)
            kfold_results[i, s] = np.linalg.norm(y_test[:, s] - y_pred)
        kfold_results_multiout = np.sum(kfold_results, axis=1)

        i += 1

    worst_fold = np.argmax(kfold_results_multiout)  # identify the worst fold
    worst_point_indices = split_indices[worst_fold]
    worst_points_X = points_X[worst_point_indices]
    worst_points_y = points_y[worst_point_indices]
    cv_error_per_point = np.zeros((len(worst_points_X), n_dim_out))

    for i in range(len(worst_point_indices)):
        X_i = worst_points_X[i, :]
        y_i = worst_points_y[i]

        idx_exc_i = np.where(np.all(worst_points_X != X_i, axis=1))
        points_X_exc_i = worst_points_X[idx_exc_i]
        points_y_exc_i = worst_points_y[idx_exc_i]

        # implements #13 #14 from [2]
        for j in range(0, n_dim_out):
            s_i = surrogate_model
            s_i.fit(points_X_exc_i, points_y_exc_i[:, j])
            # predict X[i] with surrogate
            y_pred = s_i.predict(X_i.reshape((1, -1)))

            cv_error_per_point[i, j] = np.linalg.norm(
                y_i[j] - y_pred
            )  # eq. 14 from [2]

    max_eij = np.max(cv_error_per_point)

    LOOCV_scores = np.sum(cv_error_per_point * weights + max_eij, axis=1)  #
    # eq. 16 from [2]

    return np.argmax(LOOCV_scores)


def calculate_voronoi_cells(
    points: np.ndarray,
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_simulation: int = None,
    random_points: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the Voronoi tessellation of `points` bounded by
    `domain_lower_bound` and `domain_upper_bound`.

    The algorithm is described in algorithm 1 of [1].

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
        n_simulation = 100 * points.shape[0] * points.shape[1]

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

    return random_points, distance_mx, closest_indicator_mx
