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

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from skopt.sampler import Lhs
from skopt.space import Space

# TODO add logging

# adapted from https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling/blob/master/src/adaptive_techniques/LOLA_function.m  # noqa E501
# and gitlab.com/energyincities/besos/-/blob/master/besos/

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
        model,
        train_X,
        train_y,
        test_X,
        test_y,
        domain,
        f,
        n_init=20,
        n_iteration=10,
        n_per_iteration=5,
        metric="r2",
        verbose=False,
    ):
        self.model = model
        self.dimension = len(train_X[0, :])
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.domain = domain
        self.n_init = n_init
        self.n_iteration = n_iteration
        self.n_per_iteration = n_per_iteration
        self.f = f
        self.score = np.empty((self.n_iteration + 1))
        self.verbose = verbose

        if metric == "r2":
            self.metric = r2_score
        elif metric == "mse":
            self.metric = mean_squared_error
        elif metric == "rmse":
            self.metric = lambda x, y: sqrt(mean_squared_error(x, y))

        if np.ndim(self.test_X) == 1:
            self.score[0] = self.metric(
                self.test_y, self.model.predict(self.test_X.reshape(-1, 1))
            )
        else:
            self.score[0] = self.metric(self.test_y, self.model.predict(self.test_X))

    """
    updates the surrogate model weights with newly sampled data
    """

    def update_model(self):
        self.model.update(self.new_data, self.new_data_y)

    """
    retrains the surrogate model on all current training data
    """

    def retrain_model(self):
        self.model.fit(self.train_X, self.train_y)

    """
    entry point to start the sequential algorithm
    """

    def run_sequential_design(self):
        self.N, self.S = initialize_samples(self.train_X)
        self.sample()
        self.update_model()

        for i in range(self.n_iteration):
            for train_new in self.new_data:
                self.N, self.S = update_neighbourhood(
                    self.N, self.train_X, self.S, train_new
                )
                N_new, S_new = initialize_samples(self.train_X, train_new)
                self.N = np.append(self.N, N_new, axis=2)
                self.S = np.append(self.S, S_new, axis=0)
            self.train_X = np.append(self.train_X, self.new_data, axis=0)
            self.train_y = np.append(self.train_y, self.new_data_y, axis=0)
            self.sample()
            self.update_model()
            self.score[i + 1] = self.metric(
                self.test_y, self.model.predict(self.test_X)
            )

    def sample(self):
        lola_est = lola_score(self.N, self.train_X, self.model)
        voronoi, samples = estimate_voronoi_volume(self.train_X, self.domain)
        hybrid_score = lola_voronoi_score(lola_est, voronoi)
        idx_new = np.argsort(hybrid_score)
        data_sorted = self.train_X[idx_new, :]

        ind = 2
        self.new_data = np.empty([self.n_per_iteration, self.train_X.shape[1]])

        for i in range(self.n_per_iteration):
            candidates = in_voronoi_region(
                data_sorted[-i - 1, :], self.train_X, samples
            )

            while len(candidates) <= 1:
                candidates = in_voronoi_region(
                    data_sorted[-i - ind, :], self.train_X, samples
                )
                ind += 1

            # new X data sample
            self.new_data[i, :] = select_new_sample(
                data_sorted[-i - 1, :], self.N[:, :, idx_new][:, :, -i - 1], candidates
            )

        # output from function evaluation on newly selected X samples
        self.new_data_y = self.f(self.new_data).flatten()


def select_new_sample(reference_point, neighbours, candidates):
    neighbours = np.append(neighbours, [reference_point], axis=0)
    dist_max = 0

    for candidate in candidates[1:, :]:
        d = 0
        for neighbour in neighbours:
            d = d + np.linalg.norm(candidate - neighbour)
        if d > dist_max:
            dist_max = d
            candidate_max = candidate

    return candidate_max


def in_voronoi_region(reference_point, train_X, samples):
    mask = train_X != reference_point
    train_X_temp = train_X[mask[:, 0], :]
    # candidates = np.empty([1, train_X.shape[1]])
    candidates = np.empty([1, len(train_X[0, :])])

    for s in samples:
        for train_i in train_X_temp:
            if np.linalg.norm(s - train_i) <= np.linalg.norm(s - reference_point):
                break
            else:
                continue
        if np.all(train_i == train_X[-1]):
            candidates = np.append(candidates, [s], axis=0)

    return candidates


def initialize_samples(train_X, train_new=None):
    m = 2 * len(train_X[0, :])
    d = train_X.shape[1]

    if train_new is None:
        n = len(train_X)
        Neighbourhood_points = np.empty((m, d, n))
        Score_points = np.empty((n))
        train_ref = train_X * 1.0
    else:
        if np.ndim(train_new) == 1:
            train_ref = np.expand_dims(train_new, axis=0)
        else:
            train_ref = train_new * 1.0
        n_new = len(train_ref)
        Neighbourhood_points = np.empty([m, d, n_new])
        Score_points = np.empty([n_new])

    for i in range(len(train_ref)):
        mask = train_X != train_ref[i, :]
        train_norefpoint = train_X[mask[:, 0], :]
        Neighbourhood_points[:, :, i] = train_norefpoint[0:m, :]
        Score_points[i] = neighbourhood_score(
            Neighbourhood_points[:, :, i], train_ref[i, :], train_ref.shape[1]
        )

    ind = 0

    for cand in train_X:
        ind += 1

        Neighbourhood_points, Score_points = update_neighbourhood(
            Neighbourhood_points, train_ref, Score_points, cand
        )

    return Neighbourhood_points, Score_points


def lola_voronoi_score(E, V):
    return V + E / sum(E)


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


def neighbourhood_score(neighbourhood, reference_point, dim):
    m = len(neighbourhood)

    cand_dist = np.empty((m))
    min_dist = np.empty((m))

    C = 0
    for i in range(m):
        C = C + np.linalg.norm(neighbourhood[i, :] - reference_point)
    C = C / m

    for i in range(m):
        for j in range(m):
            if not i == j:
                cand_dist[i] = np.linalg.norm(neighbourhood[i, :] - neighbourhood[j, :])
        min_dist[i] = min(cand_dist)

    if dim > 1:
        A = 1 / m * sum(min_dist)
        R = A / (np.sqrt(2) * C)
    elif dim == 1:  # 1D cases: see Crombecq p.1960
        pr1 = neighbourhood[0, :]
        pr2 = neighbourhood[1, :]
        R = 1 - (np.abs(pr1 + pr2) / (np.abs(pr1) + np.abs(pr2) + np.abs(pr1 - pr2)))

    return R / C


def lola_score(neighbours, train_X, model):
    n = len(train_X)
    idx = 0
    E = np.empty([n])

    predicted_neighbours = np.empty((neighbours.shape[0], neighbours.shape[2]))
    predicted_p = model.predict(train_X)

    for p in train_X:
        predicted_neighbours[:, idx] = model.predict(neighbours[:, :, idx]).flatten()
        grad = gradient(neighbours[:, :, idx], p, model)
        E[idx] = nonlinearity_measure(
            grad,
            neighbours[:, :, idx],
            p,
            predicted_neighbours[:, idx],
            predicted_p[idx],
        )
        idx += 1

    return E


def nonlinearity_measure(grad, neighbours, p, neighbour_prediction, predicted_p):
    E = 0

    for i in range(len(neighbours)):
        E = E + abs(
            neighbour_prediction[i]
            - (predicted_p + np.dot(grad, (neighbours[i, :] - p)))
        )

    return E


"""
 Exploration using Voronoi approximation: identify regions where sample density is
 low. lLw Voronoi volume implies low sampling density. Crombecq, Karel; Gorissen,
 Dirk; Deschrijver, Dirk; Dhaene, Tom (2011) A novel hybrid sequential design
 strategy for global surrogate modeling of computer experiments Estimation of Voronoi
 cell size, see algorithm 2. How large should n be to approximate V-size?

 alternative would be to calculated voronoi cell via Delauney tesselation. more
 expensive, not necessary according to paper.

 alternative for finding nearest, using kdtrees:
   from scipy.spatial import KDTree
   kdt = KDTree(P.T)
   kdt.query(PQ.T)
"""


def estimate_voronoi_volume(P, domain, n=100):
    V = np.zeros(len(P))
    S = hypercube_sampling(domain, n)

    for s in S:
        d = np.inf
        idx = 0
        for p in P:
            if np.linalg.norm(p - s) < d:
                d = np.linalg.norm(p - s)
                idx_fin = idx
            idx += 1
        V[idx_fin] = V[idx_fin] + 1 / len(S)

    return V, np.asarray(S)


def hypercube_sampling(domain, n_samples, method="maximin"):
    space = Space(list(map(tuple, domain)))
    lhs = Lhs(criterion=method, iterations=5000)
    samples = lhs.generate(space.dimensions, n_samples)

    return samples


def gradient(N, p, model):
    m = len(N)
    d = len(p)

    P_mtrx = np.empty((m, d))
    predicted_neighbours = model.predict(N).flatten()

    for i in range(m):
        P_mtrx[i, :] = N[i, :] - p

    gradient = np.linalg.lstsq(P_mtrx, np.transpose(predicted_neighbours), rcond=None)[
        0
    ].reshape((1, d))

    return gradient
