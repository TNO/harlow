"""
Helper functions for the adaptive sampling strategies.
"""

import numpy as np
from skopt.sampler import Lhs
from skopt.space import Space


def latin_hypercube_sampling(
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_sample: int,
    method="maximin",
):
    domain = np.vstack((domain_lower_bound, domain_upper_bound)).astype(float).T
    space = Space(list(map(tuple, domain)))
    lhs = Lhs(criterion=method, iterations=5000)
    samples = lhs.generate(space.dimensions, n_sample)

    return np.array(samples, dtype=float)
