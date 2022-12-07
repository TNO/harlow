import sys
from pathlib import Path

import numpy as np

from harlow.sampling import ProbabilisticSampler, FuzzyLolaVoronoi
from harlow.surrogating import VanillaGaussianProcess
from tests.offload_hartmann import succeeding_hartman, peaks_2d_multivariate
from tests.surrogate_loop_test import hypercube_initialization


def main():
    # domains_lower_bound = np.array([0, 0, 0, 0, 0, 0])
    # domains_upper_bound = np.array([1, 1, 1, 1, 1, 1])
    domains_lower_bound = np.array([-8, -8])
    domains_upper_bound = np.array([8, 8])
    # surrogate = GaussianProcessRegression()
    surrogate = VanillaGaussianProcess
    # sampler = ProbabilisticSampler(
    #     succeeding_hartman, surrogate, domains_lower_bound, domains_upper_bound
    # )

    sampler = FuzzyLolaVoronoi(
        peaks_2d_multivariate, surrogate, domains_lower_bound, domains_upper_bound
    )
    # Run using python3 surrogate_load_test.py <path_to_surrogates_folder>
    surrogates_folder_path = Path(sys.argv[1])
    test_points_x, test_points_y = hypercube_initialization(sampler, 50)
    sampler.load_surrogates(surrogates_folder_path)
    print(sampler.predict(test_points_x))
    print(test_points_y)


if __name__ == '__main__':
    main()
