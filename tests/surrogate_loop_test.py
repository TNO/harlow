# TODO: remove/move when finished building surrogating loop
import numpy as np

from harlow.sampling import FuzzyLolaVoronoi
from harlow.surrogating import GaussianProcessRegression
from harlow.utils.test_functions import peaks_2d


# def target_func_jo(X) -> ndarray:
#     for point in X:
#         for dim in point:





def main():
    domains_lower_bound = np.array([-8, -8])
    domains_upper_bound = np.array([8, 8])

    surrogate = GaussianProcessRegression()
    sampler = FuzzyLolaVoronoi(peaks_2d, surrogate, domains_lower_bound, domains_upper_bound)
    sampler.surrogate_loop()

if __name__ == '__main__':
    main()