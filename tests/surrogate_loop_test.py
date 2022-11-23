# TODO: remove/move when finished building surrogating loop
from pathlib import Path

import numpy as np
from loguru import logger
from sampling import FuzzyLolaVoronoi

from harlow.sampling import LolaVoronoi, Sampler
from harlow.surrogating.surrogate_model import VanillaGaussianProcess
from harlow.utils.helper_functions import latin_hypercube_sampling
from tests.offload_hartmann import succeeding_hartman

# def target_func_jo(X) -> ndarray:
#     for point in X:
#         for dim in point:


def hypercube_initialization(
    sampler: Sampler, n_initial_points: int
) -> (np.ndarray, np.ndarray):
    # latin hypercube sampling to get the initial sample of points
    points_x = latin_hypercube_sampling(
        n_sample=n_initial_points,
        domain_lower_bound=sampler.domain_lower_bound,
        domain_upper_bound=sampler.domain_upper_bound,
    )
    # evaluate the target function
    new_points_x, points_y = sampler.exec_target_function(points_x)
    return new_points_x, points_y


# To install the offloader: pip install offloader --extra-index-url
# https://ci.tno.nl/gitlab/api/v4/projects/8033/packages/pypi/simple
def offloaded_hartman(x: np.ndarray) -> np.ndarray:
    from offloader import Offloader, OffloadVector

    def pre(task_folder: Path, x: np.ndarray):
        with open(task_folder / "x.npy", "wb") as f:
            np.save(f, x)

    def post(task_folder: Path, x: np.ndarray):
        with open(task_folder / "y.npy", "rb") as f:
            y = np.load(f)
        return y

    url = "offload.dt4si.nl"

    offloader = Offloader(url, "api/v1", offload_folder="tmp")
    task_resources = {"requests": {"memory": "100Mi", "cpu": "3500m"}}
    # vector = []
    # for x_single in x:
    #     vector.append({'x': x_single})
    # print(vector)
    vector = [{"x": x}]
    command = "ls && pip install numpy && python3 offload_hartmann.py"
    off = OffloadVector(
        offloader,
        pre,
        post,
        command,
        "python:3",
        vector,
        task_resources=task_resources,
        local=False,
    )
    off.add_file("offload_hartmann.py", des_path="")
    off.get_file("y.npy")
    result = off.run()
    return result[0]


def main():
    domains_lower_bound = np.array([0, 0, 0, 0, 0, 0])
    domains_upper_bound = np.array([1, 1, 1, 1, 1, 1])

    # surrogate = GaussianProcessRegression()
    surrogate = VanillaGaussianProcess
    sampler = FuzzyLolaVoronoi(
        succeeding_hartman, surrogate, domains_lower_bound, domains_upper_bound
    )
    # sampler = LolaVoronoi(
    #     succeeding_hartman, surrogate, domains_lower_bound, domains_upper_bound
    # )
    # sampler = LatinHypercube(
    #     succeeding_hartman, surrogate, domains_lower_bound, domains_upper_bound
    # )
    # TODO: Does not work yet, _best_new_points has to be properly implemented
    # rmse_criterium = 0.05
    # sampler = ProbabilisticSampler(
    #     succeeding_hartman,
    #     surrogate,
    #     domains_lower_bound,
    #     domains_upper_bound,
    #     stopping_score=rmse_criterium,
    # )

    # Create initial set
    points_x, points_y = hypercube_initialization(sampler, 20)
    sampler.set_initial_set(points_x, points_y)
    logger.info("initial set created")

    # Create test set
    test_points_x, test_points_y = hypercube_initialization(sampler, 50)
    sampler.set_test_set(test_points_x, test_points_y)
    logger.info("test set created")

    sampler.surrogate_loop(10, 500)

    # TODO: how/when to save/store results.
    print("doneeee")


if __name__ == "__main__":
    main()
