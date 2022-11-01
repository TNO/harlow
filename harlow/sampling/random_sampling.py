import os
import pickle
import time
from typing import Callable

import numpy as np
from loguru import logger

from harlow.sampling.sampling_baseclass import Sampler
from harlow.utils.helper_functions import evaluate, latin_hypercube_sampling
from harlow.utils.log_writer import write_scores, write_timer
from harlow.utils.metrics import rmse


class LatinHypercube(Sampler):
    """ Latin Hypercube Sampler

    Lathin hypercube sampler is a space filling random sampler. This sampler
    is non-sequenctial.

    """
    def __init__(
        self,
        target_function,
        surrogate_model,
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
        super(LatinHypercube, self).__init__(
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
        n_new_points_per_iteration: int = 10,
        stopping_criterium: float = None,
    ):

        # ..........................................
        # This sampler is non iterative. This means there is not iteration that
        # adds n new samples. Instead, the sampler works with steps, testing
        # different dataset sizes, which are newly sampled each time.
        # max_n_iterations means here the number of steps
        # n_new_points_per_iterations means here the step size
        # e.g. 10 X 20, means that sets of 20, 40, 60, ..., 200 are created,
        # unless the stopping criterium is earlier reached.
        # ..........................................
        start_time = time.time()
        sets = np.linspace(n_new_points_per_iteration,
                           n_new_points_per_iteration*max_n_iterations,
                           n_new_points_per_iteration, dtype=int)

        points_x = self.fit_points_x
        points_y = self.fit_points_y

        self.step_gen_time.append(time.time() - start_time)

        surrogate_model = self.surrogate_model()
        surrogate_model.fit(points_x, points_y)
        logger.info(f"Fitted a new surrogate model in {time.time() - start_time} sec.")

        score = evaluate(
            self.logging_metrics,
            self.test_points_x,
            self.test_points_y,
        )
        self.score = score[self.evaluation_metric.__name__]
        self.step_x.append(points_x)
        self.step_y.append(points_y)
        self.step_score.append(score)
        self.step_iter.append(0)
        self.step_fit_time.append(time.time() - start_time)

        iteration = 0

        for s in sets:

            start_time = time.time()
            points_x = latin_hypercube_sampling(
                n_sample=s,
                domain_lower_bound=self.domain_lower_bound,
                domain_upper_bound=self.domain_upper_bound,
            )
            self.step_gen_time.append(time.time() - start_time)

            points_y = self.target_function(points_x).reshape((-1, 1))


            start_time = time.time()
            surrogate_model.fit(points_x, points_y)
            logger.info(f"Fitted a surrogate model in {time.time() - start_time} sec.")
            self.step_fit_time.append(time.time() - start_time)

            self.fit_points_x = points_x
            self.fit_points_y = points_y

            # Re-evaluate the surrogate model.
            predicted_y = self.surrogate_model.predict(
                self.test_points_x, as_array=True
            )
            score = evaluate(self.logging_metrics, self.test_points_y, predicted_y)

            self.score = score[self.evaluation_metric.__name__]
            logger.info(f"Score {score[self.evaluation_metric.__name__]}")
            self.step_x = points_x
            self.step_y = points_y
            self.step_score.append(score[self.evaluation_metric.__name__])
            self.step_iter.append(iteration + 1)
            timing_dict = {
                "Gen time": self.step_gen_time[iteration + 1],
                "Fit time": self.step_fit_time[iteration + 1],
            }
            write_scores(self.writer, score, iteration + 1)
            write_timer(self.writer, timing_dict, iteration + 1)

            iteration += 1

            if self.score[0] <= stopping_criterium:
                break

        save_name = self.run_name + "_{}_iters.pkl".format(self.iterations)
        save_path = os.path.join(self.save_dir, save_name)
        # Save model if converged
        with open(save_path, "wb") as file:
            pickle.dump(self.surrogate_model, file)

        logger.info(f"Algorithm converged in {iteration} iterations")
        logger.info(f"Algorithm converged with score {score}")
        self.writer.close()

        return self.fit_points_x, self.fit_points_y
