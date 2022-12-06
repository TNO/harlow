import numpy as np


class StepInfo:
    def __init__(
        self, x: np.ndarray, y: np.ndarray, score, target_func_time, gen_time, fit_time
    ):
        # TODO check if deepcopy is needed.
        self.x = x.tolist()
        self.y = y.tolist()
        self.score = score
        self.target_func_time = target_func_time
        self.gen_time = gen_time
        self.fit_time = fit_time
