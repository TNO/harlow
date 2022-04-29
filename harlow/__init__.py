from typing import List, Union

import numpy as np

# TODO
# Version number changed automatically using python-semantic-release
__version__ = "0.0.1"

# ---------------------------------
# Types
# ---------------------------------
REAL_TYPE = Union[float, int]
REAL_VECT_TYPE = Union[float, int, np.ndarray, List[REAL_TYPE]]

# ---------------------------------
# Plotting
# ---------------------------------
RANGE_FACECOLOR = "blue"
PLOT_STYLE = "seaborn-white"
DPI = 400
