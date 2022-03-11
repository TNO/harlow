import os
import warnings

# Silence Julia FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# ===================================================================
# PATHS
# ===================================================================

# Path to Julia excecutable. Must use forward slashes and end in bin/
path_julia = "C:/Users/kounei/AppData/Local/Programs/Julia-1.6.2/bin/"

# Get absolute path to __init__.py
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

# Define relative paths
path_sections = os.path.join(
    ROOT_DIR, "model_builder\\continuous_girder\\IJssel_bridge\\"
)
path_fem = os.path.join(ROOT_DIR, "model_builder\\continuous_girder\\")
meas_path = os.path.join(ROOT_DIR, "measurements\\")
data_path = os.path.join(ROOT_DIR, "model\\model_data\\")

paths = {
    "root": ROOT_DIR + "\\",
    "julia": path_julia,
    "sections": path_sections,
    "fem": path_fem,
    "measurements": meas_path,
    "model_data": data_path,
}
