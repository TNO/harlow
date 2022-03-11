"""
Model builder for the IJsselbridge twin girder betti model.

Generates inputs for the pure Python version of the model.

Requires:

    * Julia
    * PyJulia

To install PuJulia:

import julia
julia.install(julia = "path_to_julia_executable")
"""

import os
import pickle

import numpy as np
from julia import Julia
from numba import jit

from modules.IJssel_bridge_utils import paths

# ============================================================================
# IMPORT JULIA
# ============================================================================
julia_bin_path = paths["julia"]
libopenlibm_path = julia_bin_path + "libopenlibm.DLL"
try:
    jl = Julia(runtime=julia_bin_path + "julia.exe")
except Exception:
    jl = Julia(runtime=julia_bin_path + "julia.exe")
jl.eval("using Libdl")
# libopenlibm_path = jl.eval('Libdl.dlpath(Base.libm_name)')
libdl_eval_string = "Libdl.dlopen(" + '"' + libopenlibm_path + '"' ")"
jl.eval(libdl_eval_string)
jl.eval("using LinearAlgebra")
from julia import Main  # noqa: E402,I202,I100

# ============================================================================
# FEM SOLVE FUNCTIONS
# ============================================================================


@jit(nopython=True)
def find_nearest(array, values):
    # By default in case of multiple equal values the first is returned
    # (see numpy documentation for argmin())
    array = np.asarray(array)
    values = np.atleast_1d(np.asarray(values))
    idx = np.zeros(len(values), dtype=np.int32)
    for i, val in enumerate(values):
        idx[i] = np.abs(array - val).argmin()
    return idx


def find_all(array, values):
    array = np.asarray(array)
    values = np.atleast_1d(np.asarray(values))
    idx = []
    for _i, val in enumerate(values):
        idx.append(np.where(array == val))
    return np.ravel(idx)


# ============================================================================
# IMPORT JULIA FUNCTIONS
# ============================================================================
julia_path_sections = paths["sections"]
julia_path_fem = paths["fem"]

Main.include(os.path.join(julia_path_sections, "girder_sections.jl"))
Main.include(os.path.join(julia_path_fem, "FEM_girder.jl"))
Main.include(os.path.join(julia_path_fem, "FEM_utils.jl"))
Main.include(os.path.join(julia_path_fem, "utils.jl"))


# ============================================================================
# INITIALIZATION
#
# TODO: Replace the model builder class with a function.
# ============================================================================
class model_builder_twin_girder_betti:
    def __init__(self, sensor_position, E, max_elem_length, additional_node_pos=None):
        self.sensor_position = sensor_position
        self.E = E
        self.max_elem_length = max_elem_length

        if additional_node_pos is None:
            additional_node_pos = []

        # Apply hinges for Betti influence line calculation.
        self.hinge_left_pos = self.sensor_position - 0.003

        # Sensor and hinge additional node positions
        self.additional_node_positions = (
            np.array(
                np.append(
                    [self.hinge_left_pos, self.sensor_position], additional_node_pos
                )
            )
            * 1e3
        )

        # ====================================================================
        # STRUCTURE PROPERTIES FROM JULIA MODEL
        # ====================================================================

        # Get structural properties
        self.section_properties = Main.girder_sections(
            max_elem_length=self.max_elem_length,
            additional_node_positions=self.additional_node_positions,
            consider_K_braces=True,
        )

        # Copy from dict
        self.support_xs = self.section_properties["support_xs"] / 1e3
        self.node_xs = self.section_properties["node_xs"] / 1e3
        self.K_brace_xs = self.section_properties["K_brace_xs"] / 1e3
        self.elem_c_ys = self.section_properties["elem_c_ys"] / 1e3
        self.elem_h_ys = self.section_properties["elem_h_ys"] / 1e3
        self.elem_I_zs = self.section_properties["elem_I_zs"] / 1e12

        # Element stiffness
        self.elem_EIs = self.elem_I_zs * self.E
        self.W_bot_temp = self.elem_I_zs / (self.elem_h_ys - self.elem_c_ys)
        self.elem_W_bottom = np.repeat(self.W_bot_temp, 2)
        self.nelems = len(self.elem_EIs)

        # Assemble K
        (
            self.Ks0,
            self.nodes,
            self.idx_keep,
            self.lin_idx_springs,
        ) = Main.fem_general_twin_girder(
            self.node_xs,
            self.elem_EIs,
            self.support_xs,
            spring_positions=self.K_brace_xs,
            spring_stiffnesses=np.zeros(np.shape(self.K_brace_xs)),
            left_hinge_positions=[sensor_position],
        )

        # NOTE : This gives the closest node.
        # It does not ensure that sensor_pos == node_pos
        self.sensor_node = find_nearest(self.nodes[:, 0], sensor_position)[0]
        self.hinge_node = self.sensor_node
        self.hinge_left_node = find_nearest(self.nodes[:, 0], self.hinge_left_pos)[0]
        self.W_sensor = self.elem_W_bottom[self.sensor_node]

        # Convert the flat, 1-indexed indices from Julia into numpy indices
        K_shape = np.shape(self.Ks0)
        lin_idx_spring_off_diag = np.asarray(self.lin_idx_springs["off_diag"]) - 1
        lin_idx_spring_diag = np.asarray(self.lin_idx_springs["diag"]) - 1
        self.idx_spring_off_diag = np.unravel_index(lin_idx_spring_off_diag, K_shape)
        self.idx_spring_diag = np.unravel_index(lin_idx_spring_diag, K_shape)

        # TODO : Use sparse matrices for Ks to improve performance.
        self.nnode = np.shape(self.nodes)[0]
        self.ndof = np.shape(self.Ks0)[0]
        self.support_nodes = find_all(self.nodes[:, 0], self.support_xs)

        # Boolean masking for applying rotational stiffnesses to Ks
        self.Ks_stiff_mask = np.asarray(np.zeros(2 * self.nnode), dtype=bool)
        self.Ks_stiff_mask[self.support_nodes * 2 + 1] = 1
        self.Ks_stiff_mask = self.Ks_stiff_mask[self.idx_keep]

        self.model_properties = {
            "hinge_left_pos": self.hinge_left_pos,
            "support_xs": self.support_xs,
            "node_xs": self.node_xs,
            "elem_c_ys": self.elem_c_ys,
            "elem_h_ys": self.elem_h_ys,
            "elem_I_zs": self.elem_I_zs,
            "Ks0": self.Ks0,
            "nodes": self.nodes,
            "idx_keep": self.idx_keep,
            "lin_idx_springs": self.lin_idx_springs,
        }

    def save_model(self):
        """
        Returns the dictionary of model properties needed for the
        pure Python model
        """
        return self.model_properties


# ===========================================================================
# Build model
# ===========================================================================
if __name__ == "__main__":

    # Sensor names and positions
    sensor_names = [
        "H1_S",
        "H2_S",
        "H3_S",
        "H4_S",
        "H5_S",
        "H7_S",
        "H8_S",
        "H9_S",
        "H10_S",
    ]
    sensor_positions = [
        20.42,
        34.82,
        47.700,
        61.970,
        68.600,
        96.800,
        113.9,
        123.900,
        147.500,
    ]

    # Set parameters
    E = 210e6
    max_elem_length = 2.0 * 1e3

    # Loop over sensor names and positions and generate models
    model_dict = {}
    for idx_sensor, sensor_x in enumerate(sensor_positions):

        # Define model
        model = model_builder_twin_girder_betti(sensor_x, E, max_elem_length)
        model_dict[sensor_names[idx_sensor]] = model.save_model()

    # Sensor names and positions
    model_dict["sensor_names"] = sensor_names
    model_dict["sensor_positions"] = sensor_positions

    # Pickle
    fname = (
        paths["model_data"]
        + f"model_dict#max_elem_length={max_elem_length}#E={E}.pickle"
    )
    with open(fname, "wb") as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
