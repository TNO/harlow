"""
IJsselbridge twin girder Euler-Bernoulli beam model.

This modified model does not have a Julia dependency, but instead
loads the pre-computed stiffness matrix.
"""

import pickle

# ============================================================================
# IMPORTS
# ============================================================================
from copy import copy, deepcopy

import numpy as np
from numba import jit
from scipy.linalg import cho_factor, cho_solve

from harlow.utils.examples.model_data import truck_data


# ============================================================================
# FEM SOLVE FUNCTIONS
# ============================================================================
def lateral_load_func(z, c):
    z = np.asarray(z)

    # If c is a single value, assumes its a linear coefficient. otherwise its
    # assumed to be the n coefficients of a polynomial
    c = np.atleast_1d(c)
    if (not np.shape(c)) or (len(c) == 1):
        if not np.isscalar(c):
            return c[0] * (z - 2.85) + 0.5
        else:
            return c * (z - 2.85) + 0.5
    else:
        return np.polyval(c, z - 2.85)


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


# @jit(nopython = True)
def betti(nnode, cho_factor, W_sensor, hinge_node, hinge_left_node, fs, idx_keep):

    u = np.zeros(2 * nnode)

    # Note: Given equal Ks and fs, both the cholesky and conventional solve
    # will give slightly different results than the corresponding Julia
    # solve: us = Ks \ fs.
    us = cho_solve(cho_factor, fs, check_finite=False)

    u[idx_keep] = us
    r_hinge = u[hinge_left_node * 2 + 1] - u[hinge_node * 2 + 1]

    # scale to unit hinge rotation
    m_b_1 = 1 / r_hinge * u[0::4]
    m_b_2 = 1 / r_hinge * u[2::4]

    return np.asarray([m_b_1, m_b_2]) / W_sensor


def solve_inf_line(m_b, axle_pos_x, node_xs, loads):
    stress_1 = loads[:] * np.interp(axle_pos_x[:], node_xs, m_b[0])
    stress_2 = loads[:] * np.interp(axle_pos_x[:], node_xs, m_b[1])
    return np.sum(stress_1, axis=1), np.sum(stress_2, axis=1)


# ====================================================================
# INITIALIZATION
# ====================================================================
class IJssel_bridge_model:
    def __init__(
        self,
        sensor_name,
        E,
        max_elem_length,
        truck_load="TNO",
    ):
        # ====================================================================
        # INITIALIZATION
        # ====================================================================
        self.sensor_name = sensor_name
        self.max_elem_length = max_elem_length
        self.truck_load = truck_load
        self.E = E

        # Load model data
        fname = f"model_dict#max_elem_length={self.max_elem_length}#E={self.E}.dat"

        with open(fname, "rb") as handle:
            model_properties = pickle.load(handle)

        # Get sensor names and positions
        self.sensor_names = model_properties["sensor_names"]
        self.sensor_positions = model_properties["sensor_positions"]

        # Get sensor index and position for this model
        idx_sensor = model_properties["sensor_names"].index(self.sensor_name)
        self.sensor_position = model_properties["sensor_positions"][idx_sensor]

        # Check that the requested sensor exists
        if self.sensor_name not in self.sensor_names:
            raise ValueError(
                f"Specified sensor name {self.sensor_name} not found in"
                f" {self.sensor_names}"
            )

        # Keep model properties for requested sensor
        self.model_properties = model_properties[self.sensor_name]

        # ====================================================================
        # STRUCTURE PROPERTIES FROM JULIA MODEL
        # ====================================================================

        self.hinge_left_pos = self.model_properties["hinge_left_pos"]
        self.support_xs = self.model_properties["support_xs"]
        self.node_xs = self.model_properties["node_xs"]
        self.elem_c_ys = self.model_properties["elem_c_ys"]
        self.elem_h_ys = self.model_properties["elem_h_ys"]
        self.elem_I_zs = self.model_properties["elem_I_zs"]
        self.nodes = self.model_properties["nodes"]

        # Element stiffness
        self.elem_EIs = self.elem_I_zs * self.E
        self.W_bot_temp = self.elem_I_zs / (self.elem_h_ys - self.elem_c_ys)
        self.elem_W_bottom = np.repeat(self.W_bot_temp, 2)
        self.nelems = len(self.elem_EIs)

        # NOTE: This gives the closest node.
        # It does not ensure that sensor_pos == node_pos
        self.sensor_node = find_nearest(self.nodes[:, 0], self.sensor_position)[0]
        self.hinge_node = self.sensor_node
        self.hinge_left_node = find_nearest(self.nodes[:, 0], self.hinge_left_pos)[0]
        self.W_sensor = self.elem_W_bottom[self.sensor_node]

        # Load stiffness matrix, nodes, spring positions
        self.Ks0 = self.model_properties["Ks0"]
        self.idx_keep = self.model_properties["idx_keep"]
        self.lin_idx_springs = self.model_properties["lin_idx_springs"]

        # Convert the flat, 1-indexed indices from Julia into numpy indices
        K_shape = np.shape(self.Ks0)
        lin_idx_spring_off_diag = np.asarray(self.lin_idx_springs["off_diag"]) - 1
        lin_idx_spring_diag = np.asarray(self.lin_idx_springs["diag"]) - 1
        self.idx_spring_off_diag = np.unravel_index(lin_idx_spring_off_diag, K_shape)
        self.idx_spring_diag = np.unravel_index(lin_idx_spring_diag, K_shape)

        # TODO: Use sparse matrices for Ks to improve performance.
        self.nnode = np.shape(self.nodes)[0]
        self.ndof = np.shape(self.Ks0)[0]
        self.support_nodes = find_all(self.nodes[:, 0], self.support_xs)

        # Boolean masking for applying rotational stiffnesses to Ks
        self.Ks_stiff_mask = np.asarray(np.zeros(2 * self.nnode), dtype=bool)
        self.Ks_stiff_mask[self.support_nodes * 2 + 1] = 1
        self.Ks_stiff_mask = self.Ks_stiff_mask[self.idx_keep]

        # Copy and factorize Ks. Ks and K are related as follows:
        # # Ks = K[idx_keep][:, idx_keep]
        self.Ks = deepcopy(self.Ks0)
        self.Ks_factor = cho_factor(self.Ks)

        # Precalculate unit load pair for betti theorem
        f = np.zeros(self.nnode * 2)
        f[self.hinge_left_node * 2 + 1] = 1e6
        f[self.hinge_node * 2 + 1] = -1e6
        self.fs = f[self.idx_keep]

        # Precalculate unit influence line
        self.unit_il = betti(
            self.nnode,
            self.Ks_factor,
            self.W_sensor,
            self.sensor_node,
            self.hinge_left_node,
            self.fs,
            self.idx_keep,
        )

        # ====================================================================
        # TRUCK DATA
        #
        # The truck data corresponding to 'TNO' or 'fugro' measurements is
        # loaded. Default is 'TNO'.
        # ====================================================================

        # Wheel z positions
        z_truck_r_wheel_r = truck_data[truck_load]["right"]["z_wheel_r"]
        z_truck_r_wheel_l = truck_data[truck_load]["right"]["z_wheel_l"]
        z_truck_l_wheel_r = truck_data[truck_load]["left"]["z_wheel_r"]
        z_truck_l_wheel_l = truck_data[truck_load]["left"]["z_wheel_l"]

        # Forces
        right_truck_force = truck_data[truck_load]["right"]["force"]
        left_truck_force = truck_data[truck_load]["left"]["force"]

        # Axle distances
        right_truck_axle_dist_x = truck_data[truck_load]["right"]["axle_dist"]
        left_truck_axle_dist_x = truck_data[truck_load]["left"]["axle_dist"]

        # Create objects holding the parameter values for left, right and double
        # case. This is to avoid the if-else statements that were used previously.
        self.truck_z = {
            "left": [z_truck_l_wheel_r, z_truck_l_wheel_l],
            "right": [z_truck_r_wheel_r, z_truck_r_wheel_l],
            "double": [
                z_truck_r_wheel_r,
                z_truck_r_wheel_l,
                z_truck_l_wheel_r,
                z_truck_l_wheel_l,
            ],
        }
        self.truck_force = {
            "left": np.transpose(left_truck_force),
            "right": np.transpose(right_truck_force),
            "double": np.vstack(
                (np.transpose(right_truck_force), np.transpose(left_truck_force))
            ),
        }
        self.truck_axle_dist = {
            "left": np.tile(left_truck_axle_dist_x, (2, 1)),
            "right": np.tile(right_truck_axle_dist_x, (2, 1)),
            "double": np.vstack(
                (
                    np.tile(right_truck_axle_dist_x, (2, 1)),
                    np.tile(left_truck_axle_dist_x, (2, 1)),
                )
            ),
        }

        # ====================================================================
        # PRECALCULATE LOAD PATHS AND INFLUENCE LINES
        #
        # NOTES:
        #   * This is not optimized for speed. This should not be a problem
        #   since this only runs once on model initialization.
        # ====================================================================

        # Dictionaries to store precalculation output
        all_paths = ["right", "left", "double"]
        self._truck_load = dict.fromkeys(all_paths)
        self.axle_pos_x = dict.fromkeys(all_paths)
        self.loaded_nodes = dict.fromkeys(all_paths)
        self.loads = dict.fromkeys(all_paths)

        for _i, lane in enumerate(all_paths):

            # Assemble the vectors of loads and axle distances
            loads = np.atleast_2d(self.truck_force[lane])
            axle_dist_x = np.atleast_2d(self.truck_axle_dist[lane])
            ax_dist = np.cumsum(axle_dist_x[0, :])
            load_i = loads[0, :]

            ax_pos = []
            l_node = []
            load = []
            for j in range(len(self.node_xs)):
                # For each load position sum the corresponding moments from the il and
                # calculate the stress at the sensor position
                curr_pos = np.append(self.node_xs[j], self.node_xs[j] - ax_dist)
                ax_pos.append(curr_pos)
                l_node.append(find_nearest(self.node_xs, curr_pos))

                # Make load zero for points that have not fully entererd the st
                # ructure yet
                temp_load = copy(load_i)
                temp_load[curr_pos < self.node_xs[0]] = 0
                load.append(temp_load)

            # Append to global list
            self.axle_pos_x[lane] = ax_pos
            self.loaded_nodes[lane] = l_node
            self.loads[lane] = load
            self._truck_load[lane] = np.tile(
                solve_inf_line(
                    self.unit_il, self.axle_pos_x[lane], self.node_xs, self.loads[lane]
                ),
                (len(self.truck_z[lane]), 1),
            )

    # ========================================================================
    # DEFINE FUNCTIONS
    # ========================================================================

    def il_stress_truckload(self, c, lane: str, Kr=None, Kv=None):
        """
        Calculate influence line

        Args:
           c: Lateral load coefficient
           lane: String, can be "left" or "right"
           Kr: Vector with 12 elements containing the
           support rotational stiffnesses. If omitted, the
           last calculated rotational stiffness vector is used
           Kv: Scalar vertical spring stiffness
        """
        # Check if stiffnesses have been updated, if yes, update the il
        if (lane == "double") or (lane == "Double"):
            raise ValueError("Not implemented")

        if (Kr is not None) or (Kv is not None):

            # Add new stiffnesses to stiffness matrix
            self.Ks = copy(self.Ks0)
            self.Ks[self.Ks_stiff_mask, self.Ks_stiff_mask] += np.asarray(Kr)
            self.Ks[self.idx_spring_diag] += Kv
            self.Ks[self.idx_spring_off_diag] -= Kv

            # Factorize
            Ks_factor = cho_factor(self.Ks)

            # Calculate new influence line
            self.unit_il = betti(
                self.nnode,
                Ks_factor,
                self.W_sensor,
                self.sensor_node,
                self.hinge_left_node,
                self.fs,
                self.idx_keep,
            )
        # The returned stress arrays contain two stress influence lines for each
        # truck z value. lines [0::2] correspond to the right girder stress and
        # lines [1::2] correspond to the left girder stress
        self._truck_load[lane] = np.tile(
            solve_inf_line(
                self.unit_il, self.axle_pos_x[lane], self.node_xs, self.loads[lane]
            ),
            (len(self.truck_z[lane]), 1),
        )

        stress = np.zeros(len(self.node_xs))
        for i, z in enumerate(self.truck_z[lane]):
            stress += lateral_load_func(z, c) * self._truck_load[lane][2 * i, :]
            stress += lateral_load_func(z, -c) * self._truck_load[lane][2 * i + 1, :]

        return stress / 1000  # Return stress in MPa
