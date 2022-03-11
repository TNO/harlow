"""
Creates a dictionary with truck information for influence line calculation. Two
setups are included: the single sensor TNO measurement setup used initially
and the additional fugro sensors.

STRUCTURE:
    truck_data -> series -> lane -> data
"""

import numpy as np

# ============================================================================
# Assemble dictionaries
#
# Create a dictionary to store all the info. This will include the truck data
# for the two measurement series:
#   * TNO measurements from strain gauge at the second span
#   * Fugro measurements from multiple strain gauges
#
# STRUCTURE:
#     truck_data -> series -> lane -> data
# ============================================================================

keys_data = ["force", "axle_dist", "center_z", "wheel_z_dist"]
keys_lane = ["left", "right"]
keys_series = ["TNO", "fugro", "uniform"]

# Create the nested dictionary
truck_data = dict.fromkeys(keys_series)
for _i, key_i in enumerate(truck_data.keys()):
    truck_data[key_i] = dict.fromkeys(keys_lane)
    for _j, key_j in enumerate(truck_data[key_i]):
        truck_data[key_i][key_j] = dict.fromkeys(keys_data)

# ============================================================================
# TRUCK DATA
# ============================================================================

right_truck_force = np.array(
    [
        [57.518 / 2, 57.518 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
    ]
)
right_truck_axle_dist_x = np.array([1.94, 2.09, 1.345, 1.25])
right_truck_center_z = 5.700 / 2 - 3.625 / 2
right_truck_wheel_z_dist = 2.1500
right_truck_num_axle = np.shape(right_truck_force)[0]

left_truck_force = np.array(
    [
        [58.86 / 2, 58.86 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
    ]
)
left_truck_axle_dist_x = np.array([2.0, 1.82, 1.82, 1.82])
left_truck_center_z = 5.700 / 2 + 3.625 / 2
left_truck_wheel_z_dist = right_truck_wheel_z_dist
left_truck_num_axle = np.shape(left_truck_force)[0]

right_ax_flatlist = np.cumsum(right_truck_axle_dist_x)
right_ax_flatlist = np.insert(right_ax_flatlist, 0, 0)

right_truck_x = np.repeat(right_ax_flatlist, 2)
right_truck_z = np.hstack(
    (
        right_truck_center_z
        - right_truck_wheel_z_dist / 2 * np.ones(right_truck_num_axle),
        right_truck_center_z
        + right_truck_wheel_z_dist / 2 * np.ones(right_truck_num_axle),
    )
)

left_ax_flatlist = np.cumsum(left_truck_axle_dist_x)
left_ax_flatlist = np.insert(left_ax_flatlist, 0, 0)

left_truck_x = np.repeat(left_ax_flatlist, 2)
left_truck_z = np.hstack(
    (
        left_truck_center_z
        - left_truck_wheel_z_dist / 2 * np.ones(left_truck_num_axle),
        left_truck_center_z
        + left_truck_wheel_z_dist / 2 * np.ones(left_truck_num_axle),
    )
)

# Assuming that z = 0 at the center of the bridge
z_truck_r_wheel_r = right_truck_center_z - right_truck_wheel_z_dist / 2
z_truck_r_wheel_l = right_truck_center_z + right_truck_wheel_z_dist / 2
z_truck_l_wheel_r = left_truck_center_z - left_truck_wheel_z_dist / 2
z_truck_l_wheel_l = left_truck_center_z + left_truck_wheel_z_dist / 2

# Fugro truck forces: the total weight is given as 50.420 tonnes.
# The force is calculated as Wtot * g * 0.12 for the first axle and
# Wtot * g * 0.22 for the other axles
truck_axle_dist_x_fugro = np.array([2.06, 1.83, 1.82, 1.82])
truck_force_fugro = np.array(
    [
        [59.35 / 2, 59.35 / 2],
        [108.82 / 2, 108.82 / 2],
        [108.82 / 2, 108.82 / 2],
        [108.82 / 2, 108.82 / 2],
        [108.82 / 2, 108.82 / 2],
    ]
)


# ============================================================================
# Pass the values to the dictionary
# ============================================================================
truck_data["TNO"]["left"]["force"] = left_truck_force
truck_data["TNO"]["left"]["axle_dist"] = left_truck_axle_dist_x
truck_data["TNO"]["left"]["center_z"] = left_truck_center_z
truck_data["TNO"]["left"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["TNO"]["left"]["z_wheel_r"] = z_truck_l_wheel_r
truck_data["TNO"]["left"]["z_wheel_l"] = z_truck_l_wheel_l

truck_data["TNO"]["right"]["force"] = right_truck_force
truck_data["TNO"]["right"]["axle_dist"] = right_truck_axle_dist_x
truck_data["TNO"]["right"]["center_z"] = right_truck_center_z
truck_data["TNO"]["right"]["wheel_z_dist"] = right_truck_wheel_z_dist
truck_data["TNO"]["right"]["z_wheel_r"] = z_truck_r_wheel_r
truck_data["TNO"]["right"]["z_wheel_l"] = z_truck_r_wheel_l

truck_data["fugro"]["left"]["force"] = truck_force_fugro
truck_data["fugro"]["left"]["axle_dist"] = truck_axle_dist_x_fugro
truck_data["fugro"]["left"]["center_z"] = left_truck_center_z
truck_data["fugro"]["left"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["fugro"]["left"]["z_wheel_r"] = z_truck_l_wheel_r
truck_data["fugro"]["left"]["z_wheel_l"] = z_truck_l_wheel_l

truck_data["fugro"]["right"]["force"] = truck_force_fugro
truck_data["fugro"]["right"]["axle_dist"] = truck_axle_dist_x_fugro
truck_data["fugro"]["right"]["center_z"] = right_truck_center_z
truck_data["fugro"]["right"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["fugro"]["right"]["z_wheel_r"] = z_truck_r_wheel_r
truck_data["fugro"]["right"]["z_wheel_l"] = z_truck_r_wheel_l
