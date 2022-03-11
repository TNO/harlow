import os
import pickle

import numpy as np
import pandas as pd

from modules import paths

# ===========================================================================
# Model save path
# ===========================================================================
meas_path = os.path.join(paths["measurements"], "measurements_TNO.csv")


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================
def load_pickle(fname):
    with open(fname, "rb") as handle:
        data = pickle.load(handle)
    return data


def read_measurement_data(measurements_path=meas_path):
    meas = pd.read_csv(measurements_path)
    meas = meas.drop(meas.index[0])
    meas["x"] = pd.to_numeric(meas["x"])
    meas["stress"] = pd.to_numeric(meas["stress"])
    return meas


def interpolate_measurement_data(node_xs, meas):
    columns = ["x", "stress", "truck_position"]
    meas_interp = pd.DataFrame(columns=columns)
    for pos in np.unique(meas["truck_position"]):
        dfappend = pd.DataFrame(columns=columns)
        dfappend["stress"] = np.interp(
            node_xs,
            meas["x"][meas["truck_position"] == pos],
            meas["stress"][meas["truck_position"] == pos],
        )
        dfappend["x"] = node_xs
        dfappend["truck_position"] = np.repeat(pos, len(node_xs))

        meas_interp = meas_interp.append(dfappend)
    return meas_interp


def find_nearest(array, values):
    array = np.asarray(array)
    values = np.reshape(values, (-1, 1))
    idx = []
    for val in values:
        idx.append(np.abs(array - val).argmin())
    return idx


def load_meas(node_xs, sensor_name, meas_type):

    # Load measurement data and interpolate
    if meas_type == "TNO":
        measurements_path = os.path.join(paths["measurements"], "measurements_TNO.csv")
        meas_sensor = read_measurement_data(measurements_path)
    elif meas_type == "fugro":
        measurements_path = os.path.join(
            paths["measurements"], "measurements_Fugro.csv"
        )
        meas = read_measurement_data(measurements_path)
        meas_sensor = meas[meas["position"] == sensor_name]

    meas_interp = interpolate_measurement_data(node_xs, meas_sensor)

    # Interpolated measurement data as numpy array, accounting for different definition
    # of left and right between TNO and Fugro measurements
    if meas_type == "TNO":
        y_r = meas_interp["stress"][meas_interp["truck_position"] == "right"].to_numpy()
        y_l = meas_interp["stress"][meas_interp["truck_position"] == "left"].to_numpy()
        y_d = meas_interp["stress"][
            meas_interp["truck_position"] == "double"
        ].to_numpy()
    elif meas_type == "fugro":
        y_r = meas_interp["stress"][meas_interp["truck_position"] == "left"].to_numpy()
        y_l = meas_interp["stress"][meas_interp["truck_position"] == "right"].to_numpy()
        y_d = meas_interp["stress"][
            meas_interp["truck_position"] == "double"
        ].to_numpy()

    return y_l, y_r, y_d
