"""
Rectangularly bounded Voronoi diagram in 2D.

The code is largely based on: https://stackoverflow.com/a/33602171/4063376
It is changed to match the order of `filtered_regions` with the order of `towers`.
"""
import sys

import numpy as np
import scipy as sp
import scipy.spatial

eps = sys.float_info.epsilon


def in_box(towers, bounding_box):
    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]
        ),
        np.logical_and(
            bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]
        ),
    )


def bounded_voronoi_2d(towers: np.ndarray, bounding_box: list):
    # bounding_box: [x_min, x_max, y_min, y_max]
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # Filter regions
    regions = []
    idx_filtered_regions = []
    for ii, region in enumerate(vor.regions):
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bounding_box[0] - eps <= x <= bounding_box[1] + eps
                    and bounding_box[2] - eps <= y <= bounding_box[3] + eps
                ):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
            idx_filtered_regions.append(ii)

    # until we can get the order right..
    # unordered_regions = regions
    # unordered_point_regions = vor.point_region[: towers.shape[0]]
    # ordered_regions = [None] * len(regions)
    # for ii, unordered_point_region in enumerate(unordered_point_regions):
    #     idx = int(np.where(unordered_point_region == idx_filtered_regions)[0])
    #     ordered_regions[ii] = unordered_regions[idx]
    ordered_regions = regions

    vor.filtered_points = points_center
    vor.filtered_regions = ordered_regions
    return vor
