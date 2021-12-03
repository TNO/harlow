"""Corner plot visualization of R^n -> R functions. Plotting interesting 1D and 2D
sections."""
import copy
from typing import Callable, List

import labellines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from harlow import RANGE_FACECOLOR, REAL_TYPE, REAL_VECT_TYPE


def corner(
    func: Callable[[np.ndarray], np.ndarray],
    support_range: REAL_VECT_TYPE,
    title: str = "",
    n_discr: int = 40,
    fixed_var_in_support_range: REAL_TYPE = 0.5,
    func_label: str = "$f$",
    dim_labels: List[str] = None,
    iso_value: REAL_TYPE = None,
    rectangular_range: REAL_VECT_TYPE = None,
):
    """Corner plot of a function (`func`): interesting 1D and 2D sections.
    For each subplot, each not plotted independent variable (`x_i`) is fixed to a
    constant value: `x_i_fix = x_i_support_lower_bound + fixed_var_in_support_range *
    x_i_support_range`.
    """
    # .....................................
    # Initialize
    # .....................................
    n_dim = support_range.shape[1]
    fixed_vars = support_range[0, :] + fixed_var_in_support_range * np.diff(
        support_range, axis=0
    )

    z_mx_off_diag_all = np.empty((n_discr, n_discr, n_dim, n_dim))
    z_mx_off_diag_all[:] = np.nan
    x_mx_off_diag_all = copy.deepcopy(z_mx_off_diag_all)
    y_mx_off_diag_all = copy.deepcopy(z_mx_off_diag_all)
    z_mx_diag_all = np.empty((n_discr, n_dim, n_dim))
    z_mx_diag_all[:] = np.nan

    if dim_labels is None:
        dim_labels = [f"$x_{{{ii+1}}}$" for ii in range(n_dim)]

    # .....................................
    # Plot
    # .....................................
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(n_dim + 3, n_dim + 3))

    if n_dim == 1:
        axes = np.atleast_2d(axes)

    for row in range(n_dim):
        for col in range(n_dim):
            # turn off the upper triangle
            if row < col:
                axes[row, col].axis("off")

            # diagonal
            elif row == col:
                ax = axes[row, col]

                # print(f"row={row+1}/{n_dim}; \t col={col+1}/{n_dim} \t (diagonal)")
                y_labelpad = 15

                x_vec = np.linspace(
                    support_range[0, col], support_range[1, col], n_discr
                )
                all_x_vec = np.tile(fixed_vars, (n_discr, 1))

                all_x_vec[:, col] = x_vec
                z_vec = func(all_x_vec).ravel()

                z_mx_diag_all[:, row, col] = z_vec

                axes[col, col].plot(x_vec, z_vec, color="black")

                if iso_value is not None:
                    lh = ax.plot(
                        support_range[:, col], [iso_value] * 2, ls="--", color="red"
                    )
                    labellines.labelLines(lh, fontsize=7, outline_width=3)

                if rectangular_range is not None:
                    ax.axvspan(
                        rectangular_range[0, col],
                        rectangular_range[1, col],
                        alpha=0.2,
                        facecolor=RANGE_FACECOLOR,
                    )

                # left top corner
                if col == 0:
                    if n_dim == 1:
                        ax.set_xlabel(dim_labels[col])
                    else:
                        ax.set_xticks([])
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(func_label, rotation=0, labelpad=y_labelpad)

                # right bottom corner
                elif col == n_dim - 1:
                    ax.set_xlabel(dim_labels[-1])
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(func_label, rotation=0, labelpad=y_labelpad)

                # the rest
                else:
                    ax.set_xticks([])
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(func_label, rotation=0, labelpad=y_labelpad)

            # off diagonal
            else:
                # print(f"row={row+1}/{n_dim}; \t col={col+1}/{n_dim} \t (off-diagonal)")  # noqa E501
                y_labelpad = 10

                x_vec = np.linspace(
                    support_range[0, col], support_range[1, col], n_discr
                )
                y_vec = np.linspace(
                    support_range[0, row], support_range[1, row], n_discr
                )
                x_mx, y_mx = np.meshgrid(x_vec, y_vec)

                all_x_vec = np.tile(fixed_vars, (n_discr * n_discr, 1))
                all_x_vec[:, col] = x_mx.ravel()
                all_x_vec[:, row] = y_mx.ravel()

                z_vec = func(all_x_vec)
                z_mx = z_vec.reshape((n_discr, n_discr))

                z_mx_off_diag_all[:, :, row, col] = z_mx
                x_mx_off_diag_all[:, :, row, col] = x_mx
                y_mx_off_diag_all[:, :, row, col] = y_mx

                # left bottom corner
                if row == n_dim - 1 and col == 0:
                    axes[row, col].set_xlabel(dim_labels[0])
                    axes[row, col].set_ylabel(
                        dim_labels[-1], rotation=0, labelpad=y_labelpad
                    )

                # bottom edge x label
                elif row == n_dim - 1:
                    if col != 0:
                        axes[row, col].set_yticks([])
                    axes[row, col].set_xlabel(dim_labels[col])

                # left edge x label
                elif col == 0:
                    axes[row, col].set_xticks([])
                    axes[row, col].set_ylabel(
                        dim_labels[row], rotation=0, labelpad=y_labelpad
                    )

                # rest
                else:
                    axes[row, col].set_yticks([])
                    axes[row, col].set_xticks([])

            # Format the axes - for all plots
            axes[row, col].tick_params(labelsize=6)
            axes[row, col].xaxis.offsetText.set_fontsize(6)
            axes[row, col].yaxis.offsetText.set_fontsize(6)
            axes[row, col].ticklabel_format(scilimits=(-5, 5))

    # ----------------------------------------------------------------------------------
    # SET THE RANGES AND COLORS
    # ----------------------------------------------------------------------------------
    # get the range of the ordinate
    range_tot = [
        np.nanmin([np.nanmin(z_mx_diag_all), np.nanmin(z_mx_off_diag_all)]),
        np.nanmax([np.nanmax(z_mx_diag_all), np.nanmax(z_mx_off_diag_all)]),
    ]

    levels = np.linspace(
        signif_floor(range_tot[0]), signif_ceil(range_tot[1]), 10
    ).ravel()
    # padding
    r = range_tot[1] - range_tot[0]
    d = 0.05
    range_tot = [range_tot[0] - d * r, range_tot[1] + d * r]

    # Set axis ranges and colors
    for row in range(n_dim):
        for col in range(n_dim):
            ax = axes[row, col]
            # upper triangle
            if row < col:
                pass  # do nothing

            # diagonal
            elif row == col:
                ax.set_ylim([range_tot[0], range_tot[1]])

            # off diagonal
            else:
                x_mx = x_mx_off_diag_all[:, :, row, col]
                y_mx = y_mx_off_diag_all[:, :, row, col]
                z_mx = z_mx_off_diag_all[:, :, row, col]

                color = ax.contourf(
                    x_mx,
                    y_mx,
                    z_mx,
                    levels=levels,
                )

                # data range
                if rectangular_range is not None:
                    ax.add_patch(
                        Rectangle(
                            xy=(rectangular_range[0, col], rectangular_range[0, row]),
                            width=np.diff(rectangular_range[:, col])[0],
                            height=np.diff(rectangular_range[:, row])[0],
                            facecolor=RANGE_FACECOLOR,
                            edgecolor=RANGE_FACECOLOR,
                            ls="--",
                            alpha=0.3,
                        )
                    )

                # Add iso line at level `iso_value`
                if iso_value is not None:
                    iso_lines = iso_line_2d(
                        x_mx=x_mx,
                        y_mx=y_mx,
                        z_mx=z_mx,
                        iso_level=iso_value,
                    )
                    for iso_line in iso_lines:
                        lh = ax.plot(
                            iso_line[:, 0],
                            iso_line[:, 1],
                            ls="--",
                            color="red",
                            label=f"{iso_value:.2f}",
                        )
                        try:
                            labellines.labelLines(
                                lh, fontsize=6, backgroundcolor="none", outline_width=3
                            )
                        except Exception as e:
                            if str(e) == "x label location is outside data range!":
                                pass
                            else:
                                raise e

    # ----------------------------------------------------------------------------------
    # COLORBAR
    # ----------------------------------------------------------------------------------
    if n_dim > 1:
        y_labelpad = 10
        cax = plt.axes([0.85, 0.55, 0.025, 0.3])
        cb = fig.colorbar(color, cax=cax)
        cb.ax.set_ylabel(func_label, rotation=0, labelpad=y_labelpad)
        cb.ax.tick_params(labelsize=6)
        cb.ax.yaxis.get_offset_text().set_fontsize(6)
        cb.formatter.set_powerlimits((-5, 5))
        cb.update_ticks()
        if iso_value is not None:
            cb.ax.plot(
                [np.min(range_tot), np.max(range_tot)],
                [iso_value] * 2,
                ls="--",
                color="red",
            )
    else:
        plt.tight_layout()

    plt.suptitle(title)

    return fig, axes


def iso_line_2d(x_mx, y_mx, z_mx, iso_level):
    """Get coordinates for a 2d contour plot.
    Args:
        x_mx: grid coordinates along the x-axis.
        y_mx: grid coordinates along the y-axis.
        z_mx: grid coordinates along the z-axis.
        iso_level: z value for which the contour plot is obtained.
    Returns:
        p: array of coordinates of the credible region.
    """
    plt.ioff()
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(x_mx, y_mx, z_mx, levels=[iso_level])

    p = []
    for collection in cs.collections:
        for path in collection.get_paths():
            p.append(path.vertices)
    plt.close(fig_tmp)
    plt.ion()
    return p


def signif_ceil(x, digits: int = 3):
    """
    x = sci_coeff * 10**sci_exp
    """
    x = np.atleast_1d(x)
    idx = np.abs(x) < np.finfo(float).eps
    x[idx] = 3 * np.finfo(float).eps
    # scientific notation
    sci_exp = np.array([int(np.floor(np.log10(np.abs(x))))])
    sci_scale = 10.0 ** sci_exp
    sci_coeff = x / sci_scale

    # number of digits
    dig_scale = 10.0 ** digits

    sx = np.ceil(sci_coeff * dig_scale) * sci_scale / dig_scale
    return sx


def signif_floor(x, digits: int = 3):
    """
    x = sci_coeff * 10**sci_exp
    """
    x = np.atleast_1d(x)
    idx = np.abs(x) < np.finfo(float).eps
    x[idx] = 3 * np.finfo(float).eps
    # scientific notation
    sci_exp = np.array([int(np.floor(np.log10(np.abs(x))))])
    sci_scale = 10.0 ** sci_exp
    sci_coeff = x / sci_scale

    # number of digits
    dig_scale = 10.0 ** digits

    sx = np.floor(sci_coeff * dig_scale) * sci_scale / dig_scale
    return sx
