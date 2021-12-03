"""Visualization functions."""
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_cmap(n: int, name="hsv"):
    return plt.cm.get_cmap(name, n)


def plot_function(
    f: Callable,
    domain: np.ndarray,
    grid_size: int = 1000,
    plot_name: int = None,
    show: bool = True,
    save: bool = False,
):
    if domain.shape[0] > 2:
        raise NotImplementedError

    x0_range = domain[0]
    x1_range = domain[1]

    x1_vec = np.linspace(x0_range[0], x0_range[1], grid_size)
    x2_vec = np.linspace(x1_range[0], x1_range[1], grid_size)

    x1_mx, x2_mx = np.meshgrid(x1_vec, x2_vec)
    z_mx = f(
        np.hstack(
            (
                x1_mx.reshape(grid_size * grid_size, 1),
                x2_mx.reshape(grid_size * grid_size, 1),
            )
        )
    )
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    surface = ax.plot_trisurf(
        x1_mx,
        x2_mx,
        z_mx.reshape(grid_size, grid_size),
        cmap=cm.coolwarm,
        label=f.__name__ if plot_name is None else plot_name,
    )
    fig.colorbar(surface)

    if show:
        plt.show()
    if save:
        plt.savefig(f"{f.__name__ if plot_name is None else plot_name}.png")
    return plt


def plot_function_custom(
    f: Callable,
    x_mx: np.ndarray,
    y_vec: np.ndarray = None,
    plot_sample_locations: bool = False,
    plot_name: str = None,
    show: bool = True,
    save: bool = False,
    color: str = "b",
):

    if y_vec is None:
        y_vec = f(x_mx)

    fig = plt.figure()

    if len(x_mx.squeeze().shape) == 1:
        ax = fig.gca()
        ax.plot(x_mx, y_vec, label=f.__name__ if plot_name is None else plot_name)
    else:
        ax = plt.axes(projection="3d")
        surface = ax.plot_trisurf(
            x_mx[:, 0],
            x_mx[:, 1],
            y_vec,
            cmap=cm.coolwarm,
            label=f.__name__ if plot_name is None else plot_name,
        )
        fig.colorbar(surface)

    if plot_sample_locations:
        if len(x_mx.shape) == 1:
            ax.scatter(x_mx, y_vec, c=color, alpha=0.6)
        elif x_mx.shape[1] > 1:
            ax.scatter(x_mx[:, 0], x_mx[:, 1], y_vec, c=color, alpha=0.6)

    if show:
        plt.show()
    if save:
        plt.savefig(f"{f.__name__ if plot_name is None else plot_name}.png")
    return plt


def add_samples_to_plot(plot, x_mx: np.ndarray, y_vec: np.ndarray, color: str):
    ax = plot.gca()

    if x_mx.shape[1] == 1:
        ax.scatter(x_mx, y_vec, c=color)
    elif x_mx.shape[1] > 1:
        ax.scatter(x_mx[:, 0], x_mx[:, 1], y_vec, c=color)

    return plt
