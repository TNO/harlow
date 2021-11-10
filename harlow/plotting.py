"""Visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


def plot_function(f, domain, grid_size=1000, plot_name=None, show=True, save=False):
    if len(domain) > 2:
        raise NotImplementedError

    x0_range = domain[0]
    x1_range = domain[1]

    X1 = np.linspace(x0_range[0], x0_range[1], grid_size)
    X2 = np.linspace(x1_range[0], x1_range[1], grid_size)

    x1, x2 = np.meshgrid(X1, X2)
    Z = f(
        np.hstack(
            (x1.reshape(grid_size * grid_size, 1), x2.reshape(grid_size * grid_size, 1))
        )
    )
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surface = ax.plot_trisurf(
        x1,
        x2,
        Z.reshape(grid_size, grid_size),
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
    f,
    X,
    y=None,
    plot_sample_locations=False,
    plot_name=None,
    show=True,
    save=False,
    color="b",
):

    if y is None:
        Z = f(X)
    else:
        Z = y

    fig = plt.figure()

    if len(X.shape) == 1:
        ax = fig.gca()
        ax.plot(X, Z, label=f.__name__ if plot_name is None else plot_name)
    elif X.shape[1] > 1:
        ax = fig.gca(projection="3d")
        surface = ax.plot_trisurf(
            X[:, 0],
            X[:, 1],
            Z,
            cmap=cm.coolwarm,
            label=f.__name__ if plot_name is None else plot_name,
        )
        fig.colorbar(surface)

    if plot_sample_locations:
        if len(X.shape) == 1:
            ax.scatter(X, Z, c=color, alpha=0.6)
        elif X.shape[1] > 1:
            ax.scatter(X[:, 0], X[:, 1], Z, c=color, alpha=0.6)

    if show:
        plt.show()
    if save:
        plt.savefig(f"{f.__name__ if plot_name is None else plot_name}.png")
    return plt


def add_samples_to_plot(plot, X, y, color):
    ax = plot.gca()

    if X.shape[1] == 1:
        ax.scatter(X, y, c=color)
    elif X.shape[1] > 1:
        ax.scatter(X[:, 0], X[:, 1], y, c=color)

    return plt
