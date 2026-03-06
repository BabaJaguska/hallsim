"""Plotting utilities for HallSim simulations.

Provides quick visualization for simulation results from Simulator
and Scheduler, including trajectory plots, phase portraits, and
composite overviews.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(
    result,
    paths: Sequence[str] | None = None,
    title: str = "",
    figsize: tuple[float, float] = (12, 6),
    ncols: int = 1,
    ylabel: str = "Value",
    save: str | None = None,
):
    """Plot state trajectories from a SimResult or SchedulerResult.

    Parameters
    ----------
    result:
        SimResult or SchedulerResult with ``.ts`` and ``.ys``.
    paths:
        Store paths to plot. If None, plots all.
    title:
        Figure title.
    ncols:
        Number of subplot columns. If 1, all on one axes.
    save:
        If set, save figure to this path instead of showing.

    Returns
    -------
    matplotlib Figure.
    """
    ts = np.asarray(result.ts)
    if paths is None:
        paths = list(result.ys.keys())

    if ncols == 1:
        fig, ax = plt.subplots(figsize=figsize)
        for path in paths:
            vals = np.asarray(result.ys[path])
            label = path.split("/")[-1] if "/" in path else path
            ax.plot(ts, vals, label=label, linewidth=1.5)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title)
    else:
        nrows = (len(paths) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for i, path in enumerate(paths):
            ax = axes[i // ncols][i % ncols]
            vals = np.asarray(result.ys[path])
            label = path.split("/")[-1] if "/" in path else path
            ax.plot(ts, vals, linewidth=1.5)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Time", fontsize=8)
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for i in range(len(paths), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)
        if title:
            fig.suptitle(title, fontsize=13)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_phase_portrait(
    result,
    x_path: str,
    y_path: str,
    title: str = "",
    figsize: tuple[float, float] = (6, 6),
    save: str | None = None,
):
    """Plot a 2D phase portrait from simulation results.

    Parameters
    ----------
    result:
        SimResult or SchedulerResult.
    x_path, y_path:
        Store paths for x and y axes.
    """
    x = np.asarray(result.ys[x_path])
    y = np.asarray(result.ys[y_path])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=1.0, alpha=0.8)
    ax.plot(x[0], y[0], "go", markersize=8, label="start")
    ax.plot(x[-1], y[-1], "rs", markersize=8, label="end")

    x_label = x_path.split("/")[-1] if "/" in x_path else x_path
    y_label = y_path.split("/")[-1] if "/" in y_path else y_path
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_events(
    result,
    path: str,
    title: str = "",
    figsize: tuple[float, float] = (12, 4),
    save: str | None = None,
):
    """Plot a trajectory with event fire times marked.

    Parameters
    ----------
    result:
        SchedulerResult with ``.events``.
    path:
        Store path to plot as the main trajectory.
    """
    ts = np.asarray(result.ts)
    vals = np.asarray(result.ys[path])

    fig, ax = plt.subplots(figsize=figsize)
    label = path.split("/")[-1] if "/" in path else path
    ax.plot(ts, vals, linewidth=1.5, label=label)

    if hasattr(result, "events") and result.events:
        for ev in result.events:
            ax.axvline(ev.time, color="red", linestyle="--", alpha=0.7,
                       label=f"event: {ev.process}" if ev == result.events[0] else "")
            ax.annotate(ev.process, xy=(ev.time, float(vals[np.argmin(np.abs(ts - ev.time))])),
                        fontsize=7, color="red", rotation=45)

    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
