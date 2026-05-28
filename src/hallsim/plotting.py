"""Plotting utilities for HallSim simulations.

Provides quick visualization for simulation results from Scheduler
and Scheduler, including trajectory plots, phase portraits, and
composite overviews.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    """Plot state trajectories from a SchedulerResult.

    Parameters
    ----------
    result:
        SchedulerResult with ``.ts`` and ``.ys``.
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
        paths = list(result.keys)

    if ncols == 1:
        fig, ax = plt.subplots(figsize=figsize)
        for path in paths:
            vals = np.asarray(result.get(path))
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
            vals = np.asarray(result.get(path))
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
        SchedulerResult.
    x_path, y_path:
        Store paths for x and y axes.
    """
    x = np.asarray(result.get(x_path))
    y = np.asarray(result.get(y_path))

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
    vals = np.asarray(result.get(path))

    fig, ax = plt.subplots(figsize=figsize)
    label = path.split("/")[-1] if "/" in path else path
    ax.plot(ts, vals, linewidth=1.5, label=label)

    if hasattr(result, "events") and result.events:
        for ev in result.events:
            ax.axvline(
                ev.time,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"event: {ev.process}" if ev == result.events[0] else "",
            )
            ax.annotate(
                ev.process,
                xy=(ev.time, float(vals[np.argmin(np.abs(ts - ev.time))])),
                fontsize=7,
                color="red",
                rotation=45,
            )

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


# ── Composite-level run helpers (used by CalibrationProblem, demos, etc.) ──


def plot_composite_run(
    result,
    paths: Sequence[str] | None = None,
    title: str = "",
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
):
    """Multi-panel trajectory plot for a single composite run.

    Convenience wrapper over :func:`plot_trajectories` with sensible
    defaults for showing a handful of reporter store paths.
    """
    if paths is None:
        paths = list(result.keys)
    if figsize is None:
        figsize = (4.5 * ncols, 2.6 * max(1, (len(paths) + ncols - 1) // ncols))
    return plot_trajectories(
        result,
        paths=paths,
        title=title,
        figsize=figsize,
        ncols=ncols,
        save=save,
    )


def plot_runs_comparison(
    results: Mapping[str, Any],
    paths: Sequence[str],
    title: str = "",
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
):
    """Overlay multiple SchedulerResults per store path.

    Use this to compare runs side-by-side (e.g. ctrl/DDIS/RAPA at the
    same parameters; or pre-fit vs post-fit at a single arm).

    Parameters
    ----------
    results:
        ``{label: SchedulerResult}``.
    paths:
        Store paths to plot. Each gets one subplot.
    """
    n = len(paths)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.5 * ncols, 2.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, path in enumerate(paths):
        ax = axes[i // ncols][i % ncols]
        for label, res in results.items():
            ts = np.asarray(res.ts)
            try:
                vals = np.asarray(res.get(path))
            except KeyError:
                continue
            ax.plot(ts, vals, linewidth=1.5, label=label, alpha=0.85)
        ax.set_title(path.split("/")[-1] if "/" in path else path, fontsize=10)
        ax.set_xlabel("Time", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def draw_composite_graph(
    composite,
    save: str | None = None,
    title: str = "",
    figsize: tuple[float, float] = (10, 7),
    layout: str = "spring",
):
    """Render the composite's process-port-store topology graph.

    Uses :class:`hallsim.validation.GraphAnalyzer` to build a directed
    interaction graph (process nodes + store-path nodes; edges encode
    reads/writes). Renders via networkx + matplotlib.

    Parameters
    ----------
    composite:
        :class:`hallsim.composite.Composite` to visualize.
    save:
        If set, write PNG to this path.
    layout:
        ``"spring"`` (default), ``"kamada_kawai"``, ``"shell"``, or
        ``"circular"``. Picked per how connected the composite is.
    """
    import networkx as nx

    from hallsim.validation import GraphAnalyzer

    analyzer = GraphAnalyzer()
    G = analyzer.build_graph(composite.processes, composite.topology)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=1.0 / max(1, len(G.nodes) ** 0.5))
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#888888",
        arrows=True,
        arrowsize=14,
        width=1.2,
        alpha=0.75,
        connectionstyle="arc3,rad=0.07",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#4a90e2",
        node_shape="s",
        node_size=1400,
        ax=ax,
        edgecolors="#1f3a68",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)
    edge_labels = {
        (u, v): d.get("store_path", "")
        for u, v, d in G.edges(data=True)
        if d.get("store_path")
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=6,
            ax=ax,
            alpha=0.85,
            label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def save_run_results(
    results: Mapping[str, Any] | Any,
    save_path: str,
    paths: Sequence[str] | None = None,
    metadata: dict | None = None,
):
    """Write SchedulerResult(s) to JSON.

    Saves ``ts``, per-store-path ``ys`` (for the listed ``paths`` or
    all if None), plus optional metadata. Accepts either a single
    SchedulerResult or a ``{label: SchedulerResult}`` mapping.
    """
    def _serialize_one(res):
        ts = np.asarray(res.ts).tolist()
        keys = list(paths) if paths is not None else list(res.keys)
        ys = {}
        for k in keys:
            try:
                ys[k] = np.asarray(res.get(k)).tolist()
            except KeyError:
                pass
        return {"ts": ts, "ys": ys}

    if hasattr(results, "ys") and hasattr(results, "ts"):
        payload: dict = {"single": _serialize_one(results)}
    else:
        payload = {label: _serialize_one(r) for label, r in results.items()}
    if metadata:
        payload["_metadata"] = metadata

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    return save_path
