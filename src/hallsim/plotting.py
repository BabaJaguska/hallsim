"""Plotting utilities for HallSim simulations.

Provides quick visualization for simulation results from Scheduler,
including trajectory plots, phase portraits, and composite overviews.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


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
        figsize = (
            4.5 * ncols,
            2.6 * max(1, (len(paths) + ncols - 1) // ncols),
        )
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
    labels: Mapping[str, str] | None = None,
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
    labels:
        ``{store_path: panel title}``. Without it a panel is titled by its
        path's last segment, which names the model's internal species rather
        than what the panel is being read as.
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
        ax.set_title(
            (labels or {}).get(
                path, path.split("/")[-1] if "/" in path else path
            ),
            fontsize=10,
        )
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


def _node_kinds(composite) -> dict:
    """``{process: kind}`` — what each node *is*, from the process itself.

    ``model`` carries integrated species, ``edge`` is a coupling edge (it can
    be frozen), ``source`` drives an input and reads nothing.
    """
    kinds = {}
    for name, proc in composite.processes.items():
        if hasattr(proc, "_species_names"):
            kinds[name] = "model"
        elif hasattr(proc, "frozen_at"):
            kinds[name] = "edge"
        else:
            kinds[name] = "source"
    return kinds


def _layered_positions(G):
    """Sugiyama layering when netgraph is installed, spring otherwise.

    A composite is a shallow DAG plus a few feedback edges, so a layered
    layout reads as "what drives what" where a force layout reads as a ball.
    """
    import networkx as nx

    try:
        from netgraph import get_sugiyama_layout

        return get_sugiyama_layout(
            list(G.edges()), node_size=6, scale=(12.0, 6.0)
        )
    except ImportError:
        log.info("netgraph not installed; composite graph uses spring layout")
        return nx.spring_layout(
            G, seed=42, k=1.0 / max(1, len(G.nodes) ** 0.5)
        )


def draw_composite_graph(
    composite,
    save: str | None = None,
    title: str = "",
    figsize: tuple[float, float] = (13, 7),
    layout: str = "layered",
):
    """Render the composite's process interaction graph — generated, never
    drawn, so it cannot describe a composite that no longer exists.

    Nodes are processes, coloured by what they are (imported model, coupling
    edge, drive) and sized by how many states they carry; edges are labelled
    with the store path the write travels on.
    :class:`hallsim.validation.GraphAnalyzer` supplies the graph.

    Parameters
    ----------
    composite:
        :class:`hallsim.composite.Composite` to visualize.
    save:
        If set, write PNG to this path.
    layout:
        ``"layered"`` (default; Sugiyama via netgraph, spring if it is not
        installed), ``"spring"``, ``"kamada_kawai"``, ``"shell"`` or
        ``"circular"``.
    """
    import networkx as nx

    from hallsim.validation import GraphAnalyzer

    FACE = {"model": "#0173b2", "edge": "#de8f05", "source": "#94a3b8"}
    LEGEND = {
        "model": "imported model",
        "edge": "coupling edge",
        "source": "severity drive",
    }

    G = GraphAnalyzer().build_graph(composite.processes, composite.topology)
    kinds = _node_kinds(composite)

    if layout == "layered":
        pos = _layered_positions(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.0 / max(1, len(G.nodes) ** 0.5))

    widths = {
        name: len(getattr(proc, "_species_names", ()) or ())
        for name, proc in composite.processes.items()
    }
    sizes = {n: 900 + 90 * widths.get(n, 0) for n in G.nodes}

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#6b7280",
        arrows=True,
        arrowsize=13,
        width=1.2,
        alpha=0.8,
        node_size=[sizes[n] for n in G.nodes],
        connectionstyle="arc3,rad=0.08",
    )
    for kind, color in FACE.items():
        members = [n for n in G.nodes if kinds.get(n) == kind]
        if not members:
            continue
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=members,
            node_color=color,
            node_shape="s",
            node_size=[sizes[n] for n in members],
            ax=ax,
            edgecolors="white",
            linewidths=1.4,
            label=LEGEND[kind],
        )
    # Names sit under their node: a process name is longer than any box that
    # would still leave the graph readable.
    for name, (x, y) in pos.items():
        ax.annotate(
            name,
            (x, y),
            xytext=(0, -(sizes[name] ** 0.5) / 2 - 9),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=FACE[kinds.get(name, "source")],
            zorder=5,
        )
    edge_labels = {
        (u, v): d["store_path"].split("/")[-1]
        for u, v, d in G.edges(data=True)
        if d.get("store_path")
    }
    # Staggered along their edges: two labels meeting at a shared node land on
    # top of each other at a common label_pos.
    for offset, subset in (
        (0.42, dict(list(edge_labels.items())[0::2])),
        (0.62, dict(list(edge_labels.items())[1::2])),
    ):
        if subset:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=subset,
                font_size=6.5,
                font_color="#5b6b7d",
                ax=ax,
                label_pos=offset,
                rotate=False,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    fc="white",
                    ec="none",
                    alpha=0.85,
                ),
            )
    ax.margins(0.16, 0.20)
    ax.set_axis_off()
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
    )
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor="white")
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
