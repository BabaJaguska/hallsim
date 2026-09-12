"""Figures for the multi-hallmark composite — one CLI, many panels.

Consolidates the per-figure scripts into subcommands (pick with the first arg):

  schematic      wiring diagram of the composite (dials → dp14 → gz06/ih04).
  trajectories   reporter observables over the day axis, ctrl/DDIS/DDIS+rapa.
  reporter-levels calibrated reporter dynamics per condition (model units).
  concordance    measured-vs-simulated dumbbells per gene (oob + calibrated).
  temporal       out-of-the-box → calibrated log2FC trajectories vs data.
  before-after   each constituent standalone vs inside the composite.

Run: python demos/multi_hallmark_figures.py <figure>   (or `all`)
Calibration itself lives in multi_hallmark_calibrate.py; the demo
NeuralODE-hybrid swap in multi_hallmark_hybrid.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hallsim.composite import Composite  # noqa: E402
from hallsim.hallmarks import apply_hallmarks, with_hallmarks  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.calibration import load_checkpoint  # noqa: E402
from demos.models.multi_hallmark import (  # noqa: E402
    DP14_SBML_PATH,
    GZ06_SBML_PATH,
    PROCTOR07_SBML_PATH,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    }
)

ROOT = Path(__file__).resolve().parent.parent
# The most recent timestamped calibrate run (symlink maintained by
# multi_hallmark_calibrate.make_run_dir); figures track whatever last ran.
OUT_CAL = ROOT / "outputs" / "multi_hallmark_calibrate" / "latest"

_CKPT = OUT_CAL / "checkpoint.npz"


def load_fit() -> dict:
    """Fitted parameters, read live from the calibration checkpoint.

    The calibrated-model figures track whatever ``multi_hallmark_calibrate``
    last wrote — no transcribed constants to drift out of sync with the fit.
    """
    if not _CKPT.exists():
        raise FileNotFoundError(
            f"no calibration checkpoint at {_CKPT}; run "
            "`multi_hallmark_calibrate` first to produce the fit these "
            "figures plot."
        )
    params, _ = load_checkpoint(_CKPT)
    return {k: float(v) for k, v in params.items()}


def _saved_run() -> dict:
    """What the latest calibration recorded about itself — its composite
    variant and fitted set — so a figure scores the fit as it was run rather
    than as a flag re-describes it. Empty when no fit has been written."""
    path = OUT_CAL / "summary.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _problem(args):
    """The same problem the calibration built, so every figure sees the
    composite, reporters and fitted set the checkpoint belongs to — a fit
    that held a parameter is scored with that parameter left out."""
    from demos.multi_hallmark_calibrate import build_problem

    saved = _saved_run()
    fitted = list(saved["params"]) if "params" in saved else None
    if fitted is None and _CKPT.exists():
        fitted = list(load_fit())
    return build_problem(
        fitted=tuple(fitted) if fitted is not None else None,
    )


# ── schematic ────────────────────────────────────────────────────────────
# Geometry only. Membership comes from the composite, readouts from the
# reporter set and edge captions from each edge's own description; what a
# drawing cannot derive is where to put things. fig_schematic refuses to draw
# a composite whose processes have no slot rather than showing a stale
# picture. The deposit label is here because an imported process does not
# retain its accession.
MODEL_BLOCKS = {
    "dp14": dict(
        xy=(3.7, 1.95, 2.6, 2.0),
        edge="#3a3f4a",
        fill="#eef0f2",
        deposit="BIOMD582",
        body=[
            "mTOR · AMPK · FoxO3a",
            "mitophagy · ROS · DNA damage",
            "CDKN1A",
        ],
    ),
    "gz06": dict(
        xy=(8.9, 3.72, 3.1, 1.2),
        edge="#de8f05",
        fill="#fbf1e0",
        deposit="BIOMD157",
        body=["p53–Mdm2 oscillator"],
    ),
    "p07": dict(
        xy=(8.9, 0.72, 3.1, 1.42),
        edge="#0173b2",
        fill="#e7eff7",
        deposit="BIOMD105",
        body=["ubiquitin–proteasome", "misfolding · aggregation"],
    ),
}

DIAL_DRIVES = {"irradiation_pulse", "rapamycin_drive"}

MODEL_EDGES = {
    "damage_bridge": dict(
        p0=(6.4, 3.82),
        p1=(8.8, 4.05),
        rad=-0.30,
        color="#de8f05",
        at=(7.6, 4.38),
        rot=0,
    ),
    "p53_cdkn1a": dict(
        p0=(8.8, 3.75),
        p1=(6.4, 3.50),
        rad=-0.30,
        color="#de8f05",
        at=(7.6, 3.15),
        rot=0,
    ),
    "ros_identity": dict(
        p0=(6.3, 2.62),
        p1=(8.9, 1.72),
        rad=-0.11,
        color="#0173b2",
        at=(7.60, 2.36),
        rot=-17,
    ),
    "mtor_synthesis": dict(
        p0=(6.3, 2.16),
        p1=(8.9, 1.24),
        rad=-0.13,
        color="#0173b2",
        at=(7.55, 1.52),
        rot=-17,
    ),
}


def _edge_caption(proc) -> str:
    """The edge's own ``description``, trimmed to what fits on an arrow: the
    clause before the parenthetical that names the two deposits."""
    text = (proc.description or "").split("(")[0].strip().rstrip(".")
    return text or type(proc).__name__


def fig_schematic(args):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    C_DIAL, INK, DIM, BODY = "#6b7280", "#1f2530", "#5b6b7d", "#333a44"
    F_DIAL = "#f4f5f7"

    def block(ax, x, y, w, h, edge, fill, r=0.11, lw=2.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                zorder=3,
            )
        )

    def arrow(ax, p0, p1, color, rad=0.0, lw=2.0, ls="-"):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=lw,
                linestyle=ls,
                color=color,
                zorder=2,
                shrinkA=2,
                shrinkB=4,
            )
        )

    def elabel(ax, x, y, lines, color, rot):
        ax.text(
            x,
            y,
            lines,
            fontsize=9.8,
            color=color,
            ha="center",
            va="center",
            rotation=rot,
            rotation_mode="anchor",
            fontweight="bold",
            zorder=5,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none"),
        )

    problem = _problem(args)
    comp = problem.composite
    live_models = {
        n for n, pr in comp.processes.items() if hasattr(pr, "_species_names")
    }
    # Everything that is neither a constituent nor one of the two severity
    # drives (drawn as the dial arrows) is a cross-publication edge.
    live_edges = set(comp.processes) - live_models - DIAL_DRIVES
    missing = sorted(
        (live_models - set(MODEL_BLOCKS)) | (live_edges - set(MODEL_EDGES))
    )
    if missing:
        raise RuntimeError(
            f"composite_schematic has no layout entry for {missing}. Add one "
            "to MODEL_BLOCKS / MODEL_EDGES before regenerating — a schematic "
            "that silently omits a constituent is worse than no schematic."
        )

    reporters = problem.reporters

    def readouts_for(namespace):
        genes = [
            r.gene_symbol
            for r in reporters
            if r.observable.split("/")[0] == namespace
        ]
        return "readouts:  " + " · ".join(genes)

    fig, ax = plt.subplots(figsize=(12.8, 5.9))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")
    block(ax, 0.4, 3.45, 2.5, 1.02, "#94a3b8", F_DIAL, r=0.10, lw=1.7)
    ax.text(
        1.65,
        4.17,
        "Genomic Instability",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        ha="center",
    )
    ax.text(1.65, 3.86, "severity 0-1", fontsize=9.9, color=DIM, ha="center")
    ax.text(
        1.65, 3.63, "→ etoposide dose", fontsize=9.9, color=DIM, ha="center"
    )
    block(ax, 0.4, 1.1, 2.5, 1.18, "#94a3b8", F_DIAL, r=0.10, lw=1.7)
    ax.text(
        1.65,
        2.02,
        "Deregulated",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        1.65,
        1.77,
        "Nutrient Sensing",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        ha="center",
    )
    ax.text(1.65, 1.47, "severity 0-1", fontsize=9.9, color=DIM, ha="center")
    ax.text(
        1.65,
        1.24,
        "→ mTORC1 (rapamycin)",
        fontsize=9.9,
        color=DIM,
        ha="center",
    )
    for name in sorted(live_models):
        spec = MODEL_BLOCKS[name]
        x, y, w, h = spec["xy"]
        block(ax, x, y, w, h, spec["edge"], spec["fill"])
        cx, top = x + w / 2, y + h
        ax.text(
            cx,
            top - 0.33,
            name,
            fontsize=16,
            color=spec["edge"],
            fontweight="bold",
            ha="center",
        )
        ax.text(
            cx,
            top - 0.65,
            spec["deposit"],
            fontsize=10.1,
            color=DIM,
            ha="center",
        )
        for k, line in enumerate(spec["body"]):
            ax.text(
                cx,
                top - 0.91 - 0.30 * k,
                line,
                fontsize=10.3,
                color=BODY,
                ha="center",
            )
        ax.text(
            cx,
            y - 0.24,
            readouts_for(name),
            fontsize=9.2 if name == "dp14" else 9.8,
            color=spec["edge"],
            ha="center",
            va="center",
            style="italic",
            zorder=5,
        )
    arrow(ax, (2.9, 3.82), (3.7, 3.28), C_DIAL, rad=-0.14)
    arrow(ax, (2.9, 1.88), (3.7, 2.55), C_DIAL, rad=0.14)
    for name in sorted(live_edges):
        spec = MODEL_EDGES[name]
        arrow(ax, spec["p0"], spec["p1"], spec["color"], rad=spec["rad"])
        elabel(
            ax,
            *spec["at"],
            _edge_caption(comp.processes[name]),
            spec["color"],
            spec["rot"],
        )
    fig.tight_layout(pad=0.2)
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_CAL / f"composite_schematic.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"wrote composite_schematic.png/.pdf -> {OUT_CAL}", flush=True)


# ── trajectories ─────────────────────────────────────────────────────────
def fig_trajectories(args):
    from demos.models.multi_hallmark import build_multi_hallmark_composite

    arms = [
        (0.0, 0.0, "ctrl", "tab:green"),
        (1.0, 0.0, "DDIS", "tab:red"),
        (1.0, -1.0, "DDIS+rapa", "tab:blue"),
    ]
    # The run's own reporter set; plot the raw underlying state for
    # integral-based readouts (the cumulative ∫ path isn't a trajectory).
    panels = [
        (r.observable.replace("_integral", ""), r.gene_symbol)
        for r in _problem(args).reporters
    ]

    def run(gi, dns):
        base = build_multi_hallmark_composite()
        hallmarks = {"Genomic Instability": gi}
        if dns != 0.0:
            hallmarks["Deregulated Nutrient Sensing"] = dns
        comp = with_hallmarks(base, hallmarks)
        return Scheduler(auto_stiffness=True).run(
            comp,
            t_span=(0.0, 50.0),
            macro_dt=5.0,
            y0=comp.initial_state_vec(),
            save_dt=1.0,
        )

    runs = {label: run(gi, dns) for gi, dns, label, _ in arms}
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False
    )
    axf = axes.ravel()
    for i, ax in enumerate(axf):
        if i >= len(panels):
            ax.axis("off")
            continue
        path, gene = panels[i]
        for gi, dns, label, color in arms:
            ax.plot(
                np.asarray(runs[label].ts),
                np.asarray(runs[label].get(path)),
                label=label,
                color=color,
                lw=1.6,
            )
        ax.set_title(f"{gene}  [{path}]")
        ax.set_xlabel("time (days)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Multi-hallmark composite — reporter trajectories across arms "
        "(canonical day axis, rtol=1e-6)",
        fontsize=13,
    )
    fig.tight_layout()
    out = ROOT / "outputs" / "subsystem_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "composite_trajectories.png", dpi=120)
    print(f"wrote {out / 'composite_trajectories.png'}", flush=True)


# ── reporter-levels ──────────────────────────────────────────────────────
def fig_reporter_levels(args):

    conds = {
        "ctrl": ("control", "#9a9a95"),
        "DDIS": ("etoposide (DDIS)", "#2a78d6"),
        "RAPA": ("etoposide + rapamycin", "#1baf7a"),
    }
    grid_c, macro_dt, t_end = "#e6e6e2", 1.0, 14.0

    def levels(problem, params, cond, qt):
        sub = problem._substitute(problem.composite.processes, params)
        procs = apply_hallmarks(sub, problem.conditions[cond].hallmarks)
        comp = Composite(
            processes=procs,
            topology=problem.composite.topology,
            validate=False,
            semantic_validation=False,
        )
        res = problem._scheduler.run(
            comp,
            t_span=(0.0, t_end),
            macro_dt=macro_dt,
            y0=comp.initial_state_vec(),
            save_dt=macro_dt,
        )
        trajs = jnp.stack([res.ys[..., i] for i in problem._reporter_indices])
        return np.asarray(problem._reporter_summaries(res.ts, trajs, qt))

    problem = _problem(args)
    fit = {k: jnp.asarray(v) for k, v in load_fit().items()}
    qt = jnp.arange(0.5, t_end + 1e-6, 0.5)
    genes = [r.gene_symbol for r in problem.reporters]
    obs = [r.observable for r in problem.reporters]
    lv = {c: levels(problem, fit, c, qt) for c in conds}
    qt = np.asarray(qt)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True)
    for i, (ax, gene, ob) in enumerate(zip(axes.ravel(), genes, obs)):
        for c, (label, color) in conds.items():
            ax.plot(qt, lv[c][i], color=color, lw=2.2, label=label, zorder=3)
        ax.set_title(f"{gene}", fontsize=11, fontweight="bold", loc="left")
        ax.annotate(
            ob,
            (0.5, 1.005),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#6f6e6a",
        )
        ax.grid(True, color=grid_c, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel("reporter level (model units)")
        if i >= 3:
            ax.set_xlabel("day")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        "Reporter dynamics per condition (calibrated model)",
        fontsize=12.5,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_CAL / f"reporter_levels_by_condition.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    print(
        f"wrote reporter_levels_by_condition.png/.pdf → {OUT_CAL}", flush=True
    )


# ── oob (transcribed concordance table) ──────────────────────────────────
def fig_concordance(args):
    """Reporter concordance dumbbells — measured vs simulated log2FC per gene
    across all six condition/day points, computed live from the model via
    ``problem.evaluate``. Rendered for both the out-of-the-box (init) and the
    calibrated (checkpoint) parameters. ``dir N/6`` counts sign-matching
    conditions.
    """
    from matplotlib.lines import Line2D
    from hallsim.calibration_report import rows_by_gene

    C_DATA, C_MODEL, INK, DIM, BAND = (
        "#2563eb",
        "#d97706",
        "#1f2937",
        "#6b7280",
        "#f1f5f9",
    )
    problem = _problem(args)
    short = {
        "DDIS_vs_ctrl": "DDIS",
        "RAPA_vs_ctrl": "RAPA",
        "RAS_vs_ctrl": "RAS",
    }
    # Preferred display order, restricted to the arms this problem scores.
    arms_order = [a for a in short if a in problem.data]
    short = {a: short[a] for a in arms_order}
    cond = [(a, t) for a in arms_order for t in sorted(problem.data[a])]
    cond_labels = [f"{short[a]}\nD{int(t)}" for a, t in cond]
    spans, i = [], 0  # alternating per-arm bands for visual grouping
    for a in arms_order:
        n = len(sorted(problem.data[a]))
        spans.append((i, i + n))
        i += n
    order = [r.gene_symbol for r in problem.reporters]
    nC = len(cond)

    def render(params, subtitle, stem):
        ev = problem.evaluate(params)
        rows = [rows_by_gene(ev[a][t]) for a, t in cond]
        genes = [g for g in order if g in rows[0]]

        def panel(ax, gene):
            data = [rows[k][gene].delta_data for k in range(nC)]
            model = [rows[k][gene].delta_sim for k in range(nC)]
            agree = sum(rows[k][gene].sign_match for k in range(nC))
            x = range(nC)
            for bi, (lo, hi) in enumerate(spans):
                if bi % 2 == 1:
                    ax.axvspan(lo - 0.5, hi - 0.5, color=BAND, zorder=0)
            ax.axhline(0, color="#cbd5e1", lw=1.0, zorder=1)
            for xi, d, m in zip(x, data, model):
                ax.plot([xi, xi], [d, m], color="#d1d5db", lw=1.4, zorder=2)
            ax.scatter(x, data, s=46, color=C_DATA, zorder=4)
            ax.scatter(
                x,
                model,
                s=46,
                facecolors="none",
                edgecolors=C_MODEL,
                linewidths=1.8,
                zorder=4,
            )
            ax.set_title(
                gene,
                fontsize=10.5,
                color=INK,
                fontweight="bold",
                loc="left",
                pad=6,
            )
            ax.text(
                1.0,
                1.02,
                f"dir {agree}/{nC}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=DIM,
                fontweight="bold",
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(cond_labels, fontsize=8, color=DIM)
            ax.set_xlim(-0.5, nC - 0.5)
            ax.tick_params(axis="y", labelsize=8, colors=DIM)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#cbd5e1")

        ncol = 3
        nrow = -(-len(genes) // ncol)  # ceil
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(12.4, 3.3 * nrow), squeeze=False
        )
        axf = axes.ravel()
        for j, ax in enumerate(axf):
            if j >= len(genes):
                ax.axis("off")
                continue
            panel(ax, genes[j])
            ax.set_ylabel("log2 FC", fontsize=8.5, color=DIM)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=C_DATA,
                markersize=8,
                label="measured",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor=C_MODEL,
                markerfacecolor="none",
                markeredgewidth=1.8,
                markersize=8,
                label="simulated",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 1.005),
        )
        fig.suptitle(
            f"Reporter concordance — {subtitle}",
            fontsize=12.5,
            fontweight="bold",
            color=INK,
            x=0.09,
            ha="left",
            y=1.02,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4, w_pad=2.2)
        OUT_CAL.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(
                OUT_CAL / f"{stem}.{ext}",
                dpi=200,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)
        print(f"wrote {stem}.png/.pdf -> {OUT_CAL}", flush=True)

    init = problem.initial_params()
    render(init, "out-of-the-box", "reporter_concordance_oob")
    fit = {k: jnp.asarray(v) for k, v in load_fit().items()}
    render(fit, "calibrated", "reporter_concordance_calibrated")


# ── temporal (oob → calibrated log2FC vs data) ───────────────────────────
def fig_temporal(args):
    from demos.multi_hallmark_calibrate import _annotate_interventions

    C_OOB, C_FIT, C_DATA, grid_c = "#9a9a95", "#2a78d6", "#0b0b0b", "#e6e6e2"
    # Subtitles for the arms this problem may carry; only the ones it actually
    # scores get a figure, so dropping or adding an arm needs no edit here.
    ARM_SUBTITLES = {
        "DDIS_vs_ctrl": "DDIS  (etoposide, fit arm)",
        "RAPA_vs_ctrl": "rapamycin  (etoposide + rapa @ day 2, held-out)",
        "RAS_vs_ctrl": "RAS  (oncogene-induced senescence, held-out)",
    }
    t_end = 14.0

    def figure_for_arm(problem, init, fit, arm, subtitle):
        data_times = sorted(problem.data[arm])
        genes = [r.gene_symbol for r in problem.reporters]
        n = len(genes)
        ncol = 3
        nrow = -(-n // ncol)  # ceil
        # model_lfc reproduces exactly what the loss fits: within-arm
        # (X_t/X_0) fold change. Start at t>0: at exactly t=0 a window-mean
        # reporter reads a zero-width window (→0, log2 floors), a plotting-only
        # degeneracy; the fold-change is 0 as t→0⁺ by construction.
        qt = np.arange(0.1, t_end + 1e-6, 0.1)
        lfc_oob = np.asarray(problem.model_lfc(init, arm, jnp.asarray(qt)))
        lfc_fit = np.asarray(problem.model_lfc(fit, arm, jnp.asarray(qt)))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(11, 3.2 * nrow), sharex=True, squeeze=False
        )
        axf = axes.ravel()
        for i, ax in enumerate(axf):
            if i >= n:
                ax.axis("off")
                continue
            gene = genes[i]
            ax.axhline(0, color=grid_c, lw=1.2, zorder=0)
            ax.plot(
                qt,
                lfc_oob[i],
                color=C_OOB,
                lw=1.8,
                ls=(0, (4, 2)),
                zorder=2,
                label="out-of-the-box",
            )
            ax.plot(
                qt,
                lfc_fit[i],
                color=C_FIT,
                lw=2.2,
                zorder=3,
                label="calibrated",
            )
            dx = [0.0] + list(data_times)
            dy = [0.0] + [
                float(problem.data[arm][t][gene]) for t in data_times
            ]
            ax.plot(
                dx,
                dy,
                "o",
                color=C_DATA,
                ms=7,
                zorder=4,
                label="measured (GSE248823)",
            )
            _annotate_interventions(ax, arm)
            ax.set_title(gene, fontsize=11, fontweight="bold", loc="left")
            ax.grid(True, color=grid_c, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i % ncol == 0:
                ax.set_ylabel("log2 fold-change")
            if i >= n - ncol:
                ax.set_xlabel("day")
        handles, labels = axf[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9.5,
            bbox_to_anchor=(0.5, -0.005),
        )
        fig.suptitle(
            f"{subtitle} — reporter trajectories, out-of-the-box vs "
            "calibrated",
            fontsize=12.5,
            x=0.02,
            ha="left",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        stem = f"temporal_oob_vs_fit_{arm}"
        for ext in ("png", "pdf"):
            fig.savefig(
                OUT_CAL / f"{stem}.{ext}", dpi=150, bbox_inches="tight"
            )
        plt.close(fig)
        print(f"wrote {stem}.png/.pdf", flush=True)

    problem = _problem(args)
    init = problem.initial_params()
    fit = {k: jnp.asarray(v) for k, v in load_fit().items()}
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for arm in problem.data:
        figure_for_arm(problem, init, fit, arm, ARM_SUBTITLES.get(arm, arm))
    print(f"→ {OUT_CAL}", flush=True)


# ── temporal DDIS vs RAPA overlay (the rapamycin divergence) ─────────────
# Constituents the figures draw from a reaction-level population rather than
# their mean field. The calibration goes through the mean field; these
# members are reported as cells, since that is how they are run.
REACTION_LEVEL = ("p07",)

# Row headings for reporter grids laid out one row per constituent.
CONSTITUENT_LABELS = {
    "dp14": "Dalle Pezze 2014 (dp14)",
    "gz06": "Geva-Zatorsky 2006 (gz06)",
    "p07": "Proctor 2007 (p07)",
}


def _reaction_level_reporters(problem):
    """Indices of the reporters whose observable a reaction-level member
    owns, in ``problem.reporters`` order."""
    return [
        i
        for i, r in enumerate(problem.reporters)
        if r.observable.split("/")[0] in REACTION_LEVEL
    ]


def _population_lfc(problem, params, arm, qt, n_cells, seed):
    """Reaction-level members as ``n_cells`` cells on the trajectory of
    ``arm`` at ``params``: per-cell and pooled sign-aligned log2
    fold-changes at ``qt`` for the reporters those members own. Each
    cell's fold-change uses its own day-0 reference, as the loss does; the
    pooled one divides pooled readouts, as a bulk assay does. Returns
    ``(cells, pooled)`` shaped ``(n_rep, n_t, n_cells)`` and
    ``(n_rep, n_t)`` over :func:`_reaction_level_reporters`."""
    import numpy as np

    from hallsim.scheduler import Scheduler

    rep_idx = _reaction_level_reporters(problem)
    procs = dict(problem._substitute(problem.composite.processes, params))
    for name in REACTION_LEVEL:
        procs[name] = procs[name].as_stochastic()
    registry = problem._registry(params)
    cond = problem.conditions[problem.arm_pairs[arm][0]]
    comp = problem._condition_composite(procs, cond, registry=registry)
    y0 = comp.initial_state_vec()
    span = problem.t_end - problem.t_start
    res = Scheduler(**problem.scheduler_kwargs).run(
        comp,
        t_span=(problem.t_start, problem.t_end),
        macro_dt=problem.macro_dt,
        y0=jnp.tile(y0[None], (n_cells, 1)),
        save_dt=max(1e-6, span / max(1, problem.n_save - 1)),
        seed=seed,
    )
    trajs = jnp.stack(
        [res.ys[..., idx] for idx in problem._reporter_indices]
    )  # (n_rep, n_save, n_cells)
    qt = jnp.asarray(qt)

    def lfc_of(tr):
        readout = problem._reporter_summaries(res.ts, tr, qt)
        ref = problem._reporter_summaries(res.ts, tr, jnp.zeros_like(qt))
        return problem._log2_fold_change(readout, ref)

    cells = np.asarray(jax.vmap(lfc_of, in_axes=2, out_axes=2)(trajs))
    pooled = np.asarray(lfc_of(trajs.mean(axis=2)))
    return cells[rep_idx], pooled[rep_idx]


def fig_temporal_compare(args):
    """One panel per reporter overlaying the calibrated DDIS (etoposide) and
    RAPA (etoposide + rapamycin @ day 2) trajectories, with each arm's measured
    points. Directly visualizes the held-out rapamycin effect per reporter."""
    from demos.multi_hallmark_calibrate import _annotate_interventions

    ARM_STYLE = {
        "DDIS_vs_ctrl": ("DDIS", "#c0392b"),
        "RAPA_vs_ctrl": ("rapamycin", "#2a78d6"),
        "RAS_vs_ctrl": ("RAS", "#2a9d8f"),
    }
    grid_c = "#e6e6e2"
    t_end = 14.0
    qt = np.arange(0.1, t_end + 1e-6, 0.1)

    problem = _problem(args)
    arms = {a: ARM_STYLE.get(a, (a, "#6b7280")) for a in problem.data}
    init = problem.initial_params()
    fit = {k: jnp.asarray(v) for k, v in load_fit().items()}
    genes = [r.gene_symbol for r in problem.reporters]
    lfc_fit = {
        arm: np.asarray(problem.model_lfc(fit, arm, jnp.asarray(qt)))
        for arm in arms
    }
    lfc_oob = {
        arm: np.asarray(problem.model_lfc(init, arm, jnp.asarray(qt)))
        for arm in arms
    }
    # Reporters of reaction-level members come from a population on the
    # same trajectory: pooled mean for both parameter sets, and the spread
    # across cells at the calibrated one.
    pop_idx = _reaction_level_reporters(problem)
    n_cells = int(getattr(args, "n_cells", 64))
    seed = int(getattr(args, "seed", 0))
    pop = {}
    for arm in arms if pop_idx else []:
        cells_fit, pooled_fit = _population_lfc(
            problem, fit, arm, qt, n_cells, seed
        )
        _, pooled_oob = _population_lfc(problem, init, arm, qt, n_cells, seed)
        pop[arm] = (cells_fit, pooled_fit, pooled_oob)

    def series(arm, i):
        """``(out-of-the-box, calibrated, band)`` curves for reporter ``i``:
        the mean field's, or the population's with its 10–90 % band."""
        if i not in pop_idx:
            return lfc_oob[arm][i], lfc_fit[arm][i], None
        cells_fit, pooled_fit, pooled_oob = pop[arm]
        k = pop_idx.index(i)
        band = np.percentile(cells_fit[k], [10, 90], axis=1)
        return pooled_oob[k], pooled_fit[k], band

    # One row per constituent, in reporter order: the ragged right edge is
    # the composite's structure, and the spare cells hold the legend and
    # the population note.
    rows = []
    for i, r in enumerate(problem.reporters):
        ns = r.observable.split("/")[0]
        if not rows or rows[-1][0] != ns:
            rows.append((ns, []))
        rows[-1][1].append(i)
    nrow, ncol = len(rows), max(len(idxs) for _, idxs in rows)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(11, 3.2 * nrow), sharex=True, squeeze=False
    )
    spare = []
    for r, (ns, idxs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(idxs):
                ax.axis("off")
                spare.append(ax)
                continue
            i = idxs[c]
            gene = genes[i]
            ax.axhline(0, color=grid_c, lw=1.2, zorder=0)
            for arm, (label, color) in arms.items():
                oob_curve, fit_curve, band = series(arm, i)
                if band is not None:
                    ax.fill_between(
                        qt, band[0], band[1], color=color, alpha=0.15, lw=0
                    )
                ax.plot(
                    qt,
                    oob_curve,
                    color=color,
                    lw=1.5,
                    ls=(0, (4, 2)),
                    alpha=0.85,
                    zorder=2,
                    label=f"{label} — out-of-the-box",
                )
                ax.plot(
                    qt,
                    fit_curve,
                    color=color,
                    lw=2.2,
                    zorder=3,
                    label=f"{label} — calibrated",
                )
                dt = sorted(problem.data[arm])
                dx = [0.0] + list(dt)
                dy = [0.0] + [float(problem.data[arm][t][gene]) for t in dt]
                ax.plot(
                    dx,
                    dy,
                    "o",
                    color=color,
                    ms=6,
                    mfc="white",
                    mew=1.6,
                    zorder=4,
                )
            # Both arms are overlaid; annotate the one carrying the most
            # interventions so the shading is drawn once.
            _annotate_interventions(
                ax, max(arms, key=lambda a: ("rapa" in a.lower(), a))
            )
            ax.set_title(gene, fontsize=11, fontweight="bold", loc="left")
            ax.grid(True, color=grid_c, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if c == 0:
                ax.set_ylabel("log2FC")
                ax.annotate(
                    CONSTITUENT_LABELS.get(ns, ns)
                    + (
                        f" — {n_cells}-cell population"
                        if pop_idx and ns in REACTION_LEVEL
                        else ""
                    ),
                    xy=(0, 1.2),
                    xycoords="axes fraction",
                    fontsize=11,
                    color="#555",
                    ha="left",
                )
            below = r + 1 < nrow and c < len(rows[r + 1][1])
            if not below:
                ax.set_xlabel("day")
                ax.tick_params(labelbottom=True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fit_arms = set(getattr(problem, "fit_arms", None) or [])
    held_arms = set(getattr(problem, "held_out_arms", None) or [])
    grey = "#555"
    arm_entries = []
    for arm, (label, color) in arms.items():
        role = (
            " (fit)"
            if arm in fit_arms
            else " (held out)" if arm in held_arms else ""
        )
        arm_entries.append((Line2D([], [], color=color, lw=2.2), label + role))
    style_entries = [
        (Line2D([], [], color=grey, lw=1.5, ls=(0, (4, 2))), "published"),
        (Line2D([], [], color=grey, lw=2.2), "calibrated"),
    ]
    other_entries = [
        (
            Line2D(
                [], [], color=grey, marker="o", mfc="white", mew=1.6, ls="none"
            ),
            "measured",
        )
    ]
    for h, lab in zip(*axes[0, 0].get_legend_handles_labels()):
        if lab in ("etoposide", "rapamycin"):
            other_entries.append((h, lab))
    if pop_idx:
        other_entries.append(
            (Patch(color="#888", alpha=0.3), "cell-to-cell spread")
        )
    groups = [
        ("arm", arm_entries),
        ("parameters", style_entries),
        (None, other_entries),
    ]
    if spare:
        # one legend, the groups separated by a blank row, headed in bold
        blank = Patch(alpha=0)
        rows, headers = [], []
        for gi, (title, entries) in enumerate(groups):
            if gi:
                rows.append((blank, ""))
            if title:
                headers.append(len(rows))
                rows.append((blank, title))
            rows.extend(entries)
        leg = spare[0].legend(
            [h for h, _ in rows],
            [lab for _, lab in rows],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=10.5,
        )
        for i in headers:
            leg.get_texts()[i].set_fontweight("bold")
    else:
        entries = [e for _, es in groups for e in es]
        fig.legend(
            [h for h, _ in entries],
            [lab for _, lab in entries],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9.0,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.suptitle(
        "Gene reporter trajectories in treated and untreated arms",
        fontsize=13,
        x=0.5,
        ha="center",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.0 if spare else 0.05, 1, 0.96))
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_CAL / f"temporal_ddis_vs_rapa.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"wrote temporal_ddis_vs_rapa.png/.pdf -> {OUT_CAL}", flush=True)


# ── before-after (standalone vs composite) ───────────────────────────────
# One row per constituent: its deposit (for the SBML-defaults column) and the
# species worth plotting. Column 0 is the raw model at its published defaults,
# a permanent reference so a reparametrization break shows as a divergence.
BEFORE_AFTER_ROWS = [
    dict(
        namespace="dp14",
        sbml=DP14_SBML_PATH,
        ylabel="DP14\nspecies value",
        logy=True,
        dialled=True,
        vars=[
            ("dp14/mTORC1_pS2448", "mTORC1", "#6d28d9"),
            ("dp14/DNA_damage", "DNA damage", "#b91c1c"),
            ("dp14/ROS", "ROS", "#ca8a04"),
            ("dp14/CDKN1A", "p21 (CDKN1A)", "#0e7490"),
        ],
    ),
    dict(
        namespace="gz06",
        sbml=GZ06_SBML_PATH,
        ylabel="GZ06\np53 / Mdm2",
        vars=[
            ("gz06/x", "p53 (x)", "#6d28d9"),
            ("gz06/y", "Mdm2 (y)", "#b45309"),
        ],
    ),
    dict(
        namespace="p07",
        sbml=PROCTOR07_SBML_PATH,
        ylabel="UPS\nnative / misfolded / Ub",
        logy=True,
        vars=[
            ("p07/NatP", "native protein", "#0e7490"),
            ("p07/MisP", "misfolded", "#b91c1c"),
            ("p07/Ub", "free ubiquitin", "#6d28d9"),
            ("p07/AggP", "aggregates", "#ca8a04"),
        ],
    ),
]


def fig_before_after(args):
    """Standalone-vs-composite check on the composite the PIPELINE runs.

    Constituents and composite are both built from the calibration
    parameterization (``--params init``/``fit``) — not the raw SBML
    defaults — so this verifies that coupling doesn't distort the model
    actually calibrated. Constituent BEFORE ≈ composite AFTER means the
    coupling only adds the intended edges.
    """
    from hallsim.sbml_import import process_from_sbml
    from demos.models.multi_hallmark import CANONICAL_TIME_SECONDS

    out = ROOT / "outputs" / "multi_hallmark_before_after"
    # save_dt is decoupled from macro_dt: sample fine enough for the fastest
    # row (gz06's ~0.29 d p53 period) without changing the solve.
    t_end, macro_dt, save_dt = float(getattr(args, "t_end", 14.0)), 0.1, 0.001
    problem = _problem(args)
    if getattr(args, "params", "init") == "fit":
        pvals, tag = load_fit(), "calibrated fit"
    else:
        pvals = {k: float(v) for k, v in problem.initial_params().items()}
        tag = "calibration init (out-of-the-box)"
    pj = {k: jnp.asarray(v) for k, v in pvals.items()}
    cond, base = problem.arm_pairs["DDIS_vs_ctrl"]  # DDIS, control
    rows = [
        r
        for r in BEFORE_AFTER_ROWS
        if r["namespace"] in problem.composite.processes
    ]
    unknown = sorted(
        {
            n
            for n, pr in problem.composite.processes.items()
            if hasattr(pr, "_species_names")
        }
        - {r["namespace"] for r in BEFORE_AFTER_ROWS}
    )
    if unknown:
        raise RuntimeError(
            f"before_after has no row for {unknown}. Add one to "
            "BEFORE_AFTER_ROWS — a per-constituent check that silently skips "
            "a constituent checks nothing."
        )

    # Constituents and composite come from the pipeline's own substituted
    # processes; nothing is a hand-passed parameter value.
    def procs_of(cname):
        sub = problem._substitute(problem.composite.processes, pj)
        return apply_hallmarks(sub, problem.conditions[cname].hallmarks)

    def solo(proc, te, sdt):
        comp = Composite(
            {proc._name: proc},
            topology={},
            validate=False,
            semantic_validation={"check_semantics": False},
        )
        r = Scheduler(auto_stiffness=True).run(
            comp, (0.0, te), macro_dt=te, save_dt=sdt
        )
        return np.asarray(r.ts), r

    def solo_of(name, cname, te, sdt):
        return solo(procs_of(cname)[name], te, sdt)

    def run_comp(cname):
        comp = Composite(
            procs_of(cname),
            topology=problem.composite.topology,
            validate=False,
            semantic_validation={"check_semantics": False},
        )
        r = Scheduler(auto_stiffness=True).run(
            comp, (0.0, t_end), macro_dt=macro_dt, save_dt=save_dt
        )
        return np.asarray(r.ts), r

    def panel(ax, series, vars_, xlim, logy=False):
        for label_c, (t, res), ls in series:
            for path, label, col in vars_:
                ax.plot(
                    t,
                    np.asarray(res.get(path)),
                    ls,
                    color=col,
                    lw=1.4,
                    label=f"{label} · {label_c}" if ls == "-" else None,
                )
        ax.set_xlim(*xlim)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    out.mkdir(parents=True, exist_ok=True)
    ct, ctrl = run_comp(base)
    dt, ddis = run_comp(cond)
    composite_series = [
        ("control", (ct, ctrl), "-"),
        ("DDIS", (dt, ddis), "--"),
    ]

    fig, ax = plt.subplots(
        len(rows), 3, figsize=(19, 3.7 * len(rows)), squeeze=False
    )
    for title, col in (
        ("ORIGINAL — SBML defaults", 0),
        ("BEFORE — standalone (calibrated)", 1),
        ("AFTER — in composite", 2),
    ):
        ax[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i, row in enumerate(rows):
        ns, vars_, logy = row["namespace"], row["vars"], row.get("logy", False)
        default = solo(
            process_from_sbml(str(row["sbml"]), name=ns).reconciled_to(
                CANONICAL_TIME_SECONDS
            ),
            t_end,
            save_dt,
        )
        panel(ax[i, 0], [("default", default, "-")], vars_, (0, t_end), logy)
        # Only dp14 receives a severity dial standalone; the others have no
        # coupled input on their own, so one basal line is the whole story.
        standalone = (
            [
                ("control", solo_of(ns, base, t_end, save_dt), "-"),
                ("DDIS", solo_of(ns, cond, t_end, save_dt), "--"),
            ]
            if row.get("dialled")
            else [("standalone", solo_of(ns, base, t_end, save_dt), "-")]
        )
        panel(ax[i, 1], standalone, vars_, (0, t_end), logy)
        panel(ax[i, 2], composite_series, vars_, (0, t_end), logy)
        ax[i, 0].set_ylabel(row["ylabel"], fontsize=11)

    for row in ax:
        for a in row:
            a.set_xlabel("time (days)")
        row[0].legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle(
        f"Every component: SBML defaults → calibrated standalone → "
        f"in composite  ·  {tag}  (solid = control, dashed = DDIS)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"before_after.{ext}",
            dpi=140,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"wrote before_after.png/.pdf -> {out}", flush=True)


# ── composite graph (generated; cannot go stale) ─────────────────────────
def fig_composite_graph(args):
    """The composite's own interaction graph, from `hallsim.plotting`.

    Nothing here is drawn by hand: nodes, kinds, sizes and edge labels all
    come from the composite, so this figure describes whatever is actually
    composed. `composite_schematic` is the hand-laid presentation version.
    """
    from hallsim.plotting import draw_composite_graph

    problem = _problem(args)
    models = sum(
        hasattr(p, "_species_names")
        for p in problem.composite.processes.values()
    )
    fig = draw_composite_graph(
        problem.composite,
        title=f"{models} imported models on one clock, and every path "
        "between them",
    )
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_CAL / f"composite_graph.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"wrote composite_graph.png/.pdf -> {OUT_CAL}", flush=True)


# ── coupling ablation (the null a composition claim has to beat) ─────────
def fig_coupling_ablation(args):
    """Wired composite vs the same models with every edge frozen.

    Holds constituents, clock, parameters and readout fixed and removes only
    the variation the cross-publication edges transmit
    (:mod:`hallsim.ablation`), then re-scores the *same* fitted vector.
    """
    from scipy.stats import spearmanr

    from hallsim.ablation import freeze_coupling, trajectory_levels
    from demos.multi_hallmark_calibrate import build_problem

    C_WIRED, C_NULL, DIM, INK = "#0173b2", "#b0b7c3", "#5b6b7d", "#1f2530"

    problem = _problem(args)
    fitted = load_fit()
    genes = [r.gene_symbol for r in problem.reporters]
    days = sorted({d for arm in problem.data.values() for d in arm})
    qt = jnp.asarray(days)

    # Freeze at what each edge actually saw in the control arm — a source's
    # declared value is a published starting point, not a rest level.
    ctrl = Scheduler(auto_stiffness=True).run(
        with_hallmarks(
            problem.composite, problem.conditions["ctrl"].hallmarks
        ),
        t_span=(0.0, args.t_end),
        macro_dt=0.5,
        save_dt=args.t_end / 149,
    )
    levels = trajectory_levels(problem.composite, ctrl)
    null = build_problem(
        composite=freeze_coupling(problem.composite, levels),
        fitted=tuple(problem.param_refs),
    )

    rows, labels = [], []
    for arm in problem.arm_pairs:
        for j, day in enumerate(days):
            measured = np.array([problem.data[arm][day][g] for g in genes])
            scores = []
            for pr in (problem, null):
                sim = np.asarray(pr.model_lfc(fitted, arm, qt), float)[:, j]
                scores.append(
                    (
                        float(spearmanr(sim, measured).statistic),
                        int(np.sum(np.sign(sim) == np.sign(measured))),
                    )
                )
            rows.append(scores)
            labels.append(
                f"{arm.split('_')[0]}\nD{int(day):02d}"
                + ("\n(held out)" if arm in problem.held_out_arms else "")
            )

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    lo = min(r[k][0] for r in rows for k in (0, 1))
    for k, (color, name) in enumerate(
        ((C_WIRED, "wired composite"), (C_NULL, "every edge frozen"))
    ):
        bars = ax.bar(
            x + (k - 0.5) * 0.38,
            [r[k][0] for r in rows],
            0.36,
            color=color,
            label=name,
            zorder=3,
        )
        for b, r in zip(bars, rows):
            ax.text(
                b.get_x() + b.get_width() / 2,
                max(b.get_height(), 0.0) + 0.035,
                f"{r[k][1]}/{len(genes)}",
                ha="center",
                fontsize=8.4,
                color=DIM,
            )
    ax.set_xticks(x, labels, fontsize=9.2)
    ax.set_ylabel("Spearman ρ, model vs measured log2FC")
    ax.set_ylim(min(lo - 0.15, -0.05), 1.18)
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.2, loc="lower left", ncols=2)
    ax.set_title(
        "Freezing the cross-publication edges\ncosts rank agreement at "
        "every arm-day",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=26,
    )
    ax.text(
        0,
        1.015,
        "same constituents, clock, fitted parameters and readout;\nonly the "
        "signal the edges carry is removed.  labels: sign agreement",
        transform=ax.transAxes,
        fontsize=8.8,
        color=DIM,
        va="bottom",
    )
    fig.tight_layout()
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_CAL / f"coupling_ablation.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"wrote coupling_ablation.png/.pdf -> {OUT_CAL}", flush=True)


def fig_training(args):
    """Loss (top) + gradient norm (bottom), stacked on a shared epoch axis.

    Reads the persisted history from the latest run's ``summary.json`` — the
    grad-norm panel is the end-to-end-differentiability evidence (backprop
    flows through the composed stiff-ODE stack and decays to convergence)."""
    import json

    C_LOSS, C_GRAD, DIM = "#3a3f4a", "#de8f05", "#5b6b7d"
    with open(OUT_CAL / "summary.json") as f:
        s = json.load(f)
    losses = np.asarray(s["loss_history"], dtype=float)
    grads = np.asarray(s.get("grad_norm_history", []), dtype=float)
    epochs = np.arange(1, len(losses) + 1)

    fig, (axL, axG) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), gridspec_kw={"wspace": 0.22}
    )
    axL.plot(epochs, losses, color=C_LOSS, lw=2.0)
    axL.set_yscale("log")
    axL.set_ylabel("loss  (log2 FC MSE)")
    axL.set_xlabel("epoch")
    axL.set_title("Fit converges", fontsize=11, color=DIM, loc="left")

    if grads.size:
        axG.plot(epochs[: len(grads)], grads, color=C_GRAD, lw=2.0)
        axG.set_yscale("log")
        axG.set_title(
            "Gradients propagate through the composite",
            fontsize=11,
            color=DIM,
            loc="left",
        )
    else:
        axG.text(
            0.5,
            0.5,
            "no grad_norm_history in summary.json\n(re-run the fit)",
            ha="center",
            va="center",
            color=DIM,
            transform=axG.transAxes,
        )
    axG.set_ylabel("gradient norm")
    axG.set_xlabel("epoch")

    for ax in (axL, axG):
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01)

    fig.suptitle(
        "End-to-end differentiable calibration of the composite",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_CAL / f"training_curves.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"wrote training_curves.png/.pdf -> {OUT_CAL}", flush=True)


def fig_proteostasis_population(args):
    """Proctor 2007 at reaction level on the fitted trajectory, as a
    population: its two reporters per cell, the population mean, the mean
    field the calibration used, and the data.

    Calibration goes through the mean field; nothing here is fitted. Each
    cell's fold-change is the reporter machinery applied to that cell's own
    trajectory (its own day-0 reference, as the loss does); the population
    mean divides pooled readouts, as a bulk assay does.
    """
    import numpy as np

    problem = _problem(args)
    params = {k: jnp.asarray(v) for k, v in load_fit().items()}
    n_cells = int(getattr(args, "n_cells", 64))
    seed = int(getattr(args, "seed", 0))
    arms = list(problem.arm_pairs)
    rep_idx = _reaction_level_reporters(problem)
    genes = [problem.reporters[i].gene_symbol for i in rep_idx]

    rows = {}
    for arm in arms:
        days = sorted(float(t) for t in problem.data[arm])
        mean_field = np.asarray(
            problem.model_lfc(params, arm, jnp.asarray(days))
        )[rep_idx]
        cells, pooled = _population_lfc(
            problem, params, arm, days, n_cells, seed
        )
        measured = np.asarray(
            [[float(problem.data[arm][t][g]) for t in days] for g in genes]
        )
        rows[arm] = dict(
            days=days,
            cells=cells,
            pooled=pooled,
            mean_field=mean_field,
            measured=measured,
        )

    fig, axes = plt.subplots(
        len(genes),
        len(arms),
        figsize=(4.4 * len(arms), 3.2 * len(genes)),
        squeeze=False,
        sharex=True,
    )
    rng = np.random.default_rng(0)
    for j, arm in enumerate(arms):
        r = rows[arm]
        for i, gene in enumerate(genes):
            ax = axes[i, j]
            for k, day in enumerate(r["days"]):
                jitter = rng.uniform(-0.25, 0.25, n_cells)
                ax.scatter(
                    day + jitter,
                    r["cells"][i, k],
                    s=9,
                    color="#9aa0a6",
                    alpha=0.5,
                    linewidths=0,
                    label="one cell" if k == 0 else None,
                )
            ax.plot(
                r["days"],
                r["pooled"][i],
                "D-",
                color="#1f1f1f",
                ms=6,
                label=f"population mean, N={n_cells}",
            )
            ax.plot(
                r["days"],
                r["mean_field"][i],
                "s--",
                color="#d97706",
                ms=6,
                label="mean field (calibrated)",
            )
            ax.plot(
                r["days"],
                r["measured"][i],
                "o",
                color="#c62828",
                ms=7,
                mfc="none",
                mew=1.6,
                label="measured (GSE248823)",
            )
            ax.axhline(0, color="#ddd", lw=1)
            ax.set_title(
                f"{gene}  ·  {arm.replace('_vs_', ' vs ')}", fontsize=10
            )
            if j == 0:
                ax.set_ylabel("log2 fold-change")
            if i == len(genes) - 1:
                ax.set_xlabel("day")
                ax.set_xticks(r["days"])
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Proctor's reporters: one cell is quantised, the population mean "
        "tracks the mean field",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        f"Proctor 2007 at reaction level on the calibrated trajectory, "
        f"{n_cells} cells, seed {seed}; calibration through the mean field",
        ha="center",
        fontsize=9,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    stem = "proteostasis_population"
    for ext in ("png", "pdf"):
        fig.savefig(OUT_CAL / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {stem}.png -> {OUT_CAL}", flush=True)
    for arm in arms:
        r = rows[arm]
        for i, gene in enumerate(genes):
            print(
                f"  {arm} {gene}: days {r['days']} measured {r['measured'][i].round(3).tolist()} "
                f"mean-field {r['mean_field'][i].round(3).tolist()} pooled {r['pooled'][i].round(3).tolist()} "
                f"cell sd {r['cells'][i].std(axis=1).round(2).tolist()}",
                flush=True,
            )


FIGURES = {
    "schematic": fig_schematic,
    "training": fig_training,
    "trajectories": fig_trajectories,
    "reporter-levels": fig_reporter_levels,
    "concordance": fig_concordance,
    "temporal": fig_temporal,
    "temporal-compare": fig_temporal_compare,
    "before-after": fig_before_after,
    "coupling-ablation": fig_coupling_ablation,
    "composite-graph": fig_composite_graph,
    "proteostasis-population": fig_proteostasis_population,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figure", choices=list(FIGURES) + ["all"])
    ap.add_argument(
        "--t-end",
        type=float,
        default=14.0,
        help="day horizon for the before-after figure.",
    )
    ap.add_argument(
        "--params",
        choices=("init", "fit"),
        default="init",
        help="before-after parameterization: calibration init "
        "(out-of-the-box) or the saved fit.",
    )
    ap.add_argument(
        "--run",
        default=None,
        help="a calibration run directory to draw from instead of the "
        "latest (its checkpoint and summary).",
    )
    ap.add_argument(
        "--n-cells",
        type=int,
        default=64,
        help="cells in the proteostasis-population figure.",
    )
    ap.add_argument("--seed", type=int, default=0, help="population seed.")
    args = ap.parse_args()
    if args.run:
        global OUT_CAL, _CKPT
        OUT_CAL = Path(args.run).resolve()
        _CKPT = OUT_CAL / "checkpoint.npz"
    todo = FIGURES.values() if args.figure == "all" else [FIGURES[args.figure]]
    for fn in todo:
        fn(args)


if __name__ == "__main__":
    main()
