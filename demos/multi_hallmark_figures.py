"""Figures for the multi-hallmark flagship composite — one CLI, many panels.

Consolidates the per-figure scripts into subcommands (pick with the first arg):

  schematic      wiring diagram of the composite (dials → dp14 → gz06/ih04).
  trajectories   reporter observables over the day axis, ctrl/DDIS/DDIS+rapa.
  reporter-levels calibrated reporter dynamics per condition (model units).
  concordance    measured-vs-simulated dumbbells per gene (oob + calibrated).
  temporal       out-of-the-box → calibrated log2FC trajectories vs data.
  before-after   each constituent standalone vs inside the composite.

Run: python demos/multi_hallmark_figures.py <figure>   (or `all`)
Calibration itself lives in multi_hallmark_calibrate.py; the flagship
NeuralODE-hybrid swap in multi_hallmark_hybrid.py.
"""

from __future__ import annotations

import argparse
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


# ── schematic ────────────────────────────────────────────────────────────
def fig_schematic(args):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    C_DP, C_GZ, C_NF, C_DIAL = "#2563eb", "#5b3fc4", "#c0392b", "#374151"
    INK, DIM, BODY = "#111827", "#475569", "#1f2937"
    F_DP, F_GZ, F_NF, F_DIAL = "#eef4fd", "#f3f0fb", "#fdf0ef", "#f4f5f7"

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

    def arrow(ax, p0, p1, color, rad=0.0, lw=2.0):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=lw,
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

    def reporters(ax, x, y, text, color, fs=9.8):
        ax.text(
            x,
            y,
            text,
            fontsize=fs,
            color=color,
            ha="center",
            va="center",
            style="italic",
            zorder=5,
        )

    from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS

    def readouts_for(namespace):
        genes = [
            r.gene_symbol
            for r in MULTI_HALLMARK_REPORTERS
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
        fontsize=11.5,
        color=INK,
        fontweight="bold",
        ha="center",
    )
    ax.text(1.65, 3.86, "severity 0-1", fontsize=9.2, color=DIM, ha="center")
    ax.text(
        1.65, 3.63, "→ etoposide dose", fontsize=9.2, color=DIM, ha="center"
    )
    block(ax, 0.4, 1.1, 2.5, 1.18, "#94a3b8", F_DIAL, r=0.10, lw=1.7)
    ax.text(
        1.65,
        2.02,
        "Deregulated",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        1.65,
        1.77,
        "Nutrient Sensing",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
        ha="center",
    )
    ax.text(1.65, 1.47, "severity 0-1", fontsize=9.2, color=DIM, ha="center")
    ax.text(
        1.65,
        1.24,
        "→ mTORC1 (rapamycin)",
        fontsize=9.2,
        color=DIM,
        ha="center",
    )
    block(ax, 3.7, 1.95, 2.6, 2.0, C_DP, F_DP)
    ax.text(
        5.0,
        3.62,
        "dp14",
        fontsize=15,
        color=C_DP,
        fontweight="bold",
        ha="center",
    )
    ax.text(5.0, 3.30, "BIOMD582", fontsize=9.4, color=DIM, ha="center")
    for k, line in enumerate(
        ["mTOR · AMPK · FoxO3a", "mitophagy · ROS · DNA damage", "CDKN1A"]
    ):
        ax.text(
            5.0, 2.98 - 0.30 * k, line, fontsize=9.6, color=BODY, ha="center"
        )
    reporters(
        ax,
        5.0,
        1.64,
        readouts_for("dp14"),
        C_DP,
        fs=9.2,
    )
    block(ax, 8.9, 3.72, 3.1, 1.2, C_GZ, F_GZ)
    ax.text(
        10.45,
        4.60,
        "gz06",
        fontsize=15,
        color=C_GZ,
        fontweight="bold",
        ha="center",
    )
    ax.text(10.45, 4.28, "BIOMD157", fontsize=9.4, color=DIM, ha="center")
    ax.text(
        10.45,
        4.02,
        "p53–Mdm2 oscillator",
        fontsize=9.6,
        color=BODY,
        ha="center",
    )
    reporters(ax, 10.45, 3.50, readouts_for("gz06"), C_GZ)
    block(ax, 8.9, 0.9, 3.1, 1.2, C_NF, F_NF)
    ax.text(
        10.45,
        1.78,
        "ih04",
        fontsize=15,
        color=C_NF,
        fontweight="bold",
        ha="center",
    )
    ax.text(10.45, 1.46, "BIOMD230", fontsize=9.4, color=DIM, ha="center")
    ax.text(10.45, 1.20, "NF-κB / IκBα", fontsize=9.6, color=BODY, ha="center")
    reporters(ax, 10.45, 0.68, readouts_for("nfkb"), C_NF)
    arrow(ax, (2.9, 3.82), (3.7, 3.28), C_DIAL, rad=-0.14)
    arrow(ax, (2.9, 1.88), (3.7, 2.55), C_DIAL, rad=0.14)
    arrow(ax, (6.3, 3.45), (8.9, 4.25), C_GZ, rad=0.14)
    elabel(ax, 7.6, 4.05, "DNA damage → ψ", C_GZ, 17)
    arrow(ax, (6.3, 2.8), (8.9, 1.82), C_DP, rad=-0.11)
    elabel(ax, 7.6, 2.44, "mTOR → IKK", C_DP, -17)
    arrow(ax, (6.3, 2.3), (8.9, 1.32), C_NF, rad=-0.14)
    elabel(ax, 7.55, 1.6, "IKKβ → IKK", C_NF, -17)
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
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

    arms = [
        (0.0, 0.5, "ctrl", "tab:green"),
        (1.0, 1.0, "DDIS", "tab:red"),
        (1.0, 0.3, "DDIS+rapa", "tab:blue"),
    ]
    from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS

    # Read the reporter set; plot the raw underlying state for integral-based
    # readouts (the cumulative ∫ path isn't a useful trajectory).
    panels = [
        (r.observable.replace("_integral", ""), r.gene_symbol)
        for r in MULTI_HALLMARK_REPORTERS
    ]

    def run(gi, dns):
        base = build_multi_hallmark_composite()
        comp = with_hallmarks(
            base,
            {"Genomic Instability": gi, "Deregulated Nutrient Sensing": dns},
        )
        return Scheduler().run(
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
    from multi_hallmark_calibrate import build_problem

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

    problem = build_problem()
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
    from multi_hallmark_calibrate import build_problem, _rows_by_gene

    C_DATA, C_MODEL, INK, DIM, BAND = (
        "#2563eb",
        "#d97706",
        "#1f2937",
        "#6b7280",
        "#f1f5f9",
    )
    problem = build_problem()
    arms_order = ["DDIS_vs_ctrl", "RAPA_vs_ctrl", "RAS_vs_ctrl"]
    short = {
        "DDIS_vs_ctrl": "DDIS",
        "RAPA_vs_ctrl": "RAPA",
        "RAS_vs_ctrl": "RAS",
    }
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
        rows = [_rows_by_gene(ev[a][t]) for a, t in cond]
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
    from multi_hallmark_calibrate import build_problem, _annotate_interventions

    C_OOB, C_FIT, C_DATA, grid_c = "#9a9a95", "#2a78d6", "#0b0b0b", "#e6e6e2"
    arms = {
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

    problem = build_problem()
    init = problem.initial_params()
    fit = {k: jnp.asarray(v) for k, v in load_fit().items()}
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    for arm, subtitle in arms.items():
        figure_for_arm(problem, init, fit, arm, subtitle)
    print(f"→ {OUT_CAL}", flush=True)


# ── temporal DDIS vs RAPA overlay (the rapamycin divergence) ─────────────
def fig_temporal_compare(args):
    """One panel per reporter overlaying the calibrated DDIS (etoposide) and
    RAPA (etoposide + rapamycin @ day 2) trajectories, with each arm's measured
    points. Directly visualizes the held-out rapamycin effect per reporter."""
    from multi_hallmark_calibrate import build_problem, _annotate_interventions

    arms = {
        "DDIS_vs_ctrl": ("DDIS", "#c0392b"),
        "RAPA_vs_ctrl": ("rapamycin", "#2a78d6"),
    }
    grid_c = "#e6e6e2"
    t_end = 14.0
    qt = np.arange(0.1, t_end + 1e-6, 0.1)

    problem = build_problem()
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

    n = len(genes)
    ncol = 3
    nrow = -(-n // ncol)
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
        for arm, (label, color) in arms.items():
            ax.plot(
                qt,
                lfc_oob[arm][i],
                color=color,
                lw=1.5,
                ls=(0, (4, 2)),
                alpha=0.85,
                zorder=2,
                label=f"{label} — out-of-the-box",
            )
            ax.plot(
                qt,
                lfc_fit[arm][i],
                color=color,
                lw=2.2,
                zorder=3,
                label=f"{label} — calibrated",
            )
            dt = sorted(problem.data[arm])
            dx = [0.0] + list(dt)
            dy = [0.0] + [float(problem.data[arm][t][gene]) for t in dt]
            ax.plot(
                dx, dy, "o", color=color, ms=6, mfc="white", mew=1.6, zorder=4
            )
        _annotate_interventions(ax, "RAPA_vs_ctrl")
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
        fontsize=9.0,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Reporter trajectories — DDIS (fit) vs rapamycin (held-out), "
        "out-of-the-box vs calibrated",
        fontsize=12.5,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
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
def fig_before_after(args):
    """Standalone-vs-composite check on the composite the PIPELINE runs.

    Constituents and composite are both built from the calibration
    parameterization (``--params init``/``fit``) — not the raw SBML
    defaults — so this verifies that coupling doesn't distort the model
    actually calibrated. Constituent BEFORE ≈ composite AFTER means the
    coupling only adds the intended edges.
    """
    from multi_hallmark_calibrate import build_problem
    from hallsim.sbml_import import process_from_sbml
    from hallsim.models.multi_hallmark import (
        CANONICAL_TIME_SECONDS,
        DP14_SBML_PATH,
        GZ06_SBML_PATH,
        NFKB_SBML_PATH,
    )

    out = ROOT / "outputs" / "multi_hallmark_before_after"
    # save_dt is decoupled from macro_dt: sample fine enough for the fastest
    # row (NF-kB, ~100 min period) without changing the solve.
    t_end, macro_dt, save_dt = float(getattr(args, "t_end", 14.0)), 0.1, 0.001
    problem = build_problem()
    if getattr(args, "params", "init") == "fit":
        pvals, tag = load_fit(), "calibrated fit"
    else:
        pvals = {k: float(v) for k, v in problem.initial_params().items()}
        tag = "calibration init (out-of-the-box)"
    pj = {k: jnp.asarray(v) for k, v in pvals.items()}
    cond, base = problem.arm_pairs["DDIS_vs_ctrl"]  # DDIS, control
    dp14_vars = [
        ("dp14/mTORC1_pS2448", "mTORC1", "#6d28d9"),
        ("dp14/DNA_damage", "DNA damage", "#b91c1c"),
        ("dp14/ROS", "ROS", "#ca8a04"),
        ("dp14/CDKN1A", "p21 (CDKN1A)", "#0e7490"),
    ]
    gz06_vars = [
        ("gz06/x", "p53 (x)", "#6d28d9"),
        ("gz06/y", "Mdm2 (y)", "#b45309"),
    ]
    nfkb_vars = [
        ("nfkb/IKK", "IKK", "#b91c1c"),
        ("nfkb/IkBat", "IkBa transcript", "#0e7490"),
        ("nfkb/NFkBn", "NF-kB nuclear", "#6d28d9"),
    ]

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
        r = Scheduler().run(comp, (0.0, te), macro_dt=te, save_dt=sdt)
        return np.asarray(r.ts), r

    def solo_of(name, cname, te, sdt):
        return solo(procs_of(cname)[name], te, sdt)

    # Raw model at its published SBML defaults — no calibration, no
    # hallmark. A permanent reference so any reparametrization break shows
    # up as a divergence from this column.
    def solo_default(sbml_path, nm, te, sdt):
        p = process_from_sbml(str(sbml_path), name=nm).reconciled_to(
            CANONICAL_TIME_SECONDS
        )
        return solo(p, te, sdt)

    def run_comp(cname):
        comp = Composite(
            procs_of(cname),
            topology=problem.composite.topology,
            validate=False,
            semantic_validation={"check_semantics": False},
        )
        r = Scheduler().run(
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
    # gz06/nfkb have no coupled input standalone, so they're condition-
    # independent: one basal line. Both over the full run — gz06's DDIS
    # limit cycle has a delayed onset (psi ramps through its Hopf over
    # days), invisible in a short window.
    gz = solo_of("gz06", base, t_end, save_dt)
    nf = solo_of("nfkb", base, t_end, save_dt)
    # Col 0: raw models at SBML defaults (reparametrization reference).
    dp0 = solo_default(DP14_SBML_PATH, "dp14", t_end, save_dt)
    gz0 = solo_default(GZ06_SBML_PATH, "gz06", t_end, save_dt)
    nf0 = solo_default(NFKB_SBML_PATH, "nfkb", t_end, save_dt)

    fig, ax = plt.subplots(3, 3, figsize=(19, 11))
    ax[0, 0].set_title(
        "ORIGINAL — SBML defaults", fontsize=12, fontweight="bold"
    )
    ax[0, 1].set_title(
        "BEFORE — standalone (calibrated)", fontsize=12, fontweight="bold"
    )
    ax[0, 2].set_title("AFTER — in composite", fontsize=12, fontweight="bold")
    panel(ax[0, 0], [("default", dp0, "-")], dp14_vars, (0, t_end), logy=True)
    panel(
        ax[0, 1],
        [
            ("control", solo_of("dp14", base, t_end, save_dt), "-"),
            ("DDIS", solo_of("dp14", cond, t_end, save_dt), "--"),
        ],
        dp14_vars,
        (0, t_end),
        logy=True,
    )
    panel(
        ax[0, 2],
        [("control", (ct, ctrl), "-"), ("DDIS", (dt, ddis), "--")],
        dp14_vars,
        (0, t_end),
        logy=True,
    )
    ax[0, 0].set_ylabel("DP14\nspecies value", fontsize=11)
    panel(
        ax[1, 0], [("default", (gz0[0], gz0[1]), "-")], gz06_vars, (0, t_end)
    )
    panel(
        ax[1, 1], [("standalone", (gz[0], gz[1]), "-")], gz06_vars, (0, t_end)
    )
    panel(
        ax[1, 2],
        [("control", (ct, ctrl), "-"), ("DDIS", (dt, ddis), "--")],
        gz06_vars,
        (0, t_end),
    )
    ax[1, 0].set_ylabel("GZ06\np53 / Mdm2", fontsize=11)
    panel(
        ax[2, 0], [("default", (nf0[0], nf0[1]), "-")], nfkb_vars, (0, t_end)
    )
    panel(
        ax[2, 1], [("standalone", (nf[0], nf[1]), "-")], nfkb_vars, (0, t_end)
    )
    panel(
        ax[2, 2],
        [("control", (ct, ctrl), "-"), ("DDIS", (dt, ddis), "--")],
        nfkb_vars,
        (0, t_end),
    )
    ax[2, 0].set_ylabel("NFKB\nIKK / IkBa / NF-kB", fontsize=11)
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


FIGURES = {
    "schematic": fig_schematic,
    "trajectories": fig_trajectories,
    "reporter-levels": fig_reporter_levels,
    "concordance": fig_concordance,
    "temporal": fig_temporal,
    "temporal-compare": fig_temporal_compare,
    "before-after": fig_before_after,
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
    args = ap.parse_args()
    todo = FIGURES.values() if args.figure == "all" else [FIGURES[args.figure]]
    for fn in todo:
        fn(args)


if __name__ == "__main__":
    main()
