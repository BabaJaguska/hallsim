"""Temporal out-of-the-box → calibrated figure for the flagship composite.

For each gene reporter, draws the model's predicted log2 fold-change *as a
trajectory over the 14-day horizon* — out-of-the-box (dashed) vs calibrated
(solid) — with the measured microarray points overlaid at their sampling
days. Trajectories are grid-independent: the reporter summaries interpolate
each RunningIntegral at any query time.

    .venv_hallsim/bin/python demos/multi_hallmark_temporal_figure.py

Writes temporal_oob_vs_fit.{png,pdf} under outputs/multi_hallmark_calibrate/.
"""
from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from multi_hallmark_calibrate import build_problem  # noqa: E402
from hallsim.composite import Composite  # noqa: E402
from hallsim.hallmarks import apply_hallmarks  # noqa: E402

# Fitted parameters from the 150-step calibration (best loss 0.3445).
FIT = {
    "etoposide_potency": 7.724,
    "ROS_turnover": 0.8289,
    "CDKN1A_transcr": 0.0275,
    "mitophagy_inactiv": 648.37,
    "alpha_y": 0.80233,
    "psi_basal": 0.3188,
    "damage_to_nfkb": 0.066146,
    "mtor_to_nfkb": 0.14229,
}

# Role palette (dataviz skill: color by role, not per-series rainbow).
C_OOB = "#9a9a95"    # muted — out-of-the-box
C_FIT = "#2a78d6"    # series-1 blue — calibrated
C_DATA = "#0b0b0b"   # ink — measurement
GRID = "#e6e6e2"

ARMS = {
    "DDIS_vs_ctrl": "DDIS vs control  (etoposide, fit arm)",
    "RAPA_vs_DDIS": "rapamycin vs DDIS  (held-out intervention)",
    "RAS_vs_ctrl": "RAS vs control  (oncogene-induced senescence, held-out)",
}
MACRO_DT = 1.0       # finer than the fit's 3.5 for smooth curves (more accurate)
T_END = 14.0


def reporter_lfc_curve(problem, params, cond, base, qt):
    """sign·(log2 summ_cond − log2 summ_base) per reporter at query times qt."""
    sub = problem._substitute(problem.composite.processes, params)

    def run(cname):
        procs = apply_hallmarks(sub, problem.conditions[cname].hallmarks)
        comp = Composite(processes=procs, topology=problem.composite.topology,
                         validate=False, semantic_validation=False)
        res = problem._scheduler.run(comp, t_span=(0.0, T_END),
                                     macro_dt=MACRO_DT, y0=comp.initial_state_vec(),
                                     save_dt=MACRO_DT)
        trajs = jnp.stack([res.ys[..., i] for i in problem._reporter_indices])
        return res.ts, trajs

    ts_c, tr_c = run(cond)
    ts_b, tr_b = run(base)
    sc = problem._reporter_summaries(ts_c, tr_c, qt)
    sb = problem._reporter_summaries(ts_b, tr_b, qt)
    signs = jnp.asarray([float(r.sign) for r in problem.reporters])[:, None]
    return signs * (jnp.log2(jnp.maximum(sc, 1e-12))
                    - jnp.log2(jnp.maximum(sb, 1e-12)))


def figure_for_arm(problem, init, fit, arm, subtitle, out):
    cond, base = problem.arm_pairs[arm]
    data_times = sorted(problem.data[arm])
    genes = [r.gene_symbol for r in problem.reporters]

    qt = jnp.arange(0.5, T_END + 1e-6, 0.5)
    lfc_oob = np.asarray(reporter_lfc_curve(problem, init, cond, base, qt))
    lfc_fit = np.asarray(reporter_lfc_curve(problem, fit, cond, base, qt))
    qt = np.asarray(qt)
    # Anchor both curves at (0, 0): at t=0 the two conditions share the same
    # initial state, so every reporter's log2 fold-change is exactly 0.
    qt = np.concatenate([[0.0], qt])
    z = np.zeros((lfc_oob.shape[0], 1))
    lfc_oob = np.concatenate([z, lfc_oob], axis=1)
    lfc_fit = np.concatenate([z, lfc_fit], axis=1)

    # Coupling check: fit-curve value AT the sampled days vs the data.
    chk = np.asarray(reporter_lfc_curve(problem, fit, cond, base,
                                        jnp.asarray(data_times)))
    print(f"[{arm}] fit-curve @ sampled days (verify vs table):", flush=True)
    for i, g in enumerate(genes):
        vals = "  ".join(f"d{t:g}={chk[i, j]:+.3f}"
                         for j, t in enumerate(data_times))
        print(f"    {g:<9} {vals}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    for i, (ax, gene) in enumerate(zip(axes.ravel(), genes)):
        ax.axhline(0, color=GRID, lw=1.2, zorder=0)
        ax.plot(qt, lfc_oob[i], color=C_OOB, lw=1.8, ls=(0, (4, 2)),
                zorder=2, label="out-of-the-box")
        ax.plot(qt, lfc_fit[i], color=C_FIT, lw=2.2, zorder=3,
                label="calibrated")
        # Every reporter's data is a log2 fold-change vs the shared D00
        # baseline, which is 0 by definition — draw it as the common anchor
        # so all three sampling days are visible and start from one zero.
        dx = [0.0] + list(data_times)
        dy = [0.0] + [float(problem.data[arm][t][gene]) for t in data_times]
        ax.plot(dx, dy, "o", color=C_DATA, ms=7, zorder=4,
                label="measured (GSE248823)")
        ax.set_title(gene, fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel("log2 fold-change")
        if i >= 3:
            ax.set_xlabel("day")
    # Single figure-level legend below the panels — never inside an axes,
    # so its "measured" swatch can't be misread as a data point.
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{subtitle} — reporter trajectories, out-of-the-box vs "
                 "calibrated", fontsize=12.5, x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    stem = f"temporal_oob_vs_fit_{arm}"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf", flush=True)


def main():
    problem = build_problem()
    init = {k: jnp.asarray(p.init) for k, p in problem.params.items()}
    fit = {k: jnp.asarray(v) for k, v in FIT.items()}
    out = Path(__file__).resolve().parent.parent / "outputs" / \
        "multi_hallmark_calibrate"
    out.mkdir(parents=True, exist_ok=True)
    for arm, subtitle in ARMS.items():
        figure_for_arm(problem, init, fit, arm, subtitle, out)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
