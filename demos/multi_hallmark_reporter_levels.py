"""Reporter-level trajectories per condition for the flagship composite.

The intuitive view: each gene reporter's model observable over the 14-day
horizon, one line per experimental condition (control, etoposide/DDIS,
etoposide+rapamycin, oncogene/RAS), at the calibrated parameters. Shows how
the composite behaves mechanistically in each situation.

These are model observables in the model's own units (not fold-changes), so
the microarray data is not overlaid here — the temporal_oob_vs_fit figures
carry the data comparison.

    .venv_hallsim/bin/python demos/multi_hallmark_reporter_levels.py
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

FIT = {
    "etoposide_potency": 7.724, "ROS_turnover": 0.8289,
    "CDKN1A_transcr": 0.0275, "mitophagy_inactiv": 648.37,
    "alpha_y": 0.80233, "psi_basal": 0.3188,
    "damage_to_nfkb": 0.066146, "mtor_to_nfkb": 0.14229,
}

# condition -> (display label, colour). Categorical by identity, fixed order.
# The oncogene (RAS) arm is omitted here: it is mapped to the same
# full-senescence severities as etoposide and the model has no oncogene-
# specific mechanism, so its dynamics coincide exactly with the etoposide
# line. RAS earns its keep as a held-out *data* comparison (fold-change
# figures), not as a distinct model trajectory.
CONDS = {
    "ctrl": ("control", "#9a9a95"),
    "DDIS": ("etoposide (DDIS)", "#2a78d6"),
    "RAPA": ("etoposide + rapamycin", "#1baf7a"),
}
GRID = "#e6e6e2"
MACRO_DT = 1.0
T_END = 14.0


def reporter_levels(problem, params, cond_name, qt):
    """Each reporter's summary observable at query times qt (n_rep, n_qt)."""
    sub = problem._substitute(problem.composite.processes, params)
    procs = apply_hallmarks(sub, problem.conditions[cond_name].hallmarks)
    comp = Composite(processes=procs, topology=problem.composite.topology,
                     validate=False, semantic_validation=False)
    res = problem._scheduler.run(comp, t_span=(0.0, T_END), macro_dt=MACRO_DT,
                                 y0=comp.initial_state_vec(), save_dt=MACRO_DT)
    trajs = jnp.stack([res.ys[..., i] for i in problem._reporter_indices])
    return np.asarray(problem._reporter_summaries(res.ts, trajs, qt))


def main():
    problem = build_problem()
    fit = {k: jnp.asarray(v) for k, v in FIT.items()}
    qt = jnp.arange(0.5, T_END + 1e-6, 0.5)
    genes = [r.gene_symbol for r in problem.reporters]
    obs = [r.observable for r in problem.reporters]

    levels = {c: reporter_levels(problem, fit, c, qt) for c in CONDS}
    qt = np.asarray(qt)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True)
    for i, (ax, gene, ob) in enumerate(zip(axes.ravel(), genes, obs)):
        for c, (label, color) in CONDS.items():
            ax.plot(qt, levels[c][i], color=color, lw=2.2, label=label,
                    zorder=3)
        ax.set_title(f"{gene}", fontsize=11, fontweight="bold", loc="left")
        ax.annotate(ob, (0.5, 1.005), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=7.5, color="#6f6e6a")
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel("reporter level (model units)")
        if i >= 3:
            ax.set_xlabel("day")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Reporter dynamics per condition (calibrated model)",
                 fontsize=12.5, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out = Path(__file__).resolve().parent.parent / "outputs" / \
        "multi_hallmark_calibrate"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"reporter_levels_by_condition.{ext}", dpi=150,
                    bbox_inches="tight")
    print(f"wrote reporter_levels_by_condition.png/.pdf → {out}", flush=True)


if __name__ == "__main__":
    main()
