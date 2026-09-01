"""Calibrate the multi-hallmark composite against GSE248823.

The end-to-end calibration demo. Two published SBML models (DallePezze 2014 +
Geva-Zatorsky 2006), stitched by literature-grounded coupling edges, fit
against etoposide-DDIS ± rapamycin transcriptomics.

``run`` evaluates the composite out-of-the-box against every arm and writes
the OOB concordance table + trajectory figures — no fitting. ``--calibrate``
continues into the fit: it reuses that OOB evaluation as the pre-fit baseline,
fits the mechanism parameters (one per reporter axis, plus GZ06's control-side
alpha_x) on the DDIS-vs-control arm, evaluates
concordance on the held-out rapamycin arm with a magnitude-aware log2
fold-change loss, and writes the before/after comparison figures.

    simulate multi-hallmark run
    simulate multi-hallmark calibrate

Needs the GSE248823 matrix under data/FibroblastsDNA_dmg_Rapamycin/; the
SBML models download from BioModels on first import and cache locally.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hallsim.calibration import (  # noqa: E402
    CalibrationProblem,
    Condition,
    ParameterRef,
)
from hallsim.calibration_report import (  # noqa: E402
    format_table,
    plot_history,
    rows_by_gene,
    save_outputs,
)
from hallsim.io import make_run_dir  # noqa: E402
from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.gene_reporters import (  # noqa: E402
    MULTI_HALLMARK_REPORTERS,
    GeneExpressionDataset,
)
from demos.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
    GZ06_ALPHA_X_CONTROL,
    RAPA_INTERVENTION_DAY,
    DDIS_ETOPOSIDE_DOSE_WINDOW,
)


def _annotate_interventions(ax, arm: str) -> None:
    """Shade the etoposide dose window and mark the rapamycin step, so the
    experimental protocol is visible on the trajectory. The dose applies to
    every damaged arm (all but the control); rapamycin is added at washout on
    the rapamycin arm only."""
    treated = arm.split("_vs_")[0].lower()  # "DDIS_vs_ctrl" -> "ddis"
    damaged = treated != "ctrl"
    if damaged and DDIS_ETOPOSIDE_DOSE_WINDOW is not None:
        t0, t1 = DDIS_ETOPOSIDE_DOSE_WINDOW
        ax.axvspan(
            t0,
            t1,
            color="#e8a33d",
            alpha=0.15,
            lw=0,
            zorder=0,
            label="etoposide",
        )
    if "rapa" in treated:
        ax.axvline(
            RAPA_INTERVENTION_DAY,
            color="#2a78d6",
            ls=":",
            lw=1.1,
            zorder=1,
            label="rapamycin",
        )


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "FibroblastsDNA_dmg_Rapamycin"
SERIES_MATRIX = DATA_DIR / "GSE248823_series_matrix.txt"
PLATFORM = DATA_DIR / "GPL17586-45144.txt"

RUN_NAME = "multi_hallmark_calibrate"

#: Horizon for the unscored run. The scored path takes its query times
#: from the dataset; with no dataset there is none to take, so this is
#: the experiment's own endpoint (day 14).
UNSCORED_T_END = 14.0


# GSE248823 columns: etoposide DDIS sampled at D00 (baseline), D07, D14,
# and etoposide + rapamycin at D07, D14 (2 replicates each). We fit the
# whole D07 + D14 time course, not just the endpoint.
SAMPLE_POSITION_GROUPS = {
    "ETOPOSIDE_D00": [0, 1],
    "ETOPOSIDE_D07": [2, 3],
    "ETOPOSIDE_D14": [4, 5],
    "ETOPOSIDE_RAPA_D07": [6, 7],
    "ETOPOSIDE_RAPA_D14": [8, 9],
}

# The HRAS oncogene-induced arm is gone. It was wired to the same severities
# as DDIS, so the composite produced a bit-identical trajectory (max|Δ| = 0)
# and its "concordance" scored one model output against a second dataset. The
# composite has no oncogene-specific mechanism; until it does, there is
# nothing for that arm to test.
ARMS = ["DDIS_vs_ctrl", "RAPA_vs_ctrl"]

# GZ06 starts at p53 x=0; without this the day-0 reference sits in its startup.
PREROLL_DAYS = 1.0

# "baseline" = each arm vs its own day 0; "paired" = vs its reference arm at
# the same day, which needs that arm to exist in the data.
NORMALIZATION = "baseline"

ARM_PAIRS = {
    "DDIS_vs_ctrl": ("DDIS", "ctrl"),
    "RAPA_vs_ctrl": ("RAPA", "DDIS"),
}
ARM_CONDITIONS = {arm: cond for arm, (cond, _) in ARM_PAIRS.items()}


def build_problem(
    composite=None, reporters=None, equilibrate: bool = False
) -> CalibrationProblem:
    ds = (
        GeneExpressionDataset.from_series_matrix(
            SERIES_MATRIX,
            PLATFORM,
            sample_groups={},
            sample_position_groups=SAMPLE_POSITION_GROUPS,
        )
        if SERIES_MATRIX.exists()
        else None
    )
    return CalibrationProblem(
        composite=(
            composite
            if composite is not None
            else build_multi_hallmark_composite()
        ),
        reporters=(
            reporters if reporters is not None else MULTI_HALLMARK_REPORTERS
        ),
        conditions={
            "ctrl": Condition(
                "ctrl",
                {"Genomic Instability": 0.0},
            ),
            "DDIS": Condition(
                "DDIS",
                # No DNS key: etoposide perturbs DNA damage, not nutrient
                # sensing, so any mTOR change must emerge from the dynamics.
                # Homeostasis is not "mTOR off" — DP14's drive stays at basal.
                {"Genomic Instability": 1.0},
            ),
            # Rapamycin at washout: the nutrient_drive StepSource carries the
            # switch time, the severity the post-step level. Arms differ only
            # in u(t) — no rate constant, no timed parameter intervention.
            "RAPA": Condition(
                "RAPA",
                {
                    "Genomic Instability": 1.0,
                    "Deregulated Nutrient Sensing": -1.0,
                },
            ),
        },
        # Samples per arm per day. `arm_deltas` picks the reference from
        # NORMALIZATION, so the data contrast tracks the model's. The
        # rapamycin culture's day-0 is the shared etoposide D00.
        # Empty when the dataset is absent: the composite, its conditions and
        # its reporters stand on their own, so everything except scoring works.
        data=(
            ds.arm_deltas(
                {
                    "DDIS_vs_ctrl": {
                        0.0: "ETOPOSIDE_D00",
                        7.0: "ETOPOSIDE_D07",
                        14.0: "ETOPOSIDE_D14",
                    },
                    "RAPA_vs_ctrl": {
                        0.0: "ETOPOSIDE_D00",
                        7.0: "ETOPOSIDE_RAPA_D07",
                        14.0: "ETOPOSIDE_RAPA_D14",
                    },
                },
                NORMALIZATION,
                arm_pairs=ARM_PAIRS,
                arm_conditions=ARM_CONDITIONS,
            )
            if ds is not None
            else {a: {} for a in ARMS}
        ),
        normalization=NORMALIZATION,
        equilibrate=equilibrate,
        equilibration_condition="ctrl",
        arm_pairs=ARM_PAIRS,
        # Each fit param is read by ≥1 reporter and has a log-normal MAP prior.
        # See docs/coupling-edge-priors.md, docs/gz06-basal-p53.md.
        params={
            # SA-beta-gal decay: the GLB1 reporter's only lever. Its production
            # constant is not fitted — the pair sets a level, which cancels in a
            # fold change; the decay sets track-vs-accumulate, which does not.
            "sa_beta_gal_decay": ParameterRef(
                "dp14",
                "parameters.sen_ass_beta_gal_dec",
                prior=0.1548,
                prior_sigma=0.5,
            ),
            # ROS pair frozen — no ROS reporter (see diary).
            "CDKN1A_transcr": ParameterRef(
                "dp14",
                "parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
                prior=0.085,
                prior_sigma=0.5,
            ),
            # mtor_phos_rate, alpha_y, mitophagy_inactiv frozen — non-identifiable
            # here (gain-degenerate / flat gradient); see diary.
            "alpha_x_control": ParameterRef(
                "damage_bridge",
                "basal",
                prior=GZ06_ALPHA_X_CONTROL,
                prior_sigma=0.5,
            ),
            # p53 → CDKN1A edge (P53CDKN1AActivator.k_act) is fixed, not fitted.
        },
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=["RAPA_vs_ctrl"],
        prior_weight=0.03,
        t_end=14.0,
        t_start=-PREROLL_DAYS,
        macro_dt=3.5,
        # The oscillating reporters (DDB2/MDM2) read raw p53 / Mdm2 /
        # IκBα-transcript and take a zero-phase RMS/mean post-hoc, so the save
        # grid must resolve the pulse: save_dt = 14/149 ≈ 0.094 d, under the
        # ~0.145 d Nyquist for the ~0.29 d p53 period. Cost is memory (more save
        # points), not solve time. Mirror-padded edges (odd=False) keep the
        # endpoint query artifact-free — no margin needed.
        n_save=150,
    )


def plot(pre, post, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Arms may have different timepoints, so size the grid to the widest
    # arm and use each arm's own times.
    ncol = max(len(pre[a]) for a in ARMS)
    fig, axes = plt.subplots(
        len(ARMS),
        ncol,
        figsize=(6 * ncol, 4.5 * len(ARMS)),
        sharey=True,
        squeeze=False,
    )
    for ai, arm in enumerate(ARMS):
        atimes = sorted(pre[arm])
        for ti in range(ncol):
            ax = axes[ai][ti]
            if ti >= len(atimes):
                ax.axis("off")
                continue
            t = atimes[ti]
            pre_r = rows_by_gene(pre[arm][t])
            post_r = rows_by_gene(post[arm][t])
            genes = list(pre_r)
            x = np.arange(len(genes))
            w = 0.26
            ax.bar(
                x - w,
                [pre_r[g].delta_data for g in genes],
                w,
                label="measured",
                color="#333",
            )
            ax.bar(
                x,
                [pre_r[g].delta_sim for g in genes],
                w,
                label="model (out-of-box)",
                color="#bbb",
            )
            ax.bar(
                x + w,
                [post_r[g].delta_sim for g in genes],
                w,
                label="model (calibrated)",
                color="#2a7",
            )
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(genes, rotation=45, ha="right")
            tag = "FIT" if arm == "DDIS_vs_ctrl" else "HELD-OUT"
            ax.set_title(f"{tag}: {arm} · day {t:g}")
    axes[0][0].set_ylabel("log2 fold-change")
    axes[0][0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def write_concordance_table(pre, post, out_dir: Path) -> None:
    """Per-arm ρ and mean|error|, out-of-box → calibrated, as a CSV and a
    colored PNG table (green where calibration improves on out-of-box, orange
    where it worsens). Fit and held-out arms are labelled."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_arms = {"DDIS_vs_ctrl"}
    rows = []
    for arm in ARMS:
        tag = "fit" if arm in fit_arms else "held-out"
        short = arm.split("_")[0]
        for t in sorted(pre[arm]):
            rows.append(
                (
                    f"{short} ({tag})",
                    f"{t:g}",
                    pre[arm][t].spearman_r,
                    post[arm][t].spearman_r,
                    pre[arm][t].mean_abs_error,
                    post[arm][t].mean_abs_error,
                )
            )

    with open(out_dir / "concordance_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "arm",
                "day",
                "rho_oob",
                "rho_cal",
                "mean_abs_err_oob",
                "mean_abs_err_cal",
            ]
        )
        for arm, day, ro, rc, eo, ec in rows:
            w.writerow(
                [arm, day, f"{ro:.3f}", f"{rc:.3f}", f"{eo:.3f}", f"{ec:.3f}"]
            )

    IMP, REG, DIM, INK = "#1a7f4b", "#c0552b", "#6b7280", "#1f2937"
    header = [
        "Arm",
        "Day",
        "ρ (oob)",
        "ρ (cal)",
        "mean|err| (oob)",
        "mean|err| (cal)",
    ]
    text, colors = [header], [[INK] * 6]
    for arm, day, ro, rc, eo, ec in rows:
        text.append(
            [arm, day, f"{ro:+.2f}", f"{rc:+.2f}", f"{eo:.2f}", f"{ec:.2f}"]
        )
        colors.append(
            [
                INK,
                INK,
                DIM,
                IMP if rc >= ro else REG,
                DIM,
                IMP if ec <= eo else REG,
            ]
        )

    fig, ax = plt.subplots(figsize=(8.6, 0.55 + 0.42 * len(text)))
    ax.axis("off")
    tbl = ax.table(
        cellText=text,
        cellLoc="center",
        loc="center",
        colWidths=[0.26, 0.10, 0.14, 0.14, 0.18, 0.18],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dcdcdc")
        if r == 0:
            cell.set_facecolor("#f2f3f5")
            cell.set_text_props(fontweight="bold", color=INK)
        else:
            cell.get_text().set_color(colors[r][c])
            if c == 0:
                cell.get_text().set_fontweight("bold")
    ax.set_title(
        "Calibrated vs out-of-the-box concordance",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=14,
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"concordance_table.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote concordance_table.png/.csv -> {out_dir}", flush=True)


def write_reporter_table(pre, post, out_dir: Path) -> None:
    """Per-reporter measured / out-of-box / calibrated Δlog2FC, one row per
    (arm, day, gene). CSV covers every arm; the PNG renders the fit arm (DDIS)
    at each measured day, colored green where calibration shrinks |error| and
    orange where it grows (surfacing the per-reporter magnitude-vs-rank
    trade-offs the per-arm summary hides)."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_arms = {"DDIS_vs_ctrl"}
    rows = []  # (arm_short, tag, day, gene, data, oob, cal)
    for arm in ARMS:
        tag = "fit" if arm in fit_arms else "held-out"
        for t in sorted(pre[arm]):
            pr = rows_by_gene(pre[arm][t])
            po = rows_by_gene(post[arm][t])
            for g in pr:
                rows.append(
                    (
                        arm.split("_")[0],
                        tag,
                        t,
                        g,
                        pr[g].delta_data,
                        pr[g].delta_sim,
                        po[g].delta_sim,
                    )
                )

    with open(out_dir / "reporter_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "arm",
                "tag",
                "day",
                "gene",
                "measured",
                "model_oob",
                "model_cal",
                "abserr_oob",
                "abserr_cal",
            ]
        )
        for arm, tag, t, g, d, o, c in rows:
            w.writerow(
                [
                    arm,
                    tag,
                    f"{t:g}",
                    g,
                    f"{d:+.4f}",
                    f"{o:+.4f}",
                    f"{c:+.4f}",
                    f"{abs(o - d):.4f}",
                    f"{abs(c - d):.4f}",
                ]
            )

    IMP, REG, DIM, INK = "#1a7f4b", "#c0552b", "#6b7280", "#1f2937"
    header = [
        "Gene",
        "Day",
        "measured",
        "model(oob)",
        "model(cal)",
        "|err| oob→cal",
    ]
    text, colors = [header], [[INK] * 6]
    for arm, tag, t, g, d, o, c in rows:
        if arm != "DDIS":
            continue
        eo, ec = abs(o - d), abs(c - d)
        text.append(
            [
                g,
                f"{t:g}",
                f"{d:+.3f}",
                f"{o:+.3f}",
                f"{c:+.3f}",
                f"{eo:.2f}→{ec:.2f}",
            ]
        )
        colors.append(
            [
                INK,
                INK,
                DIM,
                DIM,
                IMP if ec <= eo else REG,
                IMP if ec <= eo else REG,
            ]
        )

    fig, ax = plt.subplots(figsize=(9.0, 0.55 + 0.4 * len(text)))
    ax.axis("off")
    tbl = ax.table(
        cellText=text,
        cellLoc="center",
        loc="center",
        colWidths=[0.16, 0.08, 0.16, 0.18, 0.18, 0.20],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dcdcdc")
        if r == 0:
            cell.set_facecolor("#f2f3f5")
            cell.set_text_props(fontweight="bold", color=INK)
        else:
            cell.get_text().set_color(colors[r][c])
            if c == 0:
                cell.get_text().set_fontweight("bold")
    ax.set_title(
        "Per-reporter concordance — DDIS (fit arm), out-of-the-box "
        "vs calibrated",
        fontsize=12,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=14,
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"reporter_table.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote reporter_table.png/.csv -> {out_dir}", flush=True)


# ── OOB-immediate figures (written before the fit, for review while training) ─
_ARM_STYLE = {
    "DDIS_vs_ctrl": ("#c0392b", "DDIS"),
    "RAPA_vs_ctrl": ("#2a78d6", "RAPA"),
}


def fig_oob_overview(
    problem,
    params,
    out_dir: Path,
    stem="oob_overview",
    title="Out-of-the-box: reporter trajectories vs data",
) -> None:
    """Per reporter, every arm's model trajectory (from ``params``) + its data —
    all conditions in one figure, so there is something to read while the fit
    trains. Reuses the loss's own ``model_lfc`` (starts at t>0; the t=0
    window-mean degeneracy is a plotting-only artifact)."""
    genes = [r.gene_symbol for r in problem.reporters]
    n, ncol = len(genes), 3
    nrow = -(-n // ncol)
    qt = np.arange(0.1, problem.t_end + 1e-6, 0.1)
    line_arms = list(_ARM_STYLE)
    lfc = {
        a: np.asarray(problem.model_lfc(params, a, jnp.asarray(qt)))
        for a in line_arms
    }
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(11, 3.2 * nrow), sharex=True, squeeze=False
    )
    axf = axes.ravel()
    for i, ax in enumerate(axf):
        if i >= n:
            ax.axis("off")
            continue
        g = genes[i]
        ax.axhline(0, color="#e6e6e2", lw=1.2, zorder=0)
        for a, (col, lbl) in _ARM_STYLE.items():
            if a in lfc:
                ax.plot(qt, lfc[a][i], color=col, lw=1.7, label=lbl)
            dts = sorted(problem.data.get(a, {}))
            if dts:
                ax.plot(
                    [0.0] + list(dts),
                    [0.0] + [float(problem.data[a][t][g]) for t in dts],
                    "o",
                    color=col,
                    ms=5,
                )
        ax.set_title(g, fontsize=11, fontweight="bold", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i % ncol == 0:
            ax.set_ylabel("log2 fold-change")
        if i >= n - ncol:
            ax.set_xlabel("day")
    axf[0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(title, fontsize=12.5, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png -> {out_dir}", flush=True)


def write_oob_table(pre, out_dir: Path) -> None:
    """OOB per-arm concordance (sign, rho, mean|err|) as a small table."""
    header = ["Arm", "Day", "sign", "ρ", "mean|err|"]
    text = [header]
    for arm in ARMS:
        tag = "fit" if arm == "DDIS_vs_ctrl" else "held-out"
        for t in sorted(pre[arm]):
            r = pre[arm][t]
            n_ok = sum(x.sign_match for x in r.rows)
            text.append(
                [
                    f"{arm.split('_')[0]} ({tag})",
                    f"{t:g}",
                    f"{n_ok}/{r.n_compared}",
                    f"{r.spearman_r:+.2f}",
                    f"{r.mean_abs_error:.2f}",
                ]
            )
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.42 * len(text)))
    ax.axis("off")
    tbl = ax.table(cellText=text, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.55)
    for (rr, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dcdcdc")
        if rr == 0:
            cell.set_facecolor("#f2f3f5")
            cell.set_text_props(fontweight="bold")
    ax.set_title(
        "Out-of-the-box concordance",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"oob_concordance_table.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote oob_concordance_table.png -> {out_dir}", flush=True)


# Representative internal state per constituent — the "what's happening inside"
# view, independent of the reporter mapping.
_CONSTITUENT_STATES = [
    ("gz06/x", "GZ06 p53 (x)"),
    ("gz06/y", "GZ06 Mdm2 (y)"),
    ("dp14/DNA_damage", "DP14 DNA damage"),
    ("dp14/ROS", "DP14 ROS"),
    ("dp14/FoxO3a", "DP14 FoxO3a"),
    ("dp14/mTORC1_pS2448", "DP14 mTORC1-P"),
    ("dp14/Mito_mass_new", "DP14 new mito mass"),
]


def fig_constituents(
    problem,
    init,
    final,
    out_dir: Path,
    cond="DDIS",
    stem="constituents_DDIS_pre_vs_post",
) -> None:
    """Constituent internal states, pre- vs post-fit, for one condition — the
    dynamics behind the reporters (not a fit quantity)."""
    pre = problem.simulate_all_conditions(init, n_save=200)[cond]
    post = problem.simulate_all_conditions(final, n_save=200)[cond]
    states = [
        (p, lbl) for p, lbl in _CONSTITUENT_STATES if pre.get(p) is not None
    ]
    n, ncol = len(states), 3
    nrow = -(-n // ncol)
    ts = np.asarray(pre.ts)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(12, 3.0 * nrow), sharex=True, squeeze=False
    )
    axf = axes.ravel()
    for i, ax in enumerate(axf):
        if i >= n:
            ax.axis("off")
            continue
        path, lbl = states[i]
        ax.plot(
            ts,
            np.asarray(pre.get(path)),
            color="#9a9a95",
            lw=1.6,
            ls=(0, (4, 2)),
            label="pre-fit",
        )
        ax.plot(
            ts,
            np.asarray(post.get(path)),
            color="#2a78d6",
            lw=2.0,
            label="calibrated",
        )
        _annotate_interventions(ax, cond)
        ax.set_title(lbl, fontsize=10.5, fontweight="bold", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i >= n - ncol:
            ax.set_xlabel("day")
    axf[0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(
        f"Constituent states, {cond}: pre vs post-fit",
        fontsize=12.5,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png -> {out_dir}", flush=True)


def run_oob(problem, params, out_dir: Path):
    """Composite out-of-the-box: per-arm concordance + the immediate figures,
    written before any fit. Returns the evaluation so the calibrate path can
    reuse it as its pre-fit baseline instead of re-simulating."""
    print("[oob] out-of-the-box concordance + figures ...", flush=True)
    pre = problem.evaluate(params)
    for arm in ARMS:
        for t in sorted(pre[arm]):
            print(pre[arm][t], flush=True)
    write_oob_table(pre, out_dir)
    fig_oob_overview(problem, params, out_dir)
    return pre


def _missing_data_notice() -> str:
    """Where to get the dataset the scored path needs."""
    return (
        f"Dataset not found: {SERIES_MATRIX.name}\n"
        f"  expected in : {DATA_DIR}\n"
        f"  download    : GEO accession GSE248823 (series matrix), plus the\n"
        f"                GPL17586 platform annotation ({PLATFORM.name})\n"
        f"                https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc=GSE248823\n"
        "Running the composite unscored: it will simulate and write "
        "trajectories, but concordance against measured expression is skipped."
    )


def run_unscored(equilibrate: bool, out_dir: Path):
    """The scored run minus the scoring.

    Same problem, same reporters, same overview figure — the measured points
    are simply absent, so the model trajectories are drawn on their own. No
    concordance is reported: with no arms to compare against there is nothing
    to be concordant with.
    """
    problem = build_problem(equilibrate=equilibrate)
    params = problem.initial_params()
    print(f"[unscored] reporters : {len(problem.reporters)}")
    print(f"[unscored] arms      : {list(_ARM_STYLE)}")
    fig_oob_overview(
        problem,
        params,
        out_dir,
        stem="oob_overview",
        title="Out-of-the-box: reporter trajectories (no data)",
    )
    print(f"\nunscored run → {out_dir.relative_to(ROOT)}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


def cmd_run(args) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("hallsim").setLevel(logging.INFO)
    equilibrate = getattr(args, "equilibrate", False)
    if not SERIES_MATRIX.exists():
        print(_missing_data_notice(), flush=True)
        return run_unscored(equilibrate, make_run_dir(RUN_NAME))
    problem = build_problem(equilibrate=equilibrate)
    print(f"[run] equilibrate={equilibrate}", flush=True)
    init = problem.initial_params()
    out_dir = make_run_dir(RUN_NAME)
    print(f"[run] writing to {out_dir.relative_to(ROOT)}/", flush=True)

    # ── out-of-the-box composite (always) ──
    pre = run_oob(problem, init, out_dir)

    if not args.calibrate:
        print(
            f"\nout-of-the-box run → outputs in {out_dir.relative_to(ROOT)}/",
            flush=True,
        )
        return

    # ── fit ──
    print("[fit] fitting ...", flush=True)
    steps = getattr(args, "steps", None) or 150
    base_lr = getattr(args, "lr", None) or 0.005
    # Cosine-decayed LR: fast descent into the basin, then a small-step tail
    # that doesn't overshoot the narrow valley walls.
    if getattr(args, "cosine", True):
        import optax

        lr = optax.cosine_decay_schedule(base_lr, decay_steps=steps)
        use_plateau = False
    else:
        lr = base_lr
        use_plateau = not getattr(args, "no_plateau", False)
    history = problem.fit(
        steps=steps,
        mode="reverse",
        learning_rate=lr,
        grad_clip=getattr(args, "grad_clip", None),
        reduce_on_plateau=use_plateau,
        plateau_patience=8,
        early_stop_patience=12,
        early_stop_tol=1e-3,  # relative: stop after the loss plateaus <0.1%/step
        verbose=True,
        checkpoint_path=out_dir / "checkpoint.npz",
    )

    print("[fit] calibrated concordance ...", flush=True)
    post = problem.evaluate(history.best_params)

    print(format_table(pre, post, fit_arms=problem.fit_arms))
    print("\nfitted parameters (init → fit):")
    for k in problem.param_refs:
        print(
            f"  {k:<20}{float(init[k]):>12.5g} → "
            f"{float(history.best_params[k]):>12.5g}"
        )

    # ── post-fit figures (before/after + what changed inside) ──
    print("[fit] post-fit figures ...", flush=True)
    write_concordance_table(pre, post, out_dir)
    write_reporter_table(pre, post, out_dir)
    plot_history(problem, history, out_dir / "training_history.png")
    save_outputs(problem, str(out_dir), history)
    fig_constituents(problem, init, history.best_params, out_dir)
    # Calibrated reporter figures on the fit just written, so the time-domain
    # trajectories and concordance dumbbells never lag behind the checkpoint.
    from demos.multi_hallmark_figures import (
        fig_concordance,
        fig_temporal,
        fig_temporal_compare,
    )

    fig_temporal(args)
    fig_temporal_compare(args)
    fig_concordance(args)

    print(
        f"\nbest loss {history.best_loss:.4g} over {len(history.losses)} "
        f"epochs → outputs in {out_dir.relative_to(ROOT)}/"
    )


# ── baseline: uncalibrated composite at the three arms ───────────────────
_ARMS_3 = [(0.0, 0.0, "ctrl"), (1.0, 0.0, "DDIS"), (1.0, -1.0, "DDIS+rapa")]


def _run_arms(base, gi, dns, t_end=50.0, macro_dt=5.0):
    hallmarks = {"Genomic Instability": gi}
    if dns != 0.0:
        hallmarks["Deregulated Nutrient Sensing"] = dns
    comp = with_hallmarks(base, hallmarks)
    return Scheduler().run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=macro_dt,
        y0=comp.initial_state_vec(),
        save_dt=macro_dt,
    )


def cmd_sweep(args) -> None:
    """Two-hallmark severity sweep — readouts gene-reporter validation uses."""
    base = build_multi_hallmark_composite()
    keys = [
        "dp14/DNA_damage",
        "dp14/CDKN1A",
        "dp14/mTORC1_pS2448",
        "dp14/ROS",
        "dp14/Mitophagy",
        "gz06/x",
    ]
    hdr = f"{'GI':>5} {'DNS':>5} | " + " ".join(
        f"{k.split('/')[-1]:>14}" for k in keys
    )
    print(f"Severity sweep — multi_hallmark composite\n{hdr}")
    print("-" * len(hdr))
    for gi, dns, label in _ARMS_3:
        res = _run_arms(base, gi, dns)
        vals = [
            float(res.get(k)[-1]) if res.get(k) is not None else float("nan")
            for k in keys
        ]
        print(
            f"{gi:>5.2f} {dns:>5.2f} | "
            + " ".join(f"{v:>14.4g}" for v in vals)
            + f"  # {label}"
        )


_COMMANDS = {
    "run": cmd_run,
    "sweep": cmd_sweep,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", default="run", choices=_COMMANDS)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="after the out-of-the-box run, fit the mechanism parameters and "
        "write the calibrated figures (default: out-of-the-box only)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate / cosine start (default 0.005)",
    )
    ap.add_argument(
        "--steps", type=int, default=None, help="fit steps (default 150)"
    )
    ap.add_argument(
        "--no-cosine",
        action="store_false",
        dest="cosine",
        help="use constant LR + reduce-on-plateau instead of cosine decay "
        "(cosine is the default)",
    )
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        dest="grad_clip",
        help="clip gradient global-norm to this value",
    )
    ap.add_argument(
        "--no-plateau",
        action="store_true",
        dest="no_plateau",
        help="disable reduce-on-plateau LR schedule",
    )
    ap.add_argument(
        "--equilibrate",
        action="store_true",
        help="Newton-solve the whole composite to a fixed point and share it "
        "as t=0. Off by default: this composite is mixed, and DP14 senescence "
        "is progressive with no healthy fixed point to solve for",
    )
    args = ap.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
