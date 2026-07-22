"""Calibrate the multi-hallmark flagship composite against GSE248823.

THE end-to-end calibration demo. Three published SBML models (DallePezze
2014 + Geva-Zatorsky 2006 + Ihekwaba 2004), stitched by literature-grounded
coupling edges, fit against etoposide-DDIS ± rapamycin transcriptomics.

Fits eight mechanism parameters (one per reporter axis, plus the two NF-κB
IKK edge strengths and GZ06's basal-p53 ψ) on the DDIS-vs-control arm and
evaluates concordance on the held-out rapamycin arm, using a magnitude-aware
log2 fold-change loss. Prints out-of-the-box vs calibrated vs measured per
reporter and writes the comparison plot.

    .venv_hallsim/bin/python demos/multi_hallmark_calibrate.py

Needs the GSE248823 matrix under data/FibroblastsDNA_dmg_Rapamycin/; the
SBML models download from BioModels on first import and cache locally.
"""

from __future__ import annotations

import argparse
import logging
import time
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
    HallmarkCoeffRef,
    ParamStep,
    ParameterRef,
)
from hallsim.calibration_report import save_outputs  # noqa: E402
from hallsim.composite import single_process_composite  # noqa: E402
from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.sbml_import import process_from_sbml  # noqa: E402
from hallsim.gene_reporters import (  # noqa: E402
    MULTI_HALLMARK_REPORTERS,
    GeneExpressionDataset,
)
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
    DP14_SBML_PATH,
    DP14_IRRADIATION_RATE_NAME,
    DP14_IRRADIATION_RATE_DEFAULT,
    DP14_MTOR_PHOS_RATE_NAME,
    DP14_MTOR_PHOS_RATE_DEFAULT,
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

# Each calibrate run writes to its own timestamped subfolder so results never
# overwrite; `latest` symlinks the most recent so the figure scripts follow it.
RUNS_DIR = ROOT / "outputs" / "multi_hallmark_calibrate"
LATEST_RUN = RUNS_DIR / "latest"


def make_run_dir() -> Path:
    from datetime import datetime

    run = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run.mkdir(parents=True, exist_ok=True)
    if LATEST_RUN.is_symlink() or LATEST_RUN.exists():
        if LATEST_RUN.is_symlink():
            LATEST_RUN.unlink()
        else:
            import shutil

            shutil.rmtree(LATEST_RUN)
    LATEST_RUN.symlink_to(run.name)
    return run


# GSE248823 columns: etoposide DDIS sampled at D00 (baseline), D07, D14,
# and etoposide + rapamycin at D07, D14 (2 replicates each). We fit the
# whole D07 + D14 time course, not just the endpoint.
SAMPLE_POSITION_GROUPS = {
    "ETOPOSIDE_D00": [0, 1],
    "ETOPOSIDE_D07": [2, 3],
    "ETOPOSIDE_D14": [4, 5],
    "ETOPOSIDE_RAPA_D07": [6, 7],
    "ETOPOSIDE_RAPA_D14": [8, 9],
    # Oncogene-induced senescence (HRAS) arm — a mechanistically distinct
    # senescence trigger the model is NOT calibrated on. Sampled D00/D04/D07.
    "RAS_D00": [10, 11],
    "RAS_D04": [12, 13],
    "RAS_D07": [14, 15],
}

ARMS = ["DDIS_vs_ctrl", "RAPA_vs_ctrl", "RAS_vs_ctrl"]


def build_problem(composite=None, reporters=None) -> CalibrationProblem:
    ds = GeneExpressionDataset.from_series_matrix(
        SERIES_MATRIX,
        PLATFORM,
        sample_groups={},
        sample_position_groups=SAMPLE_POSITION_GROUPS,
    )
    return CalibrationProblem(
        composite=(
            composite
            if composite is not None
            else build_multi_hallmark_composite(dose_window=None)
        ),
        reporters=(
            reporters if reporters is not None else MULTI_HALLMARK_REPORTERS
        ),
        conditions={
            "ctrl": Condition(
                "ctrl",
                {
                    "Genomic Instability": 0.0,
                    "Deregulated Nutrient Sensing": 0.5,
                },
            ),
            "DDIS": Condition(
                "DDIS",
                {
                    "Genomic Instability": 1.0,
                    "Deregulated Nutrient Sensing": 1.0,
                },
            ),
            # Etoposide + rapamycin: identical to DDIS (GI=1, DNS→mTOR at
            # base) until rapamycin is added at washout (day 2), when the mTOR
            # rate steps down to the DNS=0.3 rapamycin-suppressed level. The
            # severity sets that post-step level; the ParamStep supplies the
            # untreated pre-step level (the DDIS mTOR rate) and the switch time.
            "RAPA": Condition(
                "RAPA",
                {
                    "Genomic Instability": 1.0,
                    "Deregulated Nutrient Sensing": 0.3,
                },
                interventions=(
                    ParamStep(
                        process_name="dp14",
                        param_name=DP14_MTOR_PHOS_RATE_NAME,
                        t_step=RAPA_INTERVENTION_DAY,
                        value_before=None,
                    ),
                ),
            ),
            # Oncogene-induced senescence: mapped to the same full-senescence
            # severities as DDIS (oncogenic RAS drives replication-stress DNA
            # damage AND mTOR hyperactivation). The composite has no
            # oncogene-specific mechanism, so this is deliberately an
            # out-of-the-box generalization test — does the senescence
            # program calibrated on etoposide transfer to a different trigger?
            "RAS_OIS": Condition(
                "RAS_OIS",
                {
                    "Genomic Instability": 1.0,
                    "Deregulated Nutrient Sensing": 1.0,
                },
            ),
        },
        # Trajectory-native: each arm is a {day: Δlog2FC} time course. Model
        # time is in days (t_end=14), so the measured D07 / D14 samples map to
        # query times 7.0 and 14.0. Every arm is normalized within-arm to its
        # own D00 (fold-change-from-day-0) — matching the model's
        # ``normalization="baseline"``. The rapamycin culture's day-0 is the
        # shared etoposide D00 (rapamycin is not added until day 2), so the
        # rapamycin arm normalizes to ETOPOSIDE_D00. The drug contrast (rapa
        # vs no-rapa) is recovered post-hoc by differencing the two within-arm
        # curves, not by cross-arm normalization.
        data={
            "DDIS_vs_ctrl": {
                7.0: ds.delta("ETOPOSIDE_D07", "ETOPOSIDE_D00"),
                14.0: ds.delta("ETOPOSIDE_D14", "ETOPOSIDE_D00"),
            },
            "RAPA_vs_ctrl": {
                7.0: ds.delta("ETOPOSIDE_RAPA_D07", "ETOPOSIDE_D00"),
                14.0: ds.delta("ETOPOSIDE_RAPA_D14", "ETOPOSIDE_D00"),
            },
            # RAS vs its own pre-oncogene D00 baseline, at D04 and D07.
            "RAS_vs_ctrl": {
                4.0: ds.delta("RAS_D04", "RAS_D00"),
                7.0: ds.delta("RAS_D07", "RAS_D00"),
            },
        },
        # Model reproduces X_t/X_0 within each arm — the same reference the
        # data deltas use. `base` in arm_pairs is unused under this mode. Every
        # arm starts from the shared pre-perturbation homeostasis (the control
        # condition run to its settled state), so the within-arm t=0 is the
        # healthy baseline the data's D00 measures — not the composite's
        # arbitrary initial condition (whose relaxation transient otherwise
        # dominates and flips the damage sign).
        normalization="baseline",
        equilibrate=True,
        equilibration_condition="ctrl",
        arm_pairs={
            "DDIS_vs_ctrl": ("DDIS", "ctrl"),
            "RAPA_vs_ctrl": ("RAPA", "DDIS"),
            "RAS_vs_ctrl": ("RAS_OIS", "ctrl"),
        },
        # One mechanism knob per reporter axis, plus the two NF-κB IKK edge
        # strengths and GZ06's basal-p53 ψ. Each has a log-normal MAP prior
        # (center = literature/derived value; sigma in log10 decades) so the
        # under-constrained fit (8 params, 6 fit-arm reporters) stays
        # physical. Edges are anchored to the Ihekwaba IKK pool scale (0.1);
        # see docs/coupling-edge-priors.md and docs/gz06-basal-p53.md.
        params={
            "etoposide_potency": ParameterRef(
                "dp14",
                "parameters.DNA_damaged_by_irradiation",
                init=10.0,
                clamp=(0.01, 10000.0),
                prior=10.0,
                prior_sigma=1.0,
            ),
            "ROS_turnover": ParameterRef(
                "dp14",
                "parameters.ROS_turnover",
                init=3.231,
                clamp=(0.1, 50.0),
                prior=3.231,
                prior_sigma=0.5,
            ),
            "CDKN1A_transcr": ParameterRef(
                "dp14",
                "parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
                init=0.085,
                clamp=(0.001, 5.0),
                prior=0.085,
                prior_sigma=0.5,
            ),
            "mitophagy_inactiv": ParameterRef(
                "dp14",
                "parameters.mitophagy_inactiv_by_mTORC1_pS2448",
                init=646.0,
                clamp=(1.0, 10000.0),
                prior=646.0,
                prior_sigma=0.5,
            ),
            # The knob on the hallmark→EIF4EBP1 line: the untreated mTORC1
            # phosphorylation rate. DNS severity scales it per condition; fitting
            # the base lets the DDIS/ctrl mTOR readout match data, and the shared
            # base transfers to the held-out rapamycin arm (severity applied on
            # top, never fit there).
            "mtor_phos_rate": ParameterRef(
                "dp14",
                f"parameters.{DP14_MTOR_PHOS_RATE_NAME}",
                init=DP14_MTOR_PHOS_RATE_DEFAULT,
                clamp=(1.0, 10000.0),
                prior=DP14_MTOR_PHOS_RATE_DEFAULT,
                prior_sigma=0.5,
            ),
            "alpha_y": ParameterRef(
                "gz06",
                "parameters.alpha_y",
                init=0.8,
                clamp=(0.01, 10.0),
                prior=0.8,
                prior_sigma=0.5,
            ),
            "psi_basal": ParameterRef(
                "gz06",
                "parameters.psi",
                init=0.3,
                clamp=(0.02, 0.95),
                prior=0.3,
                prior_sigma=0.5,
            ),
            "damage_to_nfkb": ParameterRef(
                "damage_nfkb",
                "k_act",
                init=0.1,
                clamp=(1e-4, 1.0),
                prior=0.1,
                prior_sigma=0.5,
            ),
            "mtor_to_nfkb": ParameterRef(
                "mtor_nfkb",
                "k_act",
                init=0.1,
                clamp=(1e-4, 1.0),
                prior=0.1,
                prior_sigma=0.5,
            ),
            # The DNS→mTOR affine floor: mTOR's residual fraction under
            # rapamycin (severity=0). Sets the DDIS:ctrl mTOR contrast the
            # EIF4EBP1 reporter reads, and the h=0 anchor the held-out RAPA arm
            # lands on. Rides the hallmark registry, not a process field (no
            # SBML host).
            "dns_mtor_floor": HallmarkCoeffRef(
                hallmark="Deregulated Nutrient Sensing",
                param_name=f"parameters.{DP14_MTOR_PHOS_RATE_NAME}",
                init=0.3,
                clamp=(0.05, 0.95),
                prior=0.3,
                prior_sigma=0.3,
            ),
            # p53 → CDKN1A edge (P53CDKN1AActivator.k_act) is fixed, not fitted.
        },
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=["RAPA_vs_ctrl", "RAS_vs_ctrl"],
        prior_weight=0.03,
        t_end=14.0,
        macro_dt=3.5,
        # Every reporter is grid-independent — the oscillator readouts route
        # through RunningIntegral means/RMS (exact regardless of save spacing)
        # and the rest are slow DP14 states. Verified: DDIS-d14 concordance is
        # identical at n_save 29 vs 450 (max|Δ| 9e-5). So the loss uses a modest
        # grid (fast under the reverse-mode adjoint); raw-state *plots* resample
        # densely (``save_outputs(n_save_plot=…)``, above the Nyquist guardrail).
        n_save=29,
    )


def _rows_by_gene(result):
    return {r.reporter.gene_symbol: r for r in result.rows}


def print_table(pre, post) -> None:
    print("\n" + "=" * 74)
    print("OUT-OF-THE-BOX vs CALIBRATED vs MEASURED  (log2 fold-change)")
    print("=" * 74)
    for arm in ARMS:
        tag = "FIT " if arm == "DDIS_vs_ctrl" else "HELD-OUT"
        print(f"\n[{tag}] {arm}")
        for t in sorted(pre[arm]):
            pre_r = _rows_by_gene(pre[arm][t])
            post_r = _rows_by_gene(post[arm][t])
            print(
                f"  day {t:g}   {'gene':<9}{'measured':>10}"
                f"{'model(oob)':>12}{'model(cal)':>12}   {'|err|oob→cal':>14}"
            )
            for g in pre_r:
                e0 = abs(pre_r[g].delta_sim - pre_r[g].delta_data)
                e1 = abs(post_r[g].delta_sim - post_r[g].delta_data)
                print(
                    f"  {'':<9}{g:<9}{pre_r[g].delta_data:>+10.3f}"
                    f"{pre_r[g].delta_sim:>+12.4f}"
                    f"{post_r[g].delta_sim:>+12.4f}   {e0:>6.3f}→{e1:<6.3f}"
                )
            print(
                f"  {'':<9}{'mean|err|':<9}{'':>10}{'':>12}{'':>12}   "
                f"{pre[arm][t].mean_abs_error:>6.3f}→"
                f"{post[arm][t].mean_abs_error:<6.3f}   "
                f"sign {pre[arm][t].sign_agreement * 100:.0f}→"
                f"{post[arm][t].sign_agreement * 100:.0f}%  "
                f"ρ {pre[arm][t].spearman_r:+.2f}→"
                f"{post[arm][t].spearman_r:+.2f}"
            )


def plot(pre, post, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Arms may have different timepoints (etoposide D07/D14, RAS D04/D07),
    # so size the grid to the widest arm and use each arm's own times.
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
            pre_r = _rows_by_gene(pre[arm][t])
            post_r = _rows_by_gene(post[arm][t])
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


def plot_history(problem, history, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    losses = np.asarray(history.losses)
    epochs = np.arange(1, len(losses) + 1)
    best = int(np.argmin(losses))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs, losses, color="#2a7")
    ax1.scatter(
        [best + 1],
        [losses[best]],
        color="k",
        zorder=5,
        label=f"best {losses[best]:.4g} @ epoch {best + 1}",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (log2FC MSE)")
    ax1.set_title("training loss")
    ax1.legend()

    # Grad norm + effective LR — to see whether loss spikes track large
    # gradients (→ clip) or LR-schedule events (→ plateau scheduler).
    if history.grad_norms:
        gn = np.asarray(history.grad_norms)
        ax3.plot(epochs, gn, color="#c0392b", label="|grad| (global norm)")
        ax3.set_yscale("log")
        ax3.set_ylabel("|grad|", color="#c0392b")
        ax3.tick_params(axis="y", labelcolor="#c0392b")
        if history.lrs:
            axlr = ax3.twinx()
            axlr.plot(epochs, np.asarray(history.lrs), color="#2563eb")
            axlr.set_ylabel("effective LR", color="#2563eb")
            axlr.tick_params(axis="y", labelcolor="#2563eb")
        for e in epochs[np.r_[False, np.diff(losses) > 0.02]]:
            ax3.axvline(e, color="#999", lw=0.7, ls=":", zorder=0)
    ax3.set_xlabel("epoch")
    ax3.set_title("grad norm · LR scale  (dotted = loss spike)")

    # Each param's position within its clamp range, in log space (the
    # params span orders of magnitude), so trajectories are comparable.
    for name, ref in problem.param_refs.items():
        lo, hi = ref.clamp
        vals = np.asarray([float(ph[name]) for ph in history.param_history])
        norm = (np.log(vals) - np.log(lo)) / (np.log(hi) - np.log(lo))
        ax2.plot(epochs, norm, label=name)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("param (log-position in clamp range)")
    ax2.set_title("parameter trajectories")
    ax2.legend(fontsize=7, loc="best")

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


# ── OOB-immediate figures (written before the fit, for review while training) ─
_ARM_STYLE = {
    "DDIS_vs_ctrl": ("#c0392b", "DDIS"),
    "RAPA_vs_ctrl": ("#2a78d6", "RAPA"),
    "RAS_vs_ctrl": ("#2a9d5a", "RAS"),
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
    lfc = {
        a: np.asarray(problem.model_lfc(params, a, jnp.asarray(qt)))
        for a in _ARM_STYLE
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
            ax.plot(qt, lfc[a][i], color=col, lw=1.7, label=lbl)
            dts = sorted(problem.data[a])
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
    ("nfkb/NFkB", "NF-κB (free)"),
    ("nfkb/IkBa", "IκBα protein"),
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


def cmd_calibrate(args) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("hallsim").setLevel(logging.INFO)
    problem = build_problem()
    init = problem.initial_params()
    out_dir = make_run_dir()
    print(f"[run] writing to {out_dir.relative_to(ROOT)}/", flush=True)

    # ── wave 1: out-of-the-box, written immediately so there's something to
    # review while the fit trains. ──
    print("[1/4] out-of-the-box concordance + figures ...", flush=True)
    pre = problem.evaluate(init)
    for arm in ARMS:
        for t in sorted(pre[arm]):
            print(pre[arm][t], flush=True)
    write_oob_table(pre, out_dir)
    fig_oob_overview(problem, init, out_dir)

    # ── wave 2: fit ──
    print("[2/4] fitting ...", flush=True)
    steps = getattr(args, "steps", None) or 150
    base_lr = getattr(args, "lr", None) or 0.005
    # Cosine-decayed LR: fast descent into the basin, then a small-step tail
    # that doesn't overshoot the narrow valley walls.
    if getattr(args, "cosine", False):
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

    print("[3/4] calibrated concordance ...", flush=True)
    post = problem.evaluate(history.final_params)

    print_table(pre, post)
    print("\nfitted parameters (init → fit):")
    for k in problem.param_refs:
        print(
            f"  {k:<20}{float(init[k]):>12.5g} → "
            f"{float(history.final_params[k]):>12.5g}"
        )

    # ── wave 4: post-fit figures (before/after + what changed inside) ──
    print("[4/4] post-fit figures ...", flush=True)
    write_concordance_table(pre, post, out_dir)
    plot_history(problem, history, out_dir / "training_history.png")
    save_outputs(problem, str(out_dir), history)
    fig_constituents(problem, init, history.final_params, out_dir)
    # Calibrated reporter figures on the fit just written, so the time-domain
    # trajectories and concordance dumbbells never lag behind the checkpoint.
    from multi_hallmark_figures import fig_concordance, fig_temporal

    fig_temporal(args)
    fig_concordance(args)

    print(
        f"\nbest loss {history.best_loss:.4g} over {len(history.losses)} "
        f"epochs → outputs in {out_dir.relative_to(ROOT)}/"
    )


# ── baseline: uncalibrated composite at the three arms ───────────────────
_ARMS_3 = [(0.0, 0.5, "ctrl"), (1.0, 1.0, "DDIS"), (1.0, 0.3, "DDIS+rapa")]


def _run_arms(base, gi, dns, t_end=50.0, macro_dt=5.0):
    comp = with_hallmarks(
        base, {"Genomic Instability": gi, "Deregulated Nutrient Sensing": dns}
    )
    return Scheduler().run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=macro_dt,
        y0=comp.initial_state_vec(),
        save_dt=macro_dt,
    )


def cmd_baseline(args) -> None:
    """Uncalibrated composite: terminal readouts per arm (measures overload)."""
    base = build_multi_hallmark_composite()
    keys = [
        "dp14/DNA_damage",
        "dp14/CDKN1A",
        "dp14/mTORC1_pS2448",
        "nfkb/IkBat",
        "gz06/x",
        "gz06/y",
    ]
    hdr = f"{'GI':>4} {'DNS':>4} | " + " ".join(
        f"{k.split('/')[-1]:>13}" for k in keys
    )
    print("Multi-hallmark composite, uncalibrated.\n" + hdr)
    print("-" * len(hdr))
    for gi, dns, label in _ARMS_3:
        t0 = time.time()
        try:
            res = _run_arms(base, gi, dns)
        except Exception as e:  # noqa: BLE001
            print(
                f"{gi:>4.2f} {dns:>4.2f} | FAIL -> {type(e).__name__}  "
                f"# {label}"
            )
            continue
        vals = [
            float(res.get(k)[-1]) if res.get(k) is not None else float("nan")
            for k in keys
        ]
        print(
            f"{gi:>4.2f} {dns:>4.2f} | "
            + " ".join(f"{v:>13.4g}" for v in vals)
            + f"  ({time.time()-t0:.1f}s)  # {label}"
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
        "nfkb/IkBat",
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


def cmd_diagnose(args) -> None:
    """Bracket a convergence failure: DP14 alone at GI=0/1, then the whole
    composite forced into one Diffrax solve (isolates coupling vs splitting).
    """

    def try_run(label, comp, macro_dt=5.0, groups=None):
        sched = Scheduler(groups=groups) if groups else Scheduler()
        t0 = time.time()
        try:
            res = sched.run(
                comp,
                t_span=(0.0, 50.0),
                macro_dt=macro_dt,
                y0=comp.initial_state_vec(),
                save_dt=macro_dt,
            )
            print(
                f"  OK  ({time.time()-t0:5.1f}s) {label:<40} "
                f"DNA_damage[end]={float(res.get('dp14/DNA_damage')[-1]):.4g}"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  FAIL ({time.time()-t0:5.1f}s) {label:<40} "
                f"-> {type(e).__name__}"
            )

    for sev, tag in [(DP14_IRRADIATION_RATE_DEFAULT, "GI=1"), (0.0, "GI=0")]:
        print(f"\n=== DP14 alone (no GZ06, no coupling), {tag} ===")
        proc = process_from_sbml(
            str(DP14_SBML_PATH),
            name="dp14",
            parameters={
                DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
                DP14_IRRADIATION_RATE_NAME: sev,
            },
        )
        try_run(
            f"DP14 alone, {tag}", single_process_composite(proc, name="dp14")
        )

    print("\n=== Single-group sweep (whole composite, no operator split) ===")
    base = build_multi_hallmark_composite()
    for gi in [1.0, 0.5, 0.3, 0.0]:
        comp = with_hallmarks(
            base,
            {"Genomic Instability": gi, "Deregulated Nutrient Sensing": 0.5},
        )
        try_run(
            f"GI={gi:>4.2f}, DNS=0.5 (single-group)",
            comp,
            macro_dt=50.0,
            groups={"all": list(comp.processes.keys())},
        )


def cmd_earlystop(args) -> None:
    """Train on DDIS, early-stop on the held-out RAS validation arm; plot the
    train-vs-val curves (RAS U-curve ⇒ early stop helps, else it's a no-op)."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("hallsim").setLevel(logging.INFO)
    val_arm, test_arm, fit_arm = "RAS_vs_ctrl", "RAPA_vs_ctrl", "DDIS_vs_ctrl"
    problem = build_problem()
    print(
        "[1/2] fitting (train=DDIS, early-stop on RAS, patience 15) ...",
        flush=True,
    )
    history = problem.fit(
        steps=150,
        mode="reverse",
        learning_rate=0.03,
        reduce_on_plateau=True,
        plateau_patience=5,
        early_stop_patience=15,
        validation_arms=[val_arm],
        verbose=True,
    )
    val, train = np.array(history.val_losses), np.array(history.losses)
    best_e = int(np.argmin(val))
    print(
        f"\nran {len(val)} epochs; stopped={history.stopped_epoch}; "
        f"best RAS-val @ {best_e} ({val[best_e]:.4g}); "
        f"final {val[-1]:.4g}",
        flush=True,
    )
    ev = problem.evaluate(history.final_params)
    for label, arm in [
        ("FIT DDIS", fit_arm),
        ("TEST RAPA", test_arm),
        ("VAL RAS", val_arm),
    ]:
        print(f"\n--- {label} ({arm}) ---")
        for t in sorted(ev[arm]):
            print(f"  @t{t}: {ev[arm][t]}", flush=True)
    out = ROOT / "outputs" / "multi_hallmark_earlystop"
    out.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(val))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(x, train, color="#2563eb", lw=2.0, label="train — DDIS (+prior)")
    ax.plot(x, val, color="#d97706", lw=2.0, label="validation — RAS")
    ax.axvline(best_e, color="#d97706", ls="--", lw=1.2, alpha=0.7)
    ax.scatter(
        [best_e],
        [val[best_e]],
        color="#d97706",
        zorder=6,
        label=f"RAS-val min @ {best_e}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train (DDIS) vs validation (RAS)")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(
        out / "train_vs_val.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    np.savez(
        out / "curves.npz",
        train=train,
        val=val,
        best_e=best_e,
        stopped=history.stopped_epoch or len(val),
    )
    print(f"\nwrote {out.relative_to(ROOT)}/train_vs_val.png", flush=True)


_COMMANDS = {
    "calibrate": cmd_calibrate,
    "baseline": cmd_baseline,
    "sweep": cmd_sweep,
    "diagnose-convergence": cmd_diagnose,
    "earlystop": cmd_earlystop,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "command", nargs="?", default="calibrate", choices=_COMMANDS
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
        "--cosine",
        action="store_true",
        help="cosine-decay the LR over the run (replaces plateau)",
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
    args = ap.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
