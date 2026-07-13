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

import logging
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hallsim.calibration import (  # noqa: E402
    CalibrationProblem,
    Condition,
    ParameterRef,
)
from hallsim.gene_reporters import (  # noqa: E402
    MULTI_HALLMARK_REPORTERS,
    GeneExpressionDataset,
)
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "FibroblastsDNA_dmg_Rapamycin"
SERIES_MATRIX = DATA_DIR / "GSE248823_series_matrix.txt"
PLATFORM = DATA_DIR / "GPL17586-45144.txt"

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

ARMS = ["DDIS_vs_ctrl", "RAPA_vs_DDIS", "RAS_vs_ctrl"]


def build_problem() -> CalibrationProblem:
    ds = GeneExpressionDataset.from_series_matrix(
        SERIES_MATRIX,
        PLATFORM,
        sample_groups={},
        sample_position_groups=SAMPLE_POSITION_GROUPS,
    )
    return CalibrationProblem(
        composite=build_multi_hallmark_composite(),
        reporters=MULTI_HALLMARK_REPORTERS,
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
            "RAPA": Condition(
                "RAPA",
                {
                    "Genomic Instability": 1.0,
                    "Deregulated Nutrient Sensing": 0.3,
                },
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
        # time is in days (t_end=14), so the measured D07 / D14 samples map
        # to query times 7.0 and 14.0. DDIS vs the D00 baseline; rapamycin
        # vs the time-matched etoposide arm.
        data={
            "DDIS_vs_ctrl": {
                7.0: ds.delta("ETOPOSIDE_D07", "ETOPOSIDE_D00"),
                14.0: ds.delta("ETOPOSIDE_D14", "ETOPOSIDE_D00"),
            },
            "RAPA_vs_DDIS": {
                7.0: ds.delta("ETOPOSIDE_RAPA_D07", "ETOPOSIDE_D07"),
                14.0: ds.delta("ETOPOSIDE_RAPA_D14", "ETOPOSIDE_D14"),
            },
            # RAS vs its own pre-oncogene D00 baseline, at D04 and D07.
            "RAS_vs_ctrl": {
                4.0: ds.delta("RAS_D04", "RAS_D00"),
                7.0: ds.delta("RAS_D07", "RAS_D00"),
            },
        },
        arm_pairs={
            "DDIS_vs_ctrl": ("DDIS", "ctrl"),
            "RAPA_vs_DDIS": ("RAPA", "DDIS"),
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
        },
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=["RAPA_vs_DDIS", "RAS_vs_ctrl"],
        prior_weight=0.03,
        t_end=14.0,
        macro_dt=3.5,
        # Save daily so the D07 and D14 query times land on the grid and the
        # trajectory is smooth for plotting; save density is free (dense
        # output, no extra ODE solves).
        n_save=15,
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
            e0_sum = e1_sum = 0.0
            for g in pre_r:
                e0 = abs(pre_r[g].delta_sim - pre_r[g].delta_data)
                e1 = abs(post_r[g].delta_sim - post_r[g].delta_data)
                e0_sum, e1_sum = e0_sum + e0, e1_sum + e1
                print(
                    f"  {'':<9}{g:<9}{pre_r[g].delta_data:>+10.3f}"
                    f"{pre_r[g].delta_sim:>+12.4f}"
                    f"{post_r[g].delta_sim:>+12.4f}   {e0:>6.3f}→{e1:<6.3f}"
                )
            n = len(pre_r)
            print(
                f"  {'':<9}{'mean|err|':<9}{'':>10}{'':>12}{'':>12}   "
                f"{e0_sum / n:>6.3f}→{e1_sum / n:<6.3f}   "
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

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

    # Each param's position within its clamp range, in log space (the
    # params span orders of magnitude), so trajectories are comparable.
    for name, ref in problem.params.items():
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


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("hallsim").setLevel(logging.INFO)
    problem = build_problem()
    init = {k: jnp.asarray(p.init) for k, p in problem.params.items()}

    print("[1/3] out-of-the-box concordance ...", flush=True)
    pre = problem.evaluate(init)
    for arm in ARMS:
        for t in sorted(pre[arm]):
            print(pre[arm][t], flush=True)

    out_dir = ROOT / "outputs" / "multi_hallmark_calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[2/3] fitting ...", flush=True)
    history = problem.fit(
        steps=150,
        mode="reverse",
        learning_rate=0.03,
        reduce_on_plateau=True,
        plateau_patience=5,
        early_stop_patience=15,
        verbose=True,
        checkpoint_path=out_dir / "checkpoint.npz",
    )

    print("[3/3] calibrated concordance ...", flush=True)
    post = problem.evaluate(history.final_params)

    print_table(pre, post)
    print("\nfitted parameters (init → fit):")
    for k in problem.params:
        print(
            f"  {k:<20}{float(init[k]):>12.5g} → "
            f"{float(history.final_params[k]):>12.5g}"
        )

    plot(pre, post, out_dir / "oob_vs_cal_vs_measured.png")
    plot_history(problem, history, out_dir / "training_history.png")
    problem.save_outputs(str(out_dir), history)
    print(
        f"\nbest loss {history.best_loss:.4g} over {len(history.losses)} "
        f"epochs → outputs in {out_dir.relative_to(ROOT)}/"
    )


if __name__ == "__main__":
    main()
