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
    ParameterRef,
)
from hallsim.composite import Composite  # noqa: E402
from hallsim.hallmarks import apply_hallmarks  # noqa: E402
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


def build_problem(composite=None) -> CalibrationProblem:
    ds = GeneExpressionDataset.from_series_matrix(
        SERIES_MATRIX,
        PLATFORM,
        sample_groups={},
        sample_position_groups=SAMPLE_POSITION_GROUPS,
    )
    return CalibrationProblem(
        composite=composite
        if composite is not None
        else build_multi_hallmark_composite(),
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
            # p53 → CDKN1A edge (P53CDKN1AActivator.k_act) is fixed, not fitted.
        },
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=["RAPA_vs_DDIS", "RAS_vs_ctrl"],
        prior_weight=0.03,
        t_end=14.0,
        macro_dt=3.5,
        # Save every 0.5 day so the D07/D14 query times and their 2-day window
        # edges (5, 7, 12, 14) land exactly on the grid; save density is free
        # (dense output, no extra ODE solves).
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
            rows.append((
                f"{short} ({tag})", f"{t:g}",
                pre[arm][t].spearman_r, post[arm][t].spearman_r,
                pre[arm][t].mean_abs_error, post[arm][t].mean_abs_error,
            ))

    with open(out_dir / "concordance_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "day", "rho_oob", "rho_cal",
                    "mean_abs_err_oob", "mean_abs_err_cal"])
        for arm, day, ro, rc, eo, ec in rows:
            w.writerow([arm, day, f"{ro:.3f}", f"{rc:.3f}",
                        f"{eo:.3f}", f"{ec:.3f}"])

    IMP, REG, DIM, INK = "#1a7f4b", "#c0552b", "#6b7280", "#1f2937"
    header = ["Arm", "Day", "ρ (oob)", "ρ (cal)",
              "mean|err| (oob)", "mean|err| (cal)"]
    text, colors = [header], [[INK] * 6]
    for arm, day, ro, rc, eo, ec in rows:
        text.append([arm, day, f"{ro:+.2f}", f"{rc:+.2f}",
                     f"{eo:.2f}", f"{ec:.2f}"])
        colors.append([INK, INK, DIM, IMP if rc >= ro else REG,
                       DIM, IMP if ec <= eo else REG])

    fig, ax = plt.subplots(figsize=(8.6, 0.55 + 0.42 * len(text)))
    ax.axis("off")
    tbl = ax.table(cellText=text, cellLoc="center", loc="center",
                   colWidths=[0.26, 0.10, 0.14, 0.14, 0.18, 0.18])
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
    ax.set_title("Calibrated vs out-of-the-box concordance",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left",
                 pad=14)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"concordance_table.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote concordance_table.png/.csv -> {out_dir}", flush=True)


def cmd_calibrate(args) -> None:
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
    # LR kept low enough that Adam's early bias-corrected steps can't overshoot
    # a dominant-gradient parameter (CDKN1A_transcr) to its clamp floor and
    # thrash — the fit descends to a joint minimum where best ≈ final rather
    # than latching a single-epoch transient.
    history = problem.fit(
        steps=150,
        mode="reverse",
        learning_rate=0.005,
        reduce_on_plateau=True,
        plateau_patience=8,
        early_stop_patience=30,
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

    write_concordance_table(pre, post, out_dir)
    plot_history(problem, history, out_dir / "training_history.png")
    problem.save_outputs(str(out_dir), history)

    # Re-plot the calibrated figures on the fit just written, so the
    # time-domain trajectories and concordance dumbbells never lag behind the
    # checkpoint.
    from multi_hallmark_figures import fig_temporal, fig_concordance
    print("temporal reporter trajectories ...", flush=True)
    fig_temporal(args)
    print("reporter concordance dumbbells ...", flush=True)
    fig_concordance(args)

    print(
        f"\nbest loss {history.best_loss:.4g} over {len(history.losses)} "
        f"epochs → outputs in {out_dir.relative_to(ROOT)}/"
    )


# ── baseline: uncalibrated composite at the three arms ───────────────────
_ARMS_3 = [(0.0, 0.5, "ctrl"), (1.0, 1.0, "DDIS"), (1.0, 0.3, "DDIS+rapa")]


def _run_arms(base, gi, dns, t_end=50.0, macro_dt=5.0):
    procs = apply_hallmarks(base.processes, {
        "Genomic Instability": gi, "Deregulated Nutrient Sensing": dns})
    comp = Composite(processes=procs, topology=base.topology, validate=False,
                     semantic_validation={"check_semantics": False})
    return Scheduler().run(comp, t_span=(0.0, t_end), macro_dt=macro_dt,
                           y0=comp.initial_state_vec(), save_dt=macro_dt)


def cmd_baseline(args) -> None:
    """Uncalibrated composite: terminal readouts per arm (measures overload)."""
    base = build_multi_hallmark_composite()
    keys = ["dp14/DNA_damage", "dp14/CDKN1A", "dp14/mTORC1_pS2448",
            "nfkb/IkBat", "gz06/x", "gz06/y"]
    hdr = f"{'GI':>4} {'DNS':>4} | " + " ".join(
        f"{k.split('/')[-1]:>13}" for k in keys)
    print("Multi-hallmark composite, uncalibrated.\n" + hdr)
    print("-" * len(hdr))
    for gi, dns, label in _ARMS_3:
        t0 = time.time()
        try:
            res = _run_arms(base, gi, dns)
        except Exception as e:  # noqa: BLE001
            print(f"{gi:>4.2f} {dns:>4.2f} | FAIL -> {type(e).__name__}  "
                  f"# {label}")
            continue
        vals = [float(res.get(k)[-1]) if res.get(k) is not None else float("nan")
                for k in keys]
        print(f"{gi:>4.2f} {dns:>4.2f} | "
              + " ".join(f"{v:>13.4g}" for v in vals)
              + f"  ({time.time()-t0:.1f}s)  # {label}")


def cmd_sweep(args) -> None:
    """Two-hallmark severity sweep — readouts gene-reporter validation uses."""
    base = build_multi_hallmark_composite()
    keys = ["dp14/DNA_damage", "dp14/CDKN1A", "dp14/mTORC1_pS2448",
            "dp14/ROS", "dp14/Mitophagy", "nfkb/IkBat"]
    hdr = f"{'GI':>5} {'DNS':>5} | " + " ".join(
        f"{k.split('/')[-1]:>14}" for k in keys)
    print(f"Severity sweep — multi_hallmark composite\n{hdr}")
    print("-" * len(hdr))
    for gi, dns, label in _ARMS_3:
        res = _run_arms(base, gi, dns)
        vals = [float(res.get(k)[-1]) if res.get(k) is not None else float("nan")
                for k in keys]
        print(f"{gi:>5.2f} {dns:>5.2f} | "
              + " ".join(f"{v:>14.4g}" for v in vals) + f"  # {label}")


def cmd_diagnose(args) -> None:
    """Bracket a convergence failure: DP14 alone at GI=0/1, then the whole
    composite forced into one Diffrax solve (isolates coupling vs splitting)."""
    def try_run(label, comp, macro_dt=5.0, groups=None):
        sched = Scheduler(groups=groups) if groups else Scheduler()
        t0 = time.time()
        try:
            res = sched.run(comp, t_span=(0.0, 50.0), macro_dt=macro_dt,
                            y0=comp.initial_state_vec(), save_dt=macro_dt)
            print(f"  OK  ({time.time()-t0:5.1f}s) {label:<40} "
                  f"DNA_damage[end]={float(res.get('dp14/DNA_damage')[-1]):.4g}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL ({time.time()-t0:5.1f}s) {label:<40} "
                  f"-> {type(e).__name__}")

    for sev, tag in [(DP14_IRRADIATION_RATE_DEFAULT, "GI=1"), (0.0, "GI=0")]:
        print(f"\n=== DP14 alone (no GZ06, no coupling), {tag} ===")
        proc = process_from_sbml(str(DP14_SBML_PATH), name="dp14", parameters={
            DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
            DP14_IRRADIATION_RATE_NAME: sev})
        try_run(f"DP14 alone, {tag}", Composite(
            processes={"dp14": proc}, topology={}, validate=False,
            semantic_validation={"check_semantics": False}))

    print("\n=== Single-group sweep (whole composite, no operator split) ===")
    base = build_multi_hallmark_composite()
    for gi in [1.0, 0.5, 0.3, 0.0]:
        procs = apply_hallmarks(base.processes, {
            "Genomic Instability": gi, "Deregulated Nutrient Sensing": 0.5})
        comp = Composite(processes=procs, topology=base.topology,
                         validate=False,
                         semantic_validation={"check_semantics": False})
        try_run(f"GI={gi:>4.2f}, DNS=0.5 (single-group)", comp, macro_dt=50.0,
                groups={"all": list(comp.processes.keys())})


def cmd_earlystop(args) -> None:
    """Train on DDIS, early-stop on the held-out RAS validation arm; plot the
    train-vs-val curves (RAS U-curve ⇒ early stop helps, else it's a no-op)."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("hallsim").setLevel(logging.INFO)
    val_arm, test_arm, fit_arm = "RAS_vs_ctrl", "RAPA_vs_DDIS", "DDIS_vs_ctrl"
    problem = build_problem()
    print("[1/2] fitting (train=DDIS, early-stop on RAS, patience 15) ...",
          flush=True)
    history = problem.fit(
        steps=150, mode="reverse", learning_rate=0.03, reduce_on_plateau=True,
        plateau_patience=5, early_stop_patience=15, validation_arms=[val_arm],
        verbose=True)
    val, train = np.array(history.val_losses), np.array(history.losses)
    best_e = int(np.argmin(val))
    print(f"\nran {len(val)} epochs; stopped={history.stopped_epoch}; "
          f"best RAS-val @ {best_e} ({val[best_e]:.4g}); "
          f"final {val[-1]:.4g}", flush=True)
    ev = problem.evaluate(history.final_params)
    for label, arm in [("FIT DDIS", fit_arm), ("TEST RAPA", test_arm),
                       ("VAL RAS", val_arm)]:
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
    ax.scatter([best_e], [val[best_e]], color="#d97706", zorder=6,
               label=f"RAS-val min @ {best_e}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train (DDIS) vs validation (RAS)")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "train_vs_val.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    np.savez(out / "curves.npz", train=train, val=val, best_e=best_e,
             stopped=history.stopped_epoch or len(val))
    print(f"\nwrote {out.relative_to(ROOT)}/train_vs_val.png", flush=True)


_COMMANDS = {"calibrate": cmd_calibrate, "baseline": cmd_baseline,
             "sweep": cmd_sweep, "diagnose-convergence": cmd_diagnose,
             "earlystop": cmd_earlystop}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", default="calibrate",
                    choices=_COMMANDS)
    args = ap.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
