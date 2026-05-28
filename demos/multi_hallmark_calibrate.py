"""End-to-end gene-reporter validation on the multi-hallmark composite.

Wires DP14 + NFKB + GZ06 against GSE248823 (etoposide DDIS ± rapamycin)
through the high-level :class:`hallsim.calibration.CalibrationProblem`
API:

- Three experimental arms: ctrl, DDIS, RAPA — each a hallmark severity
  profile (Genomic Instability + Deregulated Nutrient Sensing).
- Two mechanism parameters fit jointly: DP14's ``DNA_damaged_by_irradiation``
  rate constant and GZ06's ``psi`` damage-signal amplitude. The Genomic
  Instability hallmark scales each by severity at its own model's
  calibrated regime — no DP14↔GZ06 topology coupling.
- Held-out validation: calibrate on DDIS_vs_ctrl, evaluate concordance
  on RAPA_vs_DDIS without using it in the loss.
- Six gene reporters (CDKN1A, DDB2, HMOX1, NFKBIA, CYCS, EIF4EBP1)
  via :data:`MULTI_HALLMARK_REPORTERS`. DDB2 routes through GZ06's
  oscillating p53 and uses cycle-averaging for phase-insensitive
  comparison to bulk RNA.

Usage::

    .venv_hallsim/bin/python demos/multi_hallmark_calibrate.py
"""

from __future__ import annotations

import time
from pathlib import Path

from hallsim.calibration import CalibrationProblem, Condition, ParameterRef
from hallsim.gene_reporters import (
    MULTI_HALLMARK_REPORTERS,
    GeneExpressionDataset,
)
from hallsim.models.multi_hallmark import build_multi_hallmark_composite


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "FibroblastsDNA_dmg_Rapamycin"
SERIES_MATRIX = DATA_DIR / "GSE248823_series_matrix.txt"
PLATFORM = DATA_DIR / "GPL17586-45144.txt"

# Sample positions in GSE248823's 20-sample series matrix.
# Same indexing as demos/concordance_reporters.py — etoposide is the
# DDIS arm, RAS is the OIS arm, RAPA is etoposide+rapamycin rescue.
SAMPLE_POSITION_GROUPS = {
    "ETOPOSIDE_D00": [0, 1],
    "ETOPOSIDE_D14": [4, 5],
    "ETOPOSIDE_RAPA_D14": [8, 9],
}


T_END = 25.0
MACRO_DT = 5.0
N_SAVE = 6


def print_concordance(results: dict) -> None:
    print(f"\n  {'arm':<20}  {'sign agree':>11}  {'Spearman r':>12}  {'n':>3}")
    print(f"  {'-'*20}  {'-'*11}  {'-'*12}  {'-'*3}")
    for arm, r in results.items():
        sa = r.sign_agreement * 100
        rho = r.spearman_r
        print(f"  {arm:<20}  {sa:>10.1f}%  {rho:>+12.3f}  {r.n_compared:>3}")


def main() -> None:
    print("=" * 78)
    print("HallSim — multi-hallmark composite gene-reporter validation")
    print("=" * 78)

    # 1. Load gene expression
    print("\n[1/4] Loading GSE248823 gene-expression matrix ...")
    ds = GeneExpressionDataset.from_series_matrix(
        SERIES_MATRIX,
        PLATFORM,
        sample_groups={},
        sample_position_groups=SAMPLE_POSITION_GROUPS,
    )
    print(f"      {ds.gene_expr.shape[0]} gene symbols × "
          f"{ds.gene_expr.shape[1]} samples")

    # 2. Build problem
    print("\n[2/4] Wiring CalibrationProblem ...")
    problem = CalibrationProblem(
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
        },
        data={
            "DDIS_vs_ctrl": ds.delta("ETOPOSIDE_D14", "ETOPOSIDE_D00"),
            "RAPA_vs_DDIS": ds.delta("ETOPOSIDE_RAPA_D14", "ETOPOSIDE_D14"),
        },
        arm_pairs={
            "DDIS_vs_ctrl": ("DDIS", "ctrl"),
            "RAPA_vs_DDIS": ("RAPA", "DDIS"),
        },
        params={
            "irradiation_rate": ParameterRef(
                process_name="dp14",
                field="parameter_overrides.DNA_damaged_by_irradiation",
                init=1.0,
                clamp=(0.0, 1e5),
                description=(
                    "DP14's exogenous-damage rate constant at full DDIS. "
                    "SBML default 9237.72 (γ-irradiation calibration) "
                    "produces DNA_damage~28000; effective dose for "
                    "etoposide DDIS is fit from data."
                ),
            ),
            "gz06_psi": ParameterRef(
                process_name="gz06",
                field="parameter_overrides.psi",
                init=1.0,
                clamp=(0.0, 10.0),
                description=(
                    "GZ06's damage-signal amplitude at full DDIS. "
                    "Geva-Zatorsky 2006 calibrated ψ near 1.0; fit "
                    "around that scale."
                ),
            ),
        },
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=["RAPA_vs_DDIS"],
        t_end=T_END,
        macro_dt=MACRO_DT,
        n_save=N_SAVE,
    )
    print("      6 reporters × 3 conditions × 2 fittable params")

    # 3. Pre-fit baseline
    print("\n[3/4] Pre-fit concordance (default parameters) ...")
    init_params = {
        k: __import__("jax").numpy.asarray(p.init)
        for k, p in problem.params.items()
    }
    t0 = time.time()
    pre_results = problem.evaluate(init_params)
    print(f"      evaluate ran in {time.time()-t0:.1f}s")
    print_concordance(pre_results)

    # 4. Fit + post-fit concordance
    print("\n[4/4] Fitting (20 Adam steps, forward-mode autodiff) ...")
    t0 = time.time()
    history = problem.fit(steps=20, learning_rate=0.05, verbose=True)
    print(f"\n      Fit ran in {time.time()-t0:.1f}s")
    print(f"      {history}")

    print("\n      Post-fit concordance:")
    post_results = problem.evaluate(history.final_params)
    print_concordance(post_results)

    print("\n      Fitted parameters:")
    for k, v in history.final_params.items():
        print(f"        {k:<25} = {float(v):.6g}")

    print(
        "\n      Held-out RAPA row is the validation number — calibrate on "
        "DDIS_vs_ctrl, evaluate on RAPA_vs_DDIS."
    )

    # 5. Artifact bundle (topology graph + per-arm trajectories +
    # pre-vs-post comparison + concordance JSON).
    out_dir = ROOT / "outputs" / "multi_hallmark_calibrate"
    print(f"\n[5/5] Writing artifacts to {out_dir.relative_to(ROOT)} ...")
    info = problem.save_outputs(str(out_dir), history, n_save_plot=50)
    for fname in info["files"]:
        print(f"      {fname}")


if __name__ == "__main__":
    main()
