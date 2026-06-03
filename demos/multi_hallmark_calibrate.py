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

# float64: the DP14 forward-mode sensitivities are large near the damage
# regime and overflow float32 to NaN during the calibration JVP (the
# primal solve is fine in float32; only the tangent overflows). Enable
# before any JAX array is created.
import jax

jax.config.update("jax_enable_x64", True)
import diffrax as dfx

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
        # Mechanism candidates — biological rate constants that vary
        # across cell states / genotypes / disease conditions, that
        # gene-expression data legitimately informs.
        #
        #   etoposide_potency ← dp14.DNA_damaged_by_irradiation
        #                       (damage per unit exposure — the dose knob.
        #                        Severity sets the Irradiation *level*; this
        #                        is what "full exposure" means in DP14 units,
        #                        recalibrated from DallePezze's γ value for
        #                        etoposide. No longer hallmark-targeted, so
        #                        it is a plain fittable mechanism parameter.)
        #   HMOX1   ← dp14.ROS_turnover           (ROS clearance)
        #   CDKN1A  ← dp14.CDKN1A_transcr_...     (p21 transcription gain)
        #   CYCS    ← dp14.mitophagy_inactiv_...  (mTOR→mitophagy)
        #   DDB2    ← gz06.alpha_y                (Mdm2 turnover, p53 ampl.)
        #
        # EIF4EBP1's mTOR phos rate is correctly hallmark-controlled.
        params={
            "etoposide_potency": ParameterRef(
                process_name="dp14",
                field="parameters.DNA_damaged_by_irradiation",
                init=10.0,
                clamp=(0.01, 10000.0),
                description=(
                    "DNA damage produced per unit Irradiation exposure. "
                    "DallePezze's published 9237.72 is the γ-irradiation "
                    "value (drives DNA_damage to ~28k); fit here to the "
                    "etoposide DDIS regime. Severity sets the exposure "
                    "level, not this potency."
                ),
            ),
            "ROS_turnover": ParameterRef(
                process_name="dp14",
                field="parameters.ROS_turnover",
                init=3.231,
                clamp=(0.1, 50.0),
                description=(
                    "DP14's ROS clearance rate. Controls steady-state "
                    "ROS pool size. Informs HMOX1 reporter via "
                    "Nrf2/ARE-driven antioxidant response."
                ),
            ),
            "CDKN1A_transcr": ParameterRef(
                process_name="dp14",
                field="parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
                init=0.085,
                clamp=(0.001, 5.0),
                description=(
                    "p21 (CDKN1A) transcription rate via "
                    "FoxO3a × DNA_damage. Direct amplitude knob for "
                    "the CDKN1A reporter — gain on the damage-to-p21 "
                    "cascade."
                ),
            ),
            "mitophagy_inactiv": ParameterRef(
                process_name="dp14",
                field="parameters.mitophagy_inactiv_by_mTORC1_pS2448",
                init=646.0,
                clamp=(1.0, 10000.0),
                description=(
                    "mTORC1's negative effect on mitophagy. Drives "
                    "the rapamycin axis (low mTOR → de-suppressed "
                    "mitophagy → mito turnover). Informs CYCS via "
                    "Mito_mass_new."
                ),
            ),
            "alpha_y": ParameterRef(
                process_name="gz06",
                field="parameters.alpha_y",
                init=0.8,
                clamp=(0.01, 10.0),
                description=(
                    "GZ06's Mdm2 protein degradation rate. Controls "
                    "p53 oscillation amplitude. Informs DDB2 via "
                    "cycle-averaged p53 level."
                ),
            ),
        },
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=["RAPA_vs_DDIS"],
        t_end=T_END,
        macro_dt=MACRO_DT,
        n_save=N_SAVE,
        # DP14's mitochondrial subsystem is stiff (λ≈-5000/day); an
        # explicit solver makes the forward sensitivity overflow to NaN.
        # A stiff (implicit, A-stable) solver integrates the variational
        # equation stably. Provisional global setting; per-group
        # auto-selection is the planned refinement.
        scheduler_kwargs={
            "solver": dfx.Kvaerno3(),
            "rtol": 1e-6,
            "atol": 1e-4,
            "max_steps": 200_000,
        },
    )
    print(
        f"      {len(problem.reporters)} reporters × "
        f"{len(problem.conditions)} conditions × "
        f"{len(problem.params)} fittable params"
    )

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
