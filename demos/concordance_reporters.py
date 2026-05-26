"""Gene-reporter validation against GSE248823 + differentiable
calibration of mechanism parameters.

Two stages:

1. **Validation.** Run the default ``damage_p53_eriq`` composite under
   control, DDIS, and DDIS+rapamycin conditions. Map late-time state
   to mechanistic observables, compute Δ_observable per arm, and
   compare to log2 fold changes of the canonical reporter genes from
   the GSE248823 expression matrix. Report sign agreement and
   Spearman correlation for the calibration arm (DDIS) and the
   held-out arms (OIS — different senescence type; rapamycin rescue
   — pharmacological intervention).

2. **Calibration.** Tune three mechanism parameters
   (``alpha``, ``MDAMAGE_SA``, ``sasp_k``) by ``jax.grad`` + ``optax``
   against the DDIS gene-level Δ_data, then re-evaluate concordance
   on the held-out arms. Demonstrates end-to-end differentiability of
   the full composite (six processes, three timescales, SBML-imported
   oscillator, all of it) with respect to biology-interpretable knobs.

Usage::

    .venv_hallsim/bin/python demos/concordance_reporters.py
    .venv_hallsim/bin/python demos/concordance_reporters.py --calibrate --steps 80 --plot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import diffrax as dfx
import jax.numpy as jnp
import numpy as np

from hallsim import Scheduler
from hallsim.calibration import Calibrator
from hallsim.gene_reporters import (
    CANONICAL_REPORTERS,
    compute_concordance,
    derive_observables,
    load_gene_expression,
    log2_fold_change,
)
from hallsim.models.damage_p53_eriq import build_damage_p53_eriq_composite


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "FibroblastsDNA_dmg_Rapamycin"
SERIES_MATRIX = DATA_DIR / "GSE248823_series_matrix.txt"
PLATFORM = DATA_DIR / "GPL17586-45144.txt"


# Sample position groups in the 20-sample expression matrix.
SAMPLE_GROUPS = {
    "ETOPOSIDE_D00": [0, 1],
    "ETOPOSIDE_D14": [4, 5],
    "ETOPOSIDE_RAPA_D14": [8, 9],
    "RAS_D00": [10, 11],
    "RAS_D07": [14, 15],
}


# ── Mechanistic runs ─────────────────────────────────────────────


def _run(
    *,
    alpha: float,
    MDAMAGE_SA: float = 1.0,
    sasp_k: float = 0.05,
    rapamycin_strength: float = 0.0,
    with_sasp_mtor: bool = True,
    t_end: float = 50.0,
):
    comp = build_damage_p53_eriq_composite(
        alpha=alpha,
        MDAMAGE_SA=MDAMAGE_SA,
        with_sasp_mtor=with_sasp_mtor,
        sasp_k=sasp_k,
        rapamycin_strength=rapamycin_strength,
    )
    return Scheduler(max_steps=2_000_000).run(
        comp, t_span=(0.0, t_end), macro_dt=5.0, save_dt=5.0
    )


def _late_state(result, last_n: int = 3) -> dict:
    """Mean of last ``last_n`` save points to smooth oscillator phase."""
    n = min(last_n, result.ys.shape[0])
    avg = jnp.mean(result.ys[-n:], axis=0)
    return {k: avg[i] for i, k in enumerate(result.keys)}


def _delta_obs(ctrl_state, pert_state) -> dict:
    o_ctrl = derive_observables(ctrl_state)
    o_pert = derive_observables(pert_state)
    return {k: o_pert[k] - o_ctrl[k] for k in o_ctrl}


# ── Differentiable loss for mechanism calibration ─────────────────


def _normalized(v: jnp.ndarray) -> jnp.ndarray:
    """Zero-mean, unit-norm. Robust to scale mismatch between sim and data."""
    v = v - jnp.mean(v)
    return v / (jnp.sqrt(jnp.sum(v**2)) + 1e-8)


def make_loss_fn(delta_data_ddis: jnp.ndarray, reporter_order: list):
    """Build the calibration loss. Returns ``loss(params_dict)`` taking
    a dict with keys ``alpha``, ``MDAMAGE_SA``, ``sasp_k``.

    The Scheduler is configured with ``dfx.ForwardMode()`` so that
    ``jax.jvp`` (the autodiff primitive used by Calibrator's
    forward-mode path) traces through ``dfx.diffeqsolve`` without
    invoking the reverse-mode adjoint machinery.
    """
    signs = jnp.asarray([r.sign for r in reporter_order], dtype=float)
    target = _normalized(delta_data_ddis)
    fwd_adjoint = dfx.ForwardMode()
    CAL_T_END = 25.0

    def loss_fn(params):
        alpha = params["alpha"]
        mdam = params["MDAMAGE_SA"]
        sk = params["sasp_k"]
        sched = Scheduler(adjoint=fwd_adjoint)
        ctrl = sched.run(
            build_damage_p53_eriq_composite(
                alpha=0.01,
                MDAMAGE_SA=mdam,
                with_sasp_mtor=True,
                sasp_k=sk,
            ),
            t_span=(0.0, CAL_T_END),
            macro_dt=5.0,
            save_dt=5.0,
        )
        ddis = sched.run(
            build_damage_p53_eriq_composite(
                alpha=alpha,
                MDAMAGE_SA=mdam,
                with_sasp_mtor=True,
                sasp_k=sk,
            ),
            t_span=(0.0, CAL_T_END),
            macro_dt=5.0,
            save_dt=5.0,
        )
        # Smooth oscillator phase by averaging last 3 save points.
        ctrl_state = {
            k: jnp.mean(ctrl.ys[-3:, i]) for i, k in enumerate(ctrl.keys)
        }
        ddis_state = {
            k: jnp.mean(ddis.ys[-3:, i]) for i, k in enumerate(ddis.keys)
        }
        do_ctrl = derive_observables(ctrl_state)
        do_ddis = derive_observables(ddis_state)
        sim_vec = jnp.stack(
            [do_ddis[r.observable] - do_ctrl[r.observable] for r in reporter_order]
        )
        sim_signed = signs * sim_vec
        sim_norm = _normalized(sim_signed)
        return jnp.mean((sim_norm - target) ** 2)

    return loss_fn


# ── Reporting helpers ─────────────────────────────────────────────


def _print_concordance(label: str, result):
    print(f"\n  {label}")
    print(f"    {result}")


def _summary_table(rows: dict[str, dict]):
    print(f"\n  {'arm':<14}  {'sign agree':>11}  {'Spearman r':>12}")
    print(f"  {'-' * 14}  {'-' * 11}  {'-' * 12}")
    for arm, r in rows.items():
        sa = r["sign_agreement"] * 100
        rho = r["spearman_r"]
        print(f"  {arm:<14}  {sa:>10.1f}%  {rho:>+12.3f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run mechanism calibration via jax.grad + optax",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("HallSim — gene-reporter validation vs GSE248823")
    print("=" * 78)

    # ── 1. Load gene expression ───────────────────────────────
    print("\n[1/4] Loading gene-level expression matrix ...")
    gene_expr = load_gene_expression(SERIES_MATRIX, PLATFORM)
    samples = list(gene_expr.columns)
    print(f"      {gene_expr.shape[0]} gene symbols × {gene_expr.shape[1]} samples")

    # Per-arm Δ_data (log2 fold change).
    def cols(group):
        return [samples[i] for i in SAMPLE_GROUPS[group]]

    delta_ddis = log2_fold_change(
        gene_expr, cols("ETOPOSIDE_D14"), cols("ETOPOSIDE_D00")
    )
    delta_ois = log2_fold_change(
        gene_expr, cols("RAS_D07"), cols("RAS_D00")
    )
    delta_rapa = log2_fold_change(
        gene_expr, cols("ETOPOSIDE_RAPA_D14"), cols("ETOPOSIDE_D14")
    )
    print(f"      Δ_data computed for DDIS, OIS, and rapamycin-rescue arms")

    # ── 2. Default-parameter validation ───────────────────────
    print("\n[2/4] Running default damage_p53_eriq composite ...")
    print("      alpha=0.01 (ctrl) / 0.05 (DDIS); +rapa at strength=0.05")
    t0 = time.time()
    ctrl_res = _run(alpha=0.01)
    ddis_res = _run(alpha=0.05)
    rapa_res = _run(alpha=0.05, rapamycin_strength=0.05)
    print(f"      3 runs in {time.time() - t0:.1f}s")

    ctrl_state = _late_state(ctrl_res)
    ddis_state = _late_state(ddis_res)
    rapa_state = _late_state(rapa_res)

    print("\n[3/4] Per-arm concordance (default composite):")

    rows_default = {}
    for arm_name, ctrl_s, pert_s, dd in [
        ("DDIS", ctrl_state, ddis_state, delta_ddis),
        ("OIS  (held-out)", ctrl_state, ddis_state, delta_ois),
        ("RAPA (held-out)", ddis_state, rapa_state, delta_rapa),
    ]:
        delta_o = _delta_obs(ctrl_s, pert_s)
        delta_o_floats = {k: float(v) for k, v in delta_o.items()}
        result = compute_concordance(
            delta_observables=delta_o_floats,
            delta_gene_expression=dd,
            condition_name=arm_name,
        )
        _print_concordance(arm_name, result)
        rows_default[arm_name] = {
            "sign_agreement": result.sign_agreement,
            "spearman_r": result.spearman_r,
            "result": result,
        }

    print("\n  Summary (default composite, no calibration):")
    _summary_table(rows_default)

    if not args.calibrate:
        print("\n" + "=" * 78)
        print("Done (no calibration). Use --calibrate for the differentiable")
        print("mechanism-parameter calibration demo.")
        print("=" * 78)
        return

    # ── 3. Differentiable calibration on DDIS arm ─────────────
    print("\n[4/4] Calibrating mechanism parameters (alpha, MDAMAGE_SA, sasp_k)")
    print("      via jax.grad + optax on DDIS Δ_data ...")

    # Filter reporters to only those whose gene is in the expression data
    # AND whose observable will appear in derive_observables.
    reporters_used = [
        r
        for r in CANONICAL_REPORTERS
        if r.gene_symbol in delta_ddis.index
    ]
    target_ddis = jnp.asarray(
        [float(delta_ddis[r.gene_symbol]) for r in reporters_used]
    )
    print(f"      Using {len(reporters_used)} reporters: "
          f"{[r.gene_symbol for r in reporters_used]}")
    print(f"      Δ_data target (log2 fold change): "
          f"{[f'{float(x):+.3f}' for x in target_ddis]}")

    loss_fn = make_loss_fn(target_ddis, reporters_used)

    print(f"      Calibrating via hallsim.calibration.Calibrator ...")
    print(f"      mode='forward' (3 params, 1 scalar output → 4× forward cost)")
    calibrator = Calibrator(
        loss_fn=loss_fn,
        init_params={
            "alpha": jnp.asarray(0.05),
            "MDAMAGE_SA": jnp.asarray(1.0),
            "sasp_k": jnp.asarray(0.05),
        },
        clamps={
            "alpha": (0.005, 0.2),
            "MDAMAGE_SA": (0.5, 5.0),
            "sasp_k": (0.0, 0.2),
        },
        mode="forward",
        learning_rate=args.lr,
        log_every=max(1, args.steps // 10),
    )
    history = calibrator.fit(steps=args.steps)
    print(f"      Calibration finished in {history.wall_time_s:.1f}s")
    loss_hist = history.losses

    # ── 4. Evaluate calibrated composite on held-out arms ─────
    fp = history.final_params
    alpha_c, mdam_c, sk_c = (
        float(fp["alpha"]), float(fp["MDAMAGE_SA"]), float(fp["sasp_k"])
    )
    print(f"\n      Calibrated params: alpha={alpha_c:.3f}  "
          f"MDAMAGE_SA={mdam_c:.3f}  sasp_k={sk_c:.4f}")
    print("      Re-running composite with calibrated params ...")
    ctrl_c = _run(alpha=0.01, MDAMAGE_SA=mdam_c, sasp_k=sk_c)
    ddis_c = _run(alpha=alpha_c, MDAMAGE_SA=mdam_c, sasp_k=sk_c)
    rapa_c = _run(
        alpha=alpha_c, MDAMAGE_SA=mdam_c, sasp_k=sk_c, rapamycin_strength=0.05
    )
    ctrl_state_c = _late_state(ctrl_c)
    ddis_state_c = _late_state(ddis_c)
    rapa_state_c = _late_state(rapa_c)

    print("\n  Per-arm concordance after calibration:")
    rows_cal = {}
    for arm_name, ctrl_s, pert_s, dd in [
        ("DDIS  (in-sample)", ctrl_state_c, ddis_state_c, delta_ddis),
        ("OIS   (held-out)", ctrl_state_c, ddis_state_c, delta_ois),
        ("RAPA  (held-out)", ddis_state_c, rapa_state_c, delta_rapa),
    ]:
        delta_o = _delta_obs(ctrl_s, pert_s)
        delta_o_floats = {k: float(v) for k, v in delta_o.items()}
        result = compute_concordance(
            delta_observables=delta_o_floats,
            delta_gene_expression=dd,
            condition_name=arm_name,
        )
        _print_concordance(arm_name, result)
        rows_cal[arm_name] = {
            "sign_agreement": result.sign_agreement,
            "spearman_r": result.spearman_r,
            "result": result,
        }

    print("\n  Summary (calibrated composite):")
    _summary_table(rows_cal)

    print(
        "\n  Δ (default → calibrated):"
        f"\n     DDIS sign agreement: {rows_default['DDIS']['sign_agreement']*100:.1f}%"
        f" → {rows_cal['DDIS  (in-sample)']['sign_agreement']*100:.1f}%"
        f"\n     OIS  sign agreement: {rows_default['OIS  (held-out)']['sign_agreement']*100:.1f}%"
        f" → {rows_cal['OIS   (held-out)']['sign_agreement']*100:.1f}%"
        f"\n     RAPA sign agreement: {rows_default['RAPA (held-out)']['sign_agreement']*100:.1f}%"
        f" → {rows_cal['RAPA  (held-out)']['sign_agreement']*100:.1f}%"
    )

    # ── Optional plot ─────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available; skipping plot")
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(loss_hist, lw=2, color="C0")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss (MSE on normalized Δ vectors)")
        ax.set_title("Calibration loss: mechanism params vs DDIS Δ_data")
        ax.grid(alpha=0.3)

        ax = axes[1]
        arms = ["DDIS", "OIS", "RAPA"]
        # Map to row keys
        keys_default = ["DDIS", "OIS  (held-out)", "RAPA (held-out)"]
        keys_cal = ["DDIS  (in-sample)", "OIS   (held-out)", "RAPA  (held-out)"]
        d_sa = [rows_default[k]["sign_agreement"] * 100 for k in keys_default]
        c_sa = [rows_cal[k]["sign_agreement"] * 100 for k in keys_cal]
        x = np.arange(len(arms))
        ax.bar(x - 0.2, d_sa, width=0.4, color="C0", label="default")
        ax.bar(x + 0.2, c_sa, width=0.4, color="C1", label="calibrated")
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_ylabel("sign agreement (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("Per-arm sign agreement (default vs calibrated)")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        out = ROOT / "demos" / "plots" / "concordance_reporters.png"
        out.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out, dpi=140)
        print(f"\nSaved {out.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
