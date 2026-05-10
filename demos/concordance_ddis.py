"""DDIS / OIS concordance demo with held-out validation.

Validates the mechanistic→pathway mapping using actual ssGSEA NES values
from GSE248823 (run scripts/run_ssgsea.py first to produce
data/ssgsea_deltas.csv).

Methodology — held-out validation
---------------------------------
The PathwayMapper has 14 calibratable Hill parameters. Calibrating them
against the same data you then evaluate against is circular; you can hit
near-perfect concordance without learning anything about mechanism. To
defend against that, we run **two evaluations**:

1. **Sign agreement (calibration-invariant primary metric).** Hill
   calibration tunes thresholds and steepness; it cannot flip a sign of
   a multi-input pathway formula because the activator/inhibitor
   structure is fixed. So sign agreement on the *default* mapper is a
   true test of mechanism — what fraction of our 7 pathway predictions
   move in the same direction as ssGSEA NES?

2. **Held-out Pearson r (calibrate on DDIS, evaluate on OIS).** We
   calibrate Hill parameters on the etoposide arm and report Pearson r
   on the held-out RAS arm. Both arms drive senescence but via
   different upstream mechanisms (DDR vs. oncogene); a calibrated mapper
   that generalizes to OIS demonstrates the formulas reflect general
   senescence biology, not arm-specific overfit.

Two model variants are compared:

- *Vanilla*: ``with_sasp_mtor=False`` — canonical ERiQ, p53 → mTOR
  inhibition.
- *SASP-corrected*: ``with_sasp_mtor=True`` — adds the chronic-stress
  mTOR activation term (Carroll 2017, Laberge 2015, Herranz 2015,
  Houssaini 2018, Fielder 2017). The literature claim about
  paradoxical mTOR in senescence is at the *kinase-activity* level;
  HALLMARK_MTORC1_SIGNALING is a *transcriptional* gene set, so we
  expect the SASP module to *not* close the mTOR sign mismatch on the
  ssGSEA side — and indeed that's what we observe. We report it
  honestly as a methodological finding (transcript-level vs.
  kinase-level mTOR activity diverge in senescence).

Usage::

    .venv_hallsim/bin/python demos/concordance_ddis.py
    .venv_hallsim/bin/python demos/concordance_ddis.py --plot --steps 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

from hallsim import (
    PATHWAY_ORDER,
    PathwayMapper,
    Scheduler,
    calibrate_pathway_mapper,
    pearson_r,
    sign_agreement,
)
from hallsim.models.damage_p53_eriq import build_damage_p53_eriq_composite
from hallsim.models.eriq import _compute_algebraic

ROOT = Path(__file__).parent.parent
DELTAS_CSV = ROOT / "data" / "ssgsea_deltas.csv"


# ── Mechanistic simulator runs ──────────────────────────────────────


def _run_condition(
    alpha: float,
    *,
    with_sasp_mtor: bool,
    rapamycin_strength: float = 0.0,
    t_end: float = 50.0,
):
    """Run damage_p53_eriq with the given alpha. ``alpha=0.01`` is the
    control / homeostatic regime; higher alphas push the composite into
    a chronic-damage senescence regime. ``rapamycin_strength > 0`` adds
    a pharmacological mTORC1 inhibitor (the DDIS-rescue arm)."""
    composite = build_damage_p53_eriq_composite(
        alpha=alpha,
        with_sasp_mtor=with_sasp_mtor,
        rapamycin_strength=rapamycin_strength,
    )
    return Scheduler(max_steps=2_000_000).run(
        composite, t_span=(0.0, t_end), macro_dt=5.0, save_dt=5.0
    )


def _late_state(result, last_n: int = 5) -> dict[str, jnp.ndarray]:
    n = min(last_n, result.ys.shape[0])
    avg = jnp.mean(result.ys[-n:], axis=0)
    return {k: avg[i] for i, k in enumerate(result.keys)}


def _eriq_observables(state: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Extract the five PathwayMapper inputs from an ERiQ state dict."""
    sub = {
        "mito_function": state["eriq/mito_function"],
        "glycolysis": state["eriq/glycolysis"],
        "mito_damage": state["eriq/mito_damage"],
        "mTOR_activity": state["eriq/mTOR_activity"],
        "p53_activity": state["eriq/p53_activity"],
        "ROS_activity": state["eriq/ROS_activity"],
        "ROS_integrator_c": state["eriq/ROS_integrator_c"],
    }
    obs = _compute_algebraic(sub)
    return {
        "p53": state["eriq/p53_activity"],
        "mtor": obs["MTOR"],
        "nfkb": obs["NFKB"],
        "ros": obs["ROS"],
        "atp": obs["ATPr"],
    }


# ── Δ vector computation ────────────────────────────────────────────


def _delta_vec(
    mapper: PathwayMapper,
    ctrl_state: dict[str, jnp.ndarray],
    pert_state: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    ctrl = mapper.from_eriq_state(ctrl_state).as_vector(PATHWAY_ORDER)
    pert = mapper.from_eriq_state(pert_state).as_vector(PATHWAY_ORDER)
    return pert - ctrl


def _print_table(
    delta_sim: jnp.ndarray,
    delta_data: jnp.ndarray,
    *,
    label: str,
):
    print(f"\n  {label}")
    print(
        f"    {'pathway':<22}  {'Δ_sim':>8}  {'Δ_data':>8}  {'sign':>6}"
    )
    print(f"    {'-' * 22}  {'-' * 8}  {'-' * 8}  {'-' * 6}")
    for i, name in enumerate(PATHWAY_ORDER):
        ds = float(delta_sim[i])
        dd = float(delta_data[i])
        sign = (
            "OK" if ds * dd > 0 else ("≈0" if ds == 0 or dd == 0 else "X")
        )
        print(f"    {name:<22}  {ds:+8.3f}  {dd:+8.3f}  {sign:>6}")


def _summary_line(
    label: str, delta_sim: jnp.ndarray, delta_data: jnp.ndarray
) -> tuple[float, float]:
    r = float(pearson_r(delta_sim, delta_data))
    sa = float(sign_agreement(delta_sim, delta_data))
    print(f"    {label:<32}  r = {r:+.3f}    sign agreement = {sa*100:5.1f}%")
    return r, sa


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip Hill calibration (default-mapper sign agreement only)",
    )
    parser.add_argument(
        "--steps", type=int, default=800, help="Calibration optimizer steps"
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save concordance figure to demos/plots/concordance_ddis.png",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("HallSim — DDIS/OIS pathway concordance vs GSE248823 (held-out)")
    print("=" * 78)

    # ── Load real Δ_data from ssGSEA ───────────────────────────────
    if not DELTAS_CSV.exists():
        raise SystemExit(
            f"Missing {DELTAS_CSV} — run scripts/run_ssgsea.py first."
        )
    deltas_df = pd.read_csv(DELTAS_CSV, index_col=0)
    deltas_df = deltas_df.reindex(list(PATHWAY_ORDER))
    delta_DDIS = jnp.asarray(deltas_df["DDIS_D14_vs_D00"].values, dtype=float)
    delta_OIS = jnp.asarray(deltas_df["OIS_D07_vs_D00"].values, dtype=float)
    delta_RAPA = jnp.asarray(
        deltas_df["DDIS_RAPA_vs_DDIS_D14"].values, dtype=float
    )
    print(
        f"\nLoaded ssGSEA Δ_data from {DELTAS_CSV.relative_to(ROOT)} "
        f"(7 pathways × DDIS+OIS arms)"
    )

    # ── Run mechanistic simulator for variants and conditions ──────
    print("\n[1/4] Running damage_p53_eriq composites ...")
    print("        Vanilla / SASP-mTOR-corrected ×")
    print("          control (α=0.01) / DDIS (α=0.05) / DDIS+rapamycin")
    sims = {}
    for variant_name, with_sasp in [("vanilla", False), ("sasp", True)]:
        ctrl = _run_condition(alpha=0.01, with_sasp_mtor=with_sasp)
        ddis = _run_condition(alpha=0.05, with_sasp_mtor=with_sasp)
        ddis_rapa = _run_condition(
            alpha=0.05, with_sasp_mtor=with_sasp, rapamycin_strength=0.05
        )
        sims[variant_name] = {
            "ctrl": _late_state(ctrl),
            "ddis": _late_state(ddis),
            "ddis_rapa": _late_state(ddis_rapa),
        }
        # Diagnostic: what does the simulator's mTOR_activity actually do?
        c = float(ctrl.get("eriq/mTOR_activity")[-1])
        d = float(ddis.get("eriq/mTOR_activity")[-1])
        r = float(ddis_rapa.get("eriq/mTOR_activity")[-1])
        print(
            f"        {variant_name:<8}  mTOR_activity ctrl={c:+.3f} | "
            f"ddis={d:+.3f} | ddis+rapa={r:+.3f}"
        )

    # ── Default mapper (no calibration): sign agreement primary ─────
    print("\n[2/4] Default Hill mapper (K=0.5, n=4) — primary mechanism test")
    print("       Sign agreement is calibration-invariant: it tests whether")
    print("       the formulas predict the right *direction* of change.")

    default = PathwayMapper()
    print("\n  ─── DDIS arm (etoposide D14 vs D00) ──────────────────────────")
    for variant in ("vanilla", "sasp"):
        d_sim = _delta_vec(
            default, sims[variant]["ctrl"], sims[variant]["ddis"]
        )
        _print_table(d_sim, delta_DDIS, label=f"variant: {variant}")
        _summary_line(f"  {variant}", d_sim, delta_DDIS)

    print("\n  ─── OIS arm (RAS D07 vs D00) — held-out target ────────────")
    print("       (Mechanistic side reuses DDIS run — both arms reach")
    print("        senescence-like states from different upstream paths.)")
    for variant in ("vanilla", "sasp"):
        d_sim = _delta_vec(
            default, sims[variant]["ctrl"], sims[variant]["ddis"]
        )
        _print_table(d_sim, delta_OIS, label=f"variant: {variant}")
        _summary_line(f"  {variant}", d_sim, delta_OIS)

    if args.no_calibrate:
        return

    # ── Calibrate on DDIS, evaluate held-out on OIS ────────────────
    print("\n[3/4] Held-out calibration: fit Hill on DDIS, evaluate on OIS ...")
    print("       This is the legitimate r number — calibration cannot have")
    print("       seen the OIS deltas, so the OIS r tests generalization.")

    hold_out_results = {}
    for variant in ("vanilla", "sasp"):
        ctrl_obs = _eriq_observables(sims[variant]["ctrl"])
        pert_obs = _eriq_observables(sims[variant]["ddis"])
        cal_inputs = {f"{k}_ctrl": v for k, v in ctrl_obs.items()}
        cal_inputs.update({f"{k}_pert": v for k, v in pert_obs.items()})

        calibrated, _ = calibrate_pathway_mapper(
            delta_sim_inputs=cal_inputs,
            delta_data=delta_DDIS,
            initial=default,
            n_steps=args.steps,
            learning_rate=args.lr,
        )
        d_sim_cal = _delta_vec(
            calibrated, sims[variant]["ctrl"], sims[variant]["ddis"]
        )

        # Same calibrated mapper, applied to the OIS held-out target
        # (mechanistic side identical — we did not retrain on OIS).
        d_sim_held_out = d_sim_cal  # state-derived Δ doesn't depend on data

        # Held-out arm 2: rapamycin rescue (DDIS+rapa vs DDIS).
        # Δ_sim_rescue = path(ddis_rapa) - path(ddis), using the same
        # DDIS-calibrated mapper. Compare to ssGSEA Δ_RAPA. This tests
        # *intervention prediction*, not just senescence transition.
        d_sim_rescue = (
            calibrated.from_eriq_state(sims[variant]["ddis_rapa"]).as_vector(
                PATHWAY_ORDER
            )
            - calibrated.from_eriq_state(sims[variant]["ddis"]).as_vector(
                PATHWAY_ORDER
            )
        )

        print(f"\n  ─── variant: {variant} ──────────────────────────────────")
        print("    Calibrated on DDIS (in-sample):")
        _print_table(d_sim_cal, delta_DDIS, label="DDIS Δ_sim_cal vs Δ_data")
        r_ddis, sa_ddis = _summary_line(
            "DDIS (in-sample)", d_sim_cal, delta_DDIS
        )
        print("    Evaluated on OIS (held-out — different senescence type):")
        _print_table(
            d_sim_held_out,
            delta_OIS,
            label="OIS Δ_sim_cal vs Δ_data",
        )
        r_ois, sa_ois = _summary_line(
            "OIS  (held-out)", d_sim_held_out, delta_OIS
        )
        print("    Evaluated on rapamycin rescue (held-out — intervention):")
        _print_table(
            d_sim_rescue,
            delta_RAPA,
            label="DDIS+rapa Δ_sim_cal vs Δ_data",
        )
        r_rapa, sa_rapa = _summary_line(
            "RAPA (held-out)", d_sim_rescue, delta_RAPA
        )

        hold_out_results[variant] = {
            "r_ddis": r_ddis,
            "r_ois": r_ois,
            "r_rapa": r_rapa,
            "sa_ddis": sa_ddis,
            "sa_ois": sa_ois,
            "sa_rapa": sa_rapa,
            "delta_sim_cal": d_sim_cal,
            "delta_sim_held_out": d_sim_held_out,
            "delta_sim_rescue": d_sim_rescue,
            "calibrated": calibrated,
        }

    # ── Headline ───────────────────────────────────────────────────
    print("\n[4/4] Headline (the numbers for the preprint):")
    print(f"       Δ_data | DDIS:   {delta_DDIS}")
    print(f"       Δ_data | OIS :   {delta_OIS}")
    print(f"       Δ_data | RAPA:   {delta_RAPA}")
    print()
    print(
        f"       {'variant':<10}  {'DDIS':>8} {'OIS':>8} {'RAPA':>8}"
        f"  {'DDIS sign':>10} {'OIS sign':>10} {'RAPA sign':>10}"
    )
    print(
        f"       {'(in-samp)':<10}  {'(held)':>8} {'(held)':>8} {'(held)':>8}"
    )
    print(f"       {'-' * 10}  {'-' * 8} {'-' * 8} {'-' * 8}"
          f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    for v in ("vanilla", "sasp"):
        r = hold_out_results[v]
        print(
            f"       {v:<10}  {r['r_ddis']:+8.3f} {r['r_ois']:+8.3f} "
            f"{r['r_rapa']:+8.3f}  {r['sa_ddis']*100:9.1f}% "
            f"{r['sa_ois']*100:9.1f}% {r['sa_rapa']*100:9.1f}%"
        )

    # ── Optional plot ──────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available; skipping plot")
            return
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        panels = [
            ("vanilla", "DDIS", delta_DDIS, "delta_sim_cal", "r_ddis",
             "vanilla / DDIS (in-sample)"),
            ("vanilla", "OIS", delta_OIS, "delta_sim_held_out", "r_ois",
             "vanilla / OIS (held-out)"),
            ("vanilla", "RAPA", delta_RAPA, "delta_sim_rescue", "r_rapa",
             "vanilla / DDIS+rapa rescue (held-out)"),
            ("sasp", "DDIS", delta_DDIS, "delta_sim_cal", "r_ddis",
             "SASP / DDIS (in-sample)"),
            ("sasp", "OIS", delta_OIS, "delta_sim_held_out", "r_ois",
             "SASP / OIS (held-out)"),
            ("sasp", "RAPA", delta_RAPA, "delta_sim_rescue", "r_rapa",
             "SASP / DDIS+rapa rescue (held-out)"),
        ]
        for ax, (variant, condition, dd, sim_key, r_key, label) in zip(
            axes.flat, panels
        ):
            d_sim = hold_out_results[variant][sim_key]
            ds = [float(x) for x in d_sim]
            dr = [float(x) for x in dd]
            ax.scatter(dr, ds, s=80, edgecolor="k", color="C0")
            for i, name in enumerate(PATHWAY_ORDER):
                ax.annotate(
                    name,
                    (dr[i], ds[i]),
                    fontsize=6,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
            lim = max(max(abs(x) for x in dr + ds), 0.1) * 1.3
            ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            r_val = hold_out_results[variant][r_key]
            ax.set_title(f"{label}\nr = {r_val:+.3f}", fontsize=10)
            ax.set_xlabel("Δ_data (ssGSEA)", fontsize=9)
            ax.set_ylabel("Δ_sim (PathwayMapper)", fontsize=9)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        out = ROOT / "demos" / "plots" / "concordance_ddis.png"
        out.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
