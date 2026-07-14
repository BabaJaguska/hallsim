"""Hybrid flagship: a learned p53 block that captures GZ06's bifurcations.

The §3.3 demonstration. Replace the mechanistic Geva-Zatorsky 2006 p53–Mdm2
oscillator in the multi-hallmark flagship with a NeuralODE block trained to
reproduce it, then compose and differentiate as if it were the original.

The block is conditioned on two inputs — the damage stimulus ψ and the Mdm2
degradation rate α_y — so the *single* learned vector field represents GZ06's
whole two-parameter family. That is what lets it reproduce GZ06's two
bifurcations (α_y-Hopf: oscillation↔fixed point; ψ-onset: damage turning
pulsing on), and it keeps α_y a live, differentiable parameter of the hybrid
instead of freezing one operating point into the weights.

Training is two-stage: derivative matching regresses the vector field, then a
shooting fine-tune integrates the learned field and matches trajectories.
Derivative matching alone undersizes the limit cycle near the Hopf (it fits
local slopes; small consistent errors shrink the emergent oscillation); the
shooting stage penalizes trajectory amplitude directly and corrects it. Both
stages are plotted against the mechanistic model so the correction is visible.

Writes to ``outputs/multi_hallmark_hybrid/``:
- ``bifurcation_recovery.png/.pdf`` — p53 amplitude across α_y and ψ:
  mechanistic vs derivative-only vs shooting-refined.
- ``flagship_ddb2.png/.pdf`` — DDB2 vs genomic-instability severity,
  mechanistic vs hybrid, in the pulsatile and calibrated-sustained regimes.
- ``provenance.json`` / ``provenance.md`` — the full run record: config, both
  stages' recovery numbers, flagship tables, gradients, and motivation.
- ``gz06_neural_block.eqx`` — the trained block.

    .venv_hallsim/bin/python demos/multi_hallmark_hybrid.py
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from hallsim.composite import Composite  # noqa: E402
from hallsim.hallmarks import apply_hallmarks  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.sbml_import import process_from_sbml  # noqa: E402
from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    GZ06_SBML_PATH, CANONICAL_TIME_SECONDS, GZ06_PSI_FULL,
    GZ06_PSI_DRIVE_K, GZ06_PSI_DRIVE_N,
    build_multi_hallmark_composite,
)
from hallsim.models.neuralode import (  # noqa: E402
    NeuralODEProcess, simulate_conditioned, fit_neuralode_derivative,
    fit_neuralode_shooting,
)

log = logging.getLogger("hallsim.demo.hybrid")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "multi_hallmark_hybrid"

FIELDS = ("x", "y0", "y")
IC = (0.0, 0.1, 0.8)          # GZ06 published initial x, y0, y
ALPHA_Y_FIT = 1.597           # calibrated α_y (above the Hopf)
TRAIN = dict(width=192, depth=3, deriv_steps=9000, shooting_steps=250,
             psi_grid=[0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05,
                       1.2, 1.3],
             # spans both α_y-Hopfs (eigenvalue-located at ~0.02 and ~1.02) —
             # from the stable fixed point below onset up past the offset — so
             # the surrogate learns the full two-Hopf bifurcation. Dense at
             # BOTH Hopfs: limit-cycle amplitude ~ sqrt(distance past the
             # bifurcation), so it is hypersensitive to field accuracy right at
             # each Hopf; sparse sampling there clips the peak and lets a smooth
             # MLP bleed a neighbouring cycle across into the fixed-point
             # regime. Data-gen runs through the Scheduler (auto-stiffness), so
             # the stiff low-α_y regime integrates fine.
             ay_grid=[0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.1, 0.15,
                      0.35, 0.55, 0.7, 0.82, 0.88, 0.92, 0.96, 1.0, 1.02, 1.05,
                      1.08, 1.15, 1.3, 1.6, 2.0],
             n_ics=6, y0_hi=1.4, t_data=2.0, n_data=200)
DDB2 = next(r for r in MULTI_HALLMARK_REPORTERS if r.gene_symbol == "DDB2")

_gz = process_from_sbml(str(GZ06_SBML_PATH), name="gz06").reconciled_to(
    CANONICAL_TIME_SECONDS
)


def _gz_with(psi, alpha_y):
    return eqx.tree_at(
        lambda p: (p.parameters["psi"], p.parameters["alpha_y"]),
        _gz, (jnp.asarray(psi), jnp.asarray(alpha_y)),
    )


def gz_rhs(u):
    proc = _gz_with(u[0], u[1])

    def rhs(t, y, args=None):
        d = proc.derivative(t, {"x": y[0], "y0": y[1], "y": y[2]})
        return jnp.stack([d["x"], d["y0"], d["y"]])

    return rhs


def hopf_points(psi=1.0):
    """The α_y Hopf bifurcations, located where the fixed point's Jacobian
    complex-conjugate pair crosses Re=0 — from eigenvalues, not amplitude.

    GZ06 has two: oscillation onsets at the lower one and dies at the upper
    one, so the p53 pulse exists only between them.
    """
    import numpy as np
    from scipy.optimize import fsolve

    grid = np.linspace(0.005, 2.0, 140)

    def fixed_point(ay):
        f = gz_rhs((psi, ay))
        fn = lambda y: np.asarray(f(0.0, jnp.asarray(y)))  # noqa: E731
        for g in ([0.4, 0.4, 0.4], [ay, ay, ay], [0.1, 0.1, 0.8]):
            s, _, ier, _ = fsolve(fn, g, full_output=True)
            if ier == 1 and np.all(np.isfinite(s)) and np.max(np.abs(s)) < 50:
                return jnp.asarray(s)
        return None

    re = []
    for ay in grid:
        fp = fixed_point(ay)
        if fp is None:
            re.append(np.nan)
            continue
        jac = jax.jacfwd(lambda y: gz_rhs((psi, ay))(0.0, y))(fp)
        ev = np.linalg.eigvals(np.asarray(jac))
        cplx = ev[np.abs(ev.imag) > 1e-9]
        re.append(float(cplx[np.argmax(cplx.real)].real)
                  if len(cplx) else np.nan)
    re = np.array(re)
    hopfs = []
    for i in range(1, len(grid)):
        if np.isfinite(re[i - 1]) and np.isfinite(re[i]) \
                and re[i - 1] * re[i] < 0:
            a0, a1, r0, r1 = grid[i - 1], grid[i], re[i - 1], re[i]
            hopfs.append(float(a0 - r0 * (a1 - a0) / (r1 - r0)))
    return hopfs


def train_stages():
    """Return (derivative-only block, shooting-refined block, (ts, ys, us))."""
    inputs = jnp.stack(jnp.meshgrid(
        jnp.array(TRAIN["psi_grid"]), jnp.array(TRAIN["ay_grid"]),
        indexing="ij"), axis=-1).reshape(-1, 2)
    ts = jnp.linspace(0.0, TRAIN["t_data"], TRAIN["n_data"])
    t0 = time.time()
    ys, us = simulate_conditioned(
        gz_rhs, ts, inputs, n_ics=TRAIN["n_ics"],
        y0_range=(0.0, TRAIN["y0_hi"]), key=jax.random.PRNGKey(0))
    log.info("training data %s in %.1fs", tuple(ys.shape), time.time() - t0)

    init = NeuralODEProcess(
        fields=FIELDS, input_fields=("psi", "alpha_y"), field_defaults=IC,
        width=TRAIN["width"], depth=TRAIN["depth"], timescale=3600.0,
        key=jax.random.PRNGKey(1))
    t0 = time.time()
    deriv = fit_neuralode_derivative(
        ts, ys, us, fields=FIELDS, input_fields=("psi", "alpha_y"),
        init=init, steps=TRAIN["deriv_steps"], lr=3e-3, batch_size=512)
    log.info("derivative fit in %.1fs", time.time() - t0)
    t0 = time.time()
    # Physics-regularized multiple shooting: plain single shooting over ~7 p53
    # periods collapses the oscillator to a fixed point (phase drift makes a
    # flat line the MSE optimum); short segments + the collocation term supply
    # the vector-field magnitude constraint that keeps the oscillation alive.
    shoot = fit_neuralode_shooting(
        ts, ys, us, fields=FIELDS, input_fields=("psi", "alpha_y"),
        init=deriv, segments=8, physics_weight=10.0,
        steps=TRAIN["shooting_steps"], lr=3e-4, batch_size=32)
    log.info("shooting fine-tune in %.1fs", time.time() - t0)
    return deriv, shoot, (ts, ys, us)


# ── standalone p53 blocks (both controls parameter-sourced) ──────────────

def _solo(proc):
    return Composite(processes={"gz06": proc}, topology={}, validate=False,
                     semantic_validation={"check_semantics": False})


def neural_solo(block, psi, alpha_y):
    b = block.with_control_param("psi", float(psi)).with_control_param(
        "alpha_y", float(alpha_y))
    return _solo(b)


def _run_traj(comp, t_end=4.0):
    r = Scheduler(auto_stiffness=True).run(
        comp, t_span=(0.0, t_end), y0=comp.initial_state_vec(),
        macro_dt=0.05, save_dt=0.01)
    return r.ts, r.get("gz06/x")


def _run_x(comp, t_end=4.0):
    return _run_traj(comp, t_end)[1]


def _amp(x):
    tail = x[len(x) // 2:]
    return float(jnp.max(tail) - jnp.min(tail))


def time_domain_figure(block):
    """p53(x) over time: mechanistic vs surrogate, across the bifurcation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [(1.0, 0.01, "before lower Hopf (fixed point)"),
             (1.0, 0.1, "onset"), (1.0, 0.4, "oscillation"),
             (1.0, 0.8, "strong oscillation"),
             (1.0, 1.0, "near upper Hopf (peak)"),
             (1.0, 1.2, "past upper Hopf (damped)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), sharex=True)
    for ax, (psi, ay, tag) in zip(axes.flat, cases):
        tm, xm = _run_traj(_solo(_gz_with(psi, ay)))
        tn, xn = _run_traj(neural_solo(block, psi, ay))
        ax.plot(tm, xm, color="#333", lw=1.8, label="mechanistic GZ06")
        ax.plot(tn, xn, color="#d97706", lw=1.6, ls="--", label="NeuralODE")
        ax.set_title(f"ψ={psi}, α_y={ay}  ({tag})", fontsize=10)
        ax.set_ylabel("p53 (x)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1, 0].set_xlabel("time (days)")
    axes[1, 1].set_xlabel("time (days)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("p53 oscillator: mechanistic vs NeuralODE surrogate "
                 "(time domain)", fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"time_domain_compare.{ext}", dpi=160,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote time_domain_compare.png/.pdf")


AY_SWEEP = [0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0]
PSI_SWEEP = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2]


def bifurcation_curves(block=None):
    """p53 amplitude across the two bifurcation axes. block=None → mech."""
    def amp_at(psi, ay):
        comp = _solo(_gz_with(psi, ay)) if block is None \
            else neural_solo(block, psi, ay)
        return _amp(_run_x(comp))
    return {
        "ay": [amp_at(1.0, a) for a in AY_SWEEP],
        "psi": [amp_at(p, 0.8) for p in PSI_SWEEP],
    }


# Held-out generalization grid: every (ψ, α_y) here is ABSENT from the training
# psi_grid × ay_grid, so amplitude error on it is a true generalization number,
# not memorization. The grid spans both Hopf approaches and the oscillatory
# interior. The guard below fails loudly if a value ever leaks onto training.
HELD_OUT_PSI = [0.52, 0.68, 0.9, 1.1]
HELD_OUT_AY = [0.08, 0.5, 0.78, 0.98, 1.25, 1.5]
assert not (set(HELD_OUT_PSI) & set(TRAIN["psi_grid"])), "held-out ψ in training"
assert not (set(HELD_OUT_AY) & set(TRAIN["ay_grid"])), "held-out α_y in training"


def held_out_recovery(block):
    """Amplitude error on (ψ, α_y) points none of which appear in training —
    the generalization number. Returns the aggregate plus per-point rows."""
    rows, errs = [], []
    for psi in HELD_OUT_PSI:
        for ay in HELD_OUT_AY:
            xn = _run_x(neural_solo(block, psi, ay))
            xm = _run_x(_solo(_gz_with(psi, ay)))
            an, am = _amp(xn), _amp(xm)
            errs.append(abs(an - am))
            rows.append(dict(psi=psi, alpha_y=ay, amp_neural=an, amp_mech=am,
                             abs_err=abs(an - am),
                             rms=float(jnp.sqrt(jnp.mean((xn - xm) ** 2)))))
    return dict(mean_abs_amp_err=sum(errs) / len(errs), n_points=len(errs),
                points=rows)


def bifurcation_figure(mech, deriv, shoot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_M, C_D, C_S = "#333", "#93c5fd", "#d97706"
    hopfs = sorted(hopf_points(psi=1.0))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(AY_SWEEP, mech["ay"], "o-", color=C_M, label="mechanistic GZ06")
    a1.plot(AY_SWEEP, deriv["ay"], "^:", color=C_D, label="NeuralODE (deriv)")
    a1.plot(AY_SWEEP, shoot["ay"], "s--", color=C_S,
            label="NeuralODE (+shooting)")
    # Eigenvalue-located Hopf points bound the oscillatory window.
    if len(hopfs) == 2:
        a1.axvspan(hopfs[0], hopfs[1], color="#f1f5f9", zorder=0)
    for h in hopfs:
        a1.axvline(h, color="#94a3b8", lw=1.1, ls="--")
        a1.text(h, a1.get_ylim()[1], f" Hopf\n α_y={h:.2f}", fontsize=7,
                color="#64748b", va="top", ha="left")
    a1.set_xlabel(r"$\alpha_y$ (Mdm2 degradation)")
    a1.set_ylabel("p53 pulse amplitude")
    a1.set_title(r"$\alpha_y$: two Hopfs bound the oscillatory window")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(PSI_SWEEP, mech["psi"], "o-", color=C_M, label="mechanistic GZ06")
    a2.plot(PSI_SWEEP, deriv["psi"], "^:", color=C_D, label="NeuralODE (deriv)")
    a2.plot(PSI_SWEEP, shoot["psi"], "s--", color=C_S,
            label="NeuralODE (+shooting)")
    a2.set_xlabel(r"ψ (damage input)")
    a2.set_ylabel("p53 pulse amplitude")
    a2.set_title(r"ψ-onset (α_y=0.8): damage turns pulsing on")
    a2.legend(frameon=False, fontsize=8)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("A learned p53 block reproduces GZ06's bifurcations",
                 fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"bifurcation_recovery.{ext}", dpi=160,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote bifurcation_recovery.png/.pdf")


# ── flagship swap: DDB2 across severities + gradient through the block ────

TOPOLOGY = build_multi_hallmark_composite(validate=False).topology

# dp14 knobs from the calibration fit (outputs/multi_hallmark_calibrate);
# with default dp14 the etoposide dose saturates ψ→1 and DDB2 goes flat, so
# the swap must be tested in the fitted regime where DNA_damage stays in the
# damage→ψ Hill's responsive band.
_FIT_KEYS = {
    "etoposide_potency": "DNA_damaged_by_irradiation",
    "ROS_turnover": "ROS_turnover",
    "CDKN1A_transcr": "CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
    "mitophagy_inactiv": "mitophagy_inactiv_by_mTORC1_pS2448",
}
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load_fitted_dp14():
    p = ROOT / "outputs" / "multi_hallmark_calibrate" / "summary.json"
    fitted = json.loads(p.read_text())["fitted_params"]
    return {_FIT_KEYS[k]: v for k, v in fitted.items() if k in _FIT_KEYS}


def _apply_dp14(dp14, fitted):
    for name, val in fitted.items():
        dp14 = eqx.tree_at(lambda p, n=name: p.parameters[n], dp14,
                           jnp.asarray(val))
    return dp14


def _ddb2_for_severity(comp_processes, severity):
    procs = apply_hallmarks(comp_processes, {
        "Genomic Instability": severity,
        "Deregulated Nutrient Sensing": 0.5,
    })
    comp = Composite(procs, TOPOLOGY, validate=False,
                     semantic_validation={"check_semantics": False})
    r = Scheduler(auto_stiffness=True).run(
        comp, t_span=(0.0, 14.0), y0=comp.initial_state_vec(),
        macro_dt=3.5, save_dt=1.0)
    return DDB2.summary(r.ts, r.get("gz06/x2_integral"), jnp.array([14.0]))[0]


def _procs_at(alpha_y, block, fitted):
    """Mechanistic and hybrid flagship process dicts at one α_y setting."""
    mech = build_multi_hallmark_composite(validate=False).processes
    mech = {**mech, "dp14": _apply_dp14(mech["dp14"], fitted),
            "gz06": eqx.tree_at(lambda p: p.parameters["alpha_y"],
                                mech["gz06"], float(alpha_y))}
    neural_gz = (
        block.with_input_driver(
            "psi", port="psi_source", basal_param="psi_basal",
            hi=GZ06_PSI_FULL, K=GZ06_PSI_DRIVE_K, n=GZ06_PSI_DRIVE_N,
            basal=0.3)
        .with_control_param("alpha_y", float(alpha_y))
    )
    return mech, {**mech, "gz06": neural_gz}


def flagship_results(block, fitted):
    out = {}
    for alpha_y in (0.8, ALPHA_Y_FIT):
        mech_p, hyb_p = _procs_at(alpha_y, block, fitted)
        mech = [float(_ddb2_for_severity(mech_p, s)) for s in SEVERITIES]
        hyb = [float(_ddb2_for_severity(hyb_p, s)) for s in SEVERITIES]
        g = float(jax.grad(lambda s: _ddb2_for_severity(hyb_p, s))(1.0))
        fd = (float(_ddb2_for_severity(hyb_p, 1.0 + 1e-3))
              - float(_ddb2_for_severity(hyb_p, 1.0 - 1e-3))) / 2e-3
        out[f"{alpha_y:g}"] = dict(
            alpha_y=alpha_y,
            regime="pulsatile" if alpha_y < 1.1 else "sustained (calibrated)",
            severities=SEVERITIES, mech=mech, hybrid=hyb,
            grad_autodiff=g, grad_finite_diff=fd)
    return out


def flagship_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_M, C_H = "#333", "#d97706"
    keys = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 4.3),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        r = results[k]
        ax.plot(r["severities"], r["mech"], "o-", color=C_M,
                label="mechanistic")
        ax.plot(r["severities"], r["hybrid"], "s--", color=C_H,
                label="hybrid (NeuralODE)")
        ax.set_xlabel("Genomic Instability severity")
        ax.set_ylabel(r"DDB2  $\sqrt{\langle x^2\rangle}$")
        ax.set_title(f"α_y={r['alpha_y']:g} · {r['regime']}\n"
                     f"∂DDB2/∂sev (hybrid) = {r['grad_autodiff']:+.4f}")
        ax.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Flagship swap: hybrid reproduces DDB2 and stays "
                 "differentiable", fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"flagship_ddb2.{ext}", dpi=160,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote flagship_ddb2.png/.pdf")


# ── provenance ───────────────────────────────────────────────────────────

def _mean_abs_amp_err(neural, mech):
    d = [abs(a - b) for a, b in zip(neural["ay"], mech["ay"])]
    p = [abs(a - b) for a, b in zip(neural["psi"], mech["psi"])]
    return sum(d + p) / len(d + p)


def write_provenance(prov):
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    r = prov["recovery"]
    md = [f"# Hybrid flagship — run provenance\n",
          f"_{prov['timestamp']}_\n",
          "## What this run did\n",
          "Recovered the Geva-Zatorsky 2006 p53–Mdm2 oscillator as a "
          "(ψ, α_y)-conditioned NeuralODE, swapped it into the multi-hallmark "
          "flagship, and checked bifurcation capture, DDB2 reproduction, and "
          "end-to-end gradient flow.\n",
          "## Recovery (held-out amplitude error)\n",
          "Both blocks are scored only on (ψ, α_y) points held out of "
          "training: the bifurcation sweeps run at ψ=1.0 and α_y=0.8 (neither "
          "in the training grid), plus a "
          f"{r['held_out_deriv']['n_points']}-point off-grid generalization "
          "set. Mean |amplitude error| vs mechanistic:\n",
          "| block | sweep (held-out) | grid (held-out) |", "|---|---|---|",
          f"| derivative | {r['deriv_amp_err']:.3f} | "
          f"{r['held_out_deriv']['mean_abs_amp_err']:.3f} |",
          f"| +shooting | {r['shoot_amp_err']:.3f} | "
          f"{r['held_out_shoot']['mean_abs_amp_err']:.3f} |",
          f"\nKept the **{prov['kept_block']}** block (lower held-out "
          "amplitude error).\n",
          "## Training config\n",
          "```\n" + json.dumps(prov["config"], indent=2) + "\n```\n",
          "## Bifurcation recovery (p53 amplitude)\n",
          "α_y-Hopf (ψ=1.0):\n",
          "| α_y | mech | deriv | +shooting |", "|---|---|---|---|"]
    for i, a in enumerate(AY_SWEEP):
        md.append(f"| {a} | {prov['bifurcation']['mech']['ay'][i]:.3f} | "
                  f"{prov['bifurcation']['deriv']['ay'][i]:.3f} | "
                  f"{prov['bifurcation']['shoot']['ay'][i]:.3f} |")
    md += ["\nψ-onset (α_y=0.8):\n",
           "| ψ | mech | deriv | +shooting |", "|---|---|---|---|"]
    for i, p in enumerate(PSI_SWEEP):
        md.append(f"| {p} | {prov['bifurcation']['mech']['psi'][i]:.3f} | "
                  f"{prov['bifurcation']['deriv']['psi'][i]:.3f} | "
                  f"{prov['bifurcation']['shoot']['psi'][i]:.3f} |")
    md.append("\n## Flagship DDB2 vs severity (shooting block)\n")
    for k, fr in prov["flagship"].items():
        md.append(f"\n**α_y={fr['alpha_y']:g} — {fr['regime']}**  "
                  f"(∂DDB2/∂severity: autodiff={fr['grad_autodiff']:+.5f}, "
                  f"finite-diff={fr['grad_finite_diff']:+.5f})\n")
        md.append("| severity | mechanistic | hybrid | |Δ| |")
        md.append("|---|---|---|---|")
        for s, m, h in zip(fr["severities"], fr["mech"], fr["hybrid"]):
            md.append(f"| {s} | {m:.4f} | {h:.4f} | {abs(m - h):.4f} |")
    (OUT / "provenance.md").write_text("\n".join(md) + "\n")
    log.info("wrote provenance.json/.md")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("hallsim").setLevel(logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    deriv, shoot, _ = train_stages()

    mech_c = bifurcation_curves(None)
    deriv_c = bifurcation_curves(deriv)
    shoot_c = bifurcation_curves(shoot)
    bifurcation_figure(mech_c, deriv_c, shoot_c)

    # Keep whichever stage recovers the amplitude better. The shooting
    # fine-tune is expected to help but can destabilise the derivative-matched
    # field on this stiff oscillator, so the choice is measured, not assumed.
    deriv_err = _mean_abs_amp_err(deriv_c, mech_c)
    shoot_err = _mean_abs_amp_err(shoot_c, mech_c)
    best, best_c, best_name = (
        (deriv, deriv_c, "derivative-only") if deriv_err <= shoot_err
        else (shoot, shoot_c, "shooting-refined"))
    log.info("kept %s block (amp err deriv=%.3f shoot=%.3f)",
             best_name, deriv_err, shoot_err)
    time_domain_figure(best)

    fitted = _load_fitted_dp14()
    flag = flagship_results(best, fitted)
    flagship_figure(flag)

    prov = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": TRAIN,
        "kept_block": best_name,
        "bifurcation": {"mech": mech_c, "deriv": deriv_c, "shoot": shoot_c},
        "recovery": {
            "deriv_amp_err": deriv_err,
            "shoot_amp_err": shoot_err,
            "held_out_deriv": held_out_recovery(deriv),
            "held_out_shoot": held_out_recovery(shoot),
        },
        "flagship": flag,
    }
    write_provenance(prov)

    eqx.tree_serialise_leaves(str(OUT / "gz06_neural_block.eqx"), best)
    print(f"\nkept {best_name}; amplitude error deriv={deriv_err:.3f} "
          f"shoot={shoot_err:.3f}")
    for k, fr in flag.items():
        print(f"α_y={fr['alpha_y']:g} [{fr['regime']}]: "
              f"∂DDB2/∂sev={fr['grad_autodiff']:+.5f}")
    print(f"outputs → {OUT.relative_to(ROOT)}/ "
          "(bifurcation_recovery, flagship_ddb2, provenance.md)")


if __name__ == "__main__":
    main()
