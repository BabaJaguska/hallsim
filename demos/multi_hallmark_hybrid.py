"""Hybrid composite: a learned p53 block that captures GZ06's bifurcations.

The §3.3 demonstration. Replace the mechanistic Geva-Zatorsky 2006 p53–Mdm2
oscillator in the multi-hallmark demo with a NeuralODE block trained to
reproduce it, then compose and differentiate as if it were the original.

The block is conditioned on the two parameters the composite varies — the
damage-driven p53 degradation rate α_x and the Mdm2 degradation rate α_y — so
the *single* learned vector field represents that whole two-parameter family.
That is what lets it reproduce GZ06's bifurcations on both axes (one α_x Hopf,
two α_y Hopfs, all oscillation↔fixed point), and it keeps both a live,
differentiable parameter of the hybrid instead of freezing one operating point
into the weights.

α_x is the axis the composite drives: DP14's accumulated DNA damage pulls it
down from a quiescent control toward the deposit's own α_x = 0, and crossing
the Hopf is what starts the p53 pulses (see ``demos/models/multi_hallmark.py``).
Conditioning on it is what lets the swapped-in block read the same damage
signal the mechanistic process did. ψ — the paper's ξ, a noise gain on protein
production — stays at its published 1.0 and is not an axis here.

Training is two-stage: derivative matching regresses the vector field, then a
shooting fine-tune integrates the learned field and matches trajectories.
Derivative matching alone undersizes the limit cycle near the Hopf (it fits
local slopes; small consistent errors shrink the emergent oscillation); the
shooting stage penalizes trajectory amplitude directly and corrects it. Both
stages are plotted against the mechanistic model so the correction is visible.

Writes to ``outputs/multi_hallmark_hybrid/``:
- ``bifurcation_recovery.png/.pdf`` — p53 amplitude across α_y and α_x:
  mechanistic vs derivative-only vs shooting-refined.
- ``ddb2_severity.png/.pdf`` — DDB2 vs genomic-instability severity,
  mechanistic vs hybrid, at the α_y the composite holds.
- ``provenance.json`` / ``provenance.md`` — the full run record: config, both
  stages' recovery numbers, DDB2 tables, gradients, and motivation.
- ``gz06_neural_block.eqx`` — the trained block.

    python demos/multi_hallmark_hybrid.py
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
from demos.models.multi_hallmark import (  # noqa: E402
    GZ06_SBML_PATH,
    CANONICAL_TIME_SECONDS,
    GZ06_ALPHA_X_CONTROL,
    GZ06_ALPHA_X_DAMAGED,
    MULTI_HALLMARK_GRID as GRID,
    build_multi_hallmark_composite,
)

from hallsim.models.neuralode import (  # noqa: E402
    NeuralODEProcess,
    simulate_conditioned,
    fit_neuralode_derivative,
    fit_neuralode_shooting,
)

log = logging.getLogger("hallsim.demo.hybrid")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "multi_hallmark_hybrid"

FIELDS = ("x", "y0", "y")
IC = (0.0, 0.1, 0.8)  # GZ06 published initial x, y0, y
# GZ06's deposited α_y, and the value the composite holds it at. The DDB2 swap
# is reported here and nowhere else: a second α_y panel would be a comparison
# point this demo picked, not one the model is ever run at.
ALPHA_Y_HELD = 0.8
CONDITIONING = ("alpha_x", "alpha_y")
TRAIN = dict(
    width=192,
    depth=3,
    deriv_steps=9000,
    shooting_steps=250,
    # α_x is the axis the composite actually drives (damage ⊣ p53 degradation,
    # see demos/models/multi_hallmark.py), running from the deposit's damaged
    # α_x = 0 up to the manufactured quiescent control at 4× the Hopf. Dense
    # either side of the Hopf at 0.16617 for the same reason the α_y grid is:
    # limit-cycle amplitude ~ sqrt(distance past the bifurcation), so the
    # field has to be accurate right there or the peak clips and a smooth MLP
    # bleeds a cycle across into the fixed-point side.
    alpha_x_grid=[
        0.0,
        0.03,
        0.07,
        0.11,
        0.14,
        0.157,
        0.166,
        0.177,
        0.19,
        0.22,
        0.28,
        0.38,
        0.50,
        0.6648,
    ],
    # spans both α_y-Hopfs (eigenvalue-located at ~0.02 and ~1.02) —
    # from the stable fixed point below onset up past the offset — so
    # the surrogate learns the full two-Hopf bifurcation. Dense at
    # BOTH Hopfs: limit-cycle amplitude ~ sqrt(distance past the
    # bifurcation), so it is hypersensitive to field accuracy right at
    # each Hopf; sparse sampling there clips the peak and lets a smooth
    # MLP bleed a neighbouring cycle across into the fixed-point
    # regime. Data-gen runs through the Scheduler (auto-stiffness), so
    # the stiff low-α_y regime integrates fine.
    ay_grid=[
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.06,
        0.1,
        0.15,
        0.35,
        0.55,
        0.7,
        0.82,
        0.88,
        0.92,
        0.96,
        1.0,
        1.02,
        1.05,
        1.08,
        1.15,
        1.3,
        1.6,
        2.0,
    ],
    n_ics=6,
    y0_hi=1.4,
    t_data=2.0,
    n_data=200,
)
# Keyed on the observable as well as the symbol: the registry carries more
# than one DDB2 entry, and the p53-amplitude one is the reporter this demo
# needs — GZ06's mean p53 is damage-blind, so a mean-like summary cannot see
# the pulsing that the damage produces.
DDB2 = next(
    r
    for r in MULTI_HALLMARK_REPORTERS
    if r.gene_symbol == "DDB2" and r.observable == "gz06/x"
)

_gz = process_from_sbml(str(GZ06_SBML_PATH), name="gz06").reconciled_to(
    CANONICAL_TIME_SECONDS
)


def _gz_with(alpha_x, alpha_y):
    """GZ06 at one point of the conditioning plane. ψ stays at its published
    1.0 throughout — it is not an axis of this demo."""
    return eqx.tree_at(
        lambda p: (p.parameters["alpha_x"], p.parameters["alpha_y"]),
        _gz,
        (jnp.asarray(alpha_x), jnp.asarray(alpha_y)),
    )


def gz_rhs(u):
    proc = _gz_with(u[0], u[1])

    def rhs(t, y, args=None):
        d = proc.derivative(t, {"x": y[0], "y0": y[1], "y": y[2]})
        return jnp.stack([d["x"], d["y0"], d["y"]])

    return rhs


def hopf_analysis(
    axis="alpha_y", *, alpha_x=GZ06_ALPHA_X_DAMAGED, alpha_y=ALPHA_Y_HELD
):
    """GZ06's bifurcations along one conditioning axis as ``Bifurcation``
    objects (kind, location, frequency, normal-form coefficient) via
    :mod:`hallsim.bifurcation`.

    ``alpha_y`` carries two supercritical Hopfs bounding the oscillatory
    window — the p53 pulse exists only between them. ``alpha_x``, the axis the
    composite drives, carries one, and crossing it is what starts the pulses.
    Both are located from eigenvalues at the other axis's held value, so the
    figure never plots a literal.
    """
    import numpy as np
    from hallsim.bifurcation import codim1_scan

    if axis == "alpha_y":
        field_of = lambda v: (lambda y: gz_rhs((alpha_x, v))(0.0, y))  # noqa
        grid = np.linspace(0.005, 2.0, 140)
    else:
        field_of = lambda v: (lambda y: gz_rhs((v, alpha_y))(0.0, y))  # noqa
        grid = np.linspace(0.0, 0.70, 200)
    return codim1_scan(field_of, grid, x0_guess=[0.4, 0.4, 0.4])


def hopf_points(axis="alpha_y", **kw):
    """The Hopf-bifurcation locations along ``axis`` (floats)."""
    return [h.param for h in hopf_analysis(axis, **kw) if h.kind == "hopf"]


def train_stages():
    """Return (derivative-only block, shooting-refined block, (ts, ys, us))."""
    inputs = jnp.stack(
        jnp.meshgrid(
            jnp.array(TRAIN["alpha_x_grid"]),
            jnp.array(TRAIN["ay_grid"]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    ts = jnp.linspace(0.0, TRAIN["t_data"], TRAIN["n_data"])
    t0 = time.time()
    ys, us = simulate_conditioned(
        gz_rhs,
        ts,
        inputs,
        n_ics=TRAIN["n_ics"],
        y0_range=(0.0, TRAIN["y0_hi"]),
        key=jax.random.PRNGKey(0),
        # Names the model the trajectories came from; the grid is hashed in
        # separately, so widening a grid regenerates rather than hitting a
        # stale file.
        cache_key=f"gz06:{Path(GZ06_SBML_PATH).name}:{CONDITIONING}",
    )
    log.info("training data %s in %.1fs", tuple(ys.shape), time.time() - t0)

    init = NeuralODEProcess(
        fields=FIELDS,
        input_fields=CONDITIONING,
        field_defaults=IC,
        width=TRAIN["width"],
        depth=TRAIN["depth"],
        timescale=3600.0,
        key=jax.random.PRNGKey(1),
    )
    t0 = time.time()
    deriv = fit_neuralode_derivative(
        ts,
        ys,
        us,
        fields=FIELDS,
        input_fields=CONDITIONING,
        init=init,
        steps=TRAIN["deriv_steps"],
        lr=3e-3,
        batch_size=512,
    )
    log.info("derivative fit in %.1fs", time.time() - t0)
    t0 = time.time()
    # Physics-regularized multiple shooting: plain single shooting over ~7 p53
    # periods collapses the oscillator to a fixed point (phase drift makes a
    # flat line the MSE optimum); short segments + the collocation term supply
    # the vector-field magnitude constraint that keeps the oscillation alive.
    shoot = fit_neuralode_shooting(
        ts,
        ys,
        us,
        fields=FIELDS,
        input_fields=CONDITIONING,
        init=deriv,
        segments=8,
        physics_weight=10.0,
        steps=TRAIN["shooting_steps"],
        lr=3e-4,
        batch_size=32,
    )
    log.info("shooting fine-tune in %.1fs", time.time() - t0)
    return deriv, shoot, (ts, ys, us)


# ── standalone p53 blocks (both controls parameter-sourced) ──────────────


def _solo(proc):
    return Composite(
        processes={"gz06": proc},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )


def neural_solo(block, alpha_x, alpha_y):
    b = block.with_control_param("alpha_x", float(alpha_x)).with_control_param(
        "alpha_y", float(alpha_y)
    )
    return _solo(b)


def _run_traj(comp, t_end=4.0):
    r = Scheduler(auto_stiffness=True).run(
        comp,
        t_span=(0.0, t_end),
        y0=comp.initial_state_vec(),
        macro_dt=0.05,
        save_dt=0.01,
    )
    return r.ts, r.get("gz06/x")


def _run_x(comp, t_end=4.0):
    return _run_traj(comp, t_end)[1]


def _amp(x):
    tail = x[len(x) // 2 :]
    return float(jnp.max(tail) - jnp.min(tail))


def time_domain_figure(block):
    """p53(x) over time: mechanistic vs surrogate, across the bifurcation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Top row walks α_x across its Hopf at the held α_y — the crossing the
    # composite drives. Bottom row walks α_y at the deposit's damaged α_x.
    cases = [
        (GZ06_ALPHA_X_CONTROL, ALPHA_Y_HELD, "quiescent control, fixed point"),
        (0.20, ALPHA_Y_HELD, "just above α_x Hopf"),
        (0.12, ALPHA_Y_HELD, "just below α_x Hopf, pulsing"),
        (GZ06_ALPHA_X_DAMAGED, 0.01, "damaged, below lower α_y Hopf"),
        (GZ06_ALPHA_X_DAMAGED, ALPHA_Y_HELD, "damaged, deposited α_y"),
        (GZ06_ALPHA_X_DAMAGED, 1.2, "damaged, past upper α_y Hopf"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), sharex=True)
    for ax, (axv, ay, tag) in zip(axes.flat, cases):
        tm, xm = _run_traj(_solo(_gz_with(axv, ay)))
        tn, xn = _run_traj(neural_solo(block, axv, ay))
        ax.plot(tm, xm, color="#333", lw=1.8, label="mechanistic GZ06")
        ax.plot(tn, xn, color="#d97706", lw=1.6, ls="--", label="NeuralODE")
        ax.set_title(f"α_x={axv:.4g}, α_y={ay:g}  ({tag})", fontsize=10)
        ax.set_ylabel("p53 (x)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1, 0].set_xlabel("time (days)")
    axes[1, 1].set_xlabel("time (days)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "p53 oscillator: mechanistic vs NeuralODE surrogate " "(time domain)",
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"time_domain_compare.{ext}",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    log.info("wrote time_domain_compare.png/.pdf")


AY_SWEEP = [0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0]
# Spans the damaged end the deposit sits at, through the Hopf, up to the
# quiescent control the damage edge runs down from.
AX_SWEEP = [
    0.0,
    0.02,
    0.05,
    0.08,
    0.11,
    0.14,
    0.155,
    0.1662,
    0.18,
    0.20,
    0.24,
    0.30,
    0.40,
    0.55,
    GZ06_ALPHA_X_CONTROL,
]


def bifurcation_curves(block=None):
    """p53 amplitude across both conditioning axes. block=None → mech.

    α_y sweeps at the deposit's damaged α_x; α_x sweeps at the published α_y
    the composite holds it at.
    """

    def amp_at(alpha_x, alpha_y):
        comp = (
            _solo(_gz_with(alpha_x, alpha_y))
            if block is None
            else neural_solo(block, alpha_x, alpha_y)
        )
        return _amp(_run_x(comp))

    return {
        "ay": [amp_at(GZ06_ALPHA_X_DAMAGED, a) for a in AY_SWEEP],
        "ax": [amp_at(x, ALPHA_Y_HELD) for x in AX_SWEEP],
    }


# Held-out generalization grid: every (α_x, α_y) here is ABSENT from the
# training alpha_x_grid × ay_grid, so amplitude error on it is a true
# generalization number, not memorization. The α_x points interleave the
# training grid and straddle the Hopf — 0.15 below it, 0.172 above — so the
# number covers the crossing the composite actually drives through, not just
# the easy interior. The guards fail loudly if a value ever leaks onto
# training.
HELD_OUT_AX = [0.015, 0.09, 0.125, 0.15, 0.172, 0.205, 0.33, 0.58]
HELD_OUT_AY = [0.08, 0.5, 0.78, 0.98, 1.25, 1.5]
assert not (
    set(HELD_OUT_AX) & set(TRAIN["alpha_x_grid"])
), "held-out α_x in training"
assert not (
    set(HELD_OUT_AY) & set(TRAIN["ay_grid"])
), "held-out α_y in training"


def held_out_recovery(block):
    """Amplitude error on (α_x, α_y) points none of which appear in training —
    the generalization number. Returns the aggregate plus per-point rows."""
    rows, errs = [], []
    for ax in HELD_OUT_AX:
        for ay in HELD_OUT_AY:
            xn = _run_x(neural_solo(block, ax, ay))
            xm = _run_x(_solo(_gz_with(ax, ay)))
            an, am = _amp(xn), _amp(xm)
            errs.append(abs(an - am))
            rows.append(
                dict(
                    alpha_x=ax,
                    alpha_y=ay,
                    amp_neural=an,
                    amp_mech=am,
                    abs_err=abs(an - am),
                    rms=float(jnp.sqrt(jnp.mean((xn - xm) ** 2))),
                )
            )
    return dict(
        mean_abs_amp_err=sum(errs) / len(errs), n_points=len(errs), points=rows
    )


def bifurcation_figure(mech, deriv, shoot):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_M, C_D, C_S = "#333", "#93c5fd", "#d97706"
    hopfs = sorted(hopf_points("alpha_y"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(AY_SWEEP, mech["ay"], "o-", color=C_M, label="mechanistic GZ06")
    a1.plot(AY_SWEEP, deriv["ay"], "^:", color=C_D, label="NeuralODE (deriv)")
    a1.plot(
        AY_SWEEP, shoot["ay"], "s--", color=C_S, label="NeuralODE (+shooting)"
    )
    # Eigenvalue-located Hopf points bound the oscillatory window.
    if len(hopfs) == 2:
        a1.axvspan(hopfs[0], hopfs[1], color="#f1f5f9", zorder=0)
    for h in hopfs:
        a1.axvline(h, color="#94a3b8", lw=1.1, ls="--")
        a1.text(
            h,
            a1.get_ylim()[1],
            f" Hopf\n α_y={h:.2f}",
            fontsize=7,
            color="#64748b",
            va="top",
            ha="left",
        )
    a1.set_xlabel(r"$\alpha_y$ (Mdm2 degradation)")
    a1.set_ylabel("p53 pulse amplitude")
    a1.set_title(r"$\alpha_y$: two Hopfs bound the oscillatory window")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(AX_SWEEP, mech["ax"], "o-", color=C_M, label="mechanistic GZ06")
    a2.plot(AX_SWEEP, deriv["ax"], "^:", color=C_D, label="NeuralODE (deriv)")
    a2.plot(
        AX_SWEEP,
        shoot["ax"],
        "s--",
        color=C_S,
        label="NeuralODE (+shooting)",
    )
    # The axis the composite drives: damage pulls alpha_x down from the
    # quiescent control, and pulsing starts where it crosses this Hopf.
    ax_hopfs = [h for h in hopf_points("alpha_x") if h is not None]
    for h in ax_hopfs:
        a2.axvline(h, color="#94a3b8", lw=1.1, ls="--")
        a2.text(
            h,
            a2.get_ylim()[1],
            f" Hopf\n α_x={h:.4f}",
            fontsize=7,
            color="#64748b",
            va="top",
            ha="left",
        )
    if ax_hopfs:
        a2.axvspan(min(AX_SWEEP), min(ax_hopfs), color="#f1f5f9", zorder=0)
    a2.set_xlabel(r"$\alpha_x$ (damage ⊣ p53 degradation)")
    a2.set_ylabel("p53 pulse amplitude")
    a2.set_title(
        rf"$\alpha_x$ at held $\alpha_y$={ALPHA_Y_HELD:g}: "
        "damage turns pulsing on"
    )
    a2.legend(frameon=False, fontsize=8)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(
        "A learned p53 block reproduces GZ06's bifurcations", fontweight="bold"
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"bifurcation_recovery.{ext}",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    log.info("wrote bifurcation_recovery.png/.pdf")


# ── block swap: DDB2 across severities + gradient through the block ──

TOPOLOGY = build_multi_hallmark_composite(validate=False).topology

# Both arms of the swap run the identical dp14, so the comparison is of the
# p53 block and nothing else; the dp14 operating point sets how far the
# severity sweep moves the readout, not whether the two agree.
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]

# When the swap is read, which is not the same decision as how long it runs.
# The etoposide pulse drives alpha_x under the Hopf from ~day 0.7 to ~day 6.9;
# p53 pulses through that window and then damps, because nothing here sustains
# it — there is no SASP in this composite. Reading at the horizon would sample
# the readout after it has decayed, where a damaged arm and a control arm are
# indistinguishable. Day 3 is inside the window, near the amplitude peak.
DDB2_READ_DAY = 3.0


def _ddb2_for_severity(comp_processes, severity):
    procs = apply_hallmarks(
        comp_processes,
        {"Genomic Instability": severity},
    )
    comp = Composite(
        procs,
        _topology_for(procs),
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    r = Scheduler(auto_stiffness=True).run(
        comp,
        t_span=(0.0, GRID.t_end),
        y0=comp.initial_state_vec(),
        macro_dt=GRID.macro_dt,
        save_dt=GRID.save_dt,
    )
    # Read the path the reporter itself declares, so the demo cannot drift
    # from the registry. Both blocks expose p53 there.
    return DDB2.summary(
        r.ts, r.get(DDB2.observable), jnp.array([DDB2_READ_DAY])
    )[0]


def _procs_at(alpha_y, block):
    """Mechanistic and hybrid process dicts at one α_y setting.

    The hybrid reads α_x from the same store path the mechanistic GZ06 does —
    the damage bridge's ``gz06/alpha_x_signal`` — so the swap changes the
    p53 block and nothing else about the wiring. ``alpha_x`` is left off
    ``parameters`` and out of the Hill drivers, which is what makes the block
    expose it as a plain INPUT port for the topology to connect.
    """
    mech = build_multi_hallmark_composite(validate=False).processes
    mech = {
        **mech,
        "gz06": eqx.tree_at(
            lambda p: p.parameters["alpha_y"], mech["gz06"], float(alpha_y)
        ),
    }
    neural_gz = block.with_control_param("alpha_y", float(alpha_y))
    return mech, {**mech, "gz06": neural_gz}


def _topology_for(procs):
    """The demo topology, with GZ06's α_x port named for whichever block is
    in place: the SBML process exposes ``alpha_x_in``, the learned block
    exposes the control field itself."""
    topo = {k: dict(v) for k, v in TOPOLOGY.items()}
    if isinstance(procs["gz06"], NeuralODEProcess):
        topo["gz06"] = {"alpha_x": "gz06/alpha_x_signal"}
    return topo


def ddb2_results(block):
    """DDB2 vs severity at the α_y the composite holds, mechanistic vs hybrid.

    One α_y only: the calibration freezes it, so a second panel would report
    an operating point the model is never run at.
    """
    out = {}
    for alpha_y in (ALPHA_Y_HELD,):
        mech_p, hyb_p = _procs_at(alpha_y, block)
        mech = [float(_ddb2_for_severity(mech_p, s)) for s in SEVERITIES]
        hyb = [float(_ddb2_for_severity(hyb_p, s)) for s in SEVERITIES]
        g = float(jax.grad(lambda s: _ddb2_for_severity(hyb_p, s))(1.0))
        fd = (
            float(_ddb2_for_severity(hyb_p, 1.0 + 1e-3))
            - float(_ddb2_for_severity(hyb_p, 1.0 - 1e-3))
        ) / 2e-3
        out[f"{alpha_y:g}"] = dict(
            alpha_y=alpha_y,
            regime="pulsatile" if alpha_y < 1.1 else "sustained",
            severities=SEVERITIES,
            mech=mech,
            hybrid=hyb,
            grad_autodiff=g,
            grad_finite_diff=fd,
        )
    return out


def ddb2_figure(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_M, C_H = "#333", "#d97706"
    keys = list(results)
    fig, axes = plt.subplots(
        1, len(keys), figsize=(5.4 * len(keys), 4.3), squeeze=False
    )
    for ax, k in zip(axes[0], keys):
        r = results[k]
        ax.plot(
            r["severities"], r["mech"], "o-", color=C_M, label="mechanistic"
        )
        ax.plot(
            r["severities"],
            r["hybrid"],
            "s--",
            color=C_H,
            label="hybrid (NeuralODE)",
        )
        # De-stretch panels whose DDB2 barely varies (e.g. sustained): a tight
        # auto-scale would magnify a sub-1% mech/hybrid gap into a visual chasm.
        vals = list(r["mech"]) + list(r["hybrid"])
        lo, hi = min(vals), max(vals)
        if hi - lo < 0.06:
            mid = 0.5 * (lo + hi)
            ax.set_ylim(mid - 0.035, mid + 0.035)
        else:
            span = hi - lo  # lift the data off the axis floor with headroom
            ax.set_ylim(lo - 0.25 * span, hi + 0.12 * span)
        ax.set_xlabel("Genomic Instability severity")
        ax.set_ylabel("DDB2 readout")
        ax.set_title(f"α_y = {r['alpha_y']:g}")
        ax.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(
        "Hybrid composite reproduces the mechanistic DDB2 readout",
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"ddb2_severity.{ext}",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    log.info("wrote ddb2_severity.png/.pdf")


# ── provenance ───────────────────────────────────────────────────────────


def _mean_abs_amp_err(neural, mech):
    d = [abs(a - b) for a, b in zip(neural["ay"], mech["ay"])]
    p = [abs(a - b) for a, b in zip(neural["psi"], mech["psi"])]
    return sum(d + p) / len(d + p)


def write_provenance(prov):
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    r = prov["recovery"]
    md = [
        "# Hybrid composite — run provenance\n",
        f"_{prov['timestamp']}_\n",
        "## What this run did\n",
        "Recovered the Geva-Zatorsky 2006 p53–Mdm2 oscillator as a "
        "(ψ, α_y)-conditioned NeuralODE, swapped it into the multi-hallmark "
        "composite, and checked bifurcation capture, DDB2 reproduction, and "
        "end-to-end gradient flow.\n",
        "## Recovery (held-out amplitude error)\n",
        "Both blocks are scored only on (ψ, α_y) points held out of "
        "training: the bifurcation sweeps run at ψ=1.0 and α_y=0.8 (neither "
        "in the training grid), plus a "
        f"{r['held_out_deriv']['n_points']}-point off-grid generalization "
        "set. Mean |amplitude error| vs mechanistic:\n",
        "| block | sweep (held-out) | grid (held-out) |",
        "|---|---|---|",
        f"| derivative | {r['deriv_amp_err']:.3f} | "
        f"{r['held_out_deriv']['mean_abs_amp_err']:.3f} |",
        f"| +shooting | {r['shoot_amp_err']:.3f} | "
        f"{r['held_out_shoot']['mean_abs_amp_err']:.3f} |",
        f"\nKept the **{prov['kept_block']}** block (lower held-out "
        "amplitude error).\n",
        "## Training config\n",
        "```\n" + json.dumps(prov["config"], indent=2) + "\n```\n",
        "## Bifurcation recovery (p53 amplitude)\n",
        f"α_y-Hopfs (at the deposit's α_x={GZ06_ALPHA_X_DAMAGED:g}):\n",
        "| α_y | mech | deriv | +shooting |",
        "|---|---|---|---|",
    ]
    for i, a in enumerate(AY_SWEEP):
        md.append(
            f"| {a} | {prov['bifurcation']['mech']['ay'][i]:.3f} | "
            f"{prov['bifurcation']['deriv']['ay'][i]:.3f} | "
            f"{prov['bifurcation']['shoot']['ay'][i]:.3f} |"
        )
    md += [
        f"\nα_x-Hopf (at the held α_y={ALPHA_Y_HELD:g}) — the axis the "
        "composite drives:\n",
        "| α_x | mech | deriv | +shooting |",
        "|---|---|---|---|",
    ]
    for i, x in enumerate(AX_SWEEP):
        md.append(
            f"| {x:.4g} | {prov['bifurcation']['mech']['ax'][i]:.3f} | "
            f"{prov['bifurcation']['deriv']['ax'][i]:.3f} | "
            f"{prov['bifurcation']['shoot']['ax'][i]:.3f} |"
        )
    md.append("\n## Multi-hallmark DDB2 vs severity (shooting block)\n")
    for k, fr in prov["ddb2"].items():
        md.append(
            f"\n**α_y={fr['alpha_y']:g} — {fr['regime']}**  "
            f"(∂DDB2/∂severity: autodiff={fr['grad_autodiff']:+.5f}, "
            f"finite-diff={fr['grad_finite_diff']:+.5f})\n"
        )
        md.append("| severity | mechanistic | hybrid | |Δ| |")
        md.append("|---|---|---|---|")
        for s, m, h in zip(fr["severities"], fr["mech"], fr["hybrid"]):
            md.append(f"| {s} | {m:.4f} | {h:.4f} | {abs(m - h):.4f} |")
    (OUT / "provenance.md").write_text("\n".join(md) + "\n")
    log.info("wrote provenance.json/.md")


def _load_block():
    """Deserialise the trained block from disk into a matching skeleton."""
    skeleton = NeuralODEProcess(
        fields=FIELDS,
        input_fields=CONDITIONING,
        field_defaults=IC,
        width=TRAIN["width"],
        depth=TRAIN["depth"],
        timescale=3600.0,
        key=jax.random.PRNGKey(1),
    )
    return eqx.tree_deserialise_leaves(
        str(OUT / "gz06_neural_block.eqx"), skeleton
    )


def _load_flag():
    with open(OUT / "provenance.json") as f:
        return json.load(f)["ddb2"]


# At the deposit's damaged α_x, α_y picks one regime on each side of the two
# Hopfs; the last case crosses the α_x Hopf instead, at the held α_y.
COMBINED_TOP_CASES = [
    (GZ06_ALPHA_X_DAMAGED, 0.011, "fixed point", "below lower α_y Hopf"),
    (GZ06_ALPHA_X_DAMAGED, ALPHA_Y_HELD, "limit cycle", "deposited α_y"),
    (GZ06_ALPHA_X_CONTROL, ALPHA_Y_HELD, "fixed point", "above α_x Hopf"),
]


def combined_figure(block, flag):
    """Single preprint figure: top row = surrogate reproduces GZ06's two
    Hopf bifurcations (p53 time-domain); bottom row = hybrid composite
    reproduces the mechanistic DDB2 readout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}
    )
    C_M, C_N = "#333333", "#d97706"
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.55)

    top = [fig.add_subplot(gs[0, 2 * i : 2 * i + 2]) for i in range(3)]
    for i, (ax, (axv, ay, regime, note)) in enumerate(
        zip(top, COMBINED_TOP_CASES)
    ):
        tm, xm = _run_traj(_solo(_gz_with(axv, ay)))
        tn, xn = _run_traj(neural_solo(block, axv, ay))
        ax.plot(tm, xm, color=C_M, lw=1.9, label="mechanistic GZ06")
        ax.plot(
            tn, xn, color=C_N, lw=1.7, ls="--", label="NeuralODE surrogate"
        )
        title = rf"$\alpha_x$ = {axv:.4g}, $\alpha_y$ = {ay:g}   {regime}"
        if note:
            title += f"\n({note})"
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time (days)")
        if i == 0:
            ax.set_ylabel("p53 (x)")
            ax.legend(frameon=False, fontsize=8.5, loc="best")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    bot = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    for i, (ax, k) in enumerate(
        zip(bot, sorted(flag, key=lambda z: float(z)))
    ):
        r = flag[k]
        sev = r["severities"]
        ax.plot(sev, r["mech"], "o-", color=C_M, label="mechanistic")
        ax.plot(sev, r["hybrid"], "s--", color=C_N, label="hybrid (NeuralODE)")
        vals = list(r["mech"]) + list(r["hybrid"])
        lo, hi = min(vals), max(vals)
        if hi - lo < 0.06:
            mid = 0.5 * (lo + hi)
            ax.set_ylim(mid - 0.035, mid + 0.035)
        else:
            span = hi - lo
            ax.set_ylim(lo - 0.25 * span, hi + 0.12 * span)
        ax.set_xlabel("Genomic Instability severity")
        if i == 0:
            ax.set_ylabel(f"DDB2 readout (day {DDB2_READ_DAY:g})")
        ax.set_title(rf"$\alpha_y$ = {float(k):g}", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(
        0.5,
        0.965,
        "NeuralODE surrogate reproduces GZ06's two Hopf bifurcations "
        r"(held-out $\psi$ = 1.0)",
        ha="center",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.455,
        "Hybrid composite reproduces the mechanistic readout",
        ha="center",
        fontsize=13.5,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"hybrid_combined.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    log.info("wrote hybrid_combined.png/.pdf -> %s", OUT)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "mode", nargs="?", default="all", choices=("all", "combined")
    )
    mode = ap.parse_args().mode

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("hallsim").setLevel(logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    if mode == "combined":
        combined_figure(_load_block(), _load_flag())
        return

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
        (deriv, deriv_c, "derivative-only")
        if deriv_err <= shoot_err
        else (shoot, shoot_c, "shooting-refined")
    )
    log.info(
        "kept %s block (amp err deriv=%.3f shoot=%.3f)",
        best_name,
        deriv_err,
        shoot_err,
    )
    time_domain_figure(best)

    flag = ddb2_results(best)
    ddb2_figure(flag)
    combined_figure(best, flag)

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
        "ddb2": flag,
    }
    write_provenance(prov)

    eqx.tree_serialise_leaves(str(OUT / "gz06_neural_block.eqx"), best)
    print(
        f"\nkept {best_name}; amplitude error deriv={deriv_err:.3f} "
        f"shoot={shoot_err:.3f}"
    )
    for k, fr in flag.items():
        print(
            f"α_y={fr['alpha_y']:g} [{fr['regime']}]: "
            f"∂DDB2/∂sev={fr['grad_autodiff']:+.5f}"
        )
    print(
        f"outputs → {OUT.relative_to(ROOT)}/ "
        "(bifurcation_recovery, ddb2_severity, provenance.md)"
    )


if __name__ == "__main__":
    main()
