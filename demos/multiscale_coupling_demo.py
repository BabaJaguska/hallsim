"""Demo: multi-timescale coupling — frozen vs interpolated Lie splitting.

This demo creates a 2-process system with genuinely different timescales
and tight coupling, then compares three integration strategies:

1. **Monolithic** (reference): single-group ODE solve, no splitting.
2. **Frozen splitting**: standard Lie splitting — each group sees a
   frozen snapshot of the other group's state at macro-step boundaries.
3. **Interpolated splitting**: Lie splitting with dense-output coupling —
   each group queries a continuous interpolant of the previous group's
   trajectory within the macro step.

The system:
- **FastOscillator** (timescale ~1s): damped oscillator driven by the
  slow variable.  x'' + 2ζω x' + ω² x = slow_drive
- **SlowIntegrator** (timescale ~100s): leaky integrator driven by the
  fast variable's amplitude.  τ y' = -y + |x|

The coupling is bidirectional: fast dynamics depend on slow state, and
slow dynamics depend on fast state. This makes splitting error visible.

Usage:
    cd /path/to/HallSim
    .venv_hallsim/bin/python demos/multiscale_coupling_demo.py
"""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.scheduler import Scheduler


# ── Processes ─────────────────────────────────────────────────────────

class FastOscillator(Process):
    """Damped harmonic oscillator driven by a slow external signal.

    State: x (position), v (velocity)
    Reads: slow_drive (from SlowIntegrator)

    dx/dt = v
    dv/dt = -2*zeta*omega*v - omega^2*x + omega^2*slow_drive

    The slow_drive shifts the oscillator's equilibrium point, so the
    oscillation center drifts as the slow process evolves. With splitting,
    the fast group sees a stale slow_drive → oscillates around the wrong
    center → visible phase/amplitude error.
    """

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 0.5  # ~0.5 second characteristic period

    omega: float = 4.0   # ~0.6 Hz natural frequency (period ~1.6s)
    zeta: float = 0.05   # very lightly damped — oscillations persist

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EXCLUSIVE, default=0.0,
                       units="dimensionless", description="Oscillator position"),
            "v": Port(role=PortRole.EXCLUSIVE, default=1.0,
                       units="1/s", description="Oscillator velocity"),
            "slow_drive": Port(role=PortRole.INPUT, default=0.0,
                               units="dimensionless",
                               description="Driving signal from slow process"),
        }

    def derivative(self, t, state):
        x = state["x"]
        v = state["v"]
        drive = state["slow_drive"]

        dx = v
        dv = (
            -2.0 * self.zeta * self.omega * v
            - self.omega ** 2 * x
            + self.omega ** 2 * drive
        )
        return {"x": dx, "v": dv}


class SlowIntegrator(Process):
    """Leaky integrator driven by the fast oscillator's amplitude.

    State: slow_drive
    Reads: x (from FastOscillator)

    tau * d(slow_drive)/dt = -slow_drive + gain * x^2

    Uses x^2 (smooth, always positive) as the coupling signal.
    The slow variable tracks the time-averaged energy of the oscillator.
    """

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 10.0  # ~10 second time constant (closer to fast → more coupling)

    tau: float = 10.0    # time constant
    gain: float = 2.0    # strong coupling to make splitting error visible

    def ports_schema(self):
        return {
            "slow_drive": Port(role=PortRole.EXCLUSIVE, default=0.0,
                               units="dimensionless",
                               description="Slow integrator output"),
            "x": Port(role=PortRole.INPUT, default=0.0,
                       units="dimensionless",
                       description="Fast oscillator position"),
        }

    def derivative(self, t, state):
        y = state["slow_drive"]
        x = state["x"]
        dy = (-y + self.gain * x ** 2) / self.tau
        return {"slow_drive": dy}


# ── Build composite ──────────────────────────────────────────────────

def build_composite():
    processes = {
        "fast": FastOscillator(),
        "slow": SlowIntegrator(),
    }
    topology = {
        "fast": {"x": "osc/x", "v": "osc/v", "slow_drive": "osc/slow_drive"},
        "slow": {"slow_drive": "osc/slow_drive", "x": "osc/x"},
    }
    return Composite(processes, topology)


# ── Run comparisons ──────────────────────────────────────────────────

def run_monolithic(composite, t_span, macro_dt, save_dt=None):
    """Reference: single-group solve via the Scheduler fast path
    (no manual groups → single Diffrax solve over the whole t_span)."""
    return Scheduler().run(
        composite, t_span=t_span, macro_dt=macro_dt, save_dt=save_dt or macro_dt
    )


def run_split(composite, t_span, macro_dt, mode="frozen", save_dt=None,
              adaptive_dt=False, splitting="lie"):
    """Lie or Strang splitting with frozen or interpolated coupling."""
    scheduler = Scheduler(
        groups={
            "fast_group": ["fast"],
            "slow_group": ["slow"],
        },
        coupling_mode=mode,
        splitting=splitting,
        adaptive_dt=adaptive_dt,
    )
    return scheduler.run(
        composite, t_span=t_span, macro_dt=macro_dt,
        save_dt=save_dt or macro_dt,
    )


def compute_error(ref, test, key):
    """RMS error between reference and test trajectories."""
    # Interpolate test to reference time points
    test_vals = jnp.interp(ref.ts, test.ts, test.get(key))
    ref_vals = ref.get(key)
    return float(jnp.sqrt(jnp.mean((test_vals - ref_vals) ** 2)))


def main():
    comp = build_composite()
    t_span = (0.0, 30.0)
    # Start with a LARGE macro_dt — deliberately too coarse for the
    # fast oscillation (~1.6s period).  This is the regime where:
    # - Frozen splitting accumulates significant coupling error
    # - Adaptive dt should detect the large residual and shrink
    macro_dt = 2.0

    print("Multi-Timescale Coupling Demo")
    print("=" * 60)
    print()
    print("System: FastOscillator (~1.6s period) <-> SlowIntegrator (tau=10s)")
    print(f"Time span: {t_span}, macro_dt: {macro_dt}")
    print()

    # 1. Monolithic reference (fine dt for ground truth)
    print("Running monolithic reference (no splitting)...")
    ref = run_monolithic(comp, t_span, macro_dt=0.05, save_dt=0.05)
    print(f"  {len(ref.ts)} time points")

    # 2. Frozen splitting
    print("Running frozen Lie splitting...")
    frozen = run_split(comp, t_span, macro_dt, mode="frozen")
    print(f"  {len(frozen.ts)} time points")

    # 3. Interpolated splitting
    print("Running interpolated Lie splitting...")
    interp = run_split(comp, t_span, macro_dt, mode="interpolated")
    print(f"  {len(interp.ts)} time points")

    # 4. Strang splitting
    print("Running Strang splitting (symmetric)...")
    strang = run_split(comp, t_span, macro_dt, mode="frozen", splitting="strang")
    print(f"  {len(strang.ts)} time points")

    # Compute errors
    print()
    print("RMS Error vs Monolithic Reference")
    print("-" * 40)

    for key, label in [("osc/x", "x (fast)"), ("osc/slow_drive", "slow_drive")]:
        err_frozen = compute_error(ref, frozen, key)
        err_interp = compute_error(ref, interp, key)
        err_strang = compute_error(ref, strang, key)
        print(f"  {label:20s}  lie={err_frozen:.6f}  interp={err_interp:.6f}  strang={err_strang:.6f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(ref.ts, ref.get("osc/x"), "k-", label="monolithic (reference)", alpha=0.8)
    ax.plot(frozen.ts, frozen.get("osc/x"), "r--", label="Lie (frozen)", alpha=0.5)
    ax.plot(interp.ts, interp.get("osc/x"), "b:", label="Lie (interpolated)", alpha=0.7, linewidth=2)
    ax.plot(strang.ts, strang.get("osc/x"), "g-.", label="Strang", alpha=0.7)
    ax.set_ylabel("x (fast oscillator)")
    ax.legend(fontsize=7)
    ax.set_title("Multi-Timescale Coupling: Lie vs Interpolated vs Strang")

    ax = axes[1]
    ax.plot(ref.ts, ref.get("osc/slow_drive"), "k-", label="monolithic", alpha=0.8)
    ax.plot(frozen.ts, frozen.get("osc/slow_drive"), "r--", label="Lie", alpha=0.5)
    ax.plot(interp.ts, interp.get("osc/slow_drive"), "b:", label="interpolated", alpha=0.7, linewidth=2)
    ax.plot(strang.ts, strang.get("osc/slow_drive"), "g-.", label="Strang", alpha=0.7)
    ax.set_ylabel("slow_drive")
    ax.legend(fontsize=7)

    # Error panel
    ax = axes[2]
    frozen_x = jnp.interp(ref.ts, frozen.ts, frozen.get("osc/x"))
    interp_x = jnp.interp(ref.ts, interp.ts, interp.get("osc/x"))
    strang_x = jnp.interp(ref.ts, strang.ts, strang.get("osc/x"))
    ax.plot(ref.ts, jnp.abs(frozen_x - ref.get("osc/x")), "r-", label="|Lie - ref|", alpha=0.5)
    ax.plot(ref.ts, jnp.abs(interp_x - ref.get("osc/x")), "b-", label="|interp - ref|", alpha=0.7)
    ax.plot(ref.ts, jnp.abs(strang_x - ref.get("osc/x")), "g-", label="|Strang - ref|", alpha=0.7)
    ax.set_ylabel("absolute error (x)")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    plt.tight_layout()
    from outdir import outdir
    out_path = str(outdir("multiscale_coupling") / "comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    print()
    print("What to look for in the plot:")
    print("  Top panel:    Fast oscillator x(t) — all 3 methods should show oscillations.")
    print("                Frozen (red) may show phase/amplitude drift vs monolithic (black).")
    print("  Middle panel: slow_drive(t) — the coupling variable. Frozen (red) diverges")
    print("                from reference because it integrates stale x values.")
    print("  Bottom panel: Absolute error of x vs monolithic reference (log scale).")
    print("                Interpolated (blue) should be consistently below frozen (red).")


if __name__ == "__main__":
    main()
