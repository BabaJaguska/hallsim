"""Which GZ06 parameter should carry DNA damage?

Geva-Zatorsky 2006's ``psi`` is the paper's ξ, a multiplicative noise gain on
protein production — not a damage variable. This scans the three degradation
channels the DNA-damage literature actually implicates for a Hopf bifurcation,
so the damage edge can be placed on a term that switches the oscillator:

    alpha_x  Mdm2-independent p53 degradation; ATM stabilises p53, so damage
             lowers it. The paper's fit sets it to 0 (Banin et al. 1998,
             Science 281:1674)
    alpha_k  Mdm2-dependent p53 degradation; same mechanism, damage lowers it
    alpha_y  Mdm2 degradation; ATM destabilises Mdm2, so damage raises it
             (Maya et al. 2001, Genes Dev 15:1067)

Run at the published ξ = 1, on the standalone process, per the
constituents-first rule. GZ06 fitted Model IV to *irradiated* cells, so the
published parameter set is the damaged state: a usable channel puts the
published value on the oscillating side and a control-side displacement on the
quiescent side.

    simulate gz06-damage-scan
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hallsim.bifurcation import (
    codim1_scan,
    critical_eigenvalue,
    equilibrium,
    field_from_composite,
)
from hallsim.composite import Composite
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
GZ06_SBML = (
    ROOT / "demos/models/sbml/zatorsky2006/zatorsky2006_BIOMD0000000157.xml"
)

PUBLISHED = {"alpha_x": 0.0, "alpha_k": 1.7, "alpha_y": 0.8}
# Wide enough to bracket the published value by a factor of ~4 either way.
SWEEPS = {
    "alpha_x": np.linspace(0.0, 3.0, 240),
    "alpha_k": np.linspace(0.2, 8.0, 240),
    "alpha_y": np.linspace(0.05, 3.2, 240),
}
# Damage direction from the cited mechanism: ATM stabilises p53 (alpha_x and
# alpha_k down), ATM destabilises Mdm2 (alpha_y up).
DAMAGE_RAISES = {"alpha_x": False, "alpha_k": False, "alpha_y": True}


def build(channel: str, value: float):
    """Standalone GZ06 at ``channel = value``, psi pinned to the published 1.0."""
    gz = process_from_sbml(
        str(GZ06_SBML),
        name="gz06",
        parameters={channel: value, "psi": 1.0},
    )
    return Composite({"gz06": gz}, {}, semantic_validation=False)


def settled(channel: str, value: float, *, hours: float = 400.0) -> tuple:
    """``(mean, peak_to_trough)`` of p53 over the last tenth of a long run.

    Long, because a damped oscillator near a Hopf decays at a rate that goes to
    zero there — a short run reports the transient and reads as a limit cycle.
    """
    comp = build(channel, value)
    res = Scheduler().run(comp, t_span=(0.0, hours), save_dt=hours / 20000)
    x = np.asarray(res.get("gz06/x"))
    tail = x[9 * len(x) // 10 :]
    return float(tail.mean()), float(tail.max() - tail.min())


def growth_rate(channel: str, value: float) -> float:
    """``Re`` of the critical eigenvalue at the equilibrium — negative means the
    fixed point is stable and any oscillation decays, whatever a finite run
    shows. Exact, unlike an amplitude read off a trajectory."""
    f = field_from_composite(build(channel, value))[0]
    eq = equilibrium(f, np.array([0.4, 0.4, 0.4]))
    if eq is None:
        return float("nan")
    return float(critical_eigenvalue(f, eq).real)


def scan_channel(channel: str) -> dict:
    def field_of(v):
        return field_from_composite(build(channel, v))[0]

    x0 = [0.4, 0.4, 0.4]
    bifs = codim1_scan(field_of, SWEEPS[channel], x0_guess=x0)
    return {"channel": channel, "bifurcations": bifs}


def report(result: dict) -> None:
    channel = result["channel"]
    bifs = result["bifurcations"]
    pub = PUBLISHED[channel]
    lo, hi = SWEEPS[channel][0], SWEEPS[channel][-1]

    print(f"\n=== {channel} (published {pub}, swept {lo:g}..{hi:g}) ===")
    if not bifs:
        print("  no codim-1 bifurcation in range")
    for b in bifs:
        print(f"  {b}")

    hopfs = [b for b in bifs if b.kind == "hopf"]
    probes = sorted(
        {lo, pub, hi}
        | {p for b in hopfs for p in (b.param * 0.5, b.param * 2, b.param * 4)}
    )
    print(
        f"  {'value':>10} {'Re(lambda)':>12} {'decay tau/h':>12} "
        f"{'p53 mean':>10} {'peak-trough':>12}"
    )
    for v in probes:
        if not lo <= v <= hi:
            continue
        re = growth_rate(channel, v)
        tau = "-" if re >= 0 else f"{-1 / re:.3g}"
        mean, ptp = settled(channel, v)
        mark = "  <- published" if abs(v - pub) < 1e-9 else ""
        print(
            f"  {v:>10.4g} {re:>12.5g} {tau:>12} {mean:>10.4g} "
            f"{ptp:>12.6g}{mark}"
        )

    if not hopfs:
        print("  VERDICT: unusable — no Hopf to switch on")
        return
    h = hopfs[0]
    pub_side_osc = growth_rate(channel, pub) > 0
    direction_ok = (pub > h.param) == DAMAGE_RAISES[channel]
    print(
        f"  VERDICT: Hopf at {h.param:.4g}, "
        f"{'supercritical' if h.supercritical else 'subcritical/degenerate'}; "
        f"published side {'oscillates' if pub_side_osc else 'is quiescent'}; "
        f"damage direction {'crosses' if direction_ok else 'moves AWAY from'} it"
    )


def main() -> None:
    logging.getLogger("hallsim").setLevel(logging.ERROR)
    for channel in ("alpha_x", "alpha_k", "alpha_y"):
        report(scan_channel(channel))


if __name__ == "__main__":
    main()
