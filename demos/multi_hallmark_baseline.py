"""Run the bridge-free multi-hallmark composite at its uncalibrated
default. Establishes the baseline that Calibrator needs to fix.

Pre-calibration expectation: DP14's DNA_damage saturates near 28000 at
GI=1.0 (SBML default irradiation rate), feeding directly into GZ06's
psi. GZ06's psi was calibrated near 1.0; passing 28000 will overload
the oscillator and likely produce nonsense or NaN. The point of this
baseline is to *measure* that overload, then fit the irradiation rate
so DP14's damage trajectory lands in GZ06's operating range.

Usage::

    .venv_hallsim/bin/python demos/multi_hallmark_baseline.py
"""

from __future__ import annotations

import time

from hallsim.composite import Composite
from hallsim.hallmarks import apply_hallmarks
from hallsim.models.multi_hallmark import build_multi_hallmark_composite
from hallsim.scheduler import Scheduler


T_END = 50.0
MACRO_DT = 5.0


def simulate(gi: float, dns: float):
    base = build_multi_hallmark_composite()
    procs = apply_hallmarks(
        base.processes,
        {"Genomic Instability": gi, "Deregulated Nutrient Sensing": dns},
    )
    comp = Composite(
        processes=procs,
        topology=base.topology,
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    sched = Scheduler()
    y0 = comp.initial_state_vec()
    t0 = time.time()
    try:
        res = sched.run(
            comp,
            t_span=(0.0, T_END),
            macro_dt=MACRO_DT,
            y0=y0,
            save_dt=MACRO_DT,
        )
        return time.time() - t0, res, None
    except Exception as e:
        return time.time() - t0, None, type(e).__name__


def main():
    arms = [
        (0.0, 0.5, "ctrl"),
        (1.0, 1.0, "DDIS"),
        (1.0, 0.3, "DDIS+rapa"),
    ]
    print("Multi-hallmark composite, bridge removed, uncalibrated.")
    print(f"T_END={T_END}, macro_dt={MACRO_DT}\n")
    keys = [
        "dp14/DNA_damage",
        "dp14/CDKN1A",
        "dp14/mTORC1_pS2448",
        "nfkb/IkBa",
        "gz06/x",
        "gz06/y",
    ]
    hdr = f"{'GI':>4} {'DNS':>4} | " + " ".join(f"{k.split('/')[-1]:>13}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for gi, dns, label in arms:
        wall, res, err = simulate(gi, dns)
        if res is None:
            print(f"{gi:>4.2f} {dns:>4.2f} | FAIL ({wall:.1f}s) -> {err}  # {label}")
            continue
        row_vals = []
        for k in keys:
            try:
                row_vals.append(float(res.get(k)[-1]))
            except KeyError:
                row_vals.append(float("nan"))
        print(f"{gi:>4.2f} {dns:>4.2f} | " + " ".join(f"{v:>13.4g}" for v in row_vals) + f"  ({wall:.1f}s)  # {label}")


if __name__ == "__main__":
    main()
