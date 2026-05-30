"""Two-hallmark severity sweep on the multi-hallmark composite.

Verifies that the DDIS-vs-rapa-vs-ctrl experimental design now maps
cleanly onto a 2D severity matrix:

  Genomic Instability        ∈ {0.0, 1.0}   — DP14's irradiation rate
  Deregulated Nutrient Sensing ∈ {0.3, 1.0} — DP14's mTORC1-phos rate

Runs the multi-hallmark composite under three conditions (control,
DDIS, DDIS+rapa) and prints terminal-time values of the readouts that
gene-reporter validation against GSE248823 will consume.

This is a sanity check that hallmark severities propagate through the
composite as expected. The actual concordance evaluation against
GSE248823 is in :mod:`demos.concordance_reporters`.

Usage::

    .venv_hallsim/bin/python demos/multi_hallmark_sweep.py
"""

from __future__ import annotations

from hallsim.composite import Composite
from hallsim.hallmarks import apply_hallmarks
from hallsim.models.multi_hallmark import build_multi_hallmark_composite
from hallsim.scheduler import Scheduler


T_END = 50.0
MACRO_DT = 5.0
KEYS_OF_INTEREST = (
    "dp14/DNA_damage",
    "dp14/CDKN1A",
    "dp14/mTORC1_pS2448",
    "dp14/ROS",
    "dp14/Mitophagy",
    "nfkb/IkBat",
    "bridge/psi",
)
ARMS = (
    # (Genomic Instability severity, Deregulated Nutrient Sensing severity, label)
    (0.0, 0.5, "ctrl"),
    (1.0, 1.0, "DDIS"),
    (1.0, 0.3, "DDIS+rapa"),
)


def simulate_arm(base, gi_severity, dns_severity):
    procs = apply_hallmarks(
        base.processes,
        {
            "Genomic Instability": gi_severity,
            "Deregulated Nutrient Sensing": dns_severity,
        },
    )
    comp = Composite(
        processes=procs,
        topology=base.topology,
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    sched = Scheduler()
    y0 = comp.initial_state_vec()
    return sched.run(
        comp, t_span=(0.0, T_END), macro_dt=MACRO_DT, y0=y0, save_dt=MACRO_DT
    )


def main():
    base = build_multi_hallmark_composite()

    print(f"\nSeverity sweep — multi_hallmark composite (T_END = {T_END})\n")
    hdr = f"{'GI':>5} {'DNS':>5} | " + " ".join(
        f"{k.split('/')[-1]:>14}" for k in KEYS_OF_INTEREST
    )
    print(hdr)
    print("-" * len(hdr))

    for gi, dns, label in ARMS:
        res = simulate_arm(base, gi, dns)
        vals = []
        for k in KEYS_OF_INTEREST:
            try:
                vals.append(float(res.get(k)[-1]))
            except KeyError:
                vals.append(float("nan"))
        row = f"{gi:>5.2f} {dns:>5.2f} | " + " ".join(
            f"{v:>14.4g}" for v in vals
        )
        print(f"{row}  # {label}")


if __name__ == "__main__":
    main()
