"""Diagnose where the multi-hallmark composite stops converging.

The two-hallmark severity sweep blows up on the control arm
(Genomic Instability = 0). This script brackets the failure: run a
range of GI severities with everything else held at DDIS settings, see
where the integrator runs out of steps. Then peek at DP14 alone (no
GZ06, no bridge) at GI=0 to isolate whether the issue is in DP14 or in
the coupling.
"""

from __future__ import annotations

import time

from hallsim.composite import Composite
from hallsim.hallmarks import apply_hallmarks
from hallsim.models.multi_hallmark import (
    DP14_IRRADIATION_RATE_DEFAULT,
    DP14_IRRADIATION_RATE_NAME,
    DP14_MTOR_PHOS_RATE_DEFAULT,
    DP14_MTOR_PHOS_RATE_NAME,
    DP14_SBML_PATH,
    build_multi_hallmark_composite,
)
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler


T_END = 50.0
MACRO_DT = 5.0


def try_run(label, composite, t_end=T_END, macro_dt=MACRO_DT, single_group=False):
    if single_group:
        # Force everything into one Diffrax solve — no operator splitting.
        all_continuous = [
            name
            for name, p in composite.processes.items()
            if getattr(p, "kind", None) is None or str(getattr(p, "kind")).endswith("CONTINUOUS")
        ]
        sched = Scheduler(groups={"all": list(composite.processes.keys())})
    else:
        sched = Scheduler()
    y0 = composite.initial_state_vec()
    t0 = time.time()
    try:
        res = sched.run(
            composite,
            t_span=(0.0, t_end),
            macro_dt=macro_dt,
            y0=y0,
            save_dt=macro_dt,
        )
        elapsed = time.time() - t0
        dmg = float(res.get("dp14/DNA_damage")[-1])
        print(f"  OK  ({elapsed:5.1f} s) {label:<40} DNA_damage[end]={dmg:.4g}")
        return res
    except Exception as e:
        elapsed = time.time() - t0
        msg = type(e).__name__
        print(f"  FAIL ({elapsed:5.1f} s) {label:<40} -> {msg}")
        return None


def sweep_gi_severity():
    print("\n=== Sweep: multi-hallmark composite, vary GI severity ===")
    print("All runs hold DNS=0.5, T_END=50, macro_dt=5\n")
    base = build_multi_hallmark_composite()
    for gi in [1.0, 0.5, 0.3, 0.1, 0.05, 0.01, 0.0]:
        procs = apply_hallmarks(
            base.processes,
            {"Genomic Instability": gi, "Deregulated Nutrient Sensing": 0.5},
        )
        comp = Composite(
            processes=procs,
            topology=base.topology,
            validate=False,
            semantic_validation={"check_semantics": False},
        )
        try_run(f"GI={gi:>4.2f}, DNS=0.5", comp)


def dp14_alone_at_gi_zero():
    print("\n=== DP14 alone (no GZ06, no bridge), GI=0 ===")
    proc = process_from_sbml(
        str(DP14_SBML_PATH),
        name="dp14",
        parameter_overrides={
            DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
            DP14_IRRADIATION_RATE_NAME: 0.0,  # explicit zero
        },
    )
    comp = Composite(
        processes={"dp14": proc},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    try_run("DP14 alone, GI=0", comp)


def dp14_alone_at_gi_full():
    print("\n=== DP14 alone (no GZ06, no bridge), GI=1 ===")
    proc = process_from_sbml(
        str(DP14_SBML_PATH),
        name="dp14",
        parameter_overrides={
            DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
            DP14_IRRADIATION_RATE_NAME: DP14_IRRADIATION_RATE_DEFAULT,
        },
    )
    comp = Composite(
        processes={"dp14": proc},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    try_run("DP14 alone, GI=1", comp)


def single_group_sweep():
    """Force the whole composite into one Diffrax solve — no splitting.

    Compares against the multi-group sweep above. If single-group runs
    that previously failed now converge, operator splitting is the
    culprit. If they still fail, the coupling itself is stiff.
    """
    print("\n=== Single-group sweep (no operator splitting) ===")
    print("All in one Diffrax solve over [0, T_END]\n")
    base = build_multi_hallmark_composite()
    for gi in [1.0, 0.5, 0.3, 0.0]:
        procs = apply_hallmarks(
            base.processes,
            {"Genomic Instability": gi, "Deregulated Nutrient Sensing": 0.5},
        )
        comp = Composite(
            processes=procs,
            topology=base.topology,
            validate=False,
            semantic_validation={"check_semantics": False},
        )
        # Run with macro_dt = T_END so there is just one macro step.
        try_run(
            f"GI={gi:>4.2f}, DNS=0.5 (single-group)",
            comp,
            macro_dt=T_END,
            single_group=True,
        )


if __name__ == "__main__":
    dp14_alone_at_gi_full()
    dp14_alone_at_gi_zero()
    single_group_sweep()
