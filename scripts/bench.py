"""Reproduce the measurements in docs/benchmarks.md.

python scripts/bench.py            # trace cost + group solve dimension
python scripts/bench.py --solver   # Newton vs VeryChord (~2 min)
python scripts/bench.py --graph    # RHS jaxpr composition
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler

logging.disable(logging.WARNING)

MULTI_HALLMARK_SPAN = (0.0, 14.0)
MULTI_HALLMARK_MACRO_DT = 5.0


class _Osc(Process):
    """Minimal 2-state block; one per timescale group."""

    timescale: float | None = None
    k: float = 1.0

    def ports_schema(self):
        return {
            "a": Port(role=PortRole.EVOLVED, default=1.0),
            "b": Port(role=PortRole.EVOLVED, default=0.5),
        }

    def derivative(self, t, state):
        return {
            "a": -self.k * state["a"] + 0.3 * state["b"],
            "b": self.k * state["a"] - 0.7 * state["b"],
        }


def _synthetic(n_groups: int) -> Composite:
    """``n_groups`` blocks with timescales far enough apart to never merge."""
    return Composite(
        processes={
            f"m{i}": _Osc(timescale=10.0 ** (3 * i)) for i in range(n_groups)
        },
        topology={
            f"m{i}": {"a": f"m{i}/a", "b": f"m{i}/b"} for i in range(n_groups)
        },
        semantic_validation=False,
    )


def _warm_ms(fn, repeats: int = 3) -> float:
    """Milliseconds per call after the first, which absorbs compilation."""
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats * 1e3


def bench_trace_cost() -> None:
    print("\n== 1. Scheduler.run per-call cost vs group count ==")
    print(f"{'groups':>7} {'warm ms':>10}")
    for n in (1, 2, 4, 8, 16):
        comp = _Osc and _synthetic(n)
        sched = Scheduler(auto_stiffness=False)
        ms = _warm_ms(
            lambda: sched.run(comp, (0.0, 10.0), macro_dt=1.0, save_dt=1.0).ys
        )
        print(f"{len(comp.auto_groups()):>7} {ms:>10.2f}")


def bench_splitting() -> None:
    """Does Lie splitting earn its keep? Both sides through Scheduler.run."""
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

    print("\n== 2. Lie splitting vs one merged group ==")
    comp = build_multi_hallmark_composite()
    auto = comp.auto_groups()
    merged = {"all": sorted({p for ps in auto.values() for p in ps})}

    for label, groups in (
        (f"split ({len(auto)} groups, as shipped)", auto),
        ("merged (1 group, fast path)", merged),
    ):
        sched = Scheduler(groups=groups)
        sched.warm_up(comp, MULTI_HALLMARK_SPAN, MULTI_HALLMARK_MACRO_DT)
        ms = _warm_ms(
            lambda: sched.run(
                comp,
                MULTI_HALLMARK_SPAN,
                macro_dt=MULTI_HALLMARK_MACRO_DT,
                save_dt=0.1,
            ).ys,
            repeats=1,
        )
        res = sched.run(
            comp,
            MULTI_HALLMARK_SPAN,
            macro_dt=MULTI_HALLMARK_MACRO_DT,
            save_dt=0.1,
        )
        steps = sum(
            int(v["num_solver_steps"])
            for v in res.stats.values()
            if isinstance(v, dict) and "num_solver_steps" in v
        )
        print(f"  {label:32s} {ms / 1e3:5.2f} s   {steps:6d} solver steps")


def bench_group_dimension() -> None:
    """Per-group Jacobian density and the cost of solving at full width.

    Sizes what ``_ReducedRHS`` buys. Both arms go through raw diffrax to
    isolate the dimension effect from the Scheduler's own machinery; for the
    shipped end-to-end number, see :func:`bench_splitting`.
    """
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

    print("\n== 3. Group solve dimension (sizes _ReducedRHS) ==")
    comp = build_multi_hallmark_composite()
    keys = comp.store_keys()
    y0 = comp.initial_state_vec()
    solver = dfx.Kvaerno5(root_finder=optx.Newton(rtol=1e-6, atol=1e-9))

    for gname, procs in comp.auto_groups().items():
        rhs, _ = comp.build_rhs(procs)
        idx = comp.evolved_indices(procs, keys)
        jac = np.asarray(jax.jacfwd(lambda y: rhs(0.0, y))(y0))
        density = 100 * int((np.abs(jac) > 0).sum()) / jac.size

        def solve(term, state, atol):
            return dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=5.0,
                dt0=1e-3,
                y0=state,
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.PIDController(rtol=1e-6, atol=atol),
                max_steps=4_000_000,
                throw=False,
            ).ys

        # Both terms are built ONCE. A closure rebuilt per call is a static
        # leaf that rehashes, so every solve would miss diffrax's cache and
        # the comparison would measure re-tracing instead of the solve.
        sub0 = y0[idx]
        full_term = dfx.ODETerm(rhs)
        sub_term = dfx.ODETerm(
            lambda t, ys, args=None, _r=rhs, _i=idx, _y=y0: _r(
                t, _y.at[_i].set(ys)
            )[_i]
        )
        full_atol = jnp.maximum(1e-9, 1e-6 * jnp.abs(y0))
        sub_atol = jnp.maximum(1e-9, 1e-6 * jnp.abs(sub0))

        full = _warm_ms(lambda: solve(full_term, y0, full_atol), repeats=1)
        restricted = _warm_ms(
            lambda: solve(sub_term, sub0, sub_atol), repeats=1
        )
        print(
            f"  {gname}: {int(idx.size)} evolving of {len(keys)}, "
            f"Jacobian {density:.1f}% dense — "
            f"full {full:.0f} ms vs restricted {restricted:.0f} ms "
            f"({full / restricted:.2f}x)"
        )


def bench_solver() -> None:
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

    print("\n== 4. Implicit root finder ==")
    comp = build_multi_hallmark_composite()
    for label, implicit in (
        ("optx.Newton (shipped)", None),
        ("VeryChord (diffrax default)", dfx.Kvaerno5()),
    ):
        kw = {} if implicit is None else {"implicit_solver": implicit}
        sched = Scheduler(**kw)
        sched.warm_up(comp, MULTI_HALLMARK_SPAN, MULTI_HALLMARK_MACRO_DT)
        t0 = time.perf_counter()
        res = sched.run(
            comp,
            MULTI_HALLMARK_SPAN,
            macro_dt=MULTI_HALLMARK_MACRO_DT,
            save_dt=0.1,
        )
        jax.block_until_ready(res.ys)
        wall = time.perf_counter() - t0
        per_group = [
            (int(v["num_solver_steps"]), int(v["num_rejected_steps"]))
            for v in res.stats.values()
            if isinstance(v, dict) and "num_solver_steps" in v
        ]
        steps = sum(s for s, _ in per_group)
        rej = sum(r for _, r in per_group)
        print(
            f"  {label:30s} steps={steps:7d} "
            f"rejected={100 * rej / max(1, steps + rej):3.0f}%  {wall:5.1f} s"
        )


def bench_graph() -> None:
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

    print("\n== 5. RHS jaxpr composition ==")
    comp = build_multi_hallmark_composite()
    rhs, _ = comp.build_rhs()
    y0 = comp.initial_state_vec()
    jaxpr = jax.make_jaxpr(lambda y: rhs(0.0, y))(y0)
    counts = Counter(str(e.primitive) for e in jaxpr.jaxpr.eqns)
    print(f"  composite RHS: {len(jaxpr.jaxpr.eqns)} equations")
    for prim, n in counts.most_common(8):
        print(f"    {prim:24s} {n:5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", action="store_true", help="item 3 (slow)")
    ap.add_argument("--graph", action="store_true", help="item 4")
    args = ap.parse_args()

    if args.solver:
        bench_solver()
    elif args.graph:
        bench_graph()
    else:
        bench_trace_cost()
        bench_splitting()
        bench_group_dimension()


if __name__ == "__main__":
    main()
