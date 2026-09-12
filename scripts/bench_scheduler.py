"""The Scheduler against a bare diffrax solve of the same reduced field, on
the shapes the Scheduler exists for.

Not a test and not a dependency of anything: run by hand, record the table
in ``docs/benchmarks.md``. Every variant integrates the same maths — the
composite's flat RHS reduced to its evolved states, the Scheduler's own
root finder, controller and tolerances — so the only thing that differs is
the orchestration. A tight bare solve is the reference every variant's
error is measured against.

    python scripts/bench_scheduler.py gz06 multi-hallmark chain:64 chain:256
    python scripts/bench_scheduler.py --macro-dt 0.25 chain:1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import diffrax as dfx  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hallsim.composite import Composite, single_process_composite  # noqa: E402
from hallsim.config import DEFAULT_ATOL, DEFAULT_RTOL  # noqa: E402
from hallsim.process import Port, PortRole, Process  # noqa: E402
from hallsim.root_finders import Chord  # noqa: E402
from hallsim.sbml_import import process_from_sbml  # noqa: E402
from hallsim.scheduler import Scheduler, _FrozenFill, _ReducedRHS  # noqa: E402


# ── the synthetic multi-rate shape ───────────────────────────────────────
class SlowChain(Process):
    """``n`` states passing a signal down a diffusion chain with a slow leak,
    node 0 driven by a sinusoid and by the stiff block's output. Unit gain
    per node, so the signal reaches the far end; rates at most ``2/tau``,
    an explicit solver's problem."""

    n: int = eqx.field(static=True, default=64)
    tau: float = 1.0
    leak: float = 0.05
    period: float = 10.0
    feedback: float = 0.5
    timescale: float = 1.0

    def ports_schema(self):
        return {
            "x": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                elements=tuple(f"x{i}" for i in range(self.n)),
            ),
            "z_in": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        x = state["x"]
        dx = (jnp.roll(x, 1, axis=-1) - x) / self.tau - self.leak * x
        dx = dx.at[..., 0].add(
            (jnp.sin(2 * jnp.pi * t / self.period) - x[..., 0]) / self.tau
            + self.feedback * state["z_in"]
        )
        return {"x": dx}


class StiffRelax(Process):
    """``m`` states relaxing to a slow input at rate ``k``: the block that
    forces an implicit solver, and small enough that its Jacobian is
    nothing to factorise."""

    m: int = eqx.field(static=True, default=8)
    k: float = 1e4
    timescale: float = 1e-4

    def ports_schema(self):
        return {
            "z": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                elements=tuple(f"z{j}" for j in range(self.m)),
            ),
            "u": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        target = state["u"][..., None] / (1.0 + 0.1 * jnp.arange(self.m))
        return {"z": -self.k * (state["z"] - target)}


def chain_composite(n: int, m: int = 8) -> Composite:
    """Block ports: one path per node, one port per block, so the port
    count is two per process however long the chain."""
    chain = {
        "x": tuple(f"chain/x{i}" for i in range(n)),
        "z_in": "stiff/z0",
    }
    stiff = {
        "z": tuple(f"stiff/z{j}" for j in range(m)),
        "u": "chain/x1",  # a short loop: x0 -> x1 -> z -> x0
    }
    return Composite(
        processes={"chain": SlowChain(n=n), "stiff": StiffRelax(m=m)},
        topology={"chain": chain, "stiff": stiff},
        validate=False,
        semantic_validation=False,
    )


def gz06_composite() -> Composite:
    from demos.models.multi_hallmark import GZ06_SBML_PATH

    return single_process_composite(
        process_from_sbml(str(GZ06_SBML_PATH), name="gz06")
    )


def multi_hallmark_composite() -> Composite:
    from demos.models.multi_hallmark import build_multi_hallmark_composite

    return build_multi_hallmark_composite(validate=False)


CASES = {
    "gz06": (gz06_composite, (0.0, 14.0)),
    "multi-hallmark": (multi_hallmark_composite, (0.0, 14.0)),
}


# ── the bare solve: same reduced field, same solver, no orchestration ────
def bare_solver(comp, span, save_dt, jump_ts, solver, rtol, atol):
    rhs, keys = comp.build_rhs()
    own = comp.evolved_indices()
    y0 = comp.initial_state_vec()
    term = dfx.ODETerm(_ReducedRHS(base=rhs, own=own, fill=_FrozenFill(y0)))
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)
    if jump_ts is not None:
        ctrl = dfx.ClipStepSizeController(ctrl, jump_ts=jump_ts)
    t0, t1 = span
    # arange accumulates i*step; the last point can land a few ulp past t1,
    # which diffrax rejects. Pin it.
    ts = jnp.minimum(jnp.arange(t0, t1 + 1e-9, save_dt), t1)

    @eqx.filter_jit
    def solve(y_own):
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=None,
            y0=y_own,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=ctrl,
            max_steps=2_000_000,
            throw=False,
        )
        return sol.ys, sol.stats["num_steps"], sol.result

    return solve, y0[own], own, ts


def timed(fn, *args, repeats=3):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        warm.append(time.perf_counter() - t0)
    return out, cold, float(np.median(warm))


def scaled_error(ts, ys, ts_ref, ref):
    """Max error over states and reference times, relative to the largest
    reference value anywhere: a solver-tolerance-sized number when the
    solve is right, order one when it is wrong."""
    ts, ys = np.asarray(ts), np.asarray(ys)
    ts_ref, ref = np.asarray(ts_ref), np.asarray(ref)
    pick = np.isin(np.round(ts, 9), np.round(ts_ref, 9))
    keep = np.isin(np.round(ts_ref, 9), np.round(ts[pick], 9))
    if pick.sum() < 5:  # Strang saves only window ends
        return float("nan")
    scale = np.max(np.abs(ref)) + 1e-12
    return float(np.max(np.abs(ys[pick] - ref[keep])) / scale)


def run_case(
    name,
    build,
    span,
    macro_dt,
    save_dt,
    repeats,
    only=(),
    reference="implicit",
):
    comp = build()
    all_procs = list(comp.continuous_processes())
    t_plan = time.perf_counter()
    plan = Scheduler().plan(comp, span, macro_dt=macro_dt, save_dt=save_dt)
    t_plan = time.perf_counter() - t_plan
    own = comp.evolved_indices()
    jump_ts = plan.jump_ts
    n_states = int(comp.evolved_indices().shape[0])
    groups = ", ".join(
        f"{g}[{len(p)} procs, {type(plan.integrators[g].solver).__name__}]"
        for g, p in plan.groups.items()
    )
    print(
        f"\n### {name}: {n_states} evolved states, span {span}, "
        f"macro_dt {macro_dt}, save_dt {save_dt}\n"
        f"auto groups: {groups}; coupling {plan.coupling}; "
        f"jumps {None if jump_ts is None else len(jump_ts)}; "
        f"plan (routing included) {t_plan:.1f} s",
        flush=True,
    )

    chord = Chord(rtol=DEFAULT_RTOL, atol=1e-6)
    # the reference: bare, tight. Implicit by default; explicit for a
    # system too large to factorise densely, where a stability-bound
    # explicit solve at tight tolerance is the cheaper exact answer.
    ref_solver = (
        dfx.Kvaerno5(root_finder=Chord(rtol=1e-10, atol=1e-8))
        if reference == "implicit"
        else dfx.Tsit5()
    )
    ref_solve, y_own, own, ts = bare_solver(
        comp, span, save_dt, jump_ts, ref_solver, 1e-10, 1e-13
    )
    ref_ys, ref_steps, ref_res = ref_solve(y_own)
    jax.block_until_ready(ref_ys)
    print(
        f"reference: {int(ref_steps)} steps, "
        f"{'ok' if ref_res == dfx.RESULTS.successful else ref_res}",
        flush=True,
    )

    rows = []

    def record(label, t, ys, cold, warm, steps, ok=True, detail=""):
        err = scaled_error(t, ys, ts, ref_ys)
        rows.append((label, warm, cold, steps, err, ok))
        print(
            f"  {label:40s} warm {warm*1e3:9.1f} ms  cold {cold:6.1f} s  "
            f"steps {steps:>8}  err {err:.2e}{'' if ok else '  FAILED'}"
            f"{'  ' + detail if detail else ''}",
            flush=True,
        )

    for label, solver in (
        ("bare Kvaerno5 (chord)", dfx.Kvaerno5(root_finder=chord)),
        ("bare Tsit5", dfx.Tsit5()),
    ):
        if only and not any(sub in label for sub in only):
            continue
        solve, y_own, _, ts = bare_solver(
            comp, span, save_dt, jump_ts, solver, DEFAULT_RTOL, DEFAULT_ATOL
        )
        (ys, steps, res), cold, warm = timed(solve, y_own, repeats=repeats)
        record(
            label,
            ts,
            ys,
            cold,
            warm,
            int(steps),
            ok=bool(res == dfx.RESULTS.successful),
        )

    variants = [
        ("Scheduler, one group (fast path)", dict(groups={"all": all_procs})),
        ("Scheduler, auto groups", {}),
    ]
    if len(plan.groups) > 1:
        variants += [
            ("Scheduler, auto, frozen", dict(coupling_mode="frozen")),
            (
                "Scheduler, auto, interpolated",
                dict(coupling_mode="interpolated"),
            ),
            ("Scheduler, auto, strang", dict(splitting="strang")),
            # a feedback loop needs the backward edge too: Gauss-Seidel
            # sweeps re-solve the window with the other group's trajectory
            (
                "Scheduler, auto, interpolated, 2 sweeps",
                dict(coupling_mode="interpolated", waveform_sweeps=2),
            ),
        ]
    for label, kw in variants:
        if only and not any(sub in label for sub in only):
            continue
        sched = Scheduler(**kw)

        def run():
            return sched.run(
                comp, t_span=span, macro_dt=macro_dt, save_dt=save_dt
            )

        res, cold, warm = timed(run, repeats=repeats)
        per_group = {
            g: int(np.asarray(v["num_solver_steps"]).sum())
            for g, v in res.stats.items()
            if isinstance(v, dict) and "num_solver_steps" in v
        }
        ys = np.asarray(res.ys)[:, own]
        record(
            label,
            res.ts,
            ys,
            cold,
            warm,
            sum(per_group.values()),
            ok=bool(np.all(res.ok)),
            detail=str(per_group) if len(per_group) > 1 else "",
        )

    print("\n| variant | warm | cold | steps | max scaled error |")
    print("|---|---|---|---|---|")
    for label, warm, cold, steps, err, ok in rows:
        print(
            f"| {label} | {warm*1e3:.0f} ms | {cold:.1f} s | {steps} | "
            f"{err:.1e}{'' if ok else ' (failed)'} |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", help="gz06, multi-hallmark, chain:<n>")
    ap.add_argument("--macro-dt", type=float, default=None)
    ap.add_argument("--save-dt", type=float, default=None)
    ap.add_argument("--span", type=float, default=None, help="chain: t_end")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--reference",
        choices=("implicit", "explicit"),
        default="implicit",
        help="reference solver: Kvaerno5 (default) or Tsit5 for a system "
        "too large to factorise densely.",
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated substrings; run only the variants whose "
        "label contains one (e.g. 'bare Kvaerno5,auto groups').",
    )
    a = ap.parse_args()
    only = tuple(sub for sub in a.only.split(",") if sub)
    for case in a.cases:
        if case.startswith("chain:") and case[6:].isdigit():
            n = int(case[6:])
            build = lambda n=n: chain_composite(n)  # noqa: E731
            span = (0.0, a.span or 40.0)
            macro_dt = a.macro_dt or 1.0
        elif case in CASES:
            build, span = CASES[case]
            macro_dt = a.macro_dt or 0.5
        else:
            raise SystemExit(
                f"unknown case {case!r}; cases are "
                f"{', '.join(CASES)} or chain:<n> (one token each)"
            )
        # Interpolated coupling saves 16 points per window; a save grid of
        # macro_dt / 15 is the one every variant's grid contains.
        save_dt = a.save_dt or macro_dt / 15.0
        run_case(
            case, build, span, macro_dt, save_dt, a.repeats, only, a.reference
        )


if __name__ == "__main__":
    main()
