"""Where a composite's derivative spends its time.

One SBML model (DallePezze 2014, 23 species), one solver, one root finder,
one controller — and five ways of presenting the same field to it:

- ``program``: the member's compiled program called bare (``y``, ``w``,
  ``c``, ``t``), no composite in the loop.
- ``generated``: the field re-emitted from its symbolic form as a single
  CSE'd function of ``(t, y)``.
- ``flat``: the composite's flat RHS reduced to the evolved states — what
  the Scheduler integrates.
- ``flat, no assign pass``: the same with the assignment-rule pass removed,
  to price that pass alone.
- ``Scheduler``: ``Scheduler.run`` on the fast path.

Per form: jaxpr size, one call, a batched call, and the compiled solve at
two tolerances with step counts and final-state agreement. The difference
between neighbouring rows is the cost of the layer between them.

Run with ``HALLSIM_COMPILATION_CACHE_DIR=off`` so cold times are real.
"""

from __future__ import annotations

import argparse
import logging
import time

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy
from sympy.printing.numpy import JaxPrinter

from hallsim.composite import _FlatRHS, single_process_composite
from hallsim.config import DEFAULT_NEWTON_ATOL
from hallsim.root_finders import Chord
from hallsim.sbml_import import process_from_sbml
from hallsim.sbml_math import TIME
from hallsim.scheduler import Scheduler, _FrozenFill, _ReducedRHS
from hallsim.structure import symbolic_field

logging.disable(logging.WARNING)

SBML = "demos/models/sbml/dallepezze2014/dallepezze2014_BIOMD0000000582.xml"
T_END, N_SAVE, BATCH = 14.0, 141, 256
TS = jnp.linspace(0.0, T_END, N_SAVE)
TOLERANCES = (("matched 1e-10/1e-12", 1e-10, 1e-12), ("defaults", 1e-6, 1e-9))


def n_eqns(jaxpr) -> int:
    """Equations in a jaxpr, sub-jaxprs included."""
    try:
        from jax.extend.core import ClosedJaxpr, Jaxpr
    except ImportError:  # older jax
        from jax.core import ClosedJaxpr, Jaxpr
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for v in eqn.params.values():
            if isinstance(v, ClosedJaxpr):
                n += n_eqns(v.jaxpr)
            elif isinstance(v, Jaxpr):
                n += n_eqns(v)
    return n


def call_us(f, *args, reps=300) -> float:
    jf = jax.jit(f)
    out = jf(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = jf(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps * 1e6


def timed(fn, *args, repeats=5):
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


def solver_for(rtol: float, atol: float):
    """The Scheduler's own solver stack, built by hand."""
    root = Chord(rtol=rtol, atol=DEFAULT_NEWTON_ATOL)
    return dfx.Kvaerno5(root_finder=root), dfx.PIDController(
        rtol=rtol, atol=atol
    )


def bare_solve(rhs, rtol, atol):
    solver, ctrl = solver_for(rtol, atol)
    term = dfx.ODETerm(lambda t, y, args: rhs(t, y))

    @jax.jit
    def run(y):
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=T_END,
            dt0=None,
            y0=y,
            saveat=dfx.SaveAt(ts=TS),
            stepsize_controller=ctrl,
            max_steps=300_000,
            throw=False,
        )
        return sol.ys, sol.stats["num_steps"]

    return run


def generated_field(comp, keys, species_keys):
    """The composite's field as one generated function of ``(t, y)`` over
    ``species_keys``, common subexpressions named once."""
    field = symbolic_field(comp, keys)
    values = {
        sympy.Symbol(n): sympy.Float(float(v), 17)
        for n, v in field.parameters.items()
    }
    Y, T = sympy.IndexedBase("y"), sympy.Symbol("t")
    paths = {sympy.Symbol(k): Y[i] for i, k in enumerate(species_keys)}
    exprs = []
    for k in species_keys:
        e = sympy.sympify(field.derivatives[k])
        e = e.xreplace(values).xreplace(paths).xreplace({TIME: T})
        stray = {
            s
            for s in e.free_symbols
            if not isinstance(s, sympy.Indexed) and s.name not in ("t", "y")
        }
        assert not stray, (k, stray)
        exprs.append(e)
    printer = JaxPrinter(
        {"fully_qualified_modules": False, "inline": True, "precision": 17}
    )
    reps, reduced = sympy.cse(exprs)
    lines = ["import jax.numpy as jnp", "from jax.numpy import *", ""]
    lines.append("def f(t, y):")
    for s, e in reps:
        lines.append(f"    {s} = {printer.doprint(e)}")
    body = ", ".join(printer.doprint(e) for e in reduced)
    lines.append(f"    return jnp.stack([{body}])")
    ns: dict = {}
    exec("\n".join(lines) + "\n", ns)
    return ns["f"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-solve", action="store_true", help="RHS rows only")
    a = ap.parse_args()

    proc = process_from_sbml(SBML, name="dp14")
    comp = single_process_composite(proc)
    keys = comp.store_keys()
    y0_full = comp.initial_state_vec(keys)
    own = comp.evolved_indices()
    species_keys = [keys[int(i)] for i in own]
    y0 = y0_full[own]

    # the member's program, bare: composite order → its own species order
    core, w0, scale = proc._model, proc._w0, proc.time_scale
    c0 = proc._constants(0.0)
    names = list(proc._species_names)
    perm = np.asarray([names.index(k.split("/", 1)[1]) for k in species_keys])
    inv = jnp.asarray(np.argsort(perm))
    perm = jnp.asarray(perm)
    frozen = jnp.asarray(proc._frozen_indices, dtype=int)

    def program(t, y):
        yp = y[..., inv]
        tn = t * scale
        w = core.boundaryfunc(yp, w0, c0, tn)
        dy = core.ratefunc(yp, tn, w, c0) * scale
        if frozen.size:
            dy = dy.at[..., frozen].set(0.0)
        return dy[..., perm]

    rhs, _ = comp.build_rhs()
    flat = _ReducedRHS(base=rhs, own=own, fill=_FrozenFill(y0_full))
    no_assign = _ReducedRHS(
        base=_FlatRHS(
            procs=rhs.procs,
            read_maps=rhs.read_maps,
            write_maps=rhs.write_maps,
            assign_procs=(),
        ),
        own=own,
        fill=_FrozenFill(y0_full),
    )
    generated = generated_field(comp, keys, species_keys)

    forms = [
        ("program", program),
        ("generated", generated),
        ("flat", lambda t, y: flat(t, y)),
        ("flat, no assign pass", lambda t, y: no_assign(t, y)),
    ]

    ref = np.asarray(program(0.0, y0))
    yb = jnp.tile(y0[None], (BATCH, 1))
    print(f"DallePezze 2014: {len(species_keys)} species, {len(keys)} slots")
    print("\n| form | jaxpr eqns | one call | batched 256 | max rel diff |")
    print("|---|---|---|---|---|")
    for label, f in forms:
        got = np.asarray(f(0.0, y0))
        diff = float(
            np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300))
        )
        eqns = n_eqns(jax.make_jaxpr(f)(0.0, y0).jaxpr)
        one = call_us(f, 0.0, y0)
        batched = call_us(jax.vmap(f, in_axes=(None, 0)), 0.0, yb, reps=100)
        print(
            f"| {label} | {eqns} | {one:.0f} µs | {batched:.0f} µs | {diff:.1e} |",
            flush=True,
        )
    if a.no_solve:
        return

    for tol_label, rtol, atol in TOLERANCES:
        print(
            f"\n### solve, {tol_label}, {N_SAVE} saved points, Kvaerno5 + chord"
        )
        print("\n| form | warm | cold | steps | max rel diff at day 14 |")
        print("|---|---|---|---|---|")
        ref_final = None
        for label, f in forms:
            (ys, steps), cold, warm = timed(
                bare_solve(f, rtol, atol), y0, repeats=a.repeats
            )
            final = np.asarray(ys[-1])
            if ref_final is None:
                ref_final = final
            diff = float(
                np.max(
                    np.abs(final - ref_final)
                    / np.maximum(np.abs(ref_final), 1e-300)
                )
            )
            print(
                f"| {label} | {warm * 1e3:.1f} ms | {cold:.1f} s | "
                f"{int(steps)} | {diff:.1e} |",
                flush=True,
            )
        sched = Scheduler(
            rtol=rtol, atol=atol, auto_stiffness=False, solver=dfx.Kvaerno5()
        )
        save_dt = T_END / (N_SAVE - 1)

        @jax.jit
        def run_sched(y):
            res = sched.run(
                comp,
                t_span=(0.0, T_END),
                macro_dt=T_END,
                save_dt=save_dt,
                y0=y,
            )
            return res.ys, sum(
                v["num_steps"]
                for v in res.stats.values()
                if isinstance(v, dict) and "num_steps" in v
            )

        (ys, steps), cold, warm = timed(run_sched, y0_full, repeats=a.repeats)
        final = np.asarray(ys[-1])[np.asarray(own)]
        diff = float(
            np.max(
                np.abs(final - ref_final)
                / np.maximum(np.abs(ref_final), 1e-300)
            )
        )
        print(
            f"| Scheduler | {warm * 1e3:.1f} ms | {cold:.1f} s | {int(steps)} | "
            f"{diff:.1e} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
