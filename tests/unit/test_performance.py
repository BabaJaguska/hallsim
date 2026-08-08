"""Performance regressions, asserted structurally rather than by wall clock.

Wall-clock thresholds are machine-dependent and flaky; the properties that
actually decide whether HallSim is fast are deterministic:

- a repeated or parameter-swept run reuses the compiled solve (0 recompiles),
- ``build_rhs`` returns a structurally stable pytree, so the JIT cache can hit,
- Process parameters are traced arrays, not values baked into the executable,
- a batched implicit run's compiled FLOP count is linear in the batch size.

Every compile-count test first asserts the *cold* run does compile, so a change
in JAX's log format fails the test instead of silently reporting zero.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class Decay(Process):
    timescale: float = 1.0
    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=10.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


def _composite():
    return Composite(
        processes={"decay": Decay()}, topology={"decay": {"x": "pool/x"}}
    )


@contextmanager
def count_compiles():
    """Count XLA compilations triggered inside the block."""
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler, log = Capture(), logging.getLogger("jax")
    prev = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    counter: list[int] = []
    try:
        with jax.log_compiles():
            yield counter
    finally:
        log.removeHandler(handler)
        log.setLevel(prev)
        counter.append(sum("Finished XLA compilation" in m for m in records))


def _run(sched, comp):
    return sched.run(comp, (0.0, 10.0), macro_dt=10.0, save_dt=1.0).ys


def test_build_rhs_is_structurally_stable():
    """Two builds of the same composite share a treedef — otherwise every
    solve is a JIT cache miss."""
    comp = _composite()
    a, _ = comp.build_rhs()
    b, _ = comp.build_rhs()
    assert jtu.tree_structure(a) == jtu.tree_structure(b)


def test_numeric_params_are_traced_arrays():
    """A Python float parameter would be static: baked into the executable,
    so every distinct value recompiles."""
    proc = Decay(rate=0.3)
    assert eqx.is_array(proc.rate)
    # timescale drives host-side grouping and stays a Python scalar.
    assert not eqx.is_array(proc.timescale)


def test_repeated_identical_run_does_not_recompile():
    comp, sched = _composite(), Scheduler()
    with count_compiles() as cold:
        jax.block_until_ready(_run(sched, comp))
    assert cold[0] > 0, "counter is broken: a cold run must compile"

    with count_compiles() as warm:
        jax.block_until_ready(_run(sched, comp))
    assert warm[0] == 0, f"identical rerun recompiled {warm[0]}x"


def test_sbml_reimport_reuses_generated_class():
    """Codegen is cached per source file. A fresh generated class per import
    is a distinct pytree node type, so no compiled solve can be reused across
    two imports of the same model."""
    from hallsim.models.multi_hallmark import GZ06_SBML_PATH
    from hallsim.sbml_import import process_from_sbml

    a = process_from_sbml(str(GZ06_SBML_PATH), name="gz06")
    b = process_from_sbml(str(GZ06_SBML_PATH), name="gz06")
    assert type(a._model) is type(b._model)
    assert jtu.tree_structure(a) == jtu.tree_structure(b)


def _flops_per_member(sched, comp, n_batch):
    y0 = comp.initial_state_vec()
    yb = jnp.broadcast_to(y0, (n_batch, y0.shape[0]))
    lowered = jax.jit(
        lambda y: sched.run(comp, (0.0, 10.0), macro_dt=10.0, y0=y).ys
    ).lower(yb)
    return lowered.compile().cost_analysis()["flops"] / n_batch


def test_batched_implicit_solve_is_linear_in_batch_size():
    """Batch members are independent, so the compiled FLOP count must be
    linear in the batch size.

    Handed a batch as one flat ``(batch, n_vars)`` state, an implicit solver
    treats it as a single unknown vector and factorizes a dense
    ``(batch·n_vars)²`` Jacobian — cubic in population size, and at N=128 that
    measured 76x *slower* than a Python loop over members. Pins the implicit
    solver rather than relying on the stiffness verdict; an explicit solver has
    no stage solve and would pass either way.

    Discrimination: without the per-member map this is 11.3x at N=16, not 1.0x.
    """
    comp = _composite()
    sched = Scheduler(solver=dfx.Kvaerno5())
    solo = _flops_per_member(sched, comp, 1)
    assert _flops_per_member(sched, comp, 16) / solo < 1.5


def test_parameter_change_does_not_recompile():
    """Sweeping a parameter reuses the executable: the value is data."""
    comp, sched = _composite(), Scheduler()
    swept = [
        eqx.tree_at(lambda c: c.processes["decay"].rate, comp, jnp.asarray(r))
        for r in (0.2, 0.3, 0.4)
    ]
    jax.block_until_ready(_run(sched, swept[0]))  # absorb the first compile

    with count_compiles() as n:
        for c in swept[1:]:
            jax.block_until_ready(_run(sched, c))
    assert n[0] == 0, f"parameter sweep recompiled {n[0]}x"
