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


# The only numeric fields that belong on the static side, per class: port
# defaults (``ports_schema()`` must stay concrete under a trace) and
# discontinuity times (read as Python floats to build the solver's jump_ts,
# and branched on in Python). A rate constant here instead is on the wrong
# side of the split — it bakes into the executable, recompiles on every
# distinct value, and is invisible to ``jax.grad``.
STRUCTURAL_NUMERIC_FIELDS = {
    "BistableLatch": {"latch_default", "target_default"},
    "ClampEdge": {"target_default"},
    # target_default defaults to None (abstains), so it is not numeric here.
    "HillActivationEdge": set(),
    "PulseSource": {"t_start", "t_end"},
    "StepSource": {"t_step", "signal_default"},
}


def _model_process_classes():
    """Every Process subclass defined under ``hallsim.models``."""
    import importlib
    import pkgutil

    import hallsim.models as models

    for mod in pkgutil.iter_modules(models.__path__):
        importlib.import_module(f"hallsim.models.{mod.name}")

    found: dict[str, type] = {}

    def walk(cls):
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("hallsim.models"):
                found[sub.__qualname__] = sub
            walk(sub)

    walk(Process)
    return found


def test_no_model_hides_a_rate_constant_on_the_static_side():
    """Sweeps every model class, not just a toy one.

    ``test_numeric_params_are_traced_arrays`` checks the mechanism on one
    hand-written Process; this checks that no *shipped* model opted out of it.
    A new static numeric field fails here until it is declared structural
    above, which makes the static/traced call a conscious one.
    """
    import dataclasses

    classes = _model_process_classes()
    assert classes, "no model Process subclasses discovered"

    offenders = {}
    for name, cls in sorted(classes.items()):
        static_numeric = {
            f.name
            for f in dataclasses.fields(cls)
            if f.metadata.get("static")
            and type(f.default) in (float, int)
            and not isinstance(f.default, bool)
        }
        expected = STRUCTURAL_NUMERIC_FIELDS.get(name, set())
        if static_numeric != expected:
            offenders[name] = {
                "unexpected": sorted(static_numeric - expected),
                "no longer present": sorted(expected - static_numeric),
            }
    assert not offenders, offenders


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
    from demos.models.multi_hallmark import GZ06_SBML_PATH
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


def test_with_params_yields_a_traced_array():
    """The supported setter coerces, as construction does: ``eqx.tree_at``
    rebuilds through ``tree_unflatten``, which skips ``__check_init__``."""
    comp = _composite().with_params({"decay.rate": 0.25})
    assert eqx.is_array(comp.processes["decay"].rate)


def test_with_params_sweep_does_not_recompile():
    """The same guarantee as ``test_parameter_change_does_not_recompile``, over
    the *public* route and a plain Python float — what a caller actually writes.
    Hand-rolling ``jnp.asarray`` in the test hides the defect it exists to
    catch."""
    comp, sched = _composite(), Scheduler()
    swept = [comp.with_params({"decay.rate": r}) for r in (0.2, 0.3, 0.4)]
    jax.block_until_ready(_run(sched, swept[0]))

    with count_compiles() as n:
        for c in swept[1:]:
            jax.block_until_ready(_run(sched, c))
    assert n[0] == 0, f"with_params sweep recompiled {n[0]}x"


def _field_eqns(n, block):
    """RHS jaxpr size for an N-path field, spelled either way."""
    from hallsim.composite import Composite

    paths = tuple(f"c/g{i}" for i in range(n))

    class Blk(Process):
        n_: int = eqx.field(static=True, default=n)

        def ports_schema(self):
            return {
                "b": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    elements=tuple(f"e{i}" for i in range(self.n_)),
                )
            }

        def derivative(self, t, state):
            return {"b": -0.3 * state["b"]}

    class Sca(Process):
        n_: int = eqx.field(static=True, default=n)

        def ports_schema(self):
            return {
                f"g{i}": Port(role=PortRole.EVOLVED, default=1.0)
                for i in range(self.n_)
            }

        def derivative(self, t, state):
            return {f"g{i}": -0.3 * state[f"g{i}"] for i in range(self.n_)}

    if block:
        comp = Composite(
            processes={"f": Blk()},
            topology={"f": {"b": paths}},
            semantic_validation=False,
        )
    else:
        comp = Composite(
            processes={"f": Sca()},
            topology={"f": {f"g{i}": paths[i] for i in range(n)}},
            semantic_validation=False,
        )
    rhs, _ = comp.build_rhs()
    y = comp.initial_state_vec()
    return len(jax.make_jaxpr(lambda v: rhs(0.0, v))(y).jaxpr.eqns)


def test_block_port_rhs_is_flat_in_width():
    """The point of block ports: graph size stops growing with the field."""
    assert _field_eqns(4, block=True) == _field_eqns(64, block=True)


def test_scalar_port_cost_per_port_does_not_regress():
    """Guards the path every existing model uses.

    Block ports were added by rewriting the scatter, and the first version
    cost the *scalar* path an extra broadcast per port — invisible to a
    correctness suite and to any test that only measures the block path.
    """
    lo, hi = _field_eqns(4, block=False), _field_eqns(64, block=False)
    slope = (hi - lo) / 60
    assert slope <= 6.0, f"scalar cost regressed to {slope:.2f} eqns/port"


def _process_eqns(n):
    """RHS jaxpr size for a composite of ``n`` single-port processes."""
    from hallsim.composite import Composite

    class Node(Process):
        i_: int = eqx.field(static=True, default=0)

        def ports_schema(self):
            return {
                "x": Port(role=PortRole.EVOLVED, default=1.0),
                "left": Port(role=PortRole.INPUT, default=1.0),
            }

        def derivative(self, t, state):
            return {"x": -0.3 * state["x"] + 0.05 * state["left"]}

    paths = tuple(f"r/n{i}" for i in range(n))
    comp = Composite(
        processes={f"p{i}": Node(i_=i) for i in range(n)},
        topology={
            f"p{i}": {"x": paths[i], "left": paths[(i - 1) % n]}
            for i in range(n)
        },
        semantic_validation=False,
    )
    rhs, _ = comp.build_rhs()
    y = comp.initial_state_vec()
    return len(jax.make_jaxpr(lambda v: rhs(0.0, v))(y).jaxpr.eqns)


def test_per_process_cost_does_not_regress():
    """The composition axis, which is what the framework is for.

    ``test_block_port_rhs_is_flat_in_width`` pins the cost of ports *within*
    one process; this pins the cost of *adding a process*. They are different
    axes, and flattening the first left the second at 23 equations per process
    with compile growing superlinearly — invisible to every test that measures
    one process.

    Fusing the per-process scatters into one brought it to 15. The cap catches
    a regression to the per-process scatter (23) while leaving room for an
    honest primitive or two.
    """
    lo, hi = _process_eqns(8), _process_eqns(64)
    slope = (hi - lo) / 56
    assert slope <= 16.0, f"per-process cost regressed to {slope:.2f} eqns"


def test_fused_scatter_still_sums_duplicate_writes():
    """EVOLVED is additive, and one scatter over concatenated indices must sum
    duplicates exactly as N sequential scatters did."""
    from hallsim.composite import Composite

    class Contrib(Process):
        gain: float = 1.0

        def ports_schema(self):
            return {"a": Port(role=PortRole.EVOLVED, default=2.0)}

        def derivative(self, t, state):
            return {"a": self.gain * state["a"]}

    n = 5
    comp = Composite(
        processes={f"c{i}": Contrib(gain=float(i + 1)) for i in range(n)},
        topology={f"c{i}": {"a": "sh/a"} for i in range(n)},
        semantic_validation=False,
    )
    rhs, _ = comp.build_rhs()
    y = comp.initial_state_vec()
    # gains 1..5 sum to 15, state is 2.0
    assert jnp.allclose(rhs(0.0, y), jnp.array([30.0]))
