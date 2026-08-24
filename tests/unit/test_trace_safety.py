"""Nothing on a traced path may need a concrete value.

``bool(x)``, ``float(x)`` and ``np.asarray(x)`` raise on a JAX tracer, so a
helper that calls one works until the first time someone puts it inside
``jit`` / ``grad`` / ``vmap`` — typically a calibration loss, several layers
below where they wrote it. Whether a given call site is reachable under trace
is a question about the whole call graph, which no grep answers; these tests
answer it by tracing the public entry points and letting the failure surface.

A new failure here is a real defect, not a test to relax: it means a traced
run reaches code that assumes concreteness. Fix it by deciding which the code
is — a diagnostic degrades (see :mod:`hallsim.tracing`), a computation raises
with an actionable message.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler
from hallsim.steady_state import conservation_laws, steady_state


class Decay(Process):
    rate: float = 0.3

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=2.0)}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


class Reader(Process):
    """Reads the same path with a *different* default, so the contested-default
    warning in ``build_initial_store`` fires on this composite."""

    def ports_schema(self):
        return {"x": Port(role=PortRole.INPUT, default=99.0)}

    def derivative(self, t, state):
        return {}


def _composite():
    return Composite(
        processes={"decay": Decay(), "reader": Reader()},
        topology={"decay": {"x": "p/x"}, "reader": {"x": "p/x"}},
        semantic_validation=False,
    )


def test_initial_state_vec_under_jit():
    """``initial_state_vec`` runs inside a traced loss via ``steady_state``,
    and rebuilds the store — including the contested-default check."""
    comp = _composite()
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    @jax.jit
    def build(p):
        return eqx.combine(p, static).initial_state_vec()

    assert jnp.all(jnp.isfinite(build(params)))


def test_steady_state_under_jit_with_precomputed_laws():
    """Conservation laws are structural, so they are resolved once eagerly and
    passed in. That is the supported traced path."""
    comp = _composite()
    laws = conservation_laws(comp, comp.initial_state_vec())
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    @jax.jit
    def solve(p):
        return steady_state(eqx.combine(p, static), laws=laws)

    assert jnp.all(jnp.isfinite(solve(params)))


def test_conservation_laws_under_trace_raises_actionably():
    """It cannot be computed from tracers, so it must say so — not surface a
    TracerArrayConversionError from three frames down."""
    comp = _composite()
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    with pytest.raises(RuntimeError, match="resolve them once eagerly"):
        jax.jit(lambda p: steady_state(eqx.combine(p, static)))(params)


def test_gradient_through_steady_state():
    comp = _composite()
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    laws = conservation_laws(comp, comp.initial_state_vec())

    def loss(p):
        return jnp.sum(steady_state(eqx.combine(p, static), laws=laws))

    assert jnp.isfinite(
        jnp.sum(jnp.asarray(jax.tree_util.tree_leaves(jax.grad(loss)(params))))
    )


def test_scheduler_run_under_jit_and_grad():
    comp = _composite()
    sched = Scheduler()
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    def summary(p):
        res = sched.run(
            eqx.combine(p, static), t_span=(0.0, 2.0), macro_dt=1.0
        )
        return jnp.sum(res.ys[-1])

    assert jnp.isfinite(jax.jit(summary)(params))
    grads = jax.tree_util.tree_leaves(jax.grad(summary)(params))
    assert all(jnp.all(jnp.isfinite(g)) for g in grads)


def test_scheduler_run_under_vmap():
    comp = _composite()
    sched = Scheduler()

    def run(rate):
        c = eqx.tree_at(lambda c: c.processes["decay"].rate, comp, rate)
        return sched.run(c, t_span=(0.0, 2.0), macro_dt=1.0).get("p/x")[-1]

    out = jax.vmap(run)(jnp.array([0.1, 0.2, 0.3]))
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("transform", [jax.jit, jax.vmap])
def test_contested_default_warning_never_raises(transform):
    """The warning path in ``build_initial_store`` compares two defaults. It
    must degrade under trace rather than raise — a warning that breaks a loss
    function is worse than no warning."""
    comp = _composite()
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    def build(p):
        return eqx.combine(p, static).initial_state_vec()

    if transform is jax.vmap:
        stacked = jax.tree_util.tree_map(lambda v: jnp.stack([v, v]), params)
        assert jnp.all(jnp.isfinite(jax.vmap(build)(stacked)))
    else:
        assert jnp.all(jnp.isfinite(jax.jit(build)(params)))
