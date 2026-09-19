"""The eager macro loop compiles each group's solve once, not once per
window: the window bounds reach the solve as arrays, never as Python floats
baked into the trace."""

import logging

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class Fast(Process):
    timescale: float = 1.0
    rate: float = 0.5

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


class SlowReader(Process):
    """Own group (far timescale) and reads the fast one: a forward edge
    across groups, which is what makes the loop interpolate."""

    timescale: float = 1000.0
    rate: float = 1e-3

    def ports_schema(self):
        return {
            "z": Port(role=PortRole.EVOLVED, default=1.0),
            "u": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"z": -self.rate * state["z"] + 1e-3 * state["u"]}


def _comp():
    """Two groups, a forward edge across them and an event: the shape that
    takes the eager macro loop rather than the compiled scan."""
    from hallsim.models.kick_event import KickEvent

    return Composite(
        processes={
            "fast": Fast(),
            "slow": SlowReader(),
            "kick": KickEvent(kick_time=3.0, deltas={"x": 1.0}),
        },
        topology={
            "fast": {"x": "a/x"},
            "slow": {"z": "b/z", "u": "a/x"},
            "kick": {"x": "a/x"},
        },
        semantic_validation=False,
    )


def _run(comp, n_steps):
    return Scheduler().run(
        comp, t_span=(0.0, 2.0 * n_steps), macro_dt=2.0, save_dt=2.0
    )


#: What a run with a new trajectory shape may compile: elementwise and
#: reduction ops over the saved trajectory in the diagnosis, never a solve.
SHAPE_ONLY = {
    "jit(convert_element_type)",
    "jit(stack)",
    "jit(abs)",
    "jit(isfinite)",
    "jit(_where)",
    "jit(_reduce_max)",
    "jit(_reduce_any)",
    "jit(invert)",
    "jit(broadcast_in_dim)",
    "jit(select_n)",
}


def test_more_macro_windows_compile_no_solve(caplog):
    comp = _comp()
    assert len(comp.auto_groups()) == 2
    _run(comp, 3)  # every solve this composite needs is compiled here
    jax.config.update("jax_log_compiles", True)
    try:
        with caplog.at_level(logging.DEBUG, logger="jax._src.dispatch"):
            res = _run(comp, 6)  # windows (6, 8), (8, 10), (10, 12) are new
    finally:
        jax.config.update("jax_log_compiles", False)
    compiled = [
        r.getMessage().split(" of ", 1)[1].split(" in ")[0]
        for r in caplog.records
        if "Finished XLA compilation" in r.getMessage()
    ]
    unexpected = [c for c in compiled if c not in SHAPE_ONLY]
    assert unexpected == [], unexpected
    assert bool(jnp.all(jnp.isfinite(res.ys)))
    # decay from 1, plus the kick of 1 which lands at the first sync point
    # after t = 3, that is t = 4; both decay at 0.5 until t = 12
    expected = jnp.exp(-0.5 * 12.0) + jnp.exp(-0.5 * 8.0)
    assert float(res.get("a/x")[-1]) == pytest.approx(
        float(expected), rel=1e-3
    )
