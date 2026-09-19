"""A reused Scheduler runs each composite's discrete and event processes at
that composite's parameters on the compiled hybrid lane.

The compiled core is cached per structure and reused across parameter
values. The continuous groups always read the composite handed to the call;
a discrete or event process bound from the closure the core was built with
would run every arm of a sweep at the first arm's parameters (the reaction
lane's regression lives in ``test_stochastic``).
"""

import jax.numpy as jnp
import numpy as np

from hallsim.composite import Composite
from hallsim.models.kick_event import KickEvent
from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.scheduler import Scheduler
import equinox as eqx


class Relax(Process):
    """dx/dt = -x."""

    timescale: float = eqx.field(static=True, default=1.0)

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -state["x"]}


class Tick(Process):
    """Adds ``step`` to a latched counter every ``dt_step``."""

    kind: ProcessKind = ProcessKind.DISCRETE
    dt_step: float = eqx.field(static=True, default=0.5)
    step: float = 1.0

    def ports_schema(self):
        return {"n": Port(role=PortRole.LATCHED, default=0.0)}

    def update(self, t, state):
        return {"n": jnp.asarray(self.step)}


def _composite(kick: float, step: float) -> Composite:
    return Composite(
        processes={
            "relax": Relax(),
            "kick": KickEvent(kick_time=0.5, deltas={"x": kick}),
            "tick": Tick(step=step),
        },
        topology={"relax": {"x": "x"}, "kick": {"x": "x"}, "tick": {"n": "n"}},
        validate=False,
        semantic_validation=False,
    )


RUN = dict(t_span=(0.0, 1.0), macro_dt=0.25, save_dt=0.25)


def _final(result, comp):
    keys = comp.store_keys()
    ys = np.asarray(result.ys)
    return float(ys[-1, keys.index("x")]), float(ys[-1, keys.index("n")])


def test_a_reused_scheduler_reads_discrete_and_event_parameters_per_call():
    loud, quiet = _composite(5.0, 1.0), _composite(0.0, 0.0)
    shared = Scheduler()
    assert shared.plan(loud, **RUN).core is not None  # the compiled lane
    x_loud, n_loud = _final(shared.run(loud, **RUN), loud)
    x_quiet, n_quiet = _final(shared.run(quiet, **RUN), quiet)

    assert n_loud > 0 and n_quiet == 0
    assert np.isclose(x_quiet, np.exp(-1.0), rtol=1e-4)
    assert x_loud > x_quiet + 1.0

    x_fresh, n_fresh = _final(Scheduler().run(quiet, **RUN), quiet)
    assert (x_quiet, n_quiet) == (x_fresh, n_fresh)


def test_the_order_of_the_sweep_does_not_matter():
    loud, quiet = _composite(5.0, 1.0), _composite(0.0, 0.0)
    shared = Scheduler()
    x_quiet, n_quiet = _final(shared.run(quiet, **RUN), quiet)
    x_loud, n_loud = _final(shared.run(loud, **RUN), loud)
    assert n_quiet == 0 and n_loud > 0
    assert x_loud > x_quiet + 1.0
