"""An eager run must not disable forward-mode differentiation afterwards."""

import jax
import jax.numpy as jnp
import numpy as np

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class Decay(Process):
    timescale: float = 1.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -state["x"]}


def test_jacfwd_works_after_an_eager_run():
    comp = Composite(
        processes={"d": Decay()},
        topology={"d": {"x": "x"}},
        validate=False,
        semantic_validation=False,
    )
    sched = Scheduler()
    y0 = jnp.array([2.0])
    eager = sched.run(
        comp, t_span=(0.0, 1.0), macro_dt=1.0, save_dt=1.0, y0=y0
    )
    assert np.isfinite(np.asarray(eager.ys)).all()

    def final(y):
        return sched.run(
            comp, t_span=(0.0, 1.0), macro_dt=1.0, save_dt=1.0, y0=y
        ).ys[-1]

    jac = jax.jacfwd(final)(y0)
    np.testing.assert_allclose(np.asarray(jac), np.exp(-1.0), rtol=1e-4)
