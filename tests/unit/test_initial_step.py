"""A group's first step comes from its field at the launch state."""

import numpy as np

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class StiffCubic(Process):
    """dx/dt = -k x³ from x = 10: a launch a thousand times faster than any
    fixed guess of the first step."""

    k: float = 1e6
    timescale: float = 1e-6

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=10.0)}

    def derivative(self, t, state):
        return {"x": -self.k * state["x"] ** 3}


class Slow(Process):
    timescale: float = 1.0

    def ports_schema(self):
        return {"z": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"z": -state["z"]}


def _run(**scheduler_kwargs):
    comp = Composite(
        processes={"fast": StiffCubic(), "slow": Slow()},
        topology={"fast": {"x": "x"}, "slow": {"z": "z"}},
        validate=False,
        semantic_validation=False,
    )
    return Scheduler(**scheduler_kwargs).run(
        comp, t_span=(0.0, 1.0), macro_dt=0.5, save_dt=0.5
    )


def _rejected(result):
    return sum(
        int(np.asarray(v["num_rejected_steps"]).sum())
        for v in result.stats.values()
        if isinstance(v, dict) and "num_rejected_steps" in v
    )


def test_the_default_first_step_opens_a_stiff_launch_without_rejections():
    assert Scheduler().dt0 is None
    chosen, pinned = _run(), _run(dt0=1e-3)
    for r in (chosen, pinned):
        assert np.isfinite(np.asarray(r.ys)).all()
    np.testing.assert_allclose(
        np.asarray(chosen.ys[-1]),
        np.asarray(pinned.ys[-1]),
        rtol=1e-4,
        atol=1e-8,
    )
    assert _rejected(pinned) > 0
    assert _rejected(chosen) < _rejected(pinned)


def test_the_eager_lane_lets_the_controller_choose_too():
    chosen = _run(progress=True)
    assert np.isfinite(np.asarray(chosen.ys)).all()
    np.testing.assert_allclose(
        np.asarray(chosen.ys[-1]),
        np.asarray(_run().ys[-1]),
        rtol=1e-4,
        atol=1e-8,
    )
