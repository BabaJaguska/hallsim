"""A piecewise-linear forcing source: its shape, its symbolic form across
every segment, and a run through the Scheduler that lands on its knots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy

from hallsim.composite import Composite
from hallsim.models.forcing import ProfileSource
from hallsim.process import Port, PortRole, Process
from hallsim.sbml_math import TIME
from hallsim.scheduler import Scheduler

#: An incremental test: rest, a ten-unit ramp to full, then an abrupt stop.
RAMP = ProfileSource(times=(2.0, 12.0, 12.0), levels=(0.0, 1.0, 0.0))


def _signal(proc, t):
    return float(proc.assign(jnp.asarray(t), {})["signal"])


def test_the_ramp_rises_linearly_and_stops_at_once():
    assert _signal(RAMP, 0.0) == 0.0
    assert _signal(RAMP, 7.0) == pytest.approx(0.5)
    assert _signal(RAMP, 11.999) == pytest.approx(0.9999)
    assert _signal(RAMP, 12.0) == 0.0
    assert _signal(RAMP, 30.0) == 0.0


def test_dose_and_amplitude_scale_the_shape():
    proc = ProfileSource(
        amplitude=0.5, dose=4.0, times=(0.0, 1.0), levels=(0.0, 1.0)
    )
    assert _signal(proc, 0.5) == pytest.approx(1.0)


def test_a_rectangle_is_four_knots():
    box = ProfileSource(times=(1.0, 1.0, 3.0, 3.0), levels=(0, 1, 1, 0))
    assert [_signal(box, t) for t in (0.5, 1.0, 2.9, 3.0)] == [0, 1, 1, 0]


@pytest.mark.parametrize(
    "proc",
    [
        RAMP,
        ProfileSource(times=(0.0, 0.5, 0.6, 0.9), levels=(0.1, 0.8, 0.3, 0.3)),
        ProfileSource(times=(1.0, 1.0, 3.0, 3.0), levels=(0, 1, 1, 0)),
    ],
)
def test_the_symbolic_form_matches_on_every_segment(proc):
    ((_, expr),) = proc.assignment_rules()
    lo, hi = proc.times[0] - 1.0, proc.times[-1] + 1.0
    for t in np.linspace(lo, hi, 97):
        subs = {TIME: sympy.Float(t), sympy.Symbol("amplitude"): 1.0}
        assert float(expr.xreplace(subs)) == pytest.approx(
            _signal(proc, t), abs=1e-12
        )


def test_the_knots_are_discontinuities_once_each():
    assert RAMP.discontinuity_times() == (2.0, 12.0)


def test_mismatched_or_decreasing_knots_are_refused():
    with pytest.raises(ValueError, match="same non-empty length"):
        ProfileSource(times=(0.0, 1.0), levels=(0.0,))
    with pytest.raises(ValueError, match="must not decrease"):
        ProfileSource(times=(1.0, 0.0), levels=(0.0, 1.0))


def test_the_amplitude_gradient_is_the_shape():
    g = jax.grad(
        lambda a: RAMP.__class__(
            amplitude=a, times=RAMP.times, levels=RAMP.levels
        ).assign(jnp.asarray(7.0), {})["signal"]
    )(1.0)
    assert float(g) == pytest.approx(0.5)


def test_integrating_the_ramp_gives_its_area():
    """x' = signal, so x at the end is the ramp's area: half of ten."""

    class _Fed(Process):
        def ports_schema(self):
            return {
                "x": Port(role=PortRole.EVOLVED, default=0.0),
                "u": Port(role=PortRole.INPUT, default=0.0),
            }

        def derivative(self, t, state):
            return {"x": state["u"]}

    comp = Composite(
        processes={"fed": _Fed(), "work": RAMP},
        topology={"fed": {"x": "p/x", "u": "p/u"}, "work": {"signal": "p/u"}},
        semantic_validation=False,
    )
    out = Scheduler().run(comp, t_span=(0.0, 20.0), macro_dt=2.0, save_dt=2.0)
    assert float(out.get("p/x")[-1]) == pytest.approx(5.0, rel=1e-4)
