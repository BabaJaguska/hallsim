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
    values = {
        sympy.Symbol(k): sympy.Float(float(v))
        for k, v in proc.symbol_values().items()
    }
    for t in np.linspace(lo, hi, 97):
        subs = {TIME: sympy.Float(t), **values}
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


def test_the_levels_are_traced_and_their_gradient_is_the_hat():
    g = jax.grad(
        lambda ls: ProfileSource(times=RAMP.times, levels=ls).assign(
            jnp.asarray(7.0), {}
        )["signal"]
    )(RAMP.levels)
    assert np.allclose(g, [0.5, 0.5, 0.0])


def test_a_panel_of_levels_is_refused_at_construction():
    with pytest.raises(ValueError, match="with_member_params"):
        ProfileSource(times=(0.0, 1.0), levels=jnp.ones((3, 2)))


class _Fed(Process):
    """x' = u: the integral of whatever drives it."""

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=0.0),
            "u": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"x": state["u"]}


def test_per_member_levels_run_each_member_on_its_own_trace():
    """Two members share the knot times and carry their own levels, so
    each integrates its own ramp's area."""
    comp = Composite(
        processes={
            "fed": _Fed(),
            "work": ProfileSource(times=(0.0, 10.0), levels=(0.0, 1.0)),
        },
        topology={"fed": {"x": "p/x", "u": "p/u"}, "work": {"signal": "p/u"}},
        semantic_validation=False,
    )
    y0 = jnp.tile(comp.initial_state_vec(), (2, 1))
    comp = comp.with_initial(y0).with_member_params(
        {"work.levels": jnp.asarray([[0.0, 1.0], [0.0, 3.0]])}
    )
    out = Scheduler().run(
        comp, t_span=(0.0, 10.0), macro_dt=10.0, save_dt=10.0
    )
    assert np.allclose(out.get("p/x")[-1], [5.0, 15.0], rtol=1e-4)


def test_integrating_the_ramp_gives_its_area():
    """x' = signal, so x at the end is the ramp's area: half of ten."""
    comp = Composite(
        processes={"fed": _Fed(), "work": RAMP},
        topology={"fed": {"x": "p/x", "u": "p/u"}, "work": {"signal": "p/u"}},
        semantic_validation=False,
    )
    out = Scheduler().run(comp, t_span=(0.0, 20.0), macro_dt=2.0, save_dt=2.0)
    assert float(out.get("p/x")[-1]) == pytest.approx(5.0, rel=1e-4)
