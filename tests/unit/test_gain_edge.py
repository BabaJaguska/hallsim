"""GainEdge: a linear bridge between two models' scales."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.models.gain_edge import GainEdge, place_gain
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class Ramp(Process):
    """dx/dt = rate, so the edge's source is a known line in time."""

    rate: float = 1.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

    def derivative(self, t, state):
        return {"x": self.rate}


class Reader(Process):
    """Integrates whatever level it is handed: y(T) = ∫ signal dt."""

    def ports_schema(self):
        return {
            "signal": Port(role=PortRole.INPUT, default=0.0),
            "y": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def derivative(self, t, state):
        return {"y": state["signal"]}


def _run(edge, t_end=2.0):
    composite = Composite(
        processes={"ramp": Ramp(), "edge": edge, "reader": Reader()},
        topology={
            "ramp": {"x": "x"},
            "edge": {"source": "x", edge.out_port: "sig"},
            "reader": {"signal": "sig", "y": "y"},
        },
        validate=False,
    )
    return Scheduler().run(
        composite, t_span=(0.0, t_end), macro_dt=t_end, save_dt=t_end
    )


def test_level_mode_is_offset_plus_gain_times_source():
    # x = t, signal = 0.5 + 3t, y(2) = ∫₀² (0.5 + 3t) dt = 1 + 6 = 7
    res = _run(GainEdge(mode="level", offset=0.5, gain=3.0))
    assert jnp.allclose(res.get("y")[-1], 7.0, rtol=1e-5)


def test_flux_mode_adds_to_the_target():
    edge = GainEdge(mode="flux", offset=0.0, gain=2.0)
    composite = Composite(
        processes={"ramp": Ramp(), "edge": edge},
        topology={"ramp": {"x": "x"}, "edge": {"source": "x", "target": "z"}},
        validate=False,
    )
    res = Scheduler().run(
        composite, t_span=(0.0, 2.0), macro_dt=2.0, save_dt=2.0
    )
    # dz/dt = 2x = 2t → z(2) = 4
    assert jnp.allclose(res.get("z")[-1], 4.0, rtol=1e-5)


def test_gain_is_differentiable():
    def loss(g):
        return jnp.squeeze(
            _run(GainEdge(mode="level", offset=0.0, gain=g)).get("y")[-1]
        )

    # y(2) = g · ∫₀² t dt = 2g → dy/dg = 2
    assert jnp.allclose(jax.grad(loss)(jnp.asarray(1.0)), 2.0, rtol=1e-4)


def test_place_gain_maps_rest_to_rest():
    assert place_gain(source_rest=14.0, target_rest=10.0) == pytest.approx(
        10.0 / 14.0
    )
    with pytest.raises(ValueError):
        place_gain(0.0, 10.0)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        GainEdge(mode="ratio")
