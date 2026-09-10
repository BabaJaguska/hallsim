"""Coupling ablation: the same models with the edges carrying no signal."""

import jax.numpy as jnp
import pytest

from hallsim.ablation import (
    coupling_edges,
    declared_levels,
    freeze_coupling,
    trajectory_levels,
)
from hallsim.composite import Composite
from hallsim.models.gain_edge import GainEdge
from hallsim.models.hill_edge import HillEdge
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
    """y(T) = ∫ signal dt — a level that never varies integrates to a line."""

    def ports_schema(self):
        return {
            "signal": Port(role=PortRole.INPUT, default=0.0),
            "y": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def derivative(self, t, state):
        return {"y": state["signal"]}


def _composite(edge):
    return Composite(
        processes={"ramp": Ramp(), "edge": edge, "reader": Reader()},
        topology={
            "ramp": {"x": "x"},
            "edge": {"source": "x", edge.out_port: "sig"},
            "reader": {"signal": "sig", "y": "y"},
        },
        validate=False,
    )


def _run(composite, t_end=2.0, save_dt=None):
    return Scheduler().run(
        composite,
        t_span=(0.0, t_end),
        macro_dt=t_end,
        save_dt=save_dt or t_end,
    )


class TestFrozenEdgesEmitAConstant:
    def test_a_gain_edge_holds_the_value_it_had_at_the_given_level(self):
        edge = GainEdge(mode="level", offset=1.0, gain=2.0)
        frozen = edge.frozen_at({"source": 3.0})

        assert float(frozen.gain) == 0.0
        assert float(frozen.offset) == pytest.approx(7.0)

    def test_a_hill_edge_holds_the_value_it_had_at_the_given_level(self):
        edge = HillEdge(mode="level", basal=0.0, hi=10.0, K=(1.0,), n=(2.0,))
        at_K = float(edge._value({"source": jnp.asarray(1.0)}))
        frozen = edge.frozen_at({"source": jnp.asarray(1.0)})

        assert float(frozen.basal) == pytest.approx(at_K)
        assert float(frozen.hi) == pytest.approx(at_K)

    def test_a_frozen_edge_stops_following_its_source(self):
        wired = _composite(GainEdge(mode="level", offset=0.0, gain=1.0))
        # source x(t) = t, so wired gives y = ∫t dt = 2 over [0, 2].
        assert float(_run(wired).get("y")[-1]) == pytest.approx(2.0, rel=1e-4)

        null = freeze_coupling(wired, {"edge": {"source": 1.0}})
        # Frozen at x = 1 the edge emits 1 for all t, so y = 2 as well — the
        # level is preserved, only the variation is gone.
        assert float(_run(null).get("y")[-1]) == pytest.approx(2.0, rel=1e-4)

        null_hi = freeze_coupling(wired, {"edge": {"source": 4.0}})
        assert float(_run(null_hi).get("y")[-1]) == pytest.approx(
            8.0, rel=1e-4
        )


class TestLevels:
    def test_declared_levels_read_the_source_port_default(self):
        comp = _composite(GainEdge(mode="level", gain=1.0))

        assert declared_levels(comp) == {"edge": {"source": 0.0}}

    def test_trajectory_levels_average_what_the_edge_actually_saw(self):
        comp = _composite(GainEdge(mode="level", gain=1.0))
        result = _run(comp, t_end=2.0, save_dt=0.01)

        # x(t) = t over [0, 2]; its mean is 1.
        assert trajectory_levels(comp, result)["edge"][
            "source"
        ] == pytest.approx(1.0, rel=1e-3)


class TestItRefusesToGuess:
    def test_a_missing_level_names_the_edge_it_is_missing_for(self):
        comp = _composite(GainEdge(mode="level", gain=1.0))

        with pytest.raises(
            ValueError, match="no level given for \\['edge'\\]"
        ):
            freeze_coupling(comp, {})

    def test_a_composite_with_no_coupling_says_so(self):
        comp = Composite(
            processes={"ramp": Ramp()},
            topology={"ramp": {"x": "x"}},
            validate=False,
        )

        assert coupling_edges(comp) == {}
        with pytest.raises(ValueError, match="no process .* exposes"):
            freeze_coupling(comp, {})
