"""The RHS carries an assignment only when something in the loop reads it."""

import numpy as np
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process


class Algebraic(Process):
    """Evolves x and publishes s = 2x, which its own derivative never reads."""

    timescale: float = 1.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=1.0),
            "s": Port(role=PortRole.ASSIGNED, default=0.0),
        }

    def assign(self, t, state):
        return {"s": 2.0 * state["x"]}

    def derivative(self, t, state):
        return {"x": -state["x"]}


class Reader(Process):
    """dz/dt = s - z through an INPUT port."""

    timescale: float = 1.0

    def ports_schema(self):
        return {
            "z": Port(role=PortRole.EVOLVED, default=0.5),
            "s": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"z": state["s"] - state["z"]}


class Sweeper(Process):
    """Reads whatever it is given, by iteration."""

    timescale: float = 1.0

    def ports_schema(self):
        return {
            "z": Port(role=PortRole.EVOLVED, default=0.5),
            "s": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"z": sum(v for k, v in state.items() if k != "z")}


def _comp(procs, topo):
    return Composite(
        processes=procs,
        topology=topo,
        validate=False,
        semantic_validation=False,
    )


def test_an_assignment_nobody_reads_leaves_the_rhs():
    comp = _comp({"a": Algebraic()}, {"a": {"x": "x", "s": "s"}})
    rhs, keys = comp.build_rhs()
    assert rhs.assign_procs == ()
    y0 = comp.initial_state_vec(keys)
    assert float(rhs(0.0, y0)[keys.index("x")]) == pytest.approx(-1.0)
    # the saved trajectory still gets it
    ys = comp.materialize_assigned(np.array([0.0]), y0[None])
    assert float(ys[0, keys.index("s")]) == pytest.approx(2.0)


@pytest.mark.parametrize("reader", [Reader, Sweeper])
def test_an_assignment_a_derivative_reads_stays(reader):
    comp = _comp(
        {"a": Algebraic(), "r": reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs, keys = comp.build_rhs()
    assert len(rhs.assign_procs) == 1
    y0 = comp.initial_state_vec(keys)
    dz = float(rhs(0.0, y0)[keys.index("z")])
    expect = 2.0 - 0.5 if reader is Reader else 2.0
    assert dz == pytest.approx(expect)


def test_a_partial_rhs_keeps_only_what_its_group_reads():
    comp = _comp(
        {"a": Algebraic(), "r": Reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs_a, _ = comp.build_rhs(["a"])
    assert rhs_a.assign_procs == ()
