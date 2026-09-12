"""A port binds exactly the paths it declares."""

import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process


class Decay(Process):
    timescale: float = 1.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -state["x"]}


class Block(Process):
    timescale: float = 1.0

    def ports_schema(self):
        return {
            "b": Port(role=PortRole.EVOLVED, default=1.0, elements=("p", "q"))
        }

    def derivative(self, t, state):
        return {"b": -state["b"]}


def test_a_plain_port_bound_to_several_paths_is_refused():
    comp = Composite(
        processes={"d": Decay()},
        topology={"d": {"x": ("a", "b", "c")}},
        validate=False,
        semantic_validation=False,
    )
    with pytest.raises(ValueError, match=r"d\.x is a plain port bound to 3"):
        comp.build_rhs()


def test_a_block_port_bound_to_the_wrong_count_is_refused():
    comp = Composite(
        processes={"k": Block()},
        topology={"k": {"b": ("p", "q", "r")}},
        validate=False,
        semantic_validation=False,
    )
    with pytest.raises(ValueError, match=r"2-wide block port bound to 3"):
        comp.build_rhs()


def test_a_block_port_bound_to_its_width_runs():
    comp = Composite(
        processes={"k": Block()},
        topology={"k": {"b": ("p", "q")}},
        validate=False,
        semantic_validation=False,
    )
    rhs, keys = comp.build_rhs()
    assert set(keys) == {"p", "q"}
