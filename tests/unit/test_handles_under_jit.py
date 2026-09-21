"""A rate mapping applies inside a jitted function: the guard against a
zero base value reads the concrete array without staging it."""

from __future__ import annotations

import equinox as eqx
import jax
import pytest

from hallsim.composite import Composite
from hallsim.handles import Handle, ParameterMapping, with_handles
from hallsim.process import Port, PortRole, Process, ProcessKind


class Decay(Process):
    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = eqx.field(static=True, default=1.0)
    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


def test_a_rate_mapping_applies_under_jit():
    comp = Composite(processes={"d": Decay()}, topology={"d": {"x": "cell/x"}})
    registry = {
        "h": Handle(
            name="h",
            mappings=[
                ParameterMapping(
                    process_name="d", param_name="rate", floor=1.0, slope=2.0
                )
            ],
        )
    }

    def rate_at(severity):
        return (
            with_handles(comp, {"h": severity}, registry=registry)
            .processes["d"]
            .rate
        )

    assert float(jax.jit(rate_at)(0.5)) == pytest.approx(0.2)
    assert float(jax.grad(rate_at)(0.5)) == pytest.approx(0.2)
