"""The path trace names the node where a perturbation stops propagating."""

import equinox as eqx
import jax.numpy as jnp

from hallsim.attenuation import trace_path
from hallsim.composite import Composite
from hallsim.handles import Handle, ParameterMapping
from hallsim.process import Port, PortRole, Process, calibratable


class Driven(Process):
    """x relaxes to the input level u."""

    u: float = calibratable(0.0, level=True)
    timescale: float | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

    def derivative(self, t, state):
        return {"x": self.u - state["x"]}


class Relay(Process):
    """y relaxes to x."""

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.INPUT, default=0.0),
            "y": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def derivative(self, t, state):
        return {"y": state["x"] - state["y"]}


class Gated(Process):
    """z has a basal supply and a gate on y that needs y near 100 to open."""

    def ports_schema(self):
        return {
            "y": Port(role=PortRole.INPUT, default=0.0),
            "z": Port(role=PortRole.EVOLVED, default=0.01),
        }

    def derivative(self, t, state):
        gate = state["y"] ** 4 / (100.0**4 + state["y"] ** 4)
        return {"z": 0.01 + gate - state["z"]}


def _chain():
    comp = Composite(
        processes={"drive": Driven(), "relay": Relay(), "gated": Gated()},
        topology={
            "drive": {"x": "pool/x"},
            "relay": {"x": "pool/x", "y": "pool/y"},
            "gated": {"y": "pool/y", "z": "pool/z"},
        },
        validate=False,
        semantic_validation=False,
    )
    registry = {
        "Exposure": Handle(
            name="Exposure",
            mappings=[ParameterMapping("drive", "u", floor=0.0, slope=1.0)],
        )
    }
    return comp, registry


def test_the_trace_names_the_gate_and_the_node_before_it():
    comp, registry = _chain()
    trace = trace_path(
        comp, "Exposure", "pool/z", registry=registry, t_end=20.0
    )
    assert [n.path for n in trace.nodes] == ["pool/x", "pool/y", "pool/z"]
    assert [n.process for n in trace.nodes] == ["drive", "relay", "gated"]
    assert trace.nodes[0].rel_change > 1e3  # x: 0 -> ~1
    assert trace.nodes[1].rel_change > 1e3  # y follows x
    assert trace.gate is not None and trace.gate.path == "pool/z"
    assert not trace.reaches
    text = str(trace)
    assert "dies at pool/z in gated, on the step pool/y -> pool/z" in text
    assert "still" in text and "pool/y" in text


def test_a_parameter_control_doubles_and_an_open_gate_reaches():
    comp, registry = _chain()
    # At u = 200 the gate opens: the signal reaches z.
    trace = trace_path(
        comp,
        "Exposure",
        "pool/z",
        registry=registry,
        t_end=20.0,
        settings=(0.0, 200.0),
    )
    assert trace.reaches and trace.gate is None
    assert "reaches pool/z" in str(trace)
    # A parameter address is run at its value and twice it.
    opened = Composite(
        {**comp.processes, "drive": Driven(u=150.0)},
        comp.topology,
        validate=False,
        semantic_validation=False,
    )
    by_param = trace_path(opened, "drive.u", "pool/y", t_end=20.0)
    assert by_param.settings == (1.0, 2.0)
    assert by_param.nodes[-1].high > by_param.nodes[-1].low
    assert float(jnp.isfinite(by_param.nodes[-1].rel_change))
