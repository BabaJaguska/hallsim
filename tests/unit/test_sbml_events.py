"""Translated SBML events — the port contract, without libsbml.

`SBMLEvent` is built directly from its IR here. Assembling events through
libsbml's own API outside a fully-populated document segfaults the
interpreter, and the deposit path is covered by network tests.
"""

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process
from hallsim.sbml_events import SBMLEvent
from hallsim.store import build_initial_store


def _event(name, assigns, param_targets=(), defaults=()):
    return SBMLEvent(
        _name=name,
        _trigger_ir=(),
        _assign_ir=tuple((t, ()) for t in assigns),
        _read_species=tuple(t for t in assigns if t not in param_targets),
        _param_targets=tuple(param_targets),
        _target_defaults=tuple(defaults),
    )


class _Owner(Process):
    """The model whose state the events assign into."""

    y0: float = 1e-26

    def ports_schema(self):
        return {"S1": Port(role=PortRole.EVOLVED, default=self.y0)}

    def derivative(self, t, state):
        return {"S1": jnp.asarray(0.0)}


class TestAssignmentTargetDefaults:
    def test_a_species_target_abstains(self):
        """The species already has an owner, so the event claims nothing about
        where it starts."""
        port = _event("m__e1", ["S1"]).ports_schema()["__set_S1"]
        assert port.role is PortRole.LATCHED
        assert port.default is None

    def test_a_parameter_target_keeps_its_published_value(self):
        """A parameter is not state, so nothing else seeds it; starting it at
        zero runs the model off the wrong constant until the event fires."""
        ev = _event(
            "m__e1",
            ["kdeg"],
            param_targets=["kdeg"],
            defaults=[("kdeg", 0.25)],
        )
        assert ev.ports_schema()["__set_kdeg"].default == 0.25

    def test_many_events_on_one_species_do_not_collide(self):
        """Four events assigning one species disagreed with its SBML initial
        value by 1e-26 and refused to build — Dwivedi2014, all four arms."""
        procs = {"m": _Owner()}
        topo = {"m": {"S1": "m/S1"}}
        for i in range(4):
            procs[f"m__e{i}"] = _event(f"m__e{i}", ["S1"])
            topo[f"m__e{i}"] = {"S1": "m/S1", "__set_S1": "m/S1"}
        store = build_initial_store(procs, topo)
        assert float(store["m/S1"]) == 1e-26, "the owner seeds the path"


def test_a_composite_that_cannot_build_is_not_reported_as_exploding():
    """Dwivedi2014 was rejected as `EXPLODING max|y|=inf` for a
    construction error it never got past. A model that never ran did not blow
    up, and `EXPLODING` sends the reader to solver tolerances."""
    from hallsim.diagnostics import screen_process

    class _Unbuildable(_Owner):
        def ports_schema(self):
            # A non-numeric default: the store cannot be assembled from it.
            return {"S1": Port(role=PortRole.EVOLVED, default="not a number")}

    r = screen_process(_Unbuildable(), t_end=1.0)
    assert r.did_not_construct is True
    assert r.exploding is False
    assert r.ok is False
    assert "did not build" in r.detail
    assert "DID-NOT-CONSTRUCT" in str(r)
