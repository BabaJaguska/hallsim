"""Translated SBML events — the port contract, without libsbml.

`SBMLEvent` is built directly from its IR here. Assembling events through
libsbml's own API outside a fully-populated document segfaults the
interpreter, and the deposit path is covered by network tests.
"""

import jax.numpy as jnp
import sympy

from hallsim.process import Port, PortRole, Process
from hallsim.sbml_events import SBMLEvent
from hallsim.store import build_initial_store


def _event(name, assigns, param_targets=(), defaults=()):
    return SBMLEvent(
        _name=name,
        _trigger=sympy.false,
        _assignments=tuple((t, sympy.Integer(0)) for t in assigns),
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


# ── event math through sympy, on the owner's clock ──────────────────

EVENT_MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="m">
<listOfFunctionDefinitions>
  <functionDefinition id="twice"><math xmlns="http://www.w3.org/1998/Math/MathML"><lambda><bvar><ci>x</ci></bvar><apply><times/><cn>2</cn><ci>x</ci></apply></lambda></math></functionDefinition>
</listOfFunctionDefinitions>
<listOfCompartments><compartment id="cell" spatialDimensions="3" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>
  <species id="X" compartment="cell" initialConcentration="1.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters><parameter id="k" value="0.5" constant="false"/></listOfParameters>
<listOfReactions>
  <reaction id="decay" reversible="false">
    <listOfReactants><speciesReference species="X" stoichiometry="1" constant="true"/></listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cell</ci><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
  </reaction>
</listOfReactions>
<listOfEvents>
  <event id="pulse" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol><apply><ci>twice</ci><cn>1</cn></apply></apply></math></trigger>
    <listOfEventAssignments>
      <eventAssignment variable="X"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><log/><logbase><cn>10</cn></logbase><cn>100</cn></apply></math></eventAssignment>
      <eventAssignment variable="k"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><root/><degree><cn>3</cn></degree><cn>27</cn></apply></math></eventAssignment>
    </listOfEventAssignments>
  </event>
</listOfEvents>
</model></sbml>
"""


def test_event_math_goes_through_sympy(tmp_path):
    """A function call in the trigger, a two-argument log and a cube root in
    the assignments: log(10, 100) is 2 and root(3, 27) is 3."""
    import pytest

    from hallsim.sbml_events import translate_events

    path = tmp_path / "ev.xml"
    path.write_text(EVENT_MODEL)
    (ev,) = translate_events(str(path), ("X",), {"k": 0.5, "cell": 1.0}, "m")
    state = {"X": jnp.asarray(1.0), "k": jnp.asarray(0.5)}
    assert not bool(ev.condition(1.9, state))
    assert bool(ev.condition(2.0, state))
    delta = ev.handler(0.0, state)
    assert float(delta["__set_X"]) == pytest.approx(2.0 - 1.0)
    assert float(delta["__set_k"]) == pytest.approx(3.0 - 0.5)


def test_a_reconciled_model_fires_its_event_on_the_composite_clock(tmp_path):
    """Native time 1 s, composite unit 2 s: the native-time-2 event fires at
    composite time 1, and not again at 2."""
    import numpy as np
    import pytest

    from hallsim.composite import single_process_composite
    from hallsim.sbml_import import process_from_sbml
    from hallsim.scheduler import Scheduler

    path = tmp_path / "ev.xml"
    path.write_text(EVENT_MODEL)
    proc = process_from_sbml(str(path), name="m").reconciled_to(2.0)
    res = Scheduler().run(
        single_process_composite(proc),
        t_span=(0.0, 3.0),
        macro_dt=0.25,
        save_dt=0.05,
    )
    ts, x = np.asarray(res.ts), np.asarray(res.get("m/X"))
    at = {round(float(t), 2): float(v) for t, v in zip(ts, x)}
    # before the event: k = 0.5 on a 2x clock decays by e^-0.25 per 0.25
    assert at[0.75] / at[0.5] == pytest.approx(np.exp(-0.25), rel=1e-3)
    # after it fires at composite t = 1: k = 3 on a 2x clock, e^-1.5
    assert at[1.5] / at[1.25] == pytest.approx(np.exp(-1.5), rel=1e-3)
    # and it does not fire again at composite t = 2
    assert at[2.5] / at[2.25] == pytest.approx(np.exp(-1.5), rel=1e-3)


def test_a_nested_composite_expands_each_event_once(tmp_path):
    from hallsim.composite import Composite, single_process_composite
    from hallsim.sbml_import import process_from_sbml

    path = tmp_path / "ev.xml"
    path.write_text(EVENT_MODEL)
    inner = single_process_composite(process_from_sbml(str(path), name="m"))
    outer = Composite({"o": inner}, semantic_validation=False, validate=False)
    events = [n for n, p in outer.processes.items() if p.kind is p.kind.EVENT]
    assert events == ["o.m__pulse"], events
