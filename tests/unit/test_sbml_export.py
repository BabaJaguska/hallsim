"""A composite written out as SBML: what the document holds."""

import libsbml
import pytest

from hallsim.composite import Composite
from hallsim.models.clamp_edge import ClampEdge
from hallsim.models.gain_edge import GainEdge
from hallsim.models.hill_edge import HillEdge
from hallsim.sbml_import import process_from_sbml
from test_sbml_core import COMPARTMENTS, RULES, _write


def build(tmp_path):
    """Two imported models sharing a species, with three edges, on a
    composite clock of 2 s per unit (so one model's rates are rescaled)."""
    a = process_from_sbml(_write(tmp_path, "a", COMPARTMENTS), name="a")
    b = process_from_sbml(
        _write(tmp_path, "b", RULES), name="b", native_time_seconds=2.0
    )
    a, b = a.reconciled_to(2.0), b.reconciled_to(2.0)
    return Composite(
        processes={
            "a": a,
            "b": b,
            "drive": HillEdge(basal=0.0, hi=0.4, K=(0.3,), n=(2.0,)),
            "bridge": GainEdge(offset=0.1, gain=2.0, mode="level"),
            "hold": ClampEdge(k_clamp=1.5),
        },
        topology={
            "drive": {"source": "a/C", "target": "b/P"},
            "bridge": {"source": "b/S", "signal": "bridge/level"},
            "hold": {"target": "a/B", "setpoint": "hold/setpoint"},
        },
        rewire={"b/S": "a/A"},
        # the shared pool starts at A's amount; B's S claimed 3.0
        initial={"hold/setpoint": 0.4, "a/A": 2.5, "a/B": 0.5},
        semantic_validation=False,
    )


def test_document_transcribes_the_composite(tmp_path):
    comp = build(tmp_path)
    out = tmp_path / "composite.xml"
    text = comp.to_sbml(str(out))
    assert out.read_text() == text
    doc = libsbml.readSBMLFromString(text)
    assert (
        doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) == 0
    ), doc.getErrorLog().toString()
    model = doc.getModel()
    species = {
        model.getSpecies(i).getId() for i in range(model.getNumSpecies())
    }
    # a shared store path is one species; boundary constants stay local
    assert {
        "a_A",
        "a_B",
        "a_C",
        "b_P",
        "b_D",
        "b_R",
        "bridge_level",
        "hold_setpoint",
        "a__E",
    } <= species
    assert "b_S" not in species
    params = {
        model.getParameter(i).getId(): model.getParameter(i).getValue()
        for i in range(model.getNumParameters())
    }
    assert (
        params["a__k1"] == 0.7
        and params["a__cyt"] == 2.5
        and params["a__release_kloc"] == 0.15
    )
    assert params["b__vmax"] == 1.2
    reactions = {
        model.getReaction(i).getId() for i in range(model.getNumReactions())
    }
    assert reactions == {
        "a__bind",
        "a__release",
        "b__convert",
        "drive__hill",
        "hold__clamp",
    }
    rules = [
        (model.getRule(i).getVariable(), model.getRule(i).isRate())
        for i in range(model.getNumRules())
    ]
    assert sorted(rules) == [
        ("b_R", True),
        ("b_drive", False),
        ("b_total", False),
        ("bridge_level", False),
    ]
    # the shared species carries A's initial amount (1.0 mM in 2.5 L)
    assert model.getSpecies("a_A").getInitialAmount() == 2.5
    # a model on a rescaled clock has its rate multiplied by the scale
    law = libsbml.formulaToL3String(
        model.getReaction("a__bind").getKineticLaw().getMath()
    )
    assert law.startswith("2 *") or "2.0 *" in law or "* 2" in law, law


def test_a_process_without_a_symbolic_form_is_refused(tmp_path):
    from hallsim.process import Port, PortRole, Process
    from hallsim.sbml_export import UnsupportedExportError

    class Opaque(Process):
        def ports_schema(self):
            return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

        def derivative(self, t, state):
            return {"x": -state["x"]}

    comp = Composite(
        {"o": Opaque()}, {}, semantic_validation=False, validate=False
    )
    with pytest.raises(UnsupportedExportError, match="Opaque"):
        comp.to_sbml()


def test_events_drivers_and_steps_export(tmp_path):
    from test_sbml_events import EVENT_MODEL

    from hallsim.composite import single_process_composite

    path = tmp_path / "ev.xml"
    path.write_text(EVENT_MODEL)
    proc = process_from_sbml(str(path), name="m").reconciled_to(2.0)
    a = process_from_sbml(
        _write(tmp_path, "a", COMPARTMENTS), name="a"
    ).with_param_step("k1", 1.5, 0.35)
    comp = Composite(
        {"o": single_process_composite(proc), "a": a},
        semantic_validation=False,
        validate=False,
    )
    doc = libsbml.readSBMLFromString(comp.to_sbml())
    assert (
        doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) == 0
    ), doc.getErrorLog().toString()
    model = doc.getModel()
    events = {
        model.getEvent(i).getId(): model.getEvent(i)
        for i in range(model.getNumEvents())
    }
    assert set(events) == {"o_m__pulse", "a__k1_step"}, set(events)
    pulse = events["o_m__pulse"]
    trigger = libsbml.formulaToL3String(pulse.getTrigger().getMath())
    assert "time" in trigger and "2" in trigger, trigger
    targets = {
        pulse.getEventAssignment(j).getVariable()
        for j in range(pulse.getNumEventAssignments())
    }
    assert targets == {"o_m_X", "o_m_k"}, targets
    # the event-driven parameter reads its store path, not a constant
    law = libsbml.formulaToL3String(
        model.getReaction("o_m__decay").getKineticLaw().getMath()
    )
    assert "o_m_k" in law, law
    # the stepped constant starts at its value before the step
    assert model.getParameter("a__k1").getValue() == 0.35
    assert not model.getParameter("a__k1").getConstant()
