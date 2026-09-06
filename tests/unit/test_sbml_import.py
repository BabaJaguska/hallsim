"""Tests for SBMLProcess timed-intervention mechanisms."""

import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from demos.models.multi_hallmark import GZ06_PSI_NAME, GZ06_SBML_PATH
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler


def _gz06(psi):
    return process_from_sbml(
        str(GZ06_SBML_PATH), name="gz06", parameters={GZ06_PSI_NAME: psi}
    )


def _solo_ys(proc, t_end=30.0):
    comp = Composite(
        processes={"gz06": proc},
        topology={"gz06": {n: f"gz06/{n}" for n in proc._species_names}},
        validate=False,
        semantic_validation=False,
    )
    res = Scheduler().run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=t_end,
        y0=comp.initial_state_vec(),
        save_dt=t_end / 30.0,
    )
    return res.ys


class TestWithParamStep:
    def test_rejects_unknown_param(self):
        with pytest.raises(KeyError, match="not an SBML constant"):
            _gz06(0.9).with_param_step("not_a_param", 5.0, 0.1)

    def test_records_step(self):
        p = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 5.0, 0.1)
        assert p._param_steps == ((GZ06_PSI_NAME, 5.0, 0.1),)

    def test_step_never_fires_holds_value_before(self):
        # t_step past the horizon → the constant stays at value_before the
        # whole run, matching a plain process pinned to value_before.
        stepped = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 1e9, 0.2)
        pinned = _gz06(0.2)
        assert jnp.allclose(_solo_ys(stepped), _solo_ys(pinned), atol=1e-6)

    def test_step_at_zero_holds_configured_value(self):
        # t_step=0 → the configured parameters value is in force from t=0,
        # matching a plain process pinned to that value (value_before unused).
        stepped = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 0.0, 0.2)
        pinned = _gz06(0.9)
        assert jnp.allclose(_solo_ys(stepped), _solo_ys(pinned), atol=1e-6)

    def test_step_changes_trajectory_midway(self):
        # A mid-run step must diverge from both endpoints' pinned runs.
        stepped = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 15.0, 0.2)
        assert not jnp.allclose(_solo_ys(stepped), _solo_ys(_gz06(0.2)))
        assert not jnp.allclose(_solo_ys(stepped), _solo_ys(_gz06(0.9)))


class TestWithParamInput:
    def _pin(self):
        return process_from_sbml(
            str(GZ06_SBML_PATH), name="gz06"
        ).with_param_input(GZ06_PSI_NAME, "psi_in")

    def test_rejects_unknown_param(self):
        with pytest.raises(KeyError, match="not a constant"):
            _gz06(0.9).with_param_input("not_a_param", "p_in")

    def test_exposes_input_port(self):
        from hallsim.process import PortRole

        assert self._pin().ports_schema()["psi_in"].role is PortRole.INPUT

    def test_identity_equals_baked_value(self):
        # Parameter driven by the port at v == the parameter pinned at v.
        direct, pin = _gz06(0.7), self._pin()
        state = {
            n: jnp.asarray(float(p.default))
            for n, p in direct.ports_schema().items()
        }
        d0 = direct.derivative(0.5, state)
        d1 = pin.derivative(0.5, {**state, "psi_in": jnp.asarray(0.7)})
        for s in direct._species_names:
            assert jnp.allclose(d0[s], d1[s], atol=1e-9)

    def test_live_signal_changes_derivative(self):
        pin = self._pin()
        state = {
            n: jnp.asarray(float(p.default))
            for n, p in pin.ports_schema().items()
            if p.role.name != "INPUT"
        }
        a = pin.derivative(0.5, {**state, "psi_in": jnp.asarray(0.7)})
        b = pin.derivative(0.5, {**state, "psi_in": jnp.asarray(0.2)})
        assert not jnp.allclose(
            jnp.stack([a[s] for s in pin._species_names]),
            jnp.stack([b[s] for s in pin._species_names]),
        )


SBML_QUAL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
      xmlns:qual="http://www.sbml.org/sbml/level3/version1/qual/version1"
      level="3" version="1" qual:required="true">
  <model id="toy">
    <listOfCompartments>
      <compartment id="c" constant="true"/>
    </listOfCompartments>
    <qual:listOfQualitativeSpecies>
      <qual:qualitativeSpecies qual:id="A" qual:compartment="c"
        qual:constant="false" qual:maxLevel="1" qual:initialLevel="0"/>
      <qual:qualitativeSpecies qual:id="B" qual:compartment="c"
        qual:constant="false" qual:maxLevel="1" qual:initialLevel="1"/>
    </qual:listOfQualitativeSpecies>
    <qual:listOfTransitions>
      <qual:transition qual:id="t_A">
        <qual:listOfInputs>
          <qual:input qual:qualitativeSpecies="B" qual:transitionEffect="none"
            qual:id="in_B"/>
        </qual:listOfInputs>
        <qual:listOfOutputs>
          <qual:output qual:qualitativeSpecies="A"
            qual:transitionEffect="assignmentLevel"/>
        </qual:listOfOutputs>
        <qual:listOfFunctionTerms>
          <qual:defaultTerm qual:resultLevel="0"/>
        </qual:listOfFunctionTerms>
      </qual:transition>
    </qual:listOfTransitions>
  </model>
</sbml>
"""


class TestSBMLQualRejected:
    """A logical model must be named as one, not fail deep in codegen.

    BioModels serves SBML qual under format "SBML" with no other signal, and
    libsbml parses it happily — the state vector just comes back empty, so
    the failure used to surface as "None is not a valid value for jnp.array".
    """

    def test_qual_model_is_rejected_by_name(self, tmp_path):
        path = tmp_path / "toy_qual.xml"
        path.write_text(SBML_QUAL)
        with pytest.raises(Exception) as exc:
            process_from_sbml(str(path), name="toy")
        assert "qual" in str(exc.value).lower()
        assert "rate laws" in str(exc.value)

    def test_kinetic_model_still_imports(self):
        proc = _gz06(0.0)
        assert len(proc.ports_schema()) > 0


SBML_EVENT_ON_NONZERO_SPECIES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="ev">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="5"
        boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants><speciesReference species="S"/></listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="reset">
        <trigger>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol
              definitionURL="http://www.sbml.org/sbml/symbols/time">t
              </csymbol><cn>1</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="S">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>
"""


class TestEventTargetKeepsPublishedInitial:
    """An event writing to a species must not claim where that species starts.

    Event translation wires a LATCHED ``__set_<species>`` port onto the
    species' own store path. Seeding it with a value makes it a second
    writer-tier claim, which collides with the ODE process's published
    initial condition for every species that does not start at zero --
    jws:conradie (GM = 1.35565) and jws:calzone1 failed to build at all,
    and the numerical screen reported it as EXPLODING with max|y|=inf.
    """

    def test_composite_builds_and_keeps_published_ic(self, tmp_path):
        from hallsim.composite import single_process_composite

        path = tmp_path / "ev.xml"
        path.write_text(SBML_EVENT_ON_NONZERO_SPECIES)
        proc = process_from_sbml(str(path), name="ev")
        comp = single_process_composite(proc, name="ev")
        store = comp.initial_state()
        assert float(store["ev/S"]) == pytest.approx(5.0)

    def test_event_set_port_abstains_for_species_target(self, tmp_path):
        from hallsim.sbml_events import translate_events
        from hallsim.sbml_import import _preprocess_sbml

        path = tmp_path / "ev.xml"
        path.write_text(SBML_EVENT_ON_NONZERO_SPECIES)
        events = translate_events(
            _preprocess_sbml(str(path)), ("S",), {"k": 0.1}, "ev"
        )
        assert events, "expected the <event> to translate"
        ports = events[0].ports_schema()
        assert ports["__set_S"].default is None
