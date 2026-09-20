"""Tests for SBMLProcess timed-intervention mechanisms."""

import logging

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


def test_sbml_import_preserves_reaction_channels_for_stochastic_execution():
    proc = _gz06(1.0)
    channels = proc.reaction_channels()
    assert channels
    assert all(channel.reaction_id for channel in channels)
    assert all(channel.rate_law for channel in channels)
    assert any(
        any(species == "x" for species, _ in channel.stoichiometry)
        for channel in channels
    )


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


class TestWithSpeciesInput:
    """A species handed over to another model's pool: read, not integrated."""

    def _shared(self, psi=0.9):
        return _gz06(psi).with_species_input("y")

    def _state(self, proc):
        return {
            n: jnp.asarray(float(p.default))
            for n, p in proc.ports_schema().items()
        }

    def test_rejects_unknown_species(self):
        with pytest.raises(KeyError, match="not species"):
            _gz06(0.9).with_species_input("not_a_species")

    def test_the_port_keeps_its_name_and_becomes_an_input(self):
        from hallsim.process import PortRole

        owned = _gz06(0.9).ports_schema()["y"]
        handed = self._shared().ports_schema()["y"]
        assert owned.role is PortRole.EVOLVED
        assert handed.role is PortRole.INPUT
        assert handed.default == owned.default
        assert handed.ontology == owned.ontology

    def test_the_model_no_longer_moves_the_species(self):
        shared = self._shared()
        d = shared.derivative(0.5, self._state(shared))
        assert "y" not in d
        assert set(d) == set(shared._species_names) - {"y"}

    def test_at_the_published_value_the_rest_is_unchanged(self):
        plain, shared = _gz06(0.9), self._shared()
        state = self._state(plain)
        d0, d1 = plain.derivative(0.5, state), shared.derivative(0.5, state)
        for s in d1:
            assert jnp.allclose(d0[s], d1[s], atol=1e-12)

    def test_the_external_value_reaches_the_rate_laws(self):
        # Mdm2 (y) degrades p53 (x): more external Mdm2, lower dx/dt. The
        # deposit starts p53 at zero, where nothing degrades, so give it some.
        shared = self._shared()
        state = {**self._state(shared), "x": jnp.asarray(1.0)}
        lo = shared.derivative(0.5, {**state, "y": jnp.asarray(0.1)})["x"]
        hi = shared.derivative(0.5, {**state, "y": jnp.asarray(2.0)})["x"]
        assert float(hi) < float(lo)

    def test_unwired_it_holds_the_published_value_for_the_whole_run(self):
        shared = self._shared()
        comp = Composite(
            processes={"gz06": shared},
            topology={"gz06": {n: f"gz06/{n}" for n in shared._species_names}},
            validate=False,
            semantic_validation=False,
        )
        res = Scheduler().run(
            comp,
            t_span=(0.0, 5.0),
            macro_dt=5.0,
            y0=comp.initial_state_vec(),
            save_dt=0.5,
        )
        y = jnp.asarray(res.get("gz06/y"))
        assert jnp.allclose(y, y[0])
        assert float(y[0]) == shared.ports_schema()["y"].default


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

        path = tmp_path / "ev.xml"
        path.write_text(SBML_EVENT_ON_NONZERO_SPECIES)
        events = translate_events(str(path), ("S",), {"k": 0.1}, "ev")
        assert events, "expected the <event> to translate"
        ports = events[0].ports_schema()
        assert ports["__set_S"].default is None


SBML_WITH_INERT_SINK = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="sink">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="5"
        boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0"
        boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="convert" reversible="false">
        <listOfReactants><speciesReference species="A"/></listOfReactants>
        <listOfProducts><speciesReference species="B"/></listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


class TestProvenance:
    """An import says where it came from and what was done to it."""

    def test_import_records_source_hash_and_changes(self):
        from hallsim.io import file_sha256

        proc = _gz06(0.123)
        meta = proc.metadata()
        assert meta["source"] == str(GZ06_SBML_PATH)
        assert meta["source_sha256"] == file_sha256(GZ06_SBML_PATH)
        assert GZ06_PSI_NAME in meta["published_parameters"]
        assert meta["modified_parameters"] == {GZ06_PSI_NAME: 0.123}
        again = proc.with_param(f"parameters.{GZ06_PSI_NAME}", 0.2)
        assert again.provenance()["modified_parameters"] == {
            GZ06_PSI_NAME: 0.2
        }

    def test_a_frozen_sink_is_named_and_warns_when_read(
        self, tmp_path, caplog
    ):
        from hallsim.composite import single_process_composite

        path = tmp_path / "sink.xml"
        path.write_text(SBML_WITH_INERT_SINK)
        proc = process_from_sbml(str(path), name="sink")
        assert proc.frozen_species() == ["B"]
        comp = single_process_composite(proc)
        assert comp.frozen_paths() == {"sink/B"}
        res = Scheduler().run(
            comp, t_span=(0.0, 1.0), macro_dt=1.0, save_dt=0.5
        )
        with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
            held = res.get("sink/B")
        assert "with_unfrozen" in caplog.text
        assert float(held[-1]) == 0.0
        assert float(res.get("sink/A")[-1]) < 5.0


def test_a_pmc_id_imports_from_the_paper_supplement(tmp_path, monkeypatch):
    """A Europe PMC hit's id is a paper; its model lives in the supplement,
    so the same id an agent gets from search imports without a detour."""
    from hallsim import literature, sbml_import

    model = tmp_path / "model.xml"
    model.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2"><model id="m">'
        '<listOfCompartments><compartment id="c" size="1" '
        'constant="true"/></listOfCompartments>'
        '<listOfSpecies><species id="X" compartment="c" '
        'initialConcentration="1" hasOnlySubstanceUnits="false" '
        'boundaryCondition="false" constant="false"/></listOfSpecies>'
        '<listOfParameters><parameter id="k" value="0.5" '
        'constant="true"/></listOfParameters>'
        '<listOfReactions><reaction id="decay" reversible="false">'
        '<listOfReactants><speciesReference species="X" stoichiometry="1" '
        'constant="true"/></listOfReactants><kineticLaw><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/>'
        "<ci>k</ci><ci>X</ci></apply></math></kineticLaw></reaction>"
        "</listOfReactions></model></sbml>\n"
    )
    monkeypatch.setattr(
        literature,
        "supplementary_model_files",
        lambda pmcid, **kw: [tmp_path / "code.m", model],
    )
    proc = sbml_import.process_from_sbml("PMC1234567")
    assert proc._name == "pmc1234567"
    assert "X" in proc._species_names

    monkeypatch.setattr(
        literature, "supplementary_model_files", lambda pmcid, **kw: []
    )
    import pytest

    with pytest.raises(LookupError, match="no SBML or COPASI file"):
        sbml_import.process_from_sbml("PMC1234567")
