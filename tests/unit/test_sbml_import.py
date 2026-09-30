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

    def test_a_species_nothing_reads_is_integrated_not_held(self, tmp_path):
        """It is a terminal product as often as a degradation counter, and a
        terminal product is what an assay measures, so its value stands."""
        from hallsim.composite import single_process_composite

        path = tmp_path / "sink.xml"
        path.write_text(SBML_WITH_INERT_SINK)
        proc = process_from_sbml(str(path), name="sink")
        assert proc.frozen_species() == []
        comp = single_process_composite(proc)
        assert comp.frozen_paths() == set()
        res = Scheduler().run(
            comp, t_span=(0.0, 1.0), macro_dt=1.0, save_dt=0.5
        )
        # B takes up what A loses, so it rises rather than sitting at zero.
        assert float(res.get("sink/B")[-1]) > 0.0
        assert float(res.get("sink/A")[-1]) < 5.0

    def test_a_named_sink_can_still_be_held(self, tmp_path, caplog):
        """Holding is opt-in now, for the case it was built for: a collector
        whose unbounded growth is costing the solve."""
        from hallsim.composite import single_process_composite

        path = tmp_path / "sink.xml"
        path.write_text(SBML_WITH_INERT_SINK)
        proc = process_from_sbml(str(path), name="sink").with_frozen("B")
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
        assert proc.with_unfrozen("B").frozen_species() == []


def test_a_pmc_id_imports_from_the_paper_supplement(tmp_path, monkeypatch):
    """A Europe PMC hit's id is a paper; its model lives in the supplement,
    so the same id an agent gets from search imports without a detour."""
    from hallsim import sbml_import
    from hallsim.search import literature

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


def test_jws_source_scheme_resolves_without_touching_biomodels(monkeypatch):
    """`jws:<slug>` must route to JWS, not fall through to a BioModels fetch."""
    from pathlib import Path

    import pytest

    from hallsim import sbml_import

    monkeypatch.setattr(
        sbml_import,
        "download_jws_model",
        lambda slug: Path(f"/tmp/{slug}.xml"),
    )
    monkeypatch.setattr(
        sbml_import,
        "download_biomodel_main",
        lambda i: pytest.fail("routed to BioModels"),
    )
    path, name = sbml_import._resolve_source("jws:achcar2", None)
    assert path.endswith("achcar2.xml")
    assert name == "jws_achcar2"


def test_an_assignment_rule_species_keeps_its_annotation():
    """A curated deposit states its observables as assignment-rule species and
    annotates those, so dropping their ontology loses exactly the ones a
    readout is built from."""
    from hallsim.process import PortRole
    from hallsim.sbml_import import process_from_sbml
    from demos.models.sbml import sbml_source

    path = sbml_source(
        "rateitschak2012",
        "rateitschak2012_BIOMD0000000585.xml",
        "BIOMD0000000585",
    )
    schema = process_from_sbml(str(path), name="stat1").ports_schema()
    annotated = {
        n: s.ontology
        for n, s in schema.items()
        if s.role is PortRole.ASSIGNED and s.ontology
    }
    assert annotated, "assignment-rule ports lost their MIRIAM annotations"
    assert annotated["Stat1ex"]["hgnc.symbol"] == "STAT1"


class TestLiveParameterDrivers:
    """An assignment rule reading a driven constant must see the driven
    value, and a promoted constant must not start at zero."""

    def _proc(self):
        from hallsim.sbml_import import process_from_sbml
        from demos.models.sbml import sbml_source

        path = sbml_source(
            "rateitschak2012",
            "rateitschak2012_BIOMD0000000585.xml",
            "BIOMD0000000585",
        )
        return process_from_sbml(str(path), name="s").with_param_input(
            "scale_Stat1Pcex", "scale_in"
        )

    def test_assign_sees_the_driven_value(self):
        import jax.numpy as jnp

        proc = self._proc()
        published = float(proc.parameters["scale_Stat1Pcex"])
        state = {
            n: jnp.asarray(float(p.default))
            for n, p in proc.ports_schema().items()
        }
        state["Stat1Pd"] = jnp.asarray(1.0)
        at_published = proc.assign(
            0.0, {**state, "scale_in": jnp.asarray(published)}
        )["Stat1Pcex"]
        at_triple = proc.assign(
            0.0, {**state, "scale_in": jnp.asarray(published * 3.0)}
        )["Stat1Pcex"]
        assert float(at_published) != 0.0
        assert float(at_triple) == pytest.approx(3.0 * float(at_published))

    def test_a_promoted_constant_defaults_to_its_published_value(self):
        proc = self._proc()
        port = proc.ports_schema()["scale_in"]
        assert port.default == pytest.approx(
            float(proc.parameters["scale_Stat1Pcex"])
        )
        assert port.default != 0.0


def test_timescale_is_not_inferred_from_the_time_unit():
    """`timescale` is a rate; the unit a model is written in is not evidence
    of one, and auto_groups clusters on it."""
    from hallsim.sbml_import import process_from_sbml
    from demos.models.sbml import sbml_source

    path = str(
        sbml_source(
            "rateitschak2012",
            "rateitschak2012_BIOMD0000000585.xml",
            "BIOMD0000000585",
        )
    )
    assert process_from_sbml(path, name="s").timescale is None
    supplied = process_from_sbml(path, name="s", timescale=120.0)
    assert supplied.timescale == pytest.approx(120.0)


def test_an_unwired_promoted_constant_is_reported(caplog):
    """Neither default is safe — zero loses the published value and the
    published value is the paper's own experiment, not a resting state — so
    composing says which driver ports nothing writes."""
    import logging

    from hallsim.composite import single_process_composite
    from hallsim.sbml_import import process_from_sbml
    from demos.models.sbml import sbml_source

    path = str(
        sbml_source(
            "rateitschak2012",
            "rateitschak2012_BIOMD0000000585.xml",
            "BIOMD0000000585",
        )
    )
    proc = process_from_sbml(path, name="s").with_param_input(
        "scale_Stat1Pcex", "scale_in"
    )
    with caplog.at_level(logging.WARNING, logger="hallsim.composite"):
        single_process_composite(proc, "s")
    assert any("scale_in" in r.message for r in caplog.records)


class TestIdentitySurvivesRoleChange:
    """A quantity keeps its identity and its value when its role changes.

    The importer addresses a model two ways — ports, which carry a role, a
    default and an ontology, and `parameters`, which carries none of those. A
    quantity promoted from the second to the first used to arrive bare, which
    is one cause behind several separately-reported silences: a readout that
    joins to nothing, and a driver that runs the model at zero until wired.
    """

    PATH = (
        "demos/models/sbml/dallepezze2014/"
        "dallepezze2014_BIOMD0000000582.xml"
    )

    def _proc(self):
        from hallsim.sbml_import import process_from_sbml

        return process_from_sbml(self.PATH, name="dp14")

    def test_a_promoted_parameter_keeps_its_annotation(self):
        """`Insulin` carries a ChEBI id, so a dataset measuring insulin has to
        reach it by identity however it is addressed."""
        port = (
            self._proc()
            .with_param_input("Insulin", "insulin_in")
            .ports_schema()["insulin_in"]
        )
        assert port.ontology.get("chebi") == "CHEBI:5931"

    def test_a_driven_boundary_input_keeps_its_annotation(self):
        port = (
            self._proc()
            .with_input_driver("Irradiation", "dose_in")
            .ports_schema()["dose_in"]
        )
        assert port.ontology.get("sbo") == "SBO:0000405"

    def test_a_driven_boundary_input_keeps_its_published_value(self):
        """Defaulting a driver port to 0 runs the model at zero insulin until
        something wires it, which is a wrong answer rather than a missing one.
        """
        proc = self._proc()
        port = proc.with_param_input("Insulin", "insulin_in").ports_schema()[
            "insulin_in"
        ]
        assert port.default == pytest.approx(float(proc.parameters["Insulin"]))
        assert port.default != 0.0

    def test_identity_is_kept_for_undriven_inputs_too(self):
        """Identity belongs to the quantity, not to whichever port is
        addressing it, so it must not depend on having called a driver."""
        proc = self._proc()
        assert proc.identity_of("Insulin").get("chebi") == "CHEBI:5931"
        assert proc.identity_of("Amino_Acids").get("chebi") == "CHEBI:33709"
        assert proc.identity_of("Irradiation").get("sbo") == "SBO:0000405"

    def test_identity_does_not_depend_on_which_vector_holds_it(self):
        """Insulin is a constants-vector parameter and Irradiation a boundary
        input; both are annotated species and both must resolve."""
        proc = self._proc()
        assert "Insulin" in proc._param_names
        assert "Irradiation" in proc._w_names
        for name in ("Insulin", "Irradiation"):
            assert proc.identity_of(name)


class TestAConstantBoundaryInputIsSettable:
    """A boundary species whose rule is a constant is a level the experiment
    held fixed, so setting it has to reach the field. One whose rule reads time
    is a protocol and must keep it."""

    PATH = (
        "demos/models/sbml/dallepezze2014/"
        "dallepezze2014_BIOMD0000000582.xml"
    )

    def _proc(self):
        from hallsim.sbml_import import process_from_sbml

        return process_from_sbml(self.PATH, name="dp14")

    def test_it_becomes_a_real_parameter(self):
        proc = self._proc()
        assert "Insulin" in proc._param_names
        assert "Amino_Acids" in proc._param_names

    def test_setting_it_moves_the_vector_field(self):
        """It used to accept the write and move nothing, so a handle aimed at
        it swept a flat line that looked like biology."""
        import equinox as eqx
        import jax.numpy as jnp

        from hallsim.composite import single_process_composite

        proc = self._proc()
        comp = single_process_composite(proc, "dp14")
        keys = comp.store_keys()
        y0 = comp.initial_state_vec(keys)
        base, _ = comp.build_rhs()
        half = eqx.tree_at(
            lambda m: m.parameters["Insulin"], proc, jnp.asarray(0.5)
        )
        treated, _ = single_process_composite(half, "dp14").build_rhs()
        moved = [
            keys[i]
            for i, d in enumerate(
                abs(treated(0.0, y0, None) - base(0.0, y0, None))
            )
            if float(d) > 0
        ]
        assert moved, "setting Insulin changed no derivative"
        assert any("Akt" in path for path in moved)

    def test_a_time_varying_input_keeps_its_rule(self):
        """The dose gate stays a protocol: it is not a parameter, and its
        pulse stays in the symbolic field where the analysis reads it."""
        proc = self._proc()
        assert "Irradiation" in proc._w_names
        assert "Irradiation" not in proc._param_names
