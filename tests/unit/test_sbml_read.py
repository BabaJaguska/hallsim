"""Everything one SBML file says, read once into a single record."""

import pytest
from click.testing import CliRunner

from hallsim.cli import supply
from hallsim.sbml_read import _references, read_sbml

SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m">
  <listOfCompartments>
   <compartment id="cyt" size="1"/><compartment id="nuc" size="1"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="A" name="alpha" compartment="cyt" initialConcentration="2"/>
   <species id="B" compartment="nuc" initialConcentration="0"/>
   <species id="drive" compartment="cyt" initialConcentration="1"
            boundaryCondition="true" constant="true"/>
   <species id="orphan" compartment="cyt" initialConcentration="5"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k" value="0.3"/></listOfParameters>
  <listOfFunctionDefinitions>
   <functionDefinition id="mm">
    <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
     <bvar><ci>s</ci></bvar><bvar><ci>v</ci></bvar>
     <apply><divide/><apply><times/><ci>v</ci><ci>s</ci></apply>
      <apply><plus/><ci>s</ci><cn>1</cn></apply></apply>
    </lambda></math>
   </functionDefinition>
  </listOfFunctionDefinitions>
  <listOfRules>
   <assignmentRule variable="B">
    <math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><times/><ci>k</ci><ci>A</ci></apply></math>
   </assignmentRule>
  </listOfRules>
  <listOfReactions>
   <reaction id="r1" reversible="false">
    <listOfReactants><speciesReference species="A"/></listOfReactants>
    <listOfProducts><speciesReference species="B"/></listOfProducts>
    <listOfModifiers><modifierSpeciesReference species="drive"/></listOfModifiers>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><ci>mm</ci><ci>A</ci><ci>k</ci></apply></math></kineticLaw>
   </reaction>
   <reaction id="r2" reversible="true">
    <listOfReactants><speciesReference species="B"/></listOfReactants>
    <listOfProducts><speciesReference species="A"/></listOfProducts>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""


@pytest.fixture
def deposit(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(SBML)
    return read_sbml(str(path), "m")


def test_species_carry_their_compartment_and_markers(deposit):
    """A two-compartment deposit listed without compartments shows every
    species twice with no way to tell the copies apart."""
    by_id = {s.id: s for s in deposit.species}
    assert by_id["A"].compartment == "cyt"
    assert by_id["B"].compartment == "nuc"
    assert by_id["drive"].boundary and by_id["drive"].constant
    assert not by_id["A"].boundary
    assert "cyt" in str(by_id["A"]) and "alpha" in str(by_id["A"])


def test_a_species_in_nothing_is_marked_unused(deposit):
    """Declared, named, and referenced by no reaction and no rule, so an
    intervention wired to it drives nothing."""
    assert [s.id for s in deposit.unused_species] == ["orphan"]
    assert "UNUSED" in str({s.id: s for s in deposit.species}["orphan"])


def test_a_species_only_a_rule_reads_is_not_unused(deposit):
    by_id = {s.id: s for s in deposit.species}
    assert by_id["A"].in_reactions and by_id["A"].in_rules


def test_reactions_report_stoichiometry_and_the_law_as_written(deposit):
    r1 = {r.id: r for r in deposit.reactions}["r1"]
    assert r1.reactants == ("A",) and r1.products == ("B",)
    assert r1.modifiers == ("drive",)
    assert not r1.reversible
    assert "mm(" in r1.law.replace(" ", "")
    assert "A" in str(r1) and "B" in str(r1)


def test_a_reaction_with_no_kinetic_law_is_reported_as_such(deposit):
    assert [r.id for r in deposit.law_free_reactions] == ["r2"]
    assert "no kinetic law" in str({r.id: r for r in deposit.reactions}["r2"])


def test_a_law_that_is_a_call_points_at_the_function_body(deposit):
    """A deposit whose laws are all calls keeps its mechanism where the
    reaction bodies do not show it."""
    r1 = {r.id: r for r in deposit.reactions}["r1"]
    assert r1.calls == ("mm",)
    assert [r.id for r in deposit.reactions_calling_functions] == ["r1"]
    assert "mm" in deposit.functions
    assert "/" in deposit.functions["mm"]


def test_rules_are_reported_with_their_target(deposit):
    assert deposit.rules and deposit.rules[0][1] == "B"
    assert "k" in deposit.rules[0][2]


def test_a_missing_model_element_raises_rather_than_reading_empty(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text('<?xml version="1.0"?><sbml level="2" version="4"/>')
    with pytest.raises(ValueError, match="no SBML model"):
        read_sbml(str(path), "bad")


@pytest.mark.parametrize(
    "formula,identifier,expected",
    [
        ("k * A", "A", True),
        ("k * Ax", "A", False),
        ("xA + 1", "A", False),
        ("f(A)", "A", True),
        ("", "A", False),
        ("comp.A", "A", False),
    ],
)
def test_a_rule_reference_matches_on_a_token_boundary(
    formula, identifier, expected
):
    """Substring matching finds A inside Ax and reports every short species
    id as read by every rule."""
    assert _references(formula, identifier) is expected


#: The importer refuses a reaction with no kinetic law, so the delta tests need
#: a deposit it will accept — which is itself the point of the file reader:
#: it reads deposits the import rejects, and names the reaction responsible.
IMPORTABLE = SBML.replace(
    """   <reaction id="r2" reversible="true">
    <listOfReactants><speciesReference species="B"/></listOfReactants>
    <listOfProducts><speciesReference species="A"/></listOfProducts>
   </reaction>""",
    """   <reaction id="r2" reversible="true">
    <listOfReactants><speciesReference species="B"/></listOfReactants>
    <listOfProducts><speciesReference species="A"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><times/><ci>k</ci><ci>B</ci></apply></math></kineticLaw>
   </reaction>""",
)


def test_a_law_free_reaction_is_named_so_a_refusal_can_be_traced(tmp_path):
    """The law-free reaction above makes this file unimportable; reading it is
    how you find out which reaction is responsible."""
    path = tmp_path / "m.xml"
    path.write_text(SBML)
    assert [r.id for r in read_sbml(str(path), "m").law_free_reactions] == [
        "r2"
    ]


def test_the_command_prints_a_mechanism(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(SBML)
    result = CliRunner().invoke(supply, ["reactions", str(path)])
    assert result.exit_code == 0, result.output
    assert "2 reactions" in result.output
    assert "UNUSED" in result.output
    assert "functionDefinitions" in result.output
    assert "r1: A -> B" in result.output


def test_the_command_says_a_rule_based_deposit_is_not_empty(tmp_path):
    """Zero reactions reads as an empty model, and three contestants have
    written one off for it."""
    path = tmp_path / "rules.xml"
    path.write_text(
        SBML.replace(
            SBML[SBML.index("  <listOfReactions>") :], " </model>\n</sbml>\n"
        )
    )
    result = CliRunner().invoke(supply, ["reactions", str(path)])
    assert result.exit_code == 0, result.output
    assert "mechanism is in its rules" in result.output


EVENTFUL = IMPORTABLE.replace(
    "  <listOfReactions>",
    """  <listOfEvents>
   <event id="dose">
    <trigger><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><geq/><csymbol encoding="text"
      definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
      <cn>5</cn></apply></math></trigger>
    <listOfEventAssignments>
     <eventAssignment variable="drive">
      <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>200</cn></math>
     </eventAssignment>
     <eventAssignment variable="k">
      <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.6</cn></math>
     </eventAssignment>
    </listOfEventAssignments>
   </event>
  </listOfEvents>
  <listOfReactions>""",
    1,
)


class TestEvents:
    """A deposit usually keeps its protocol in its events — the dose, when it
    starts, the knockdown at a given day — so a mechanism dump without them
    omits the intervention the paper performed."""

    def _deposit(self, tmp_path):
        path = tmp_path / "eventful.xml"
        path.write_text(EVENTFUL)
        return read_sbml(str(path), "m")

    def test_the_trigger_and_its_assignments_are_read(self, tmp_path):
        events = self._deposit(tmp_path).events
        assert len(events) == 1
        event = events[0]
        assert event.id == "dose"
        assert "5" in event.trigger
        assert dict(event.assignments) == {"drive": "200", "k": "0.6"}

    def test_it_names_what_an_event_overwrites(self, tmp_path):
        """The trap: a handle set on one of these works until the protocol
        fires and then quietly stops working, because an event assigns
        outright."""
        deposit = self._deposit(tmp_path)
        assert set(deposit.event_written) == {"drive", "k"}
        assert set(deposit.events[0].writes) == {"drive", "k"}

    def test_a_deposit_with_no_events_reports_none(self, tmp_path):
        path = tmp_path / "plain.xml"
        path.write_text(IMPORTABLE)
        deposit = read_sbml(str(path), "m")
        assert deposit.events == ()
        assert deposit.event_written == ()

    def test_the_command_prints_events_and_the_warning(self, tmp_path):
        path = tmp_path / "eventful.xml"
        path.write_text(EVENTFUL)
        result = CliRunner().invoke(
            supply, ["reactions", str(path), "--no-species"]
        )
        assert result.exit_code == 0, result.output
        assert "Events" in result.output
        assert "sets drive = 200" in result.output
        assert "overwritten when it fires" in result.output


ANNOTATED = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"
      timeUnits="second" substanceUnits="mole">
 <model id="m" timeUnits="second" substanceUnits="substance">
  <notes><body xmlns="http://www.w3.org/1999/xhtml"><p>A model of nothing.</p></body></notes>
  <listOfUnitDefinitions>
   <unitDefinition id="substance">
    <listOfUnits><unit kind="mole" scale="-9" exponent="1" multiplier="1"/></listOfUnits>
   </unitDefinition>
  </listOfUnitDefinitions>
  <listOfCompartments>
   <compartment id="cyt" size="0.5" spatialDimensions="3" constant="true">
    <notes><body xmlns="http://www.w3.org/1999/xhtml"><p>Measured by stereology.</p></body></notes>
   </compartment>
  </listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="cyt" initialAmount="7" substanceUnits="substance"
            hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false">
    <notes><body xmlns="http://www.w3.org/1999/xhtml"><p>Counted, not measured.</p></body></notes>
   </species>
   <species id="B" compartment="cyt" initialConcentration="1"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="kdeg" value="0" constant="true">
    <notes><body xmlns="http://www.w3.org/1999/xhtml"><p>Zero is the inhibited arm; normal is 1e-3.</p></body></notes>
   </parameter>
   <parameter id="kon" value="2" units="second" constant="true"/>
  </listOfParameters>
  <listOfInitialAssignments>
   <initialAssignment symbol="B">
    <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>kon</ci></math>
   </initialAssignment>
  </listOfInitialAssignments>
  <listOfConstraints>
   <constraint>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><lt/><ci>A</ci><cn>100</cn></apply></math>
    <message><p xmlns="http://www.w3.org/1999/xhtml">A above 100 is outside the fit.</p></message>
   </constraint>
  </listOfConstraints>
  <listOfReactions>
   <reaction id="r1" reversible="false">
    <listOfReactants><speciesReference species="A" stoichiometry="2"/></listOfReactants>
    <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
    <kineticLaw>
     <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><times/><ci>kloc</ci><ci>A</ci></apply></math>
     <listOfLocalParameters><localParameter id="kloc" value="0.25"/></listOfLocalParameters>
    </kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""


class TestItReadsEverythingTheFileCarries:
    """Every construct below was in the file and dropped before.

    The corpus counts are from the 2,531 cached BioModels deposits: 2,322 carry
    a model note, 966 a stoichiometry other than 1, 892 rate constants local to
    a kinetic law, 316 an initialAssignment, 206 a note on a parameter.
    """

    @pytest.fixture
    def d(self, tmp_path):
        path = tmp_path / "annotated.xml"
        path.write_text(ANNOTATED)
        return read_sbml(str(path), "m")

    def test_a_note_on_a_parameter_is_read_and_kept_with_it(self, d):
        """The Proctor case: the deposit ships a rate at zero and says in a note
        on that parameter that zero is its inhibited condition."""
        kdeg = {p.id: p for p in d.parameters}["kdeg"]
        assert kdeg.value == 0.0 and kdeg.zero
        assert "inhibited arm" in kdeg.notes
        assert [p.id for p in d.zero_parameters] == ["kdeg"]
        assert "ZERO" in str(kdeg) and "inhibited arm" in str(kdeg)

    def test_every_note_is_reachable_with_where_it_hangs(self, d):
        where = dict(d.annotated_notes)
        assert "A model of nothing." in where["model"]
        assert "inhibited arm" in where["parameter kdeg"]
        assert "Counted, not measured." in where["species A"]
        assert "Measured by stereology." in where["compartment cyt"]

    def test_stoichiometry_is_kept_and_shown(self, d):
        """2 A -> B is not A -> B, and 966 deposits have a coefficient."""
        r1 = d.reactions[0]
        assert ("reactants", "A", 2.0) in r1.stoichiometry
        assert "2 A ->" in str(r1)

    def test_a_constant_local_to_a_kinetic_law_is_read(self, d):
        """No parameter list holds these, so nothing could see them before."""
        r1 = d.reactions[0]
        assert r1.local_parameters == (("kloc", 0.25, ""),)
        assert "local to this law: kloc = 0.25" in str(r1)
        assert "kloc" not in {p.id for p in d.parameters}

    def test_amount_and_concentration_are_told_apart(self, d):
        """They differ by the compartment's size, which here is 0.5."""
        by_id = {s.id: s for s in d.species}
        assert by_id["A"].initial == 7 and by_id["A"].initial_kind == "amount"
        assert by_id["B"].initial_kind == "concentration"
        assert by_id["A"].only_substance_units
        assert "(amount)" in str(by_id["A"])

    def test_the_compartment_size_is_read_not_just_its_id(self, d):
        cyt = d.compartments[0]
        assert cyt.id == "cyt" and cyt.size == 0.5
        assert "size 0.5" in str(cyt)

    def test_unit_definitions_are_rendered(self, d):
        assert "1e-9 mole" in str(d.units[0])
        assert dict(d.model_units)["time"] == "second"

    def test_an_initial_assignment_is_read(self, d):
        """It overrides the initial value on the species, so reading the
        species list alone gives the wrong starting point."""
        assert d.initial_assignments[0].symbol == "B"
        assert "kon" in d.initial_assignments[0].math

    def test_a_constraint_carries_the_authors_own_message(self, d):
        assert "outside the fit" in d.constraints[0].message
        assert "100" in d.constraints[0].math

    def test_the_sbml_level_and_version_are_reported(self, d):
        assert (d.sbml_level, d.sbml_version) == (3, 1)

    def test_rules_still_unpack_as_the_triple_they_were(self, deposit):
        """Existing readers index these positionally; that keeps working."""
        kind, target, formula = deposit.rules[0]
        assert (kind, target) == ("assignment", "B")
        assert deposit.rules[0][1] == "B" and deposit.rules[0].target == "B"

    def test_the_command_prints_the_parameters_and_the_zero_note(
        self, tmp_path
    ):
        path = tmp_path / "annotated.xml"
        path.write_text(ANNOTATED)
        result = CliRunner().invoke(
            supply, ["reactions", str(path), "--no-species"]
        )
        assert result.exit_code == 0, result.output
        assert "Parameters" in result.output
        assert "kdeg = 0  [ZERO]" in result.output
        assert "inhibited arm" in result.output
        assert "1 of these is exactly zero" in result.output
        assert "yours to decide" in result.output
        assert "Constraints" in result.output
        assert "initialAssignments" in result.output

    def test_the_notes_flag_prints_every_note_in_full(self, tmp_path):
        path = tmp_path / "annotated.xml"
        path.write_text(ANNOTATED)
        result = CliRunner().invoke(
            supply,
            ["reactions", str(path), "--no-species", "--notes"],
        )
        assert result.exit_code == 0, result.output
        assert "[parameter kdeg]" in result.output
        assert "[compartment cyt]" in result.output


QUALIFIED = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m" metaid="mm">
  <listOfCompartments><compartment id="cyt" size="1"/></listOfCompartments>
  <listOfSpecies>
   <species id="exact" compartment="cyt" initialConcentration="1" metaid="s1">
    <annotation>
     <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
              xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
      <rdf:Description rdf:about="#s1">
       <bqbiol:is><rdf:Bag>
        <rdf:li rdf:resource="http://identifiers.org/chebi/CHEBI:16240"/>
       </rdf:Bag></bqbiol:is>
      </rdf:Description>
     </rdf:RDF>
    </annotation>
   </species>
   <species id="loose" compartment="cyt" initialConcentration="1" metaid="s2">
    <annotation>
     <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
              xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
      <rdf:Description rdf:about="#s2">
       <bqbiol:isVersionOf><rdf:Bag>
        <rdf:li rdf:resource="http://identifiers.org/uniprot/P04040"/>
       </rdf:Bag></bqbiol:isVersionOf>
      </rdf:Description>
     </rdf:RDF>
    </annotation>
   </species>
  </listOfSpecies>
  <listOfReactions>
   <reaction id="r1" reversible="false">
    <listOfReactants><speciesReference species="exact"/></listOfReactants>
    <listOfProducts><speciesReference species="loose"/></listOfProducts>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""


class TestItKeepsTheAnnotationQualifier:
    """``is`` licenses an identity join; the others do not, and the importer
    keeps the term while dropping which one it was."""

    @pytest.fixture
    def d(self, tmp_path):
        path = tmp_path / "qualified.xml"
        path.write_text(QUALIFIED)
        return read_sbml(str(path), "m")

    def test_the_qualifier_comes_back_with_the_term(self, d):
        by_id = {s.id: s for s in d.species}
        assert by_id["exact"].annotations == (("is", "chebi", "CHEBI:16240"),)
        assert by_id["loose"].annotations == (
            ("isVersionOf", "uniprot", "P04040"),
        )

    def test_only_the_weaker_ones_are_called_out(self, d):
        assert d.inexact_species == (
            ("loose", "isVersionOf", "uniprot:P04040"),
        )

    def test_the_rendering_shows_the_qualifier(self, d):
        by_id = {s.id: s for s in d.species}
        assert "is chebi:CHEBI:16240" in str(by_id["exact"])
        assert "isVersionOf uniprot:P04040" in str(by_id["loose"])

    def test_the_command_warns_on_a_join_that_is_not_identity(self, tmp_path):
        path = tmp_path / "qualified.xml"
        path.write_text(QUALIFIED)
        result = CliRunner().invoke(supply, ["reactions", str(path)])
        assert result.exit_code == 0, result.output
        assert "weaker than identity" in result.output
        assert "loose isVersionOf uniprot:P04040" in result.output

    def test_a_species_annotated_is_raises_no_caution(self, tmp_path):
        path = tmp_path / "plain.xml"
        path.write_text(IMPORTABLE)
        assert read_sbml(str(path), "m").inexact_species == ()
