"""Reading a deposit's mechanism from its file, and diffing it against import."""

import pytest
from click.testing import CliRunner

from hallsim.cli import supply
from hallsim.sbml_inspect import _references, import_delta, inspect_sbml

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
    return inspect_sbml(str(path), "m")


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
        inspect_sbml(str(path), "bad")


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


def test_the_file_reads_a_deposit_the_import_refuses(tmp_path):
    """The law-free reaction above makes this file unimportable; reading it is
    how a contestant finds out which reaction is responsible."""
    from hallsim.sbml_core import UnsupportedSBMLFeatureError

    path = tmp_path / "m.xml"
    path.write_text(SBML)
    assert [r.id for r in inspect_sbml(str(path), "m").law_free_reactions] == [
        "r2"
    ]
    with pytest.raises(UnsupportedSBMLFeatureError, match="r2"):
        import_delta(str(path), "m")

    result = CliRunner().invoke(supply, ["reactions", str(path)])
    assert result.exit_code == 0, result.output
    assert "the import refuses this deposit" in result.output
    assert "r2" in result.output


class TestImportDelta:
    """The half reading the file cannot give you."""

    def test_it_counts_what_the_import_exposed(self, tmp_path):
        path = tmp_path / "m.xml"
        path.write_text(IMPORTABLE)
        delta = import_delta(str(path), "m")
        assert delta.declared_species == 4
        assert delta.ported_species > 0

    def test_an_annotation_the_ports_lose_is_named(self, tmp_path):
        """The silent failure this exists for: the file carries an id, the port
        does not, and a readout joins to nothing while reporting success."""
        annotated = IMPORTABLE.replace(
            '<species id="drive" compartment="cyt" initialConcentration="1"\n'
            '            boundaryCondition="true" constant="true"/>',
            '<species id="drive" compartment="cyt" initialConcentration="1"'
            ' boundaryCondition="true" constant="true" metaid="d1">'
            "<annotation>"
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            ' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">'
            '<rdf:Description rdf:about="#d1"><bqbiol:is><rdf:Bag>'
            '<rdf:li rdf:resource="http://identifiers.org/chebi/CHEBI:5931"/>'
            "</rdf:Bag></bqbiol:is></rdf:Description></rdf:RDF>"
            "</annotation></species>",
        )
        path = tmp_path / "annotated.xml"
        path.write_text(annotated)
        delta = import_delta(str(path), "m")
        assert "drive" in delta.annotated_in_file
        # Whether a boundary constant should be a port is a design call; that
        # its identifier vanishes either way is not.
        if "drive" not in delta.annotated_in_ports:
            assert "drive" in delta.lost_annotations
            assert "ANNOTATION LOST" in str(delta)


def test_the_command_prints_a_mechanism(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(SBML)
    result = CliRunner().invoke(supply, ["reactions", str(path), "--no-delta"])
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
    result = CliRunner().invoke(supply, ["reactions", str(path), "--no-delta"])
    assert result.exit_code == 0, result.output
    assert "mechanism is in its rules" in result.output
