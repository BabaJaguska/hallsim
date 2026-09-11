"""The compiled SBML core: layout, initial values, rates, refusals."""

import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallsim.sbml_core import (
    UnsupportedSBMLFeatureError,
    compile_sbml,
    unsupported_features,
)

HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="m">
"""
TAIL = "</model></sbml>\n"

COMPARTMENTS = (
    HEAD
    + """
<listOfCompartments>
  <compartment id="cyt" spatialDimensions="3" size="2.5" constant="true"/>
  <compartment id="nuc" spatialDimensions="3" size="0.4" constant="true"/>
</listOfCompartments>
<listOfSpecies>
  <species id="A" compartment="cyt" initialConcentration="1.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="B" compartment="cyt" initialAmount="0.5" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
  <species id="C" compartment="nuc" initialConcentration="0.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="E" compartment="cyt" initialConcentration="0.3" hasOnlySubstanceUnits="false" boundaryCondition="true" constant="true"/>
</listOfSpecies>
<listOfParameters><parameter id="k1" value="0.7" constant="true"/></listOfParameters>
<listOfReactions>
  <reaction id="bind" reversible="false">
    <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="C" stoichiometry="1" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cyt</ci><ci>k1</ci><ci>A</ci><ci>B</ci><ci>E</ci></apply></math></kineticLaw>
  </reaction>
  <reaction id="release" reversible="false">
    <listOfReactants><speciesReference species="C" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="A" stoichiometry="2" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>nuc</ci><ci>kloc</ci><ci>C</ci></apply></math>
      <listOfLocalParameters><localParameter id="kloc" value="0.15"/></listOfLocalParameters>
    </kineticLaw>
  </reaction>
</listOfReactions>
"""
    + TAIL
)

RULES = (
    HEAD
    + """
<listOfFunctionDefinitions>
  <functionDefinition id="hill"><math xmlns="http://www.w3.org/1998/Math/MathML"><lambda><bvar><ci>x</ci></bvar><bvar><ci>K</ci></bvar><bvar><ci>n</ci></bvar>
    <apply><divide/><apply><power/><ci>x</ci><ci>n</ci></apply><apply><plus/><apply><power/><ci>K</ci><ci>n</ci></apply><apply><power/><ci>x</ci><ci>n</ci></apply></apply></apply></lambda></math></functionDefinition>
</listOfFunctionDefinitions>
<listOfCompartments><compartment id="cell" spatialDimensions="3" size="1.5" constant="true"/></listOfCompartments>
<listOfSpecies>
  <species id="S" compartment="cell" initialConcentration="2.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="P" compartment="cell" initialConcentration="0.1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="D" compartment="cell" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="R" compartment="cell" initialConcentration="0.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters>
  <parameter id="vmax" value="1.2" constant="true"/>
  <parameter id="K" value="0.8" constant="true"/>
  <parameter id="kdeg" value="0.3" constant="true"/>
  <parameter id="drive" constant="false"/>
  <parameter id="total" constant="false"/>
</listOfParameters>
<listOfInitialAssignments>
  <initialAssignment symbol="D"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>0.5</cn><ci>S</ci></apply></math></initialAssignment>
</listOfInitialAssignments>
<listOfRules>
  <assignmentRule variable="drive"><math xmlns="http://www.w3.org/1998/Math/MathML"><piecewise><piece><apply><times/><cn>2</cn><ci>total</ci></apply><apply><lt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol><cn>3</cn></apply></piece><otherwise><ci>total</ci></otherwise></piecewise></math></assignmentRule>
  <assignmentRule variable="total"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>S</ci><ci>P</ci></apply></math></assignmentRule>
  <rateRule variable="R"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><apply><times/><cn>0.4</cn><ci>drive</ci></apply><apply><times/><ci>kdeg</ci><ci>R</ci></apply></apply></math></rateRule>
</listOfRules>
<listOfReactions>
  <reaction id="convert" reversible="false">
    <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="P" stoichiometry="1" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cell</ci><ci>vmax</ci><apply><ci>hill</ci><ci>S</ci><ci>K</ci><cn>2</cn></apply><apply><ci>hill</ci><ci>D</ci><cn>0.5</cn><cn>1</cn></apply></apply></math></kineticLaw>
  </reaction>
</listOfReactions>
"""
    + TAIL
)


def _write(tmp_path, name, text):
    path = tmp_path / f"{name}.xml"
    path.write_text(textwrap.dedent(text))
    return str(path)


def test_compartments_layout_initials_and_rates(tmp_path):
    core = compile_sbml(_write(tmp_path, "compartments", COMPARTMENTS))
    assert list(core.y_indexes) == ["A", "B", "C"]
    assert core.w_indexes == {}
    assert list(core.c_indexes) == ["k1", "E", "cyt", "nuc", "release_kloc"]
    # amounts: A is 1.0 mM in 2.5 L; B is an amount; E is a boundary amount
    assert core.y0 == (2.5, 0.5, 0.0)
    assert core.c0 == (0.7, 0.75, 2.5, 0.4, 0.15)
    assert core.reaction_ids == ("bind", "release")
    assert core.stoichiometry == ((-1.0, 2.0), (-1.0, 0.0), (1.0, -1.0))
    y, w, c = jnp.asarray(core.y0), jnp.asarray(core.w0), jnp.asarray(core.c0)
    v = np.asarray(core.reaction_velocities(y, w, c, 0.0))
    # cyt * k1 * [A] * B * [E] = 2.5 * 0.7 * 1.0 * 0.5 * 0.3
    assert v == pytest.approx([0.2625, 0.0])
    dy = np.asarray(jax.jit(core.ratefunc)(y, 0.0, w, c))
    assert dy == pytest.approx([-0.2625, -0.2625, 0.2625])
    assert core.reads == (frozenset({"A", "B"}), frozenset({"C"}))


def test_rules_initial_assignment_and_rate_rule(tmp_path):
    core = compile_sbml(_write(tmp_path, "rules", RULES))
    assert list(core.y_indexes) == ["S", "P", "D", "R"]
    assert list(core.w_indexes) == ["total", "drive"]  # dependency order
    # D = 0.5 * [S] = 1.0 mM, stored as an amount in 1.5 L
    assert core.y0 == pytest.approx((3.0, 0.15, 1.5, 0.0))
    # total = [S] + [P] = 2.1; drive = 2 * total before t = 3
    assert core.w0 == pytest.approx((2.1, 4.2))
    y, w, c = jnp.asarray(core.y0), jnp.asarray(core.w0), jnp.asarray(core.c0)
    w_later = np.asarray(core.assignmentfunc(y, w, c, 5.0))
    assert w_later == pytest.approx([2.1, 2.1])
    dy = np.asarray(core.ratefunc(y, 0.0, w, c))
    hill = lambda x, K, n: x**n / (K**n + x**n)  # noqa: E731
    v = 1.5 * 1.2 * hill(2.0, 0.8, 2) * hill(1.0, 0.5, 1)
    # dR/dt is a concentration rate, scaled by the compartment into the store
    assert dy == pytest.approx([-v, v, 0.0, 1.5 * (0.4 * 4.2 - 0.0)])
    assert core.reads[0] == frozenset({"S", "D"})
    grad = jax.grad(lambda yy: core.ratefunc(yy, 0.0, w, c)[1])(y)
    assert np.isfinite(np.asarray(grad)).all()


def test_core_is_cached_per_file(tmp_path):
    path = _write(tmp_path, "compartments", COMPARTMENTS)
    assert compile_sbml(path) is compile_sbml(path)


@pytest.mark.parametrize(
    "snippet,message",
    [
        (
            '<listOfRules><algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>A</ci><ci>B</ci></apply></math></algebraicRule></listOfRules>',
            "algebraic",
        ),
        (
            '<listOfRules><assignmentRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/delay">delay</csymbol><ci>B</ci><cn>1</cn></apply></math></assignmentRule></listOfRules>',
            "FUNCTION_DELAY",
        ),
    ],
)
def test_unsupported_constructs_are_named(tmp_path, snippet, message):
    text = COMPARTMENTS.replace(
        "<listOfParameters>", snippet + "<listOfParameters>"
    )
    path = _write(tmp_path, "bad", text)
    issues = unsupported_features(path)
    assert any(message in i for i in issues), issues
    with pytest.raises((UnsupportedSBMLFeatureError, Exception)):
        compile_sbml(path)


def test_unset_initial_value_is_refused(tmp_path):
    text = COMPARTMENTS.replace(' initialAmount="0.5"', "")
    with pytest.raises(
        UnsupportedSBMLFeatureError, match="'B' has no initial"
    ):
        compile_sbml(_write(tmp_path, "noinit", text))
