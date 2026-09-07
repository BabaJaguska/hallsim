"""Rate laws written as stochastic propensities.

`k*x*(x-1)/2` counts distinct pairs among x molecules — a Gillespie
propensity. As an ODE it is negative for 0 < x < 1, and small pools sit
there. Found by hand on Hui 2016 and again on Proctor 2013, the second time
only after a reviewer traced the negative through seven downstream laws.
"""

import textwrap

import pytest

from hallsim.intake import combinatorial_propensities

libsbml = pytest.importorskip("libsbml")

SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="c" initialAmount="10"/>
   <species id="B" compartment="c" initialAmount="0"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
  <listOfReactions>
   <reaction id="{rid}">
    <listOfReactants><speciesReference species="A" stoichiometry="2"/>
    </listOfReactants>
    <listOfProducts><speciesReference species="B"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     {math}
    </math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""

PROPENSITY = """<apply><times/><ci>k</ci><ci>A</ci>
  <apply><minus/><ci>A</ci><cn>1</cn></apply></apply>"""
MEAN_FIELD = """<apply><times/><ci>k</ci><ci>A</ci><ci>A</ci></apply>"""


def _write(tmp_path, math, rid="dimerise"):
    path = tmp_path / "m.xml"
    path.write_text(textwrap.dedent(SBML.format(rid=rid, math=math)))
    return path


def test_a_propensity_is_reported(tmp_path):
    (hit,) = combinatorial_propensities(_write(tmp_path, PROPENSITY))
    assert hit.species == "A"
    assert hit.reaction == "dimerise"
    assert "mean-field" in str(hit)


def test_the_mean_field_form_is_not_reported(tmp_path):
    assert combinatorial_propensities(_write(tmp_path, MEAN_FIELD)) == ()


def test_a_parameter_minus_one_is_not_a_propensity(tmp_path):
    """`k*(k-1)` on a parameter is arithmetic, not a pair count — only a
    species can be a molecule number."""
    math = """<apply><times/><ci>k</ci>
      <apply><minus/><ci>k</ci><cn>1</cn></apply></apply>"""
    assert combinatorial_propensities(_write(tmp_path, math)) == ()


def test_a_missing_file_yields_nothing_rather_than_raising(tmp_path):
    assert combinatorial_propensities(tmp_path / "absent.xml") == ()
