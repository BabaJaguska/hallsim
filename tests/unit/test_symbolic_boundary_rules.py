"""A boundary species' time-dependent rule is a rule in the symbolic
field, not a constant at its initial value."""

import sympy

from hallsim.composite import single_process_composite
from hallsim.sbml_import import process_from_sbml
from hallsim.sbml_math import TIME
from hallsim.structure import symbolic_field

SBML = "demos/models/sbml/dallepezze2014/dallepezze2014_BIOMD0000000582.xml"


def test_the_irradiation_pulse_is_in_the_field_not_in_the_parameters():
    comp = single_process_composite(process_from_sbml(SBML, name="dp14"))
    field = symbolic_field(comp)
    damage = sympy.sympify(field.derivatives["dp14/DNA_damage"])
    assert damage.has(TIME)  # the pulse is a piecewise in time
    assert "dp14.parameters.Irradiation" not in field.parameters
    # the pulse is on during the first five minutes and off after
    on = damage.xreplace({TIME: sympy.Float(0.001)})
    off = damage.xreplace({TIME: sympy.Float(0.5)})
    assert on != off
