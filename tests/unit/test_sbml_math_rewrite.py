"""MathML ``<log/>`` with a base and ``<root/>`` import as ``ln/ln`` and a power."""

import libsbml
import jax.numpy as jnp
import pytest

from hallsim.sbml_import import _rewrite_math_functions, process_from_sbml


def _model_with_log(base: int | None):
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("logmodel")
    c = m.createCompartment()
    c.setId("cell")
    c.setSize(1.0)
    c.setConstant(True)
    s = m.createSpecies()
    s.setId("x")
    s.setCompartment("cell")
    s.setInitialConcentration(100.0)
    s.setHasOnlySubstanceUnits(False)
    s.setBoundaryCondition(False)
    s.setConstant(False)
    r = m.createReaction()
    r.setId("decay")
    r.setReversible(False)
    sr = r.createReactant()
    sr.setSpecies("x")
    sr.setStoichiometry(1.0)
    sr.setConstant(True)
    kl = r.createKineticLaw()
    logbase = "" if base is None else f"<logbase><cn>{base}</cn></logbase>"
    kl.setMath(
        libsbml.readMathMLFromString(
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"<apply><log/>{logbase}<ci>x</ci></apply></math>"
        )
    )
    return doc


@pytest.mark.parametrize("base", [None, 10, 2])
def test_rewrite_prints_natural_logs(base):
    doc = _model_with_log(base)
    formula = libsbml.formulaToString(
        doc.getModel().getReaction(0).getKineticLaw().getMath()
    )
    assert "log10" in formula or "log(" in formula
    assert _rewrite_math_functions(doc.getModel()) >= 1
    rewritten = libsbml.formulaToString(
        doc.getModel().getReaction(0).getKineticLaw().getMath()
    )
    assert "log10" not in rewritten
    # The base prints in exponent form, which is what makes it a float once
    # the translator emits it as Python source.
    assert rewritten == f"log(x) / log({base or 10}e0)"


def test_rewrite_is_idempotent_on_natural_logs():
    doc = _model_with_log(10)
    _rewrite_math_functions(doc.getModel())
    assert _rewrite_math_functions(doc.getModel()) == 0


def test_an_integer_literal_becomes_a_float_but_an_exponent_does_not():
    """``pow(1500, 6)`` folds to 1.139e19 and overflows int64, rejecting the
    whole model. Coercing the base fixes it; coercing the exponent would
    change ``x ** 6`` from defined at negative ``x`` to NaN."""
    doc = _model_with_log(10)
    kl = doc.getModel().getReaction(0).getKineticLaw()
    kl.setMath(
        libsbml.readMathMLFromString(
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><power/>'
            "<cn type='integer'>1500</cn><cn type='integer'>6</cn>"
            "</apply></math>"
        )
    )
    _rewrite_math_functions(doc.getModel())
    out = libsbml.formulaToString(kl.getMath())
    assert out == "pow(1500e0, 6)", out
    # The base is float, so the fold is float and 1.139e19 is representable;
    # the exponent stays an int, so a negative base would still be defined.
    assert isinstance(eval(out), float)


def test_log_base_model_imports_and_evaluates(tmp_path):
    path = tmp_path / "logmodel.xml"
    libsbml.writeSBMLToFile(_model_with_log(10), str(path))
    proc = process_from_sbml(str(path), name="lg")
    rhs = proc.derivative(0.0, {"x": jnp.asarray(100.0)})
    # d x/dt = -log10(100) = -2
    assert jnp.allclose(rhs["x"], -2.0, rtol=1e-6)


def _model_with_root(degree: int | None):
    doc = _model_with_log(10)
    kl = doc.getModel().getReaction(0).getKineticLaw()
    deg = "" if degree is None else f"<degree><cn>{degree}</cn></degree>"
    kl.setMath(
        libsbml.readMathMLFromString(
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"<apply><root/>{deg}<ci>x</ci></apply></math>"
        )
    )
    return doc


@pytest.mark.parametrize(
    "degree,expect", [(None, -10.0), (3, -(100 ** (1 / 3)))]
)
def test_root_imports_as_a_power(tmp_path, degree, expect):
    path = tmp_path / "rootmodel.xml"
    libsbml.writeSBMLToFile(_model_with_root(degree), str(path))
    proc = process_from_sbml(str(path), name="rt")
    rhs = proc.derivative(0.0, {"x": jnp.asarray(100.0)})
    assert jnp.allclose(rhs["x"], expect, rtol=1e-6)
