"""libsbml ↔ sympy ↔ JAX translation of SBML math."""

import glob
import math

import jax
import jax.numpy as jnp
import libsbml
import numpy as np
import pytest
import sympy

from hallsim.sbml_math import (
    AVOGADRO_VALUE,
    TIME,
    UnsupportedMathError,
    function_definitions,
    inline_functions,
    to_ast,
    to_jax,
    to_sympy,
)

x, y = sympy.symbols("x y")
POINT = {"x": 2.5, "y": -1.5}

# (L3 formula, expected value at POINT, t=0.75)
CASES = [
    ("x + y * 2 - 1", 2.5 - 3.0 - 1),
    ("-x", -2.5),
    ("x / y", 2.5 / -1.5),
    ("x ^ 3", 2.5**3),
    ("pow(x, 3)", 2.5**3),
    ("root(3, 27)", 3.0),
    ("sqrt(x)", math.sqrt(2.5)),
    ("log(2, 8)", 3.0),
    ("log(100)", 2.0),
    ("log10(1000)", 3.0),
    ("ln(x)", math.log(2.5)),
    ("exp(y)", math.exp(-1.5)),
    ("abs(y)", 1.5),
    ("floor(x) + ceiling(y)", 2.0 + -1.0),
    ("factorial(4)", 24.0),
    ("min(x, y) + max(x, 0)", -1.5 + 2.5),
    ("piecewise(1, x > 3, 2, x > 2, 0)", 2.0),
    ("piecewise(1, x < 0)", float("nan")),
    ("piecewise(7, x > 0 && y < 0, 0)", 7.0),
    ("piecewise(7, x < 0 || y < 0, 0)", 7.0),
    ("piecewise(7, !(x < 0), 0)", 7.0),
    ("piecewise(7, x >= 2.5 && x <= 2.5 && x == 2.5 && y != 0, 0)", 7.0),
    ("piecewise(1, true, 0) + piecewise(1, false, 0)", 1.0),
    (
        "sin(x) + cos(x) + tan(x)",
        math.sin(2.5) + math.cos(2.5) + math.tan(2.5),
    ),
    (
        "sec(x) + csc(x) + cot(x)",
        1 / math.cos(2.5) + 1 / math.sin(2.5) + 1 / math.tan(2.5),
    ),
    (
        "arcsin(0.5) + arccos(0.5) + arctan(x)",
        math.asin(0.5) + math.acos(0.5) + math.atan(2.5),
    ),
    (
        "arcsec(x) + arccsc(x) + arccot(x)",
        math.acos(1 / 2.5) + math.asin(1 / 2.5) + math.atan(1 / 2.5),
    ),
    (
        "sinh(y) + cosh(y) + tanh(y)",
        math.sinh(-1.5) + math.cosh(-1.5) + math.tanh(-1.5),
    ),
    (
        "sech(y) + csch(y) + coth(y)",
        1 / math.cosh(-1.5) + 1 / math.sinh(-1.5) + 1 / math.tanh(-1.5),
    ),
    (
        "arcsinh(x) + arccosh(x) + arctanh(0.5)",
        math.asinh(2.5) + math.acosh(2.5) + math.atanh(0.5),
    ),
    (
        "arcsech(0.5) + arccsch(x) + arccoth(x)",
        math.acosh(2.0) + math.asinh(1 / 2.5) + math.atanh(1 / 2.5),
    ),
    ("pi + exponentiale", math.pi + math.e),
    ("quotient(7, 2) + rem(7, 2)", 3.0 + 1.0),
    ("quotient(-7, 2) + rem(-7, 2)", -3.0 + -1.0),
    ("time * 4", 3.0),
    ("avogadro / 1e23", AVOGADRO_VALUE / 1e23),
]


def _evaluate(expr):
    f = to_jax(expr, [x, y, TIME])
    return float(jax.jit(f)(POINT["x"], POINT["y"], 0.75))


@pytest.mark.parametrize("formula,expected", CASES, ids=[c[0] for c in CASES])
def test_formula_evaluates_under_jit(formula, expected):
    node = libsbml.parseL3Formula(formula)
    assert node is not None, libsbml.getLastParseL3Error()
    got = _evaluate(to_sympy(node))
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("formula,expected", CASES, ids=[c[0] for c in CASES])
def test_round_trip_through_libsbml(formula, expected):
    expr = to_sympy(libsbml.parseL3Formula(formula))
    node = to_ast(expr)
    assert node.isWellFormedASTNode()
    again = to_sympy(node)
    a, b = _evaluate(expr), _evaluate(again)
    if math.isnan(a):
        assert math.isnan(b)
    else:
        assert b == pytest.approx(a, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "formula,name",
    [("delay(x, 2)", "FUNCTION_DELAY"), ("rateOf(x)", "FUNCTION_RATE_OF")],
)
def test_unsupported_nodes_are_refused_by_name(formula, name):
    node = libsbml.parseL3Formula(formula)
    with pytest.raises(UnsupportedMathError, match=name):
        to_sympy(node)


def test_no_sympy_form_is_refused_by_name():
    with pytest.raises(UnsupportedMathError, match="cot"):
        to_ast(sympy.cot(x))


def test_float_literals_print_exactly():
    f = to_jax(to_sympy(libsbml.parseL3Formula("0.1 * x + 1e-300")), [x])
    assert float(f(1.0)) == 0.1 * 1.0 + 1e-300


def test_big_integer_literals_do_not_overflow_under_jit():
    expr = to_sympy(libsbml.parseL3Formula("pow(1500, 6) * x"))
    assert expr.atoms(sympy.Integer)  # folded to 1.139e19 by sympy
    f = jax.jit(to_jax(expr, [x]))
    assert float(f(1.0)) == pytest.approx(1500.0**6)
    node = to_ast(expr)
    assert node.isWellFormedASTNode()
    assert _evaluate(to_sympy(node)) == pytest.approx(1500.0**6 * 2.5)


def test_symbols_that_are_python_keywords_or_csymbols_lambdify():
    lam = sympy.Symbol("lambda")
    f = to_jax(lam * x + TIME, [lam, x, TIME])
    assert float(jax.jit(f)(2.0, 3.0, 0.5)) == 6.5


def test_integer_exponents_stay_integers():
    expr = to_sympy(libsbml.parseL3Formula("x ^ 6"))
    assert expr.exp == 6 and expr.exp.is_Integer
    f = to_jax(expr, [x])
    assert float(f(-2.0)) == 64.0


def test_grad_and_vmap_through_piecewise():
    expr = to_sympy(
        libsbml.parseL3Formula("piecewise(x ^ 2, x > 1, 2 * x - 1)")
    )
    f = to_jax(expr, [x])
    grads = jax.vmap(jax.grad(f))(jnp.asarray([0.0, 0.5, 2.0]))
    assert np.allclose(grads, [2.0, 2.0, 4.0])


FUNCTIONS_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m">
    <listOfFunctionDefinitions>
      <functionDefinition id="f">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>a</ci></bvar><bvar><ci>b</ci></bvar>
            <apply><plus/><apply><times/><ci>a</ci><ci>b</ci></apply><cn>1</cn></apply>
          </lambda>
        </math>
      </functionDefinition>
      <functionDefinition id="g">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>u</ci></bvar>
            <apply><ci>f</ci><ci>u</ci><cn>2</cn></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies><species id="S" compartment="c" initialAmount="3"/></listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5"/></listOfParameters>
    <listOfReactions>
      <reaction id="r">
        <listOfReactants><speciesReference species="S"/></listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><apply><ci>g</ci><ci>S</ci></apply></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_function_definitions_inline_recursively():
    model = libsbml.readSBMLFromString(FUNCTIONS_SBML).getModel()
    defs = function_definitions(model)
    assert set(defs) == {"f", "g"}
    law = to_sympy(model.getReaction(0).getKineticLaw().getMath())
    k, S = sympy.symbols("k S")
    assert sympy.simplify(inline_functions(law, defs) - k * (2 * S + 1)) == 0
    with pytest.raises(UnsupportedMathError, match="undefined function"):
        inline_functions(sympy.Function("h")(x), defs)
    lam = to_ast(defs["g"])
    assert lam.getType() == libsbml.AST_LAMBDA


def _random_point(symbols, rng):
    return {
        s: float(v)
        for s, v in zip(symbols, rng.uniform(0.5, 2.0, len(symbols)))
    }


@pytest.mark.parametrize(
    "path",
    sorted(glob.glob("demos/models/sbml/*/*.xml")),
    ids=lambda p: p.split("/")[-1],
)
def test_vendored_corpus_translates_and_round_trips(path):
    from hallsim.sbml_core import collect_math_nodes

    model = libsbml.readSBMLFromFile(path).getModel()
    defs = function_definitions(model)
    rng = np.random.default_rng(0)
    n = 0
    for node in collect_math_nodes(model):
        expr = to_sympy(node)
        if isinstance(expr, sympy.Lambda):
            continue
        expr = inline_functions(expr, defs)
        again = to_sympy(to_ast(expr))
        symbols = sorted(expr.free_symbols | again.free_symbols, key=str)
        point = _random_point(symbols, rng)
        a, b = expr.subs(point), again.subs(point)
        if isinstance(a, sympy.logic.boolalg.Boolean):
            assert bool(a) == bool(b), (path, expr)
        else:
            a, b = complex(a.evalf()), complex(b.evalf())
            if not (math.isnan(a.real) and math.isnan(b.real)):
                assert b == pytest.approx(a, rel=1e-9, abs=1e-9), (path, expr)
        n += 1
    assert n > 0
