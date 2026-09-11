"""SBML math as sympy: libsbml ``ASTNode`` ↔ sympy expression → JAX callable.

One translation serves every place SBML carries an expression — kinetic
laws, rules, initial assignments, event triggers and assignments, function
definitions — so the ODE velocity vector, the SSA propensity vector and an
exported model cannot drift from one another. Sympy holds the expression in
between: its free symbols are the sparsity pattern, ``sympy.diff`` is the
symbolic Jacobian, and :func:`to_jax` prints it for JAX.

Every libsbml node type is either translated here or refused by name; there
is no fall-through. Argument order follows SBML: ``log(base, x)`` and
``root(degree, x)`` put the base and degree first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import libsbml
import sympy
from sympy.core.function import AppliedUndef
from sympy.printing.numpy import JaxPrinter

#: The SBML ``time`` csymbol. Not a valid SBML identifier, so it cannot
#: collide with a model's own names.
TIME = sympy.Symbol("<time>")
#: The SBML ``avogadro`` csymbol, substituted by value in :func:`to_jax`.
AVOGADRO = sympy.Symbol("<avogadro>")
AVOGADRO_VALUE = libsbml.ASTNode(libsbml.AST_NAME_AVOGADRO).getValue()

# A Python int past this cannot be handed to a jitted computation (an
# SBML rate constant written as ``pow(1500, 6)`` folds to 1.1e19), so
# :func:`to_jax` demotes larger literals to floats.
_INT_LIMIT = 2**53
# libsbml stores INTEGER and RATIONAL parts as C ``long``.
_LONG_LIMIT = 2**31 - 1

_TYPE_NAMES = {
    value: name[4:]
    for name, value in vars(libsbml).items()
    if name.startswith("AST_") and isinstance(value, int)
}


class UnsupportedMathError(ValueError):
    """An SBML math construct with no translation, named by node type."""


def _type_name(node) -> str:
    return _TYPE_NAMES.get(node.getType(), str(node.getType()))


# ── libsbml → sympy ──────────────────────────────────────────────────


def _reciprocal(f):
    return lambda x: f(1 / x)


_UNARY = {
    libsbml.AST_FUNCTION_ABS: sympy.Abs,
    libsbml.AST_FUNCTION_ARCCOS: sympy.acos,
    libsbml.AST_FUNCTION_ARCCOSH: sympy.acosh,
    libsbml.AST_FUNCTION_ARCCOT: _reciprocal(sympy.atan),
    libsbml.AST_FUNCTION_ARCCOTH: _reciprocal(sympy.atanh),
    libsbml.AST_FUNCTION_ARCCSC: _reciprocal(sympy.asin),
    libsbml.AST_FUNCTION_ARCCSCH: _reciprocal(sympy.asinh),
    libsbml.AST_FUNCTION_ARCSEC: _reciprocal(sympy.acos),
    libsbml.AST_FUNCTION_ARCSECH: _reciprocal(sympy.acosh),
    libsbml.AST_FUNCTION_ARCSIN: sympy.asin,
    libsbml.AST_FUNCTION_ARCSINH: sympy.asinh,
    libsbml.AST_FUNCTION_ARCTAN: sympy.atan,
    libsbml.AST_FUNCTION_ARCTANH: sympy.atanh,
    libsbml.AST_FUNCTION_CEILING: sympy.ceiling,
    libsbml.AST_FUNCTION_COS: sympy.cos,
    libsbml.AST_FUNCTION_COSH: sympy.cosh,
    libsbml.AST_FUNCTION_COT: lambda x: 1 / sympy.tan(x),
    libsbml.AST_FUNCTION_COTH: lambda x: 1 / sympy.tanh(x),
    libsbml.AST_FUNCTION_CSC: lambda x: 1 / sympy.sin(x),
    libsbml.AST_FUNCTION_CSCH: lambda x: 1 / sympy.sinh(x),
    libsbml.AST_FUNCTION_EXP: sympy.exp,
    libsbml.AST_FUNCTION_FACTORIAL: sympy.factorial,
    libsbml.AST_FUNCTION_FLOOR: sympy.floor,
    libsbml.AST_FUNCTION_LN: sympy.log,
    libsbml.AST_FUNCTION_SEC: lambda x: 1 / sympy.cos(x),
    libsbml.AST_FUNCTION_SECH: lambda x: 1 / sympy.cosh(x),
    libsbml.AST_FUNCTION_SIN: sympy.sin,
    libsbml.AST_FUNCTION_SINH: sympy.sinh,
    libsbml.AST_FUNCTION_TAN: sympy.tan,
    libsbml.AST_FUNCTION_TANH: sympy.tanh,
    libsbml.AST_LOGICAL_NOT: sympy.Not,
}

_NARY = {
    libsbml.AST_PLUS: sympy.Add,
    libsbml.AST_TIMES: sympy.Mul,
    libsbml.AST_FUNCTION_MIN: sympy.Min,
    libsbml.AST_FUNCTION_MAX: sympy.Max,
    libsbml.AST_LOGICAL_AND: sympy.And,
    libsbml.AST_LOGICAL_OR: sympy.Or,
    libsbml.AST_LOGICAL_XOR: sympy.Xor,
}

_BINARY = {
    libsbml.AST_POWER: sympy.Pow,
    libsbml.AST_FUNCTION_POWER: sympy.Pow,
    libsbml.AST_RELATIONAL_LT: sympy.Lt,
    libsbml.AST_RELATIONAL_LEQ: sympy.Le,
    libsbml.AST_RELATIONAL_GT: sympy.Gt,
    libsbml.AST_RELATIONAL_GEQ: sympy.Ge,
    libsbml.AST_RELATIONAL_EQ: sympy.Eq,
    libsbml.AST_RELATIONAL_NEQ: sympy.Ne,
}

_LEAVES = {
    libsbml.AST_CONSTANT_TRUE: lambda node: sympy.true,
    libsbml.AST_CONSTANT_FALSE: lambda node: sympy.false,
    libsbml.AST_CONSTANT_PI: lambda node: sympy.pi,
    libsbml.AST_CONSTANT_E: lambda node: sympy.E,
    libsbml.AST_NAME_TIME: lambda node: TIME,
    libsbml.AST_NAME_AVOGADRO: lambda node: AVOGADRO,
    libsbml.AST_NAME: lambda node: sympy.Symbol(node.getName()),
    libsbml.AST_INTEGER: lambda node: sympy.Integer(node.getInteger()),
    libsbml.AST_REAL: lambda node: sympy.Float(node.getValue()),
    libsbml.AST_REAL_E: lambda node: sympy.Float(node.getValue()),
    libsbml.AST_RATIONAL: lambda node: sympy.Rational(
        node.getNumerator(), node.getDenominator()
    ),
}


def _truncated_quotient(a, b):
    q = a / b
    return sympy.sign(q) * sympy.floor(sympy.Abs(q))


def _arity(node, children, expected: int) -> None:
    if len(children) != expected:
        raise UnsupportedMathError(
            f"{_type_name(node)} takes {expected} argument(s), "
            f"got {len(children)}"
        )


def to_sympy(node) -> sympy.Basic:
    """Translate a libsbml ``ASTNode`` tree into a sympy expression.

    Function definitions arrive as :class:`sympy.Lambda`; calls to them as
    applied undefined functions, inlined by :func:`inline_functions`.
    ``time`` and ``avogadro`` map to :data:`TIME` and :data:`AVOGADRO`.
    """
    if node is None:
        raise UnsupportedMathError("missing math")
    kind = node.getType()
    if kind in _LEAVES:
        _arity(node, [], node.getNumChildren() and 0)
        return _LEAVES[kind](node)
    children = [
        to_sympy(node.getChild(i)) for i in range(node.getNumChildren())
    ]
    if kind in _UNARY:
        _arity(node, children, 1)
        return _UNARY[kind](children[0])
    if kind in _NARY:
        return _NARY[kind](*children)
    if kind in _BINARY:
        _arity(node, children, 2)
        return _BINARY[kind](*children)
    if kind == libsbml.AST_MINUS:
        if len(children) == 1:
            return -children[0]
        _arity(node, children, 2)
        return children[0] - children[1]
    if kind == libsbml.AST_DIVIDE:
        _arity(node, children, 2)
        return children[0] / children[1]
    if kind == libsbml.AST_FUNCTION_ROOT:
        if len(children) == 1:
            return sympy.sqrt(children[0])
        _arity(node, children, 2)
        degree, radicand = children
        return sympy.Pow(radicand, 1 / degree)
    if kind == libsbml.AST_FUNCTION_LOG:
        if len(children) == 1:
            return sympy.log(children[0], 10)
        _arity(node, children, 2)
        base, argument = children
        return sympy.log(argument, base)
    if kind == libsbml.AST_FUNCTION_QUOTIENT:
        _arity(node, children, 2)
        return _truncated_quotient(*children)
    if kind == libsbml.AST_FUNCTION_REM:
        _arity(node, children, 2)
        a, b = children
        return a - b * _truncated_quotient(a, b)
    if kind == libsbml.AST_FUNCTION_PIECEWISE:
        pieces = [
            (children[i], children[i + 1])
            for i in range(0, len(children) - 1, 2)
        ]
        if len(children) % 2:
            pieces.append((children[-1], sympy.true))
        if not pieces:
            return sympy.nan
        return sympy.Piecewise(*pieces)
    if kind == libsbml.AST_LAMBDA:
        if not children:
            raise UnsupportedMathError("LAMBDA with no body")
        return sympy.Lambda(tuple(children[:-1]), children[-1])
    if kind == libsbml.AST_FUNCTION:
        return sympy.Function(node.getName())(*children)
    raise UnsupportedMathError(
        f"SBML math node {_type_name(node)} has no translation"
    )


def function_definitions(model) -> dict[str, sympy.Lambda]:
    """``{id: Lambda}`` for every ``<functionDefinition>`` in ``model``."""
    out = {}
    for i in range(model.getNumFunctionDefinitions()):
        fd = model.getFunctionDefinition(i)
        lam = to_sympy(fd.getMath())
        if not isinstance(lam, sympy.Lambda):
            raise UnsupportedMathError(
                f"function definition {fd.getId()!r} is not a lambda"
            )
        out[fd.getId()] = lam
    return out


def inline_functions(
    expr: sympy.Basic, definitions: Mapping[str, sympy.Lambda]
) -> sympy.Basic:
    """Replace every call to a defined function by its body, recursively."""
    for _ in range(64):
        calls = expr.atoms(AppliedUndef)
        if not calls:
            return expr
        subs = {}
        for call in calls:
            name = call.func.__name__
            if name not in definitions:
                raise UnsupportedMathError(
                    f"call to undefined function {name!r}"
                )
            lam = definitions[name]
            if len(lam.variables) != len(call.args):
                raise UnsupportedMathError(
                    f"{name!r} takes {len(lam.variables)} argument(s), "
                    f"called with {len(call.args)}"
                )
            subs[call] = lam(*call.args)
        expr = expr.xreplace(subs)
    raise UnsupportedMathError("function definitions nest too deeply")


# ── sympy → JAX ──────────────────────────────────────────────────────


def to_jax(
    expr: sympy.Basic, symbols: Sequence[sympy.Symbol], *, cse: bool = False
):
    """A JAX callable ``f(*values)`` evaluating ``expr`` at ``symbols``.

    Traceable under ``jit``, ``grad`` and ``vmap``. Integer literals past
    the jit limit and the Avogadro csymbol are substituted as floats;
    everything else prints exactly. ``cse`` factors the subexpressions a
    tuple of expressions shares into temporaries computed once, which is
    the difference between a model's laws and a model's laws each
    re-deriving what the others already have.
    """
    demote = {AVOGADRO: sympy.Float(AVOGADRO_VALUE)}
    for n in expr.atoms(sympy.Integer):
        if abs(int(n)) > _INT_LIMIT:
            demote[n] = sympy.Float(n)
    for r in expr.atoms(sympy.Rational):
        if not r.is_Integer and (abs(r.p) > _INT_LIMIT or r.q > _INT_LIMIT):
            demote[r] = sympy.Float(r)
    printer = JaxPrinter(
        {
            "fully_qualified_modules": False,
            "inline": True,
            "allow_unknown_functions": False,
            "precision": 17,
        }
    )
    return sympy.lambdify(
        list(symbols),
        expr.xreplace(demote),
        modules="jax",
        printer=printer,
        cse=cse,
    )


# ── sympy → libsbml ──────────────────────────────────────────────────

_SYMPY_UNARY = {
    sympy.exp: libsbml.AST_FUNCTION_EXP,
    sympy.log: libsbml.AST_FUNCTION_LN,
    sympy.sin: libsbml.AST_FUNCTION_SIN,
    sympy.cos: libsbml.AST_FUNCTION_COS,
    sympy.tan: libsbml.AST_FUNCTION_TAN,
    sympy.asin: libsbml.AST_FUNCTION_ARCSIN,
    sympy.acos: libsbml.AST_FUNCTION_ARCCOS,
    sympy.atan: libsbml.AST_FUNCTION_ARCTAN,
    sympy.sinh: libsbml.AST_FUNCTION_SINH,
    sympy.cosh: libsbml.AST_FUNCTION_COSH,
    sympy.tanh: libsbml.AST_FUNCTION_TANH,
    sympy.asinh: libsbml.AST_FUNCTION_ARCSINH,
    sympy.acosh: libsbml.AST_FUNCTION_ARCCOSH,
    sympy.atanh: libsbml.AST_FUNCTION_ARCTANH,
    sympy.Abs: libsbml.AST_FUNCTION_ABS,
    sympy.floor: libsbml.AST_FUNCTION_FLOOR,
    sympy.ceiling: libsbml.AST_FUNCTION_CEILING,
    sympy.factorial: libsbml.AST_FUNCTION_FACTORIAL,
    sympy.sign: None,
}

_SYMPY_NARY = {
    sympy.Add: libsbml.AST_PLUS,
    sympy.Min: libsbml.AST_FUNCTION_MIN,
    sympy.Max: libsbml.AST_FUNCTION_MAX,
    sympy.And: libsbml.AST_LOGICAL_AND,
    sympy.Or: libsbml.AST_LOGICAL_OR,
    sympy.Xor: libsbml.AST_LOGICAL_XOR,
    sympy.Not: libsbml.AST_LOGICAL_NOT,
    sympy.StrictLessThan: libsbml.AST_RELATIONAL_LT,
    sympy.LessThan: libsbml.AST_RELATIONAL_LEQ,
    sympy.StrictGreaterThan: libsbml.AST_RELATIONAL_GT,
    sympy.GreaterThan: libsbml.AST_RELATIONAL_GEQ,
    sympy.Equality: libsbml.AST_RELATIONAL_EQ,
    sympy.Unequality: libsbml.AST_RELATIONAL_NEQ,
}


def _node(kind, *children):
    node = libsbml.ASTNode(kind)
    for child in children:
        node.addChild(child)
    return node


def _name(kind, name):
    node = libsbml.ASTNode(kind)
    node.setName(name)
    return node


def _number(value):
    if isinstance(value, sympy.Integer):
        if abs(int(value)) <= _LONG_LIMIT:
            node = libsbml.ASTNode(libsbml.AST_INTEGER)
            node.setValue(int(value))
            return node
        value = sympy.Float(value)
    if isinstance(value, sympy.Rational) and not isinstance(
        value, sympy.Float
    ):
        if abs(value.p) <= _LONG_LIMIT and value.q <= _LONG_LIMIT:
            node = libsbml.ASTNode(libsbml.AST_RATIONAL)
            node.setValue(int(value.p), int(value.q))
            return node
        value = sympy.Float(value)
    node = libsbml.ASTNode(libsbml.AST_REAL)
    node.setValue(float(value))
    return node


def to_ast(expr: sympy.Basic):
    """Translate a sympy expression into a well-formed libsbml ``ASTNode``.

    The inverse of :func:`to_sympy` up to canonical form: a division comes
    back as ``DIVIDE``, a ``Piecewise`` whose last condition is ``True``
    gets an ``otherwise``, and :data:`TIME` / :data:`AVOGADRO` become their
    csymbols.
    """
    node = _to_ast(expr)
    if not node.isWellFormedASTNode():
        raise ValueError(f"no well-formed SBML form for {expr}")
    return node


def _to_ast(expr):
    if expr is TIME:
        return _name(libsbml.AST_NAME_TIME, "time")
    if expr is AVOGADRO:
        return _name(libsbml.AST_NAME_AVOGADRO, "avogadro")
    if isinstance(expr, sympy.Symbol):
        return _name(libsbml.AST_NAME, expr.name)
    if expr is sympy.true:
        return libsbml.ASTNode(libsbml.AST_CONSTANT_TRUE)
    if expr is sympy.false:
        return libsbml.ASTNode(libsbml.AST_CONSTANT_FALSE)
    if expr is sympy.pi:
        return libsbml.ASTNode(libsbml.AST_CONSTANT_PI)
    if expr is sympy.E:
        return libsbml.ASTNode(libsbml.AST_CONSTANT_E)
    if isinstance(expr, sympy.Number):
        return _number(expr)
    if isinstance(expr, sympy.Pow):
        base, exponent = expr.args
        if exponent == -1:
            return _node(
                libsbml.AST_DIVIDE, _number(sympy.Integer(1)), _to_ast(base)
            )
        return _node(libsbml.AST_POWER, _to_ast(base), _to_ast(exponent))
    if isinstance(expr, sympy.Mul):
        numerator, denominator = expr.as_numer_denom()
        if denominator != 1:
            return _node(
                libsbml.AST_DIVIDE, _to_ast(numerator), _to_ast(denominator)
            )
        return _node(libsbml.AST_TIMES, *(_to_ast(a) for a in expr.args))
    if isinstance(expr, sympy.Piecewise):
        node = libsbml.ASTNode(libsbml.AST_FUNCTION_PIECEWISE)
        for value, condition in expr.args:
            node.addChild(_to_ast(value))
            if condition is not sympy.true:
                node.addChild(_to_ast(condition))
        return node
    if isinstance(expr, sympy.Lambda):
        return _node(
            libsbml.AST_LAMBDA,
            *(_to_ast(v) for v in expr.variables),
            _to_ast(expr.expr),
        )
    if isinstance(expr, AppliedUndef):
        node = _name(libsbml.AST_FUNCTION, expr.func.__name__)
        for arg in expr.args:
            node.addChild(_to_ast(arg))
        return node
    for cls, kind in _SYMPY_NARY.items():
        if isinstance(expr, cls):
            return _node(kind, *(_to_ast(a) for a in expr.args))
    kind = _SYMPY_UNARY.get(type(expr))
    if kind is not None:
        return _node(kind, *(_to_ast(a) for a in expr.args))
    raise UnsupportedMathError(
        f"no SBML form for sympy {type(expr).__name__}: {expr}"
    )
