"""XPP auto-import — convert XPPAUT ``.ode`` files into Process instances.

XPP/XPPAUT (Ermentrout's phase-plane tool) stores an ODE system as a
plain-text ``.ode`` file: ``par`` for constants, ``init`` for initial
conditions, ``x'=...`` / ``dx/dt=...`` for the rate laws, plus fixed
formulas, ``aux`` outputs, and user-defined functions. A large share of
the dynamical-systems literature (relaxation oscillators, neural models,
the Gérard/Goldbeter IL-6/STAT3 SASP loop) ships in this format and never
in SBML.

This module parses the ``.ode`` directly and translates each right-hand
side into a JAX expression, then wraps the system as a :class:`Process`
whose state variables are EVOLVED ports and whose ``par`` constants are
the substitutable, calibratable :attr:`XPPProcess.parameters` surface —
the same shape :func:`hallsim.sbml_import.process_from_sbml` produces, so
imported XPP models compose, calibrate, and reconcile clocks exactly like
imported SBML models.

Example
-------
>>> proc = process_from_xpp("gerard2014_il6.ode")
>>> proc.ports_schema()        # one EVOLVED port per state variable
>>> proc.calibratable_params() # one entry per `par` constant

The XPP math dialect differs from Python in three ways, all handled by
:func:`_xpp_expr_to_python`: ``^`` is exponentiation, ``if(c)then(a)else(b)``
is the conditional, and identifiers are case-insensitive. Features that
have no differentiable continuous-time meaning (``wiener`` noise,
``global`` events, ``table``/``special`` lookups, ``0=`` algebraic
constraints) are rejected with a clear message rather than silently
mistranslated.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process

log = logging.getLogger(__name__)


class XPPParseError(Exception):
    """Raised when a ``.ode`` file is malformed or structurally ambiguous."""


class UnsupportedXPPFeatureError(Exception):
    """Raised when a ``.ode`` file uses a construct with no continuous-time,
    differentiable translation (stochastic noise, discrete events, table
    lookups, algebraic/DAE constraints)."""


# ---------------------------------------------------------------------------
# XPP math dialect → JAX
# ---------------------------------------------------------------------------

# Built-in names an XPP expression may reference, mapped to their JAX
# equivalents. These shadow Python builtins (abs/min/max/pow) inside the
# eval namespace so traced arrays never hit a Python comparison. XPP's
# ``log`` is the natural logarithm (``log10`` is base 10); ``heav`` is the
# step function with heav(0)=1; ``max``/``min``/``mod`` are the two-argument
# elementwise forms.
_XPP_MATH: dict[str, Any] = {
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "atan2": jnp.arctan2,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "exp": jnp.exp,
    "log": jnp.log,
    "ln": jnp.log,
    "log10": jnp.log10,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
    "sign": jnp.sign,
    "floor": jnp.floor,
    "flr": jnp.floor,
    "ceil": jnp.ceil,
    "heav": lambda x: jnp.heaviside(x, 1.0),
    "mod": jnp.mod,
    "max": jnp.maximum,
    "min": jnp.minimum,
    "pow": jnp.power,
    "where": jnp.where,
    "logical_and": jnp.logical_and,
    "logical_or": jnp.logical_or,
    "pi": jnp.pi,
}


# --- expression tokenizer + precedence parser ------------------------------
#
# XPP's expression dialect is close to Python but differs in ways that plain
# textual substitution cannot fix: ``&`` / ``|`` are logical AND / OR (in
# Python they bind *tighter* than comparisons, so ``x>0 & y>0`` would mean
# ``x > (0 & y) > 0``), ``^`` is exponentiation, and identifiers are
# case-insensitive. Correctly rewriting those needs operand boundaries, i.e.
# a parse. This is a small precedence-climbing parser that emits an
# equivalent Python expression over the JAX ops in ``_XPP_MATH``; the emitted
# string is what the derivative eval's each step, so the RHS stays
# JIT-compilable and differentiable.

_TOKEN_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<NUMBER>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
    | (?P<NAME>[A-Za-z_]\w*)
    | (?P<OP><=|>=|==|!=|<>|[-+*/^()<>,&|])
    """,
    re.VERBOSE,
)

# operator -> (binding precedence, associativity); higher binds tighter.
_BINPREC: dict[str, tuple[int, str]] = {
    "|": (1, "L"),
    "&": (2, "L"),
    "<": (3, "L"),
    ">": (3, "L"),
    "<=": (3, "L"),
    ">=": (3, "L"),
    "==": (3, "L"),
    "!=": (3, "L"),
    "<>": (3, "L"),
    "+": (4, "L"),
    "-": (4, "L"),
    "*": (5, "L"),
    "/": (5, "L"),
    "^": (7, "R"),
}
_UNARY_PREC = 6


def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    for m in _TOKEN_RE.finditer(expr):
        if m.start() != pos:
            raise XPPParseError(
                f"unexpected character {expr[pos]!r} in {expr!r}"
            )
        pos = m.end()
        if m.lastgroup != "WS":
            tokens.append((m.lastgroup, m.group()))
    if pos != len(expr):
        raise XPPParseError(f"unexpected character {expr[pos]!r} in {expr!r}")
    return tokens


def _emit_bin(op: str, left: str, right: str) -> str:
    if op == "&":
        return f"logical_and({left}, {right})"
    if op == "|":
        return f"logical_or({left}, {right})"
    if op == "^":
        return f"({left} ** {right})"
    if op == "<>":
        op = "!="
    return f"({left} {op} {right})"


class _XPPExprParser:
    """Precedence-climbing parser: XPP expression tokens -> Python/JAX string."""

    def __init__(self, tokens: list[tuple[str, str]], src: str) -> None:
        self.toks = tokens
        self.src = src
        self.i = 0

    def _peek(self) -> tuple[str, str]:
        return self.toks[self.i] if self.i < len(self.toks) else ("EOF", "")

    def _next(self) -> tuple[str, str]:
        tok = self._peek()
        self.i += 1
        return tok

    def _expect_op(self, op: str) -> None:
        kind, val = self._next()
        if kind != "OP" or val != op:
            raise XPPParseError(
                f"expected {op!r} but found {val!r} in {self.src!r}"
            )

    def _expect_name(self, name: str) -> None:
        kind, val = self._next()
        if kind != "NAME" or val.lower() != name:
            raise XPPParseError(
                f"expected {name!r} but found {val!r} in {self.src!r}"
            )

    def parse(self) -> str:
        expr = self._parse_expr(0)
        if self.i != len(self.toks):
            raise XPPParseError(
                f"trailing tokens after {self.toks[self.i][1]!r} in {self.src!r}"
            )
        return expr

    def _parse_expr(self, min_prec: int) -> str:
        left = self._parse_unary()
        while True:
            kind, val = self._peek()
            if kind != "OP" or val not in _BINPREC:
                break
            prec, assoc = _BINPREC[val]
            if prec < min_prec:
                break
            self._next()
            next_min = prec + 1 if assoc == "L" else prec
            right = self._parse_expr(next_min)
            left = _emit_bin(val, left, right)
        return left

    def _parse_unary(self) -> str:
        kind, val = self._peek()
        if kind == "OP" and val in ("-", "+"):
            self._next()
            operand = self._parse_expr(_UNARY_PREC)
            return f"(-{operand})" if val == "-" else operand
        return self._parse_atom()

    def _parse_atom(self) -> str:
        kind, val = self._next()
        if kind == "NUMBER":
            return val
        if kind == "OP" and val == "(":
            inner = self._parse_expr(0)
            self._expect_op(")")
            return f"({inner})"
        if kind == "NAME":
            name = val.lower()
            if name == "if":
                return self._parse_if()
            if self._peek() == ("OP", "("):
                self._next()  # consume '('
                args: list[str] = []
                if self._peek() != ("OP", ")"):
                    args.append(self._parse_expr(0))
                    while self._peek() == ("OP", ","):
                        self._next()
                        args.append(self._parse_expr(0))
                self._expect_op(")")
                return f"{name}({', '.join(args)})"
            return name
        raise XPPParseError(f"unexpected token {val!r} in {self.src!r}")

    def _parse_if(self) -> str:
        """XPP ``if(c)then(a)else(b)`` -> ``where((c), (a), (b))``."""
        self._expect_op("(")
        cond = self._parse_expr(0)
        self._expect_op(")")
        self._expect_name("then")
        self._expect_op("(")
        then_e = self._parse_expr(0)
        self._expect_op(")")
        self._expect_name("else")
        self._expect_op("(")
        else_e = self._parse_expr(0)
        self._expect_op(")")
        return f"where(({cond}), ({then_e}), ({else_e}))"


def _xpp_expr_to_python(expr: str) -> str:
    """Translate one XPP right-hand-side string into a Python/JAX expression.

    Handles XPP's case-insensitive identifiers, ``^`` exponentiation,
    logical ``&`` / ``|`` (mapped to ``logical_and`` / ``logical_or`` so
    Python's tighter bitwise precedence can't corrupt a comparison), and
    ``if()then()else()`` conditionals. The result is evaluated against
    :data:`_XPP_MATH` plus the current state/parameter values, so it is
    JIT-compilable and differentiable.
    """
    expr = expr.strip()
    if not expr:
        raise XPPParseError("empty expression")
    return _XPPExprParser(_tokenize(expr), expr).parse()


# ---------------------------------------------------------------------------
# .ode file parsing
# ---------------------------------------------------------------------------

# name = number, tolerating comma- or space-separated lists and surrounding
# whitespace (``par a=1, b=2`` and ``par a = 1 b = 2`` both parse).
_ASSIGN_NUM = re.compile(
    r"([A-Za-z_]\w*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)

_KEYWORDS_PARAM = {
    "par",
    "param",
    "parameter",
    "parameters",
    "p",
    "number",
    "num",
}
_KEYWORDS_INIT = {"init", "initial", "i"}
_UNSUPPORTED = {
    "wiener",
    "global",
    "table",
    "special",
    "markov",
    "volt",
    "bndry",
    "bdry",
}


class _ParsedXPP:
    """Structured contents of a parsed ``.ode`` file."""

    def __init__(self) -> None:
        self.states: list[str] = []
        self.ode_src: list[str] = []  # parallel to states, XPP syntax
        self.inits: dict[str, float] = {}
        self.params: dict[str, float] = {}  # insertion order preserved
        self.inter_names: list[str] = []
        self.inter_src: list[str] = []  # fixed formulas + '!' vars, in order
        self.aux_names: list[str] = []
        self.aux_src: list[str] = []
        self.func_names: list[str] = []
        self.func_args: list[tuple[str, ...]] = []
        self.func_src: list[str] = []


def _parse_xpp_text(text: str) -> _ParsedXPP:
    p = _ParsedXPP()

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()  # strip comments and whitespace
        if not line:
            continue
        low = line.lower()

        if low in ("done", "d"):
            break
        # Numerical/UI directives carry no dynamics.
        if line.startswith("@") or low.split()[0] in ("set", "only"):
            continue

        # Algebraic/DAE constraint (`0 = f(...)`) — no explicit RHS to integrate.
        if re.match(r"^0\s*=", line):
            raise UnsupportedXPPFeatureError(
                f"algebraic constraint (DAE) not supported: {line!r}"
            )

        head = low.split(maxsplit=1)[0].rstrip("(")
        if head in _UNSUPPORTED:
            raise UnsupportedXPPFeatureError(
                f"XPP '{head}' construct has no differentiable continuous-time "
                f"translation: {line!r}"
            )

        # XPP identifiers are case-insensitive; expressions are lowercased in
        # translation, so every declared name is lowercased here to match.

        # dX/dt = ...
        m = re.match(
            r"^d([A-Za-z_]\w*)\s*/\s*dt\s*=(.*)$", line, re.IGNORECASE
        )
        if m:
            p.states.append(m.group(1).lower())
            p.ode_src.append(m.group(2).strip())
            continue

        # X' = ...
        m = re.match(r"^([A-Za-z_]\w*)\s*'\s*=(.*)$", line)
        if m:
            p.states.append(m.group(1).lower())
            p.ode_src.append(m.group(2).strip())
            continue

        # X(0) = value   (initial condition; the numeric arg distinguishes it
        # from a function definition like f(u) = ...).
        m = re.match(r"^([A-Za-z_]\w*)\s*\(\s*0\s*\)\s*=(.*)$", line)
        if m:
            p.inits[m.group(1).lower()] = float(m.group(2).strip())
            continue

        # keyword-led declarations: par / number / init
        kw = low.split(maxsplit=1)[0]
        if kw in _KEYWORDS_PARAM:
            for name, val in _ASSIGN_NUM.findall(line[len(kw) :]):
                p.params[name.lower()] = float(val)
            continue
        if kw in _KEYWORDS_INIT:
            for name, val in _ASSIGN_NUM.findall(line[len(kw) :]):
                p.inits[name.lower()] = float(val)
            continue
        if kw == "aux":
            m = re.match(
                r"^aux\s+([A-Za-z_]\w*)\s*=(.*)$", line, re.IGNORECASE
            )
            if m:
                p.aux_names.append(m.group(1).lower())
                p.aux_src.append(m.group(2).strip())
            continue

        # user function:  f(a, b) = expr   (args are identifiers)
        m = re.match(
            r"^([A-Za-z_]\w*)\s*\(([A-Za-z_][\w\s,]*)\)\s*=(.*)$", line
        )
        if m:
            p.func_names.append(m.group(1).lower())
            p.func_args.append(
                tuple(a.strip().lower() for a in m.group(2).split(","))
            )
            p.func_src.append(m.group(3).strip())
            continue

        # fixed formula:  !v = expr   (evaluated each step from params)
        #            or:   v = expr   (fixed/intermediate variable)
        m = re.match(r"^!?\s*([A-Za-z_]\w*)\s*=(.*)$", line)
        if m:
            p.inter_names.append(m.group(1).lower())
            p.inter_src.append(m.group(2).strip())
            continue

        raise XPPParseError(f"unrecognized XPP line: {raw.strip()!r}")

    if not p.states:
        raise XPPParseError("no ODEs (x'=... or dx/dt=...) found in file")
    return p


# ---------------------------------------------------------------------------
# Process wrapper
# ---------------------------------------------------------------------------


class XPPProcess(Process):
    """Process auto-generated from an XPPAUT ``.ode`` file.

    State variables become EVOLVED ports; ``par`` constants become the
    substitutable :attr:`parameters` surface (auto-populated at
    construction, discoverable via :meth:`calibratable_params`). Not
    constructed directly — use :func:`process_from_xpp`.

    Each right-hand side is stored as a translated Python-expression string
    (static metadata) and evaluated against a namespace of JAX ops plus the
    current state and parameter values at every derivative call. Because the
    namespace uses only ``jax.numpy`` operations, the derivative is
    shape-polymorphic (auto-vmaps over a leading batch axis), JIT-compilable,
    and differentiable — matching :class:`hallsim.sbml_import.SBMLProcess`.
    """

    _state_names: tuple[str, ...] = ()
    _state_y0: tuple[float, ...] = ()
    _ode_py: tuple[str, ...] = ()
    _inter_names: tuple[str, ...] = ()
    _inter_py: tuple[str, ...] = ()
    _func_names: tuple[str, ...] = ()
    _func_args: tuple[tuple[str, ...], ...] = ()
    _func_py: tuple[str, ...] = ()
    _aux_names: tuple[str, ...] = ()
    _aux_py: tuple[str, ...] = ()
    # Seconds per native time unit the .ode rate laws are written in. XPP
    # files carry no unit metadata, so this is supplied at import (default
    # 1.0 = the model runs on its own clock). Mirror of
    # SBMLProcess.native_time_seconds / time_scale.
    native_time_seconds: float = 1.0
    time_scale: float = 1.0
    parameters: dict[str, float] = None  # type: ignore[assignment]
    _param_names: tuple[str, ...] = ()
    _name: str = ""

    def ports_schema(self):
        return {
            name: Port(
                role=PortRole.EVOLVED,
                default=float(y0),
                units="dimensionless",
                description=f"XPP state variable: {name}",
            )
            for name, y0 in zip(self._state_names, self._state_y0)
        }

    def coupling_structure(self) -> dict:
        """XPP equation structure for the coupling-wiring check: ``par``/``p``
        constants, ``dx/dt`` states, and ``fixed``/``!``/``aux`` intermediates
        as the algebraic-rule graph. See :mod:`hallsim.coupling_wiring`."""
        import re

        universe = (
            set(self._state_names)
            | set(self._param_names)
            | set(self._inter_names)
            | set(self._aux_names)
        )

        def deps(expr: str) -> frozenset:
            ids = set(re.findall(r"[A-Za-z_]\w*", expr))
            return frozenset(ids & universe)

        rules = tuple(
            (name, deps(src))
            for name, src in list(zip(self._inter_names, self._inter_py))
            + list(zip(self._aux_names, self._aux_py))
        )
        return {
            "param_constant": {p: True for p in self._param_names},
            "param_sbo": {},
            "variables": frozenset(
                set(self._state_names)
                | set(self._inter_names)
                | set(self._aux_names)
            ),
            "rules": rules,
            "boundary": frozenset(),
        }

    def _eval_namespace(self, t, state):
        """Build the evaluation namespace for one derivative call.

        Order matters: math builtins, then time, state, and parameters,
        then user functions (which close over this dict so they see params
        and each other), then fixed/intermediate formulas in file order.
        """
        ns = dict(_XPP_MATH)
        ns["t"] = t * self.time_scale
        for name in self._state_names:
            ns[name] = state[name]
        for name in self._param_names:
            ns[name] = jnp.asarray(self.parameters[name])
        for name, args, src in zip(
            self._func_names, self._func_args, self._func_py
        ):
            ns[name] = eval(
                f"lambda {','.join(args)}: ({src})", ns
            )  # noqa: S307
        for name, src in zip(self._inter_names, self._inter_py):
            ns[name] = eval(src, ns)  # noqa: S307
        return ns

    def derivative(self, t, state):
        ns = self._eval_namespace(t, state)
        # Rate laws are written in native time; map canonical dy/dt back by
        # the same chain-rule factor SBMLProcess uses. time_scale=1.0 is a
        # no-op (model on its own clock).
        return {
            name: eval(src, ns) * self.time_scale  # noqa: S307
            for name, src in zip(self._state_names, self._ode_py)
        }

    def reconciled_to(self, canonical_time_seconds: float) -> "XPPProcess":
        """Return a copy on a shared canonical clock (see
        :meth:`hallsim.sbml_import.SBMLProcess.reconciled_to`)."""
        import equinox as eqx

        scale = canonical_time_seconds / self.native_time_seconds
        return eqx.tree_at(lambda p: p.time_scale, self, float(scale))

    def metadata(self):
        base = super().metadata()
        base["xpp_name"] = self._name
        base["n_states"] = len(self._state_names)
        base["n_parameters"] = len(self._param_names)
        base["native_time_seconds"] = self.native_time_seconds
        base["time_scale"] = self.time_scale
        return base

    def calibratable_params(self) -> list:
        """Expose every ``par`` constant as a fittable ``parameters.<name>``
        candidate, mirroring SBMLProcess so the XPP model plugs into
        :meth:`hallsim.composite.Composite.calibration_targets` unchanged."""
        from hallsim.calibration import CalibratableParam, default_clamp

        out = super().calibratable_params()
        for name, value in self.parameters.items():
            v = float(value)
            out.append(
                CalibratableParam(
                    process_name="",
                    field=f"parameters.{name}",
                    default=v,
                    clamp=default_clamp(v),
                    description=f"XPP parameter {name!r} on {self._name}",
                )
            )
        return out


def process_from_xpp(
    path: str,
    name: str | None = None,
    timescale: float | None = None,
    native_time_seconds: float = 1.0,
    parameters: dict[str, float] | None = None,
) -> XPPProcess:
    """Load an XPPAUT ``.ode`` file and wrap it as a :class:`Process`.

    Parameters
    ----------
    path:
        Path to a local ``.ode`` file.
    name:
        Human-readable process name. Defaults to the filename stem.
    timescale:
        Characteristic timescale (seconds) for multi-rate Scheduler
        grouping. Defaults to ``native_time_seconds``.
    native_time_seconds:
        Seconds per unit of the model's own time axis (XPP files carry no
        unit metadata). Used by :meth:`XPPProcess.reconciled_to` to place
        the model on a composite's canonical clock. Default ``1.0``.
    parameters:
        Optional ``{name: value}`` overrides applied to specific ``par``
        constants at construction. Each name must exist in the file.

    Returns
    -------
    XPPProcess with one EVOLVED port per state variable and every ``par``
    constant on its substitutable, calibratable :attr:`~XPPProcess.parameters`.

    Raises
    ------
    XPPParseError
        If the file is malformed or an override names an unknown parameter.
    UnsupportedXPPFeatureError
        If the file uses ``wiener``/``global``/``table``/``special`` or a
        ``0=`` algebraic constraint.
    """
    import os

    name = name or os.path.splitext(os.path.basename(path))[0]
    log.info(f"Loading XPP file '{path}' as '{name}'...")
    with open(path) as f:
        parsed = _parse_xpp_text(f.read())

    if parameters:
        overrides = {k.lower(): float(v) for k, v in parameters.items()}
        missing = [k for k in overrides if k not in parsed.params]
        if missing:
            raise XPPParseError(
                f"parameters {missing} not found in {path!r}; available: "
                f"{sorted(parsed.params)}"
            )
        parsed.params.update(overrides)

    log.info(
        f"Loaded {len(parsed.states)} states {tuple(parsed.states)}, "
        f"{len(parsed.params)} parameters"
    )

    proc = object.__new__(XPPProcess)
    object.__setattr__(proc, "_state_names", tuple(parsed.states))
    object.__setattr__(
        proc,
        "_state_y0",
        tuple(float(parsed.inits.get(s, 0.0)) for s in parsed.states),
    )
    object.__setattr__(
        proc, "_ode_py", tuple(_xpp_expr_to_python(s) for s in parsed.ode_src)
    )
    object.__setattr__(proc, "_inter_names", tuple(parsed.inter_names))
    object.__setattr__(
        proc,
        "_inter_py",
        tuple(_xpp_expr_to_python(s) for s in parsed.inter_src),
    )
    object.__setattr__(proc, "_func_names", tuple(parsed.func_names))
    object.__setattr__(proc, "_func_args", tuple(parsed.func_args))
    object.__setattr__(
        proc,
        "_func_py",
        tuple(_xpp_expr_to_python(s) for s in parsed.func_src),
    )
    object.__setattr__(proc, "_aux_names", tuple(parsed.aux_names))
    object.__setattr__(
        proc, "_aux_py", tuple(_xpp_expr_to_python(s) for s in parsed.aux_src)
    )
    object.__setattr__(proc, "native_time_seconds", float(native_time_seconds))
    object.__setattr__(proc, "time_scale", 1.0)
    object.__setattr__(proc, "parameters", dict(parsed.params))
    object.__setattr__(proc, "_param_names", tuple(parsed.params.keys()))
    object.__setattr__(proc, "_name", name)
    object.__setattr__(
        proc,
        "timescale",
        (
            float(timescale)
            if timescale is not None
            else float(native_time_seconds)
        ),
    )
    return proc
