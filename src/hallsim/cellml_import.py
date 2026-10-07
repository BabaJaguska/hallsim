"""Import a CellML model by converting it to SBML on the way in.

CellML is the Physiome Model Repository's format. A file holds components,
each with variables and MathML equations; units the file defines itself;
connections that make one component's variable another's; and an
encapsulation tree. Nothing in it is a species or a reaction — the author
writes the differential equations out — so a CellML model is the rule-based
shape ``hallsim.sbml_core`` compiles already: states under rate rules,
derived quantities under assignment rules, constants as parameters.

Reading is Myokit's (``myokit.formats.cellml``: CellML 1.0, 1.1 and 2.0,
connections and encapsulation resolved, units checked), and this module
writes what was read as an SBML L3V2 document. ``process_from_sbml`` routes a
``.cellml`` path here, so every downstream check — triage, the numerical
screen, reporters, calibration — sees an ordinary SBML import.

    from hallsim.sbml_import import process_from_sbml
    proc = process_from_sbml("luo_rudy_1991.cellml")

An SBML id is the component and variable joined by an underscore
(``membrane_V``); the SBML ``name`` keeps the qualified ``membrane.V``. A
connected variable is one quantity under the id of the component that
defines it. Every variable's unit becomes a unit definition and the time
variable's becomes the model's ``timeUnits``, so the native clock is
declared, not assumed.

Refused, with the reason: ``<import>`` elements (Myokit does not resolve
them), a variable bound to anything but time, and a right-hand side the SBML
math layer has no form for.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from xml.sax.saxutils import escape

import libsbml
import sympy

from hallsim.io import file_sha256
from hallsim.sbml_export import _add_rule, _Ids, _sid
from hallsim.sbml_math import TIME

log = logging.getLogger(__name__)

#: SBML level/version written. L3V2 carries the constructs the importer
#: already understands.
SBML_LEVEL, SBML_VERSION = 3, 2

#: Myokit states a unit as exponents over gram, metre, second, ampere,
#: kelvin, candela and mole, in that order, times a multiplier. SBML's mass
#: base is the kilogram, so gram is kilogram at scale -3.
_BASES = (
    (libsbml.UNIT_KIND_KILOGRAM, -3),
    (libsbml.UNIT_KIND_METRE, 0),
    (libsbml.UNIT_KIND_SECOND, 0),
    (libsbml.UNIT_KIND_AMPERE, 0),
    (libsbml.UNIT_KIND_KELVIN, 0),
    (libsbml.UNIT_KIND_CANDELA, 0),
    (libsbml.UNIT_KIND_MOLE, 0),
)


class UnsupportedCellMLFeatureError(Exception):
    """The file uses a construct with no translation here."""


def is_cellml(path) -> bool:
    """Whether ``path`` looks like a CellML file, by extension."""
    return isinstance(path, (str, Path)) and str(path).lower().endswith(
        ".cellml"
    )


def read_cellml(path):
    """The file as a validated ``myokit.Model``: connections resolved and
    every variable classed as state, derived quantity or constant."""
    import myokit
    from myokit.formats.cellml import CellMLImporter

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such CellML file: {path}")
    try:
        model = CellMLImporter().model(str(path))
        model.validate()
    except myokit.MyokitError as exc:
        raise UnsupportedCellMLFeatureError(f"{path.name}: {exc}") from exc
    return model


def document_from_cellml(path) -> libsbml.SBMLDocument:
    """The CellML file as an in-memory SBML document."""
    return _document(read_cellml(path), Path(path))


def cellml_to_sbml(
    cellml_path: str | Path, out_path: str | Path | None = None
) -> str:
    """Convert a ``.cellml`` to SBML and return the path written.

    Caches under ``~/.cache/hallsim/cellml/`` keyed on the file's own
    contents, so re-importing an unchanged model is a local read and an
    edited one is converted again.
    """
    cellml_path = Path(cellml_path)
    if not cellml_path.is_file():
        raise FileNotFoundError(f"no such CellML file: {cellml_path}")
    if out_path is None:
        digest = hashlib.sha256(cellml_path.read_bytes()).hexdigest()[:16]
        out = Path.home() / ".cache" / "hallsim" / "cellml"
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{cellml_path.stem}_{digest}.xml"
        if out_path.exists():
            log.info("Using cached SBML for '%s'.", cellml_path.name)
            return str(out_path)
    out_path = Path(out_path)
    document = document_from_cellml(cellml_path)
    libsbml.writeSBMLToFile(document, str(out_path))
    model = document.getModel()
    log.info(
        "Converted CellML '%s' to SBML L%dV%d: %d rate rules, %d "
        "assignment rules, %d constants.",
        cellml_path.name,
        SBML_LEVEL,
        SBML_VERSION,
        sum(model.getRule(i).isRate() for i in range(model.getNumRules())),
        sum(
            model.getRule(i).isAssignment() for i in range(model.getNumRules())
        ),
        sum(
            model.getParameter(i).getConstant()
            for i in range(model.getNumParameters())
        ),
    )
    return str(out_path)


class _UnitTable:
    """One SBML unit definition per distinct unit, keyed on Myokit's
    printed form; the SBML built-in ``dimensionless`` for a bare number."""

    def __init__(self, sbml, ids: _Ids):
        self.sbml = sbml
        self.ids = ids
        self.by_unit: dict[str, str] = {}

    def id_for(self, unit) -> str:
        if unit is None:
            return ""
        key = str(unit)
        if key not in self.by_unit:
            exponents = [float(e) for e in unit.exponents()]
            multiplier = float(unit.multiplier())
            self.by_unit[key] = (
                "dimensionless"
                if not any(exponents) and multiplier == 1.0
                else self._define(key, exponents, multiplier)
            )
        return self.by_unit[key]

    def _define(self, key, exponents, multiplier) -> str:
        label = key.strip("[]")
        want = _sid(label.replace("/", "_per_").replace("^", ""))
        if libsbml.UnitKind_isValidUnitKindString(
            want, SBML_LEVEL, SBML_VERSION
        ):
            want += "_unit"
        uid = self.ids.fresh(want)
        definition = self.sbml.createUnitDefinition()
        definition.setId(uid)
        definition.setName(label)
        bases = [(i, e) for i, e in enumerate(exponents) if e]
        # A multiple of one base unit carries the multiplier on that unit
        # (millisecond is second × 1e-3); any other unit carries it on a
        # dimensionless factor, which SBML's unit algebra multiplies in.
        on_base = len(bases) == 1 and bases[0][1] == 1.0
        for i, exponent in bases:
            kind, scale = _BASES[i]
            unit = definition.createUnit()
            unit.setKind(kind)
            unit.setExponent(exponent)
            unit.setScale(scale)
            unit.setMultiplier(multiplier if on_base else 1.0)
        if multiplier != 1.0 and not on_base:
            unit = definition.createUnit()
            unit.setKind(libsbml.UNIT_KIND_DIMENSIONLESS)
            unit.setExponent(1.0)
            unit.setScale(0)
            unit.setMultiplier(multiplier)
        return uid


def _parameter(sbml, sid, name, units, value, *, constant):
    p = sbml.createParameter()
    p.setId(sid)
    p.setName(name)
    p.setConstant(constant)
    if value is not None:
        p.setValue(float(value))
    if units:
        p.setUnits(units)
    return p


def _starting_value(variable):
    """The derived quantity's value at the deposit's initial state, or
    ``None`` where it cannot be evaluated there."""
    try:
        return float(variable.eval())
    except Exception:
        return None


def _notes(model, source: Path, myokit_version: str) -> str:
    lines = [
        f"Converted from CellML file {source.name} (sha256 "
        f"{file_sha256(source)}) by hallsim.cellml_import, read with "
        f"Myokit {myokit_version}."
    ]
    documentation = (model.meta.get("documentation") or "").strip()
    if documentation:
        lines.append(
            "CellML documentation: " + documentation.splitlines()[0].strip()
        )
    body = "".join(f"<p>{escape(line)}</p>" for line in lines)
    return f'<body xmlns="http://www.w3.org/1999/xhtml">{body}</body>'


def _document(model, source: Path) -> libsbml.SBMLDocument:
    import myokit
    from myokit.formats.sympy import SymPyExpressionWriter

    time = model.time()
    if time is None:
        raise UnsupportedCellMLFeatureError(
            f"{source.name}: no variable is bound to time"
        )
    for v in model.variables(deep=True):
        if v.binding() not in (None, "time"):
            raise UnsupportedCellMLFeatureError(
                f"{source.name}: {v.qname()} is bound to {v.binding()!r}, "
                "an external input the file does not define"
            )

    document = libsbml.SBMLDocument(SBML_LEVEL, SBML_VERSION)
    sbml = document.createModel()
    ids = _Ids()
    model_name = model.name() or source.stem
    sbml.setId(ids.fresh(_sid(model_name)))
    sbml.setName(model_name)
    sbml.setNotes(_notes(model, source, myokit.__version__))
    units = _UnitTable(sbml, ids)
    sbml.setTimeUnits(units.id_for(time.unit()) or "dimensionless")

    variables = [v for v in model.variables(deep=True) if v is not time]
    sid_of = {v.qname(): ids.fresh(_sid(v.qname())) for v in variables}
    symbols = {
        sympy.Symbol(qname): sympy.Symbol(sid) for qname, sid in sid_of.items()
    }
    symbols[sympy.Symbol(time.qname())] = TIME
    known = set(symbols.values())
    writer = SymPyExpressionWriter()

    def expr_of(variable):
        expr = writer.ex(variable.rhs()).xreplace(symbols)
        stray = sorted(str(s) for s in expr.free_symbols if s not in known)
        if stray:
            raise UnsupportedCellMLFeatureError(
                f"{source.name}: {variable.qname()} reads "
                f"{', '.join(stray)}, which no variable defines"
            )
        return expr

    initial = dict(
        zip(
            (s.qname() for s in model.states()),
            model.initial_values(as_floats=True),
        )
    )
    for v in variables:
        sid, uid = sid_of[v.qname()], units.id_for(v.unit())
        if v.is_state():
            _parameter(
                sbml, sid, v.qname(), uid, initial[v.qname()], constant=False
            )
            _add_rule(sbml, sid, expr_of(v), rate=True)
        elif isinstance(v.rhs(), myokit.Number):
            _parameter(
                sbml, sid, v.qname(), uid, v.rhs().eval(), constant=True
            )
        else:
            _parameter(
                sbml, sid, v.qname(), uid, _starting_value(v), constant=False
            )
            _add_rule(sbml, sid, expr_of(v), rate=False)
    return document
