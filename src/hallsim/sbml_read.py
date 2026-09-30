"""Everything one SBML file says, read once into a single record.

This is the low-level read. It parses the document exactly once and returns a
:class:`Deposit` holding what the file states — species, reactions, rules,
events, parameters, compartments, units, initial assignments, constraints, the
notes on every one of those, and the MIRIAM annotations *with their qualifiers*.
Every other consumer reads that record instead of re-opening the file:
:func:`hallsim.sbml_import.process_from_sbml` builds a Process from it and
:mod:`hallsim.intake` screens from it. There is deliberately nothing that
*diffs* the record against the built model — anything in here that is model
content belongs on the model, so a diff would only ever display a defect
nobody then fixes.

Reading it all is the point. A deposit says more than a model can hold, and the
part that gets dropped is repeatedly the part that mattered: BIOMD0000000105
ships ``k69 = 0`` with a note *on that parameter* saying zero is its
proteasome-inhibited condition and 1e-3 is normal, and a constant scoped inside
a kinetic law appears in no parameter list at all.

    from hallsim.sbml_read import read_sbml
    d = read_sbml("BIOMD0000000585")
    [s.id for s in d.species if s.unused]     # named, referenced by nothing
    d.reactions[0].law                        # the rate law, as written
    d.zero_parameters                         # set to zero by someone, and why
    d.inexact_species                         # annotated weaker than identity

:func:`read_deposit` is the same read starting from an already-parsed libsbml
model, for a caller that has one and must not pay for a second parse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _references(formula: str, identifier: str) -> bool:
    """Whether ``formula`` names ``identifier``, on a token boundary.

    Substring matching would find ``x`` inside ``xy`` and report every short
    species id as read by every rule.
    """
    if not formula:
        return False
    return (
        re.search(rf"(?<![\w.]){re.escape(identifier)}(?![\w.])", formula)
        is not None
    )


_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def _prose(raw: str) -> str:
    """The text of an SBML ``<notes>`` block, with its XHTML markup dropped.

    Notes are XHTML, so the payload is wrapped in ``<body>``/``<p>`` and often
    carries links and line breaks. Only the prose is wanted here.
    """
    if not raw:
        return ""
    return _SPACE.sub(" ", _TAGS.sub(" ", raw)).strip()


def _note_lines(notes: str, indent: str = "    ") -> str:
    """``notes`` as an indented continuation, or nothing when there are none."""
    if not notes:
        return ""
    return f"\n{indent}note: {notes}"


def _read_notes(element) -> str:
    """The prose of an element's own ``<notes>``, if it has one.

    Every SBML element may carry notes, and this is where an author writes the
    thing the numbers cannot say — that a constant is set to its inhibited
    value, that a rate came from a different cell type, that a unit is wrong.
    Nothing else in the framework reads them.
    """
    try:
        if not element.isSetNotes():
            return ""
        return _prose(element.getNotesString() or "")
    except Exception:  # pragma: no cover - libsbml version differences
        log.debug("notes unreadable on %s", element, exc_info=True)
        return ""


_IDENTIFIERS = re.compile(r"https?://identifiers\.org/([^/]+)/(.+)$")


def _annotations(element) -> tuple[tuple[str, str, str], ...]:
    """MIRIAM terms on one element, as ``(qualifier, namespace, id)``.

    The qualifier is the part that is usually thrown away, and it changes what
    the annotation licenses: ``is`` asserts identity, so a measurement of that
    molecule is a measurement of this species; ``isVersionOf`` says only that it
    is *a kind of* that thing; ``hasPart`` says the species is a complex that
    contains it. Joining data on the last two as if they were the first is how a
    readout ends up pointing at the wrong quantity while reporting a match.
    """
    import libsbml

    out: list[tuple[str, str, str]] = []
    try:
        count = element.getNumCVTerms()
    except Exception:  # pragma: no cover - libsbml version differences
        return ()
    for i in range(count):
        cv = element.getCVTerm(i)
        if cv.getQualifierType() == libsbml.BIOLOGICAL_QUALIFIER:
            qualifier = libsbml.BiolQualifierType_toString(
                cv.getBiologicalQualifierType()
            )
        elif cv.getQualifierType() == libsbml.MODEL_QUALIFIER:
            qualifier = libsbml.ModelQualifierType_toString(
                cv.getModelQualifierType()
            )
        else:
            qualifier = ""
        for j in range(cv.getNumResources()):
            match = _IDENTIFIERS.match(cv.getResourceURI(j))
            if match:
                out.append((qualifier or "?", match.group(1), match.group(2)))
    return tuple(dict.fromkeys(out))


def _render_annotations(terms: tuple[tuple[str, str, str], ...]) -> str:
    """Annotations as an indented continuation, grouped by qualifier."""
    if not terms:
        return ""
    by_qualifier: dict[str, list[str]] = {}
    for qualifier, namespace, identifier in terms:
        by_qualifier.setdefault(qualifier, []).append(
            f"{namespace}:{identifier}"
        )
    shown = "; ".join(
        f"{qualifier} {', '.join(ids)}"
        for qualifier, ids in by_qualifier.items()
    )
    return f"\n    annotated: {shown}"


def _sbo(element) -> str:
    """The element's SBO term, as ``SBO:0000009``, or an empty string.

    SBO says what kind of thing an element *is* — a kinetic constant, a
    catalyst, a degradation — independently of what it was named.
    """
    try:
        return element.getSBOTermID() or ""
    except Exception:  # pragma: no cover - libsbml version differences
        return ""


@dataclass(frozen=True)
class Species:
    """One species as the file declares it."""

    id: str
    name: str
    compartment: str
    boundary: bool
    constant: bool
    initial: float | None
    initial_kind: str = ""
    substance_units: str = ""
    only_substance_units: bool = False
    conversion_factor: str = ""
    sbo: str = ""
    notes: str = ""
    annotations: tuple[tuple[str, str, str], ...] = ()
    in_reactions: tuple[str, ...] = ()
    in_rules: tuple[str, ...] = ()

    @property
    def unused(self) -> bool:
        """Declared and then referenced by no reaction and no rule.

        A species like this is inert whatever its name promises, so wiring an
        intervention to it drives nothing. Deposits do ship them.
        """
        return not self.in_reactions and not self.in_rules

    def __str__(self) -> str:
        marks = [
            m
            for m, on in (
                ("boundary", self.boundary),
                ("constant", self.constant),
                ("substance units only", self.only_substance_units),
                ("UNUSED", self.unused),
            )
            if on
        ]
        where = f"@{self.compartment}" if self.compartment else ""
        tail = f"  [{', '.join(marks)}]" if marks else ""
        label = f" — {self.name}" if self.name and self.name != self.id else ""
        # amount and concentration differ by the compartment's size, so which
        # one the file declared is part of the value.
        start = ""
        if self.initial is not None:
            units = f" {self.substance_units}" if self.substance_units else ""
            start = f" = {self.initial:g}{units} ({self.initial_kind})"
        head = f"{self.id}{where}{label}{start}{tail}"
        return (
            head
            + _render_annotations(self.annotations)
            + _note_lines(self.notes)
        )


@dataclass(frozen=True)
class Reaction:
    """One reaction and the rate law as written, not as imported."""

    id: str
    name: str
    reversible: bool
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    modifiers: tuple[str, ...]
    law: str
    calls: tuple[str, ...] = ()
    stoichiometry: tuple[tuple[str, str, float | None], ...] = ()
    local_parameters: tuple[tuple[str, float | None, str], ...] = ()
    compartment: str = ""
    fast: bool = False
    sbo: str = ""
    notes: str = ""
    annotations: tuple[tuple[str, str, str], ...] = ()

    def _side(self, role: str, names: tuple[str, ...]) -> str:
        """One side of the arrow, with a coefficient wherever it is not 1.

        Stoichiometry changes what the reaction *means* — ``2 A -> B`` is not
        ``A -> B`` — and printing only species names loses it.
        """
        coeff = {
            species: n
            for where, species, n in self.stoichiometry
            if where == role
        }
        parts = []
        for species in names:
            n = coeff.get(species)
            prefix = "" if n is None or n == 1 else f"{n:g} "
            parts.append(f"{prefix}{species}")
        return " + ".join(parts) or "∅"

    def __str__(self) -> str:
        left = self._side("reactants", self.reactants)
        right = self._side("products", self.products)
        arrow = "<->" if self.reversible else "->"
        head = f"{self.id}: {left} {arrow} {right}"
        if self.modifiers:
            head += f"   (modifiers: {', '.join(self.modifiers)})"
        if self.fast:
            head += "   [fast]"
        out = f"{head}\n    {self.law or '(no kinetic law)'}"
        if self.local_parameters:
            # Scoped to this law, so they never appear in the model's own
            # parameter list and a handle cannot reach them.
            shown = ", ".join(
                f"{pid} = {'?' if val is None else format(val, 'g')}"
                f"{' ' + units if units else ''}"
                for pid, val, units in self.local_parameters
            )
            out += f"\n    local to this law: {shown}"
        return (
            out
            + _render_annotations(self.annotations)
            + _note_lines(self.notes)
        )


@dataclass(frozen=True)
class Event:
    """One event: when it fires and what it overwrites.

    This is usually where a deposit keeps its *protocol* — the dose, when it
    starts, when it stops, the knockdown at a given day — so a mechanism dump
    without events omits the intervention the paper actually performed. It also
    hides a trap: an event assigns to a variable outright, so a handle mapped
    onto a parameter an event writes is silently overwritten when it fires.
    """

    id: str
    name: str
    trigger: str
    assignments: tuple[tuple[str, str], ...] = ()
    delay: str = ""
    from_trigger_time: bool = True
    persistent: bool = True
    initial_value: bool = True
    priority: str = ""
    sbo: str = ""
    notes: str = ""

    @property
    def writes(self) -> tuple[str, ...]:
        return tuple(target for target, _ in self.assignments)

    def __str__(self) -> str:
        label = f" — {self.name}" if self.name and self.name != self.id else ""
        head = f"{self.id or '(unnamed)'}{label}: when {self.trigger}"
        if self.delay:
            head += f", after a delay of {self.delay}"
        if self.priority:
            head += f", priority {self.priority}"
        if not self.persistent:
            # A non-persistent trigger can arm and then disarm before the event
            # fires, so the assignment below is not guaranteed to happen.
            head += ", non-persistent"
        if not self.initial_value:
            head += ", may fire at t=0"
        sets = ", ".join(f"{k} = {v}" for k, v in self.assignments)
        body = f"{head}\n    sets {sets}" if sets else head
        return body + _note_lines(self.notes)


@dataclass(frozen=True)
class Parameter:
    """One parameter as the file declares it, with whatever the author said.

    A deposit's parameter list is where its quantitative claims live, and until
    now nothing in the framework would show you one. The note matters as much as
    the value: BIOMD0000000105 ships ``k69 = 0`` and says, in the note on that
    parameter, that the value is its *proteasome-inhibited* condition and the
    normal one is 1e-3. Reading the value without the note gets you a model
    whose misfolded protein climbs to 998 of 1000 and calling it aging.
    """

    id: str
    name: str = ""
    value: float | None = None
    units: str = ""
    constant: bool = True
    sbo: str = ""
    notes: str = ""
    annotations: tuple[tuple[str, str, str], ...] = ()

    @property
    def zero(self) -> bool:
        """Set to exactly zero — reported, not judged.

        Nobody fits a rate to 0.0, so the value was chosen; the reason is in the
        note, in the paper, or nowhere. What it means is the modeller's call.
        """
        return self.value == 0.0

    def __str__(self) -> str:
        label = f" — {self.name}" if self.name and self.name != self.id else ""
        val = "(unset)" if self.value is None else format(self.value, "g")
        units = f" {self.units}" if self.units else ""
        marks = []
        if not self.constant:
            marks.append("varies")
        if self.zero:
            marks.append("ZERO")
        tail = f"  [{', '.join(marks)}]" if marks else ""
        return (
            f"{self.id}{label} = {val}{units}{tail}"
            + _render_annotations(self.annotations)
            + _note_lines(self.notes)
        )


@dataclass(frozen=True)
class Compartment:
    """One compartment: its size is a divisor on every concentration in it."""

    id: str
    name: str = ""
    size: float | None = None
    spatial_dimensions: float | None = None
    units: str = ""
    constant: bool = True
    outside: str = ""
    sbo: str = ""
    notes: str = ""
    annotations: tuple[tuple[str, str, str], ...] = ()

    def __str__(self) -> str:
        label = f" — {self.name}" if self.name and self.name != self.id else ""
        size = "(unset)" if self.size is None else format(self.size, "g")
        units = f" {self.units}" if self.units else ""
        dims = (
            f", {self.spatial_dimensions:g}D"
            if self.spatial_dimensions is not None
            else ""
        )
        where = f", inside {self.outside}" if self.outside else ""
        varies = "" if self.constant else ", varies"
        return (
            f"{self.id}{label}: size {size}{units}{dims}{where}{varies}"
            + _render_annotations(self.annotations)
            + _note_lines(self.notes)
        )


@dataclass(frozen=True)
class UnitDefinition:
    """One unit definition, rendered as the product of powers it is.

    A model declaring ``substance`` as ``1e-9 mole`` is in nanomoles, and every
    rate constant in it is scaled accordingly. The census records only whether a
    time unit exists; this is what it says.
    """

    id: str
    name: str = ""
    definition: str = ""

    def __str__(self) -> str:
        label = f" — {self.name}" if self.name and self.name != self.id else ""
        return f"{self.id}{label} = {self.definition or '(empty)'}"


@dataclass(frozen=True)
class InitialAssignment:
    """A value computed at t=0, overriding the declared initial value.

    The number written on the species is then not the number the model starts
    from, which makes a deposit's stated initial condition misleading to read
    off the species list alone.
    """

    symbol: str
    math: str = ""
    sbo: str = ""
    notes: str = ""

    def __str__(self) -> str:
        return f"{self.symbol} := {self.math}" + _note_lines(self.notes)


@dataclass(frozen=True)
class Constraint:
    """A range the author says the model is only valid inside.

    This is the author stating the model's own boundary, in the file, with a
    human-readable message attached. Solving past it is out of scope by the
    deposit's own declaration.
    """

    math: str = ""
    message: str = ""
    sbo: str = ""
    notes: str = ""

    def __str__(self) -> str:
        head = f"valid only while {self.math}" if self.math else "constraint"
        msg = f"\n    message: {self.message}" if self.message else ""
        return head + msg + _note_lines(self.notes)


@dataclass(frozen=True)
class Rule:
    """One rule: which kind, what it writes, and the expression it writes.

    Indexable as the ``(kind, target, formula)`` triple this used to be, so
    existing readers keep working while the note and units are now here too.
    """

    kind: str
    target: str
    formula: str
    units: str = ""
    sbo: str = ""
    notes: str = ""

    def __getitem__(self, index: int) -> str:
        return (self.kind, self.target, self.formula)[index]

    def __iter__(self):
        return iter((self.kind, self.target, self.formula))

    def __str__(self) -> str:
        units = f" [{self.units}]" if self.units else ""
        return (
            f"{self.kind} {self.target} = {self.formula}{units}"
            + _note_lines(self.notes)
        )


@dataclass(frozen=True)
class Deposit:
    """A deposit's mechanism, as its own file states it."""

    name: str
    species: tuple[Species, ...] = ()
    reactions: tuple[Reaction, ...] = ()
    functions: dict[str, str] = field(default_factory=dict)
    rules: tuple[Rule, ...] = ()
    compartments: tuple[Compartment, ...] = ()
    events: tuple[Event, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    units: tuple[UnitDefinition, ...] = ()
    initial_assignments: tuple[InitialAssignment, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    notes: str = ""
    annotations: tuple[tuple[str, str, str], ...] = ()
    model_units: tuple[tuple[str, str], ...] = ()
    history: tuple[tuple[str, str], ...] = ()
    sbml_level: int = 0
    sbml_version: int = 0
    packages: tuple[str, ...] = ()

    @property
    def zero_parameters(self) -> tuple[Parameter, ...]:
        """Parameters set to exactly zero, with their notes if they have any.

        Surfaced rather than ruled on: a term switched off may be the deposit's
        own experimental arm, vestigial structure, or a genuine baseline. Only
        the modeller can say which, and only if they are told it is there.
        """
        return tuple(p for p in self.parameters if p.zero)

    @property
    def inexact_species(self) -> tuple[tuple[str, str, str], ...]:
        """Species annotated by something weaker than identity.

        Returned as ``(species, qualifier, term)``. The importer keeps the term
        and drops the qualifier, so these reach a port looking exactly like an
        ``is``, and a readout joining on one is measuring a different quantity
        than it reports: ``isVersionOf`` is a kind of that molecule, ``hasPart``
        is a complex containing it.
        """
        return tuple(
            (species.id, qualifier, f"{namespace}:{identifier}")
            for species in self.species
            for qualifier, namespace, identifier in species.annotations
            if qualifier not in ("is", "?")
        )

    @property
    def annotated_notes(self) -> tuple[tuple[str, str], ...]:
        """Every note in the file, as (where it hangs, what it says).

        The place is part of the meaning — the same sentence on the model and on
        one parameter says two different things.
        """
        out: list[tuple[str, str]] = []
        if self.notes:
            out.append(("model", self.notes))
        for group, items in (
            ("parameter", self.parameters),
            ("species", self.species),
            ("reaction", self.reactions),
            ("compartment", self.compartments),
            ("event", self.events),
            ("initialAssignment", self.initial_assignments),
        ):
            for item in items:
                if item.notes:
                    label = getattr(item, "id", None) or getattr(
                        item, "symbol", "?"
                    )
                    out.append((f"{group} {label}", item.notes))
        for constraint in self.constraints:
            if constraint.notes:
                out.append(("constraint", constraint.notes))
        return tuple(out)

    @property
    def event_written(self) -> tuple[str, ...]:
        """Names an event assigns to — do not aim a handle at one of these.

        A handle sets a parameter once; an event sets it again at its own time,
        and the event wins from then on. The effect is a handle that works until
        the protocol fires and then quietly stops working.
        """
        return tuple(
            dict.fromkeys(
                target for event in self.events for target in event.writes
            )
        )

    @property
    def unused_species(self) -> tuple[Species, ...]:
        return tuple(s for s in self.species if s.unused)

    @property
    def law_free_reactions(self) -> tuple[Reaction, ...]:
        return tuple(r for r in self.reactions if not r.law)

    @property
    def reactions_calling_functions(self) -> tuple[Reaction, ...]:
        """Reactions whose law is a call, so the mechanism is in `functions`."""
        return tuple(r for r in self.reactions if r.calls)


def _law(libsbml, reaction) -> str:
    kinetic = reaction.getKineticLaw()
    if kinetic is None or not kinetic.isSetMath():
        return ""
    return libsbml.formulaToL3String(kinetic.getMath()) or ""


def _local_parameters(
    reaction,
) -> tuple[tuple[str, float | None, str], ...]:
    """Parameters scoped to this reaction's kinetic law.

    These are invisible to the model's own parameter list, so a constant living
    here cannot be read off it and cannot be reached by a handle aimed at the
    model.
    """
    kinetic = reaction.getKineticLaw()
    if kinetic is None:
        return ()
    out = []
    for i in range(kinetic.getNumParameters()):
        p = kinetic.getParameter(i)
        out.append(
            (
                p.getId() or p.getName() or "?",
                p.getValue() if p.isSetValue() else None,
                p.getUnits() if p.isSetUnits() else "",
            )
        )
    return tuple(out)


def _reaction_notes(reaction) -> str:
    """Notes on the reaction and on its kinetic law, which are separate.

    An author explaining where a rate law came from usually writes it on the
    law rather than the reaction, and either place is easy to miss.
    """
    own = _read_notes(reaction)
    kinetic = reaction.getKineticLaw()
    law = _read_notes(kinetic) if kinetic is not None else ""
    if own and law:
        return f"{own} — on its rate law: {law}"
    return own or law


def _called(libsbml, reaction, known: set[str]) -> tuple[str, ...]:
    kinetic = reaction.getKineticLaw()
    if kinetic is None or not kinetic.isSetMath():
        return ()
    found: list[str] = []

    def walk(node):
        if node is None:
            return
        if node.getType() == libsbml.AST_FUNCTION:
            called = node.getName()
            if called in known and called not in found:
                found.append(called)
        for i in range(node.getNumChildren()):
            walk(node.getChild(i))

    walk(kinetic.getMath())
    return tuple(found)


def _render_units(libsbml, definition) -> str:
    """A unit definition as the product of powers it is, e.g. ``1e-9 mole``.

    ``substance = 1e-9 mole`` means every amount in the model is a nanomole and
    every rate constant is scaled to match, which no species or parameter line
    says on its own.
    """
    parts = []
    for i in range(definition.getNumUnits()):
        u = definition.getUnit(i)
        kind = libsbml.UnitKind_toString(u.getKind())
        scale, exponent, multiplier = (
            u.getScale(),
            u.getExponentAsDouble(),
            u.getMultiplier(),
        )
        text = kind
        if exponent != 1:
            text += f"^{exponent:g}"
        if scale:
            text = f"1e{scale} {text}"
        if multiplier != 1:
            text = f"{multiplier:g} {text}"
        parts.append(text)
    return " · ".join(parts)


_MODEL_UNIT_GETTERS = (
    ("time", "TimeUnits"),
    ("substance", "SubstanceUnits"),
    ("volume", "VolumeUnits"),
    ("area", "AreaUnits"),
    ("length", "LengthUnits"),
    ("extent", "ExtentUnits"),
    ("conversionFactor", "ConversionFactor"),
)


def _model_units(model) -> tuple[tuple[str, str], ...]:
    """The model's own default units, for the ones it sets.

    The census records whether a time unit is declared; this is what it says,
    and the other five are never looked at anywhere.
    """
    out = []
    for label, suffix in _MODEL_UNIT_GETTERS:
        is_set = getattr(model, f"isSet{suffix}", None)
        get = getattr(model, f"get{suffix}", None)
        if is_set is None or get is None:
            continue
        try:
            if is_set():
                out.append((label, get()))
        except Exception:  # pragma: no cover - libsbml version differences
            continue
    return tuple(out)


def _history(model) -> tuple[tuple[str, str], ...]:
    """Who deposited it and when, from the model's RDF history."""
    try:
        if not model.isSetModelHistory():
            return ()
        history = model.getModelHistory()
    except Exception:  # pragma: no cover - libsbml version differences
        return ()
    out = []
    for label, getter in (("created", "getCreatedDate"),):
        try:
            date = getattr(history, getter)()
            if date is not None:
                out.append((label, date.getDateAsString()))
        except Exception:
            pass
    try:
        for i in range(history.getNumModifiedDates()):
            out.append(
                ("modified", history.getModifiedDate(i).getDateAsString())
            )
    except Exception:
        pass
    try:
        people = []
        for i in range(history.getNumCreators()):
            c = history.getCreator(i)
            who = " ".join(
                x for x in (c.getGivenName(), c.getFamilyName()) if x
            )
            org = c.getOrganisation() or ""
            people.append(f"{who} ({org})" if org else who)
        if people:
            out.append(("creators", "; ".join(p for p in people if p)))
    except Exception:
        pass
    return tuple(out)


def _packages(document) -> tuple[str, ...]:
    """SBML packages the file enables, beyond core.

    A package carries model content core SBML cannot express, so an importer
    that ignores one is dropping part of the model: ``comp`` holds submodels,
    ``fbc`` flux bounds and objectives, ``distrib`` parameter uncertainty.
    ``layout`` and ``render`` are presentation only and say nothing about the
    mechanism.
    """
    out = []
    try:
        for i in range(document.getNumPlugins()):
            name = document.getPlugin(i).getPackageName()
            if name and name != "core":
                out.append(name)
    except Exception:  # pragma: no cover - libsbml version differences
        return ()
    return tuple(dict.fromkeys(out))


def read_sbml(source, name: str | None = None) -> Deposit:
    """Everything ``source`` says, from one parse of it.

    ``source`` is a local SBML path or a BioModels accession, resolved and
    cached the way every other importer resolves one. Raises whatever the
    resolution raises: a file that cannot be read is not an empty deposit.
    """
    import libsbml

    from hallsim.sbml_import import _resolve_source

    path, resolved = _resolve_source(source, name or "deposit")
    document = libsbml.readSBMLFromFile(str(path))
    model = document.getModel()
    if model is None:
        raise ValueError(f"{path}: no SBML model element")
    return read_deposit(model, document=document, name=resolved)


def read_deposit(model, document=None, name: str = "deposit") -> Deposit:
    """The same read, from a libsbml model a caller already parsed.

    This is the one place that turns a parsed model into a :class:`Deposit`, so
    a consumer holding a document pays for no second parse — which matters: the
    largest deposit in BioModels takes 24 s to parse.
    """
    import libsbml

    if document is None:
        document = model.getSBMLDocument()

    functions = {}
    for i in range(model.getNumFunctionDefinitions()):
        fd = model.getFunctionDefinition(i)
        body = fd.getBody()
        functions[fd.getId()] = (
            libsbml.formulaToL3String(body) if body is not None else ""
        )

    rules = []
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        kind = type(rule).__name__.replace("Rule", "").lower() or "rule"
        math = rule.getMath()
        rules.append(
            Rule(
                kind=kind,
                target=rule.getVariable() if rule.isSetVariable() else "",
                formula=(
                    libsbml.formulaToL3String(math) if math is not None else ""
                ),
                units=rule.getUnits() if hasattr(rule, "getUnits") else "",
                sbo=_sbo(rule),
                notes=_read_notes(rule),
            )
        )

    reactions = []
    touches: dict[str, list[str]] = {}
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        parts = {}
        stoichiometry: list[tuple[str, str, float | None]] = []
        for role, getter, count in (
            ("reactants", rxn.getReactant, rxn.getNumReactants()),
            ("products", rxn.getProduct, rxn.getNumProducts()),
            ("modifiers", rxn.getModifier, rxn.getNumModifiers()),
        ):
            refs = []
            for j in range(count):
                ref = getter(j)
                sid = ref.getSpecies()
                refs.append(sid)
                touches.setdefault(sid, []).append(rxn.getId())
                if role != "modifiers":
                    stoichiometry.append(
                        (
                            role,
                            sid,
                            (
                                ref.getStoichiometry()
                                if ref.isSetStoichiometry()
                                else None
                            ),
                        )
                    )
            parts[role] = tuple(refs)
        reactions.append(
            Reaction(
                id=rxn.getId(),
                name=rxn.getName() or "",
                reversible=bool(rxn.getReversible()),
                law=_law(libsbml, rxn),
                calls=_called(libsbml, rxn, set(functions)),
                stoichiometry=tuple(stoichiometry),
                local_parameters=_local_parameters(rxn),
                compartment=(
                    rxn.getCompartment() if rxn.isSetCompartment() else ""
                ),
                fast=bool(rxn.getFast()) if rxn.isSetFast() else False,
                sbo=_sbo(rxn),
                notes=_reaction_notes(rxn),
                annotations=_annotations(rxn),
                **parts,
            )
        )

    in_rules: dict[str, list[str]] = {}
    for kind, target, formula in rules:
        if target:
            in_rules.setdefault(target, []).append(f"{kind}:{target}")

    species = []
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        sid = s.getId()
        mentioned = [
            f"{kind}:{target or '?'}"
            for kind, target, formula in rules
            if _references(formula, sid)
        ]
        species.append(
            Species(
                id=sid,
                name=s.getName() or "",
                compartment=s.getCompartment() or "",
                boundary=bool(s.getBoundaryCondition()),
                constant=bool(s.getConstant()),
                initial=(
                    s.getInitialConcentration()
                    if s.isSetInitialConcentration()
                    else (
                        s.getInitialAmount()
                        if s.isSetInitialAmount()
                        else None
                    )
                ),
                # The two differ by the compartment's size, so which one the
                # file declared is part of the value.
                initial_kind=(
                    "concentration"
                    if s.isSetInitialConcentration()
                    else ("amount" if s.isSetInitialAmount() else "")
                ),
                substance_units=(
                    s.getSubstanceUnits() if s.isSetSubstanceUnits() else ""
                ),
                only_substance_units=bool(s.getHasOnlySubstanceUnits()),
                conversion_factor=(
                    s.getConversionFactor()
                    if s.isSetConversionFactor()
                    else ""
                ),
                sbo=_sbo(s),
                notes=_read_notes(s),
                annotations=_annotations(s),
                in_reactions=tuple(dict.fromkeys(touches.get(sid, ()))),
                in_rules=tuple(
                    dict.fromkeys(in_rules.get(sid, []) + mentioned)
                ),
            )
        )

    def formula(node):
        return libsbml.formulaToL3String(node) if node is not None else ""

    events = []
    for i in range(model.getNumEvents()):
        e = model.getEvent(i)
        trigger = e.getTrigger()
        delay = e.getDelay()
        events.append(
            Event(
                id=e.getId() or "",
                name=e.getName() or "",
                trigger=formula(
                    trigger.getMath() if trigger is not None else None
                ),
                assignments=tuple(
                    (
                        e.getEventAssignment(j).getVariable(),
                        formula(e.getEventAssignment(j).getMath()),
                    )
                    for j in range(e.getNumEventAssignments())
                ),
                delay=formula(delay.getMath() if delay is not None else None),
                from_trigger_time=bool(e.getUseValuesFromTriggerTime()),
                persistent=(
                    bool(trigger.getPersistent())
                    if trigger is not None and trigger.isSetPersistent()
                    else True
                ),
                initial_value=(
                    bool(trigger.getInitialValue())
                    if trigger is not None and trigger.isSetInitialValue()
                    else True
                ),
                priority=formula(
                    e.getPriority().getMath()
                    if e.isSetPriority() and e.getPriority() is not None
                    else None
                ),
                sbo=_sbo(e),
                notes=_read_notes(e),
            )
        )

    parameters = tuple(
        Parameter(
            id=p.getId() or p.getName() or "?",
            name=p.getName() or "",
            value=p.getValue() if p.isSetValue() else None,
            units=p.getUnits() if p.isSetUnits() else "",
            constant=bool(p.getConstant()) if p.isSetConstant() else True,
            sbo=_sbo(p),
            notes=_read_notes(p),
            annotations=_annotations(p),
        )
        for p in (
            model.getParameter(i) for i in range(model.getNumParameters())
        )
    )

    compartments = tuple(
        Compartment(
            id=c.getId(),
            name=c.getName() or "",
            size=c.getSize() if c.isSetSize() else None,
            spatial_dimensions=(
                c.getSpatialDimensions()
                if c.isSetSpatialDimensions()
                else None
            ),
            units=c.getUnits() if c.isSetUnits() else "",
            constant=bool(c.getConstant()) if c.isSetConstant() else True,
            outside=c.getOutside() if c.isSetOutside() else "",
            sbo=_sbo(c),
            notes=_read_notes(c),
            annotations=_annotations(c),
        )
        for c in (
            model.getCompartment(i) for i in range(model.getNumCompartments())
        )
    )

    units = tuple(
        UnitDefinition(
            id=u.getId(),
            name=u.getName() or "",
            definition=_render_units(libsbml, u),
        )
        for u in (
            model.getUnitDefinition(i)
            for i in range(model.getNumUnitDefinitions())
        )
    )

    initial_assignments = tuple(
        InitialAssignment(
            symbol=a.getSymbol(),
            math=formula(a.getMath()),
            sbo=_sbo(a),
            notes=_read_notes(a),
        )
        for a in (
            model.getInitialAssignment(i)
            for i in range(model.getNumInitialAssignments())
        )
    )

    constraints = tuple(
        Constraint(
            math=formula(c.getMath()),
            message=(
                _prose(c.getMessageString() or "") if c.isSetMessage() else ""
            ),
            sbo=_sbo(c),
            notes=_read_notes(c),
        )
        for c in (
            model.getConstraint(i) for i in range(model.getNumConstraints())
        )
    )

    return Deposit(
        name=name,
        species=tuple(species),
        reactions=tuple(reactions),
        functions=functions,
        rules=tuple(rules),
        compartments=compartments,
        events=tuple(events),
        parameters=parameters,
        units=units,
        initial_assignments=initial_assignments,
        constraints=constraints,
        notes=_read_notes(model),
        annotations=_annotations(model),
        model_units=_model_units(model),
        history=_history(model),
        sbml_level=document.getLevel(),
        sbml_version=document.getVersion(),
        packages=_packages(document),
    )
