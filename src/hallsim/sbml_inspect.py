"""What a deposit holds, read from its SBML and nothing else.

Choosing a deposit means arguing from its species *and its reactions*, because a
model whose title is about a pathway need not carry one: the commonest way a week
is lost here is a deposit chosen on its name. Nothing else in the framework shows
a reaction, so that argument has been made by hand-written regex over cached
files, which is how a species that appears in no reaction at all reads as
mechanism.

This reads the file, deliberately, rather than the imported model: what the
deposit *says* is the question here, and an import is entitled to drop what it
cannot use, so going through it would hide the gaps worth seeing. A deposit whose
reactions are all ``functionDefinition`` calls keeps its mechanism somewhere the
reaction bodies do not show it, and the bodies are here too.

Then :func:`import_delta` reads the same deposit *through* the importer and
reports the difference, which is the half no amount of hand-written regex over
the file can produce: an annotation the file carries and the port schema does
not, a state the file declares and the model does not integrate. Every silent
join failure found in this project so far has been one of those two.

    from hallsim.sbml_inspect import inspect_sbml, import_delta
    d = inspect_sbml("BIOMD0000000585")
    [s.id for s in d.species if s.unused]     # named, referenced by nothing
    d.reactions[0].law                        # the rate law, as written
    import_delta("BIOMD0000000585")           # what the import did not keep
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


@dataclass(frozen=True)
class Species:
    """One species as the file declares it."""

    id: str
    name: str
    compartment: str
    boundary: bool
    constant: bool
    initial: float | None
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
                ("UNUSED", self.unused),
            )
            if on
        ]
        where = f"@{self.compartment}" if self.compartment else ""
        tail = f"  [{', '.join(marks)}]" if marks else ""
        label = f" — {self.name}" if self.name and self.name != self.id else ""
        return f"{self.id}{where}{label}{tail}"


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

    def __str__(self) -> str:
        left = " + ".join(self.reactants) or "∅"
        right = " + ".join(self.products) or "∅"
        arrow = "<->" if self.reversible else "->"
        head = f"{self.id}: {left} {arrow} {right}"
        if self.modifiers:
            head += f"   (modifiers: {', '.join(self.modifiers)})"
        return f"{head}\n    {self.law or '(no kinetic law)'}"


@dataclass(frozen=True)
class Deposit:
    """A deposit's mechanism, as its own file states it."""

    name: str
    species: tuple[Species, ...] = ()
    reactions: tuple[Reaction, ...] = ()
    functions: dict[str, str] = field(default_factory=dict)
    rules: tuple[tuple[str, str, str], ...] = ()
    compartments: tuple[str, ...] = ()

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


def inspect_sbml(source, name: str | None = None) -> Deposit:
    """Read ``source`` and report the mechanism it declares.

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
            (
                kind,
                rule.getVariable() if rule.isSetVariable() else "",
                libsbml.formulaToL3String(math) if math is not None else "",
            )
        )

    reactions = []
    touches: dict[str, list[str]] = {}
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        parts = {}
        for role, getter, count in (
            ("reactants", rxn.getReactant, rxn.getNumReactants()),
            ("products", rxn.getProduct, rxn.getNumProducts()),
            ("modifiers", rxn.getModifier, rxn.getNumModifiers()),
        ):
            refs = []
            for j in range(count):
                sid = getter(j).getSpecies()
                refs.append(sid)
                touches.setdefault(sid, []).append(rxn.getId())
            parts[role] = tuple(refs)
        reactions.append(
            Reaction(
                id=rxn.getId(),
                name=rxn.getName() or "",
                reversible=bool(rxn.getReversible()),
                law=_law(libsbml, rxn),
                calls=_called(libsbml, rxn, set(functions)),
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
                in_reactions=tuple(dict.fromkeys(touches.get(sid, ()))),
                in_rules=tuple(
                    dict.fromkeys(in_rules.get(sid, []) + mentioned)
                ),
            )
        )

    return Deposit(
        name=resolved,
        species=tuple(species),
        reactions=tuple(reactions),
        functions=functions,
        rules=tuple(rules),
        compartments=tuple(
            model.getCompartment(i).getId()
            for i in range(model.getNumCompartments())
        ),
    )


@dataclass(frozen=True)
class ImportDelta:
    """What the importer did not keep from a deposit's own file.

    Neither list is a defect on its own: a boundary constant is *meant* not to
    be integrated, and a species the model computes algebraically is still
    exposed. What matters is that they are visible, because the silent failures
    in this project have all been of the second kind — an identifier the file
    carries, the port schema lacks, and a readout then joins to nothing while
    reporting success.
    """

    name: str
    declared_species: int
    ported_species: int
    annotated_in_file: tuple[str, ...] = ()
    annotated_in_ports: tuple[str, ...] = ()
    missing_ports: tuple[str, ...] = ()

    @property
    def lost_annotations(self) -> tuple[str, ...]:
        """Species the file annotates whose port carries no ontology id."""
        return tuple(
            s
            for s in self.annotated_in_file
            if s not in self.annotated_in_ports
        )

    def __str__(self) -> str:
        lines = [
            f"{self.name}: {self.declared_species} species declared, "
            f"{self.ported_species} exposed as ports"
        ]
        if self.missing_ports:
            lines.append(
                f"  declared, not a port: {', '.join(self.missing_ports[:12])}"
            )
        lines.append(
            f"  annotated in the file: {len(self.annotated_in_file)}; "
            f"carried onto a port: {len(self.annotated_in_ports)}"
        )
        if self.lost_annotations:
            lines.append(
                "  ANNOTATION LOST (a readout joining on these finds nothing): "
                + ", ".join(self.lost_annotations[:12])
            )
        return "\n".join(lines)


def _file_annotations(source, name: str | None = None) -> dict[str, dict]:
    """Ontology ids the file attaches to each species, by species id.

    The importer's own MIRIAM reader, so the comparison below is against what
    the import *could* have seen rather than against a second parser of mine
    that might disagree with it for its own reasons.
    """
    from hallsim.sbml_import import _extract_species_ontology, _resolve_source

    path, _ = _resolve_source(source, name or "deposit")
    return {
        species: terms
        for species, terms in _extract_species_ontology(str(path)).items()
        if terms
    }


def import_delta(source, name: str = "deposit") -> ImportDelta:
    """Read a deposit twice — as a file and through the importer — and diff it.

    This is what importing the framework buys over reading the SBML by hand: the
    file alone cannot say what was lost on the way in.
    """
    from hallsim.sbml_import import process_from_sbml

    deposit = inspect_sbml(source, name)
    annotated = _file_annotations(source, name)
    schema = process_from_sbml(source, name=name).ports_schema()

    ported_with_ontology = {
        port
        for port, spec in schema.items()
        if getattr(spec, "ontology", None)
    }
    declared = {s.id for s in deposit.species}
    return ImportDelta(
        name=deposit.name,
        declared_species=len(deposit.species),
        ported_species=len(schema),
        annotated_in_file=tuple(sorted(annotated)),
        annotated_in_ports=tuple(
            sorted(ported_with_ontology & set(annotated))
        ),
        missing_ports=tuple(sorted(declared - set(schema))),
    )
