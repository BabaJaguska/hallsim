"""Where a search hit meets the framework.

:mod:`hallsim.search` says what a repository holds. Whether the framework
can use it is a different question, answered here by reading the thing
itself. A model candidate is opened and asked what role a quantity plays
in it (:func:`screen_species_roles`): annotation search answers "is this
model about IL6", composing needs to know whether the model produces IL6,
consumes it, reads it in a rate law, or holds it as a fixed input. Each
is a different use. A producer supplies the quantity; a consumer or a
reader is downstream biology waiting for a producer to be attached by an
edge. A dataset candidate is read against a composite's annotated store
paths (:func:`coverage`): a deposit is a calibration target only if it
measures something the composite carries. :func:`search_species` and
:func:`search_measuring` are the two searches with their screen applied,
and :func:`loader_route` says how the expression loader would read a
platform, or why it cannot.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hallsim.search.datasets import DatasetCandidate, search_for_dataset
from hallsim.search.models import model_files, search_for_model

log = logging.getLogger(__name__)


# ── The role a quantity plays in a model ────────────────────────────

#: What a deposit can do with a species, each a different use when
#: composing. ``produces``: a product of some reaction, so the deposit can
#: supply it. ``consumes``: a reactant, so the deposit removes it and a
#: producer attached upstream drives it. ``reads``: a modifier or a symbol
#: in a rate law, so it regulates without being used up. ``input``: a
#: boundary or constant species the deposit takes as given, the plainest
#: place to attach a producer.
ROLES = ("produces", "consumes", "reads", "input")


@dataclass(frozen=True)
class SpeciesScreen:
    """Which roles the species matching a pattern play in one deposit."""

    model_id: str
    #: ``match`` | ``no-match`` | ``qualitative`` (SBML-qual, a logical
    #: model) | ``constraint-based`` (SBML-fbc) | ``no-reactions`` |
    #: ``no-rate-laws`` | ``no-sbml`` | ``unreadable`` | ``fetch-failed`` |
    #: ``unscreenable``
    status: str
    #: ``((species label, (role, ...)), ...)`` for every matching species.
    roles: tuple[tuple[str, tuple[str, ...]], ...] = ()
    n_species: int = 0
    n_reactions: int = 0
    #: Why, when the deposit could not be screened.
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "match"

    def having(self, role: str) -> tuple[str, ...]:
        """The matching species that play ``role``."""
        return tuple(s for s, rs in self.roles if role in rs)

    @property
    def produced(self) -> tuple[str, ...]:
        return self.having("produces")


def _screen_xpp(
    model_id: str, path, pattern: str, wanted: tuple[str, ...]
) -> "SpeciesScreen":
    """Screen an XPPAUT model, which declares dynamics and no reactions.

    Without reactions the only roles readable off the derivative are
    whether a state has a source term (``produces``) or only loses mass
    (``consumes``)."""
    from hallsim.intake import emitted_species
    from hallsim.process import PortRole
    from hallsim.xpp_import import process_from_xpp

    try:
        proc = process_from_xpp(str(path), name="m")
    except Exception as exc:
        return SpeciesScreen(
            model_id, "unreadable", note=f"{type(exc).__name__}: {exc}"
        )
    rx = re.compile(pattern, re.I)
    produced = set(emitted_species(proc, pattern))
    states = [
        s for s in proc.ports_with_role(PortRole.EVOLVED) if rx.search(s)
    ]
    roles = tuple(
        (s, ("produces",) if s in produced else ("consumes",))
        for s in sorted(states)
    )
    return SpeciesScreen(
        model_id,
        "match" if any(set(r) & set(wanted) for _, r in roles) else "no-match",
        roles,
        n_species=len(proc.ports_schema()),
        note="XPPAUT; screened on the derivative, not on reactions",
    )


def _math_names(node) -> set[str]:
    """Every symbol a libsbml formula reads."""
    if node is None:
        return set()
    out = {node.getName()} if node.isName() else set()
    for i in range(node.getNumChildren()):
        out |= _math_names(node.getChild(i))
    return out


def sbml_species_roles(model, pattern: str) -> dict[str, tuple[str, ...]]:
    """``{species label: roles}`` for the species of a libsbml model
    whose id or display name matches ``pattern``.

    The label is the display name where one is declared: a CellDesigner
    export gives every species a UUID id and the gene symbol as its name.
    Only reactions carrying a kinetic law count; an arrow in a drawn map
    produces and consumes nothing.
    """
    rx = re.compile(pattern, re.I)
    label, fixed = {}, set()
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        sid = s.getId()
        if rx.search(sid) or rx.search(s.getName() or ""):
            label[sid] = s.getName() or sid
            if s.getBoundaryCondition() or s.getConstant():
                fixed.add(sid)
    found: dict[str, set[str]] = {sid: set() for sid in label}
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        if not reaction.isSetKineticLaw():
            continue
        products = {
            reaction.getProduct(j).getSpecies()
            for j in range(reaction.getNumProducts())
        }
        reactants = {
            reaction.getReactant(j).getSpecies()
            for j in range(reaction.getNumReactants())
        }
        modifiers = {
            reaction.getModifier(j).getSpecies()
            for j in range(reaction.getNumModifiers())
        }
        law = _math_names(reaction.getKineticLaw().getMath())
        for sid in found:
            if sid in products:
                found[sid].add("produces")
            if sid in reactants:
                found[sid].add("consumes")
            if sid in modifiers or (
                sid in law and sid not in products | reactants
            ):
                found[sid].add("reads")
    for sid in fixed:
        if found[sid]:
            found[sid].add("input")
    out: dict[str, set[str]] = {}
    for sid, rs in found.items():
        if rs:
            out.setdefault(label[sid], set()).update(rs)
    return {
        name: tuple(r for r in ROLES if r in rs)
        for name, rs in sorted(out.items())
    }


def _readable_model_files(paths):
    """``paths`` with any COPASI or CellML file converted to SBML.

    A deposit's supplement often ships the ``.cps`` and no SBML at all, and
    the Physiome repository ships CellML only; a deposit is screened on its
    content, not skipped for its format.
    """
    from hallsim.cellml_import import cellml_to_sbml, is_cellml
    from hallsim.cps_import import cps_to_sbml, is_cps

    converters = ((is_cps, cps_to_sbml), (is_cellml, cellml_to_sbml))
    out = []
    for path in paths:
        convert = next((c for test, c in converters if test(path)), None)
        if convert is None:
            out.append(path)
            continue
        try:
            out.append(convert(path))
        except Exception as exc:
            log.warning("could not convert %s: %s", path, exc)
    return out


def _first_readable_sbml(paths):
    """``(model, note)`` for the first path libsbml reads as SBML.

    A deposit is a directory of many files — OWL, MATLAB, Octave, PNG, PDF and
    a ``manifest.xml`` that is XML but not SBML — so "the .xml file" is not a
    well-defined thing to open.
    """
    import libsbml

    seen = []
    for path in paths:
        if not str(path).endswith((".xml", ".sbml")):
            continue
        doc = libsbml.SBMLReader().readSBMLFromFile(str(path))
        model = doc.getModel()
        if model is not None:
            return model, ""
        seen.append(f"{Path(path).name}({doc.getNumErrors()} errors)")
    if not seen:
        return None, "deposit contains no .xml/.sbml file"
    return None, "no readable SBML among " + ", ".join(seen)


def _reactionless_formalism(model) -> tuple[str, str]:
    """``(status, note)`` for a parsed SBML model with no reactions, from the
    package it carries: SBML-qual is a logical model, SBML-fbc a
    constraint-based one, and neither is a rate-law model to screen."""
    if model.getPlugin("qual") is not None:
        return (
            "qualitative",
            "SBML-qual: a logical model over discrete levels, with update "
            "rules rather than rate laws; nothing to integrate or screen",
        )
    if model.getPlugin("fbc") is not None:
        return (
            "constraint-based",
            "SBML-fbc: a constraint-based model with flux bounds and an "
            "objective rather than rate laws; nothing to integrate or screen",
        )
    return (
        "no-reactions",
        "parsed, but declares no reactions and no qual or fbc package — "
        "not a rate-law model",
    )


def screen_species_roles(
    candidates,
    pattern: str,
    *,
    roles: tuple[str, ...] = ROLES,
    timeout: float = 60.0,
) -> list[SpeciesScreen]:
    """What role each species matching ``pattern`` plays in each of
    ``candidates``, with ``match`` where one plays any of ``roles``.

    ``pattern`` is a case-insensitive regex matched against species ids and
    display names. A deposit that consumes or reads a quantity it never
    produces is not a failure: it is the downstream half of a composite,
    and the producer is the edge to attach. Narrow ``roles`` to
    ``("produces",)`` when the deposit itself must supply the quantity.

    Every input yields a row, including the ones that could not be read — a
    silent skip hides an unreadable deposit as an uninteresting one.
    """
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; choose {ROLES}")
    wanted = tuple(roles)
    out: list[SpeciesScreen] = []
    for cand in candidates:
        model_id = getattr(cand, "id", cand)
        source = getattr(cand, "source", "biomodels")
        # A repository that re-hosts BioModels embeds the accession in its own
        # id (BioSimulations: "BIOMD0000000582_tellurium_..."). Screening the
        # underlying deposit beats reporting a duplicate as unreadable.
        embedded = re.search(r"BIOMD\d{10}", str(model_id))
        if source != "biomodels" and embedded:
            model_id, source = embedded.group(0), "biomodels"
        try:
            paths = model_files(model_id, source, timeout)
        except LookupError as exc:
            # A source without a fetcher is reported, never dropped — an
            # unscreened hit is still a hit, and silently losing it looks
            # like the repository had none.
            out.append(
                SpeciesScreen(
                    str(model_id),
                    "unscreenable",
                    note=f"{exc} (format {getattr(cand, 'format', '?')})",
                )
            )
            continue
        except Exception as exc:  # network, 404, malformed accession
            out.append(
                SpeciesScreen(
                    str(model_id),
                    "fetch-failed",
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        paths = _readable_model_files(paths)
        # An XPPAUT .ode has no reactions to scan, so it is screened on the
        # derivative instead (intake.emitted_species). Reported with the same
        # statuses so the two formats read alike in the outcome table.
        ode = [q for q in paths if str(q).lower().endswith(".ode")]
        if ode and not [
            q for q in paths if str(q).lower().endswith((".xml", ".sbml"))
        ]:
            out.append(_screen_xpp(str(model_id), ode[0], pattern, wanted))
            continue
        try:
            model, note = _first_readable_sbml(paths)
        except ImportError as exc:
            out.append(SpeciesScreen(str(model_id), "no-sbml", note=str(exc)))
            continue
        if model is None:
            status = "no-sbml" if "no .xml" in note else "unreadable"
            out.append(SpeciesScreen(str(model_id), status, note=note))
            continue
        n_rx = model.getNumReactions()
        if n_rx and not any(
            model.getReaction(i).isSetKineticLaw() for i in range(n_rx)
        ):
            # A CellDesigner disease map draws reactions with no rate law.
            # Nothing is produced by an arrow, so counting one as production
            # promotes a diagram to a model (Wu2010 has 254 such reactions).
            out.append(
                SpeciesScreen(
                    str(model_id),
                    "no-rate-laws",
                    n_species=model.getNumSpecies(),
                    n_reactions=n_rx,
                    note=f"{n_rx} reactions, none with a kinetic law — a "
                    f"drawn pathway map, not an integrable model",
                )
            )
            continue
        if n_rx == 0:
            # SBML-qual and other non-reaction formalisms parse fine and
            # present an empty core model. Reporting that as ``no-match`` is
            # indistinguishable from a deposit whose reactions were read and
            # produced nothing, which is the opposite conclusion. The
            # package the file carries says which formalism it is.
            status, note = _reactionless_formalism(model)
            out.append(
                SpeciesScreen(
                    str(model_id),
                    status,
                    n_species=model.getNumSpecies(),
                    note=note,
                )
            )
            continue
        found = sbml_species_roles(model, pattern)
        hit = any(set(rs) & set(wanted) for rs in found.values())
        out.append(
            SpeciesScreen(
                str(model_id),
                "match" if hit else "no-match",
                tuple(found.items()),
                model.getNumSpecies(),
                model.getNumReactions(),
            )
        )
    return out


def search_species(
    query: str,
    pattern: str,
    *,
    roles: tuple[str, ...] = ROLES,
    limit: int = 40,
    sources=None,
    **kwargs,
) -> list[SpeciesScreen]:
    """Search **every** repository for ``query`` and say what role
    ``pattern`` plays in each hit.

    The composable version of a text search: a hit is useful when the
    quantity takes part in its dynamics, whether the deposit supplies it or
    is what a supplied quantity would drive. Defaults to all sources —
    searching one repository and concluding the model does not exist is how
    a candidate gets missed.
    """
    hits = search_for_model(query, limit=limit, sources=sources, **kwargs)
    return screen_species_roles(hits, pattern, roles=roles)


# ── A dataset that measures what a composite carries ────────────────


@dataclass(frozen=True)
class Coverage:
    """Which of a composite's annotated store paths a dataset measures,
    and through what: ``direct`` (the path's own id is measured),
    ``complete`` (the modality measures every quantity of the path's
    kind), ``regulon`` (a transcriptome reads a transcription factor
    through its targets), or ``none``."""

    paths: tuple[str, ...] = ()
    via: str = "none"

    def __bool__(self) -> bool:
        return bool(self.paths)


def coverage(candidate: DatasetCandidate, composite, ontmap=None) -> Coverage:
    """What ``candidate`` measures of ``composite``, by ontology."""
    from hallsim.reporter_wiring import (
        paths_measuring,
        store_ontology_map,
        tf_observables,
    )

    ontmap = store_ontology_map(composite) if ontmap is None else ontmap
    m = candidate.measured

    def with_ns(ns: str) -> list[str]:
        return sorted(p for p, o in ontmap.items() if ns in o)

    def listed(ns: str) -> list[str]:
        return sorted(paths_measuring(ontmap, ns, m.ids))

    if m.modality == "proteomics":
        hit = listed("uniprot")
        if hit:
            return Coverage(tuple(hit), "direct")
        if m.complete:
            return Coverage(tuple(with_ns("uniprot")), "complete")
    elif m.modality == "metabolomics":
        hit = listed("chebi")
        if hit:
            return Coverage(tuple(hit), "direct")
        if m.complete:
            return Coverage(tuple(with_ns("chebi")), "complete")
    elif m.modality == "expression" and m.complete:
        return Coverage(
            tuple(sorted(tf_observables(composite, ontmap))), "regulon"
        )
    elif m.modality == "imaging":
        hit = listed("uniprot")
        if hit:
            return Coverage(tuple(hit), "direct")
    return Coverage()


def search_measuring(
    query: str, composite, *, limit: int = 25, sources=None, **kwargs
) -> list[tuple[DatasetCandidate, Coverage]]:
    """Search every repository for ``query`` and keep the hits that
    measure something ``composite`` carries, best covered and longest time
    course first."""
    from hallsim.reporter_wiring import store_ontology_map

    ontmap = store_ontology_map(composite)
    hits = search_for_dataset(query, limit=limit, sources=sources, **kwargs)
    kept = []
    for hit in hits:
        cov = coverage(hit, composite, ontmap)
        if cov:
            kept.append((hit, cov))
    kept.sort(key=lambda hc: (-len(hc[1].paths), -hc[0].design.n_timepoints))
    return kept


def loader_route(frame: pd.DataFrame) -> str:
    """How :func:`hallsim.gene_reporters.load_gene_expression` would map
    this platform's probes to genes, or why it cannot."""
    from hallsim.gene_reporters import choose_annotation

    if frame.empty or "ID" not in frame.columns:
        return "no platform table in the series' SOFT"
    try:
        col, route = choose_annotation(frame)
    except ValueError as exc:
        return f"the loader cannot map it: {exc}"
    via = " via MyGene.info" if route == "accession" else ""
    return f"the loader reads {route}s from column {col!r}{via}"
