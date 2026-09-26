"""Literature-mined mechanisms around a model's species, from INDRA's
database, as candidate pieces for composition.

A composed model fails a held-out contrast for a reason a paper has often
already stated: a phosphatase the deposit left out, the transcript a kinase
drives, a feedback. INDRA reads the literature into typed statements —
phosphorylation, dephosphorylation, activation, inhibition, a change in
amount, complex formation — each backed by the sentences and papers it
came from, and serves them at ``db.indra.bio``. :func:`mechanisms` asks for
one gene or one directed pair. :func:`around` asks about each of a model's
own species and sorts what comes back into mechanisms *among* them, which
the model may already carry, and mechanisms reaching one step *outside*,
which are the candidate attachments, ranked by evidence. Every mechanism
carries a sentence and a PMID, so a piece is cited before it is composed.
What a mechanism's kinetics are the literature rarely says; that stays
declared unknown and is calibrated.

    from hallsim.mechanisms import around, model_symbols
    hood = around(model_symbols(process))
    hood.outside[:10]      # the best-attested pieces the model lacks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hallsim.datasets import _cached_json, _retrying
from hallsim.discovery import _get_json

log = logging.getLogger(__name__)

INDRA_DB = "https://db.indra.bio/statements/from_agents"
#: The statement kinds that name a mechanism a process could carry.
KINDS = (
    "Phosphorylation",
    "Dephosphorylation",
    "Activation",
    "Inhibition",
    "IncreaseAmount",
    "DecreaseAmount",
    "Complex",
    "Ubiquitination",
    "Deubiquitination",
    "Acetylation",
    "Translocation",
)
_SOURCE_ROLES = ("subj", "enz", "gef", "gap", "agent")
_TARGET_ROLES = ("obj", "sub", "ras", "rho")


@dataclass(frozen=True)
class Mechanism:
    """One literature-mined statement: ``source`` acts on ``target`` by
    ``kind`` (``members`` for a complex), attested by ``evidence`` sentences
    of which ``sentence`` from ``pmid`` is the first."""

    kind: str
    source: str
    target: str
    members: tuple[str, ...]
    evidence: int
    belief: float
    sentence: str
    pmid: str
    hash: str
    #: Every agent is a human gene or a protein family (HGNC or FamPlex),
    #: so the statement names molecules a process could carry — not a GO
    #: process, a cell type or a mis-grounded protein from another phylum.
    grounded: bool = True
    #: ``(name, FamPlex family)`` per agent, the family empty for a gene.
    refs: tuple[tuple[str, str], ...] = ()

    @property
    def agents(self) -> tuple[str, ...]:
        return self.members or tuple(
            a for a in (self.source, self.target) if a
        )

    @property
    def complete(self) -> bool:
        """Both ends named: a mechanism that could be composed, unlike
        "X is phosphorylated" by an unnamed kinase."""
        if self.members:
            return len(self.members) >= 2
        if self.kind == "Autophosphorylation":
            return bool(self.source)
        return bool(self.source and self.target)

    def __str__(self) -> str:
        who = (
            " + ".join(self.members)
            if self.members
            else f"{self.source} → {self.target}"
        )
        cite = f" (PMID {self.pmid})" if self.pmid else ""
        return (
            f"{self.kind:<17} {who:<28} {self.evidence:>4} ev  "
            f"{self.sentence[:90]}{cite}"
        )


@dataclass
class Neighbourhood:
    """What the literature says around a set of species: ``among`` holds
    mechanisms whose agents all belong to the set, ``outside`` those that
    reach one agent beyond it, both ranked by evidence."""

    symbols: tuple[str, ...]
    among: list[Mechanism] = field(default_factory=list)
    outside: list[Mechanism] = field(default_factory=list)


def _name(agent) -> str:
    return str((agent or {}).get("name") or "")


#: Groundings that name a human gene or a protein family.
_MOLECULAR = ("HGNC", "FPLX")
_FAMILIES: dict[str, frozenset] | None = None


def _grounded(agent) -> bool:
    refs = (agent or {}).get("db_refs") or {}
    return any(ns in refs for ns in _MOLECULAR)


def _family(agent) -> str:
    return str(((agent or {}).get("db_refs") or {}).get("FPLX") or "")


def families_of() -> dict[str, frozenset]:
    """``gene symbol → FamPlex families it belongs to``, transitively, from
    the vendored FamPlex relations table. INDRA grounds "ERK" to the family
    ERK, not to MAPK1, so a model that carries MAPK1 carries ERK's
    statements too."""
    global _FAMILIES
    if _FAMILIES is not None:
        return _FAMILIES
    from hallsim.reporter_wiring import _data_dir

    path = _data_dir() / "famplex" / "relations.csv"
    parents: dict[str, set[str]] = {}
    members: dict[str, set[str]] = {}
    if not path.exists():
        log.warning(
            "%s is missing: family-level statements (ERK for MAPK1) will "
            "count as outside a model rather than among its species",
            path,
        )
        _FAMILIES = {}
        return _FAMILIES
    for line in path.read_text().splitlines():
        ns, name, _rel, pns, parent = (line.split(",") + [""] * 5)[:5]
        if pns != "FPLX":
            continue
        if ns == "FPLX":
            parents.setdefault(name, set()).add(parent)
        elif ns == "HGNC":
            members.setdefault(name.upper(), set()).add(parent)
    out = {}
    for symbol, fams in members.items():
        closed, frontier = set(), list(fams)
        while frontier:
            f = frontier.pop()
            if f not in closed:
                closed.add(f)
                frontier.extend(parents.get(f, ()))
        out[symbol] = frozenset(closed)
    _FAMILIES = out
    return out


def _parse(payload: dict) -> list[Mechanism]:
    counts = payload.get("evidence_counts") or {}
    out = []
    for h, s in (payload.get("statements") or {}).items():
        raw = list(s.get("members") or []) or [
            s[r] for r in _SOURCE_ROLES + _TARGET_ROLES if s.get(r)
        ]
        members = tuple(_name(m) for m in s.get("members") or [])
        source = next((_name(s[r]) for r in _SOURCE_ROLES if s.get(r)), "")
        target = next((_name(s[r]) for r in _TARGET_ROLES if s.get(r)), "")
        ev = s.get("evidence") or [{}]
        out.append(
            Mechanism(
                kind=str(s.get("type", "")),
                source=source,
                target=target,
                members=members,
                evidence=int(counts.get(h) or len(ev)),
                belief=float(s.get("belief") or 0.0),
                sentence=str(ev[0].get("text") or "").strip(),
                pmid=str(ev[0].get("pmid") or ""),
                hash=str(h),
                grounded=bool(raw) and all(_grounded(a) for a in raw),
                refs=tuple((_name(a), _family(a)) for a in raw),
            )
        )
    return sorted(out, key=lambda m: -m.evidence)


def _fetch(params: dict, timeout: float) -> dict:
    query = {k: v for k, v in params.items() if v is not None}
    query.update(format="json", best_first="true")
    key = "indra " + " ".join(f"{k}={query[k]}" for k in sorted(query))
    return _cached_json(
        key, lambda: _retrying(lambda: _get_json(INDRA_DB, query, timeout))
    )


def mechanisms(
    *,
    agent: str | None = None,
    subject: str | None = None,
    object: str | None = None,
    kind: str | None = None,
    ev_limit: int = 2,
    max_stmts: int = 200,
    timeout: float = 60.0,
) -> list[Mechanism]:
    """Statements about one gene (``agent``, in any role) or one directed
    pair (``subject`` acting on ``object``), best-attested first. ``kind``
    keeps one statement type, e.g. ``"Dephosphorylation"``. Symbols are
    HGNC gene symbols; INDRA resolves family names such as ERK itself."""
    if not (agent or subject or object):
        raise ValueError("give agent=, or subject= and/or object=")
    payload = _fetch(
        {
            "agent": agent,
            "subject": subject,
            "object": object,
            "type": kind,
            "ev_limit": ev_limit,
            "max_stmts": max_stmts,
        },
        timeout,
    )
    return _parse(payload)


def around(
    symbols,
    *,
    kind: str | None = None,
    min_evidence: int = 2,
    max_stmts: int = 200,
    ev_limit: int = 2,
    timeout: float = 60.0,
) -> Neighbourhood:
    """The literature's mechanisms among ``symbols`` and one step outside
    them, one query per symbol. A statement returned for two of the symbols
    involves both, so it is *among* them whatever name INDRA gave the
    agents; one returned for a single symbol reaches outside, and is a
    candidate piece when both its ends are named and every agent is a
    human gene or a protein family. A family-level statement about a
    member of ``symbols`` (ERK for MAPK1) is not returned for the member,
    so it lands outside; the FamPlex crosswalk that would fold it in is
    not carried yet."""
    symbols = tuple(dict.fromkeys(s for s in symbols if s))
    seen: dict[str, Mechanism] = {}
    hits: dict[str, set[str]] = {}
    for sym in symbols:
        for m in mechanisms(
            agent=sym,
            kind=kind,
            ev_limit=ev_limit,
            max_stmts=max_stmts,
            timeout=timeout,
        ):
            if m.evidence < min_evidence or not m.complete:
                continue
            seen.setdefault(m.hash, m)
            hits.setdefault(m.hash, set()).add(sym)
    inside = {s.upper() for s in symbols}
    families = families_of()
    inside_families = set().union(
        *(families.get(s, frozenset()) for s in inside)
    )

    def within(ref) -> bool:
        name, family = ref
        return name.upper() in inside or (
            bool(family) and family in inside_families
        )

    hood = Neighbourhood(symbols)
    for h, m in seen.items():
        if len(hits[h]) >= 2 or all(within(r) for r in m.refs):
            hood.among.append(m)
        elif m.grounded:
            hood.outside.append(m)
    hood.among.sort(key=lambda m: -m.evidence)
    hood.outside.sort(key=lambda m: -m.evidence)
    log.info(
        "around %d symbols: %d mechanisms among them, %d one step outside",
        len(symbols),
        len(hood.among),
        len(hood.outside),
    )
    return hood


def model_symbols(model) -> list[str]:
    """The HGNC symbols of the proteins a process or composite annotates
    with UniProt, resolved through the reporter wiring's crosswalk."""
    from hallsim.reporter_wiring import _human_symbol

    schemas = (
        [p.ports_schema() for p in model.processes.values()]
        if hasattr(model, "processes")
        else [model.ports_schema()]
    )
    accessions = []
    for schema in schemas:
        for port in schema.values():
            ont = {
                k.lower(): v
                for k, v in (getattr(port, "ontology", None) or {}).items()
            }
            if ont.get("uniprot"):
                accessions.append(str(ont["uniprot"]))
    symbols = []
    for acc in dict.fromkeys(accessions):
        sym = _human_symbol(acc)[0]
        if sym and sym not in symbols:
            symbols.append(sym)
    return symbols
