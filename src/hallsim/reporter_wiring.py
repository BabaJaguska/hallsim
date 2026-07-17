"""Reporter wiring validation — is a transcript reporter faithful to the
mechanistic node it reads?

A bulk-transcript measurement can faithfully track only a quantity that is
itself transcriptionally coupled: a transcription factor's activity read
off one of its target genes, or a modeled transcript. A metabolite, a
physical structure (organelle mass, protein complex), or a non-TF
protein/kinase activity mapped onto a gene is a category error — a
transcript cannot follow a phospho-level or a mitochondrial mass.

Classification is objective, from data the framework already retains plus
vendored reference tables — not a hand-maintained keyword list:

- molecular *kind* from each species' SBML MIRIAM annotation
  (:attr:`hallsim.process.Port.ontology`: ChEBI ⇒ metabolite, UniProt ⇒
  protein, GO aspect ⇒ physical vs process) via ``data/ontology/``,
- TF identity, target membership and sign from a vendored CollecTRI
  snapshot (``data/collectri/``), crosswalked UniProt→symbol.

The residual the annotations cannot resolve — a phospho/active *form*
distinction (``mTORC1_pS2448`` annotates as the TORC1 complex, not its
active state) — is flagged low-confidence, never silently asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

import pandas as pd

from hallsim.process import PortRole

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
_PHOSPHO = re.compile(r"_p[STY]\d")


class ObservableKind(Enum):
    TRANSCRIPT = "transcript"
    PROTEIN = "protein"
    METABOLITE = "metabolite"
    PHYSICAL = "physical"  # GO cellular_component: complex / organelle / mass
    PROCESS = "process"  # GO biological_process / molecular_function
    UNKNOWN = "unknown"


# ── Vendored reference tables ──────────────────────────────────────


@lru_cache(maxsize=1)
def _go_aspect() -> dict[str, str]:
    df = pd.read_csv(_DATA / "ontology" / "go_aspect.tsv", sep="\t")
    return dict(zip(df.go_id, df.aspect))


@lru_cache(maxsize=1)
def _uniprot_symbol() -> dict[str, tuple[str, str]]:
    """UniProt accession → (gene symbol, NCBI taxon id)."""
    df = pd.read_csv(
        _DATA / "ontology" / "uniprot_symbol.tsv", sep="\t", dtype=str
    )
    return {
        a: (s, t)
        for a, s, t in zip(df.uniprot, df.symbol, df.taxon)
        if pd.notna(s)
    }


@lru_cache(maxsize=1)
def _orthologs() -> dict[str, str]:
    """Mouse gene symbol → human ortholog symbol (MGI)."""
    df = pd.read_csv(
        _DATA / "ontology" / "ortholog_mouse_human.tsv", sep="\t", dtype=str
    )
    return dict(zip(df.mouse_symbol, df.human_symbol))


@lru_cache(maxsize=1)
def _collectri() -> tuple[frozenset, dict, dict]:
    """(TF symbols, {(tf, target): sign}, {target: {tf: sign}})."""
    df = pd.read_csv(_DATA / "collectri" / "collectri_human.tsv", sep="\t")
    edges = {
        (t, g): int(w) for t, g, w in zip(df.source, df.target, df.weight)
    }
    regulators: dict[str, dict[str, int]] = {}
    for (tf, g), w in edges.items():
        regulators.setdefault(g, {})[tf] = w
    return frozenset(df.source.unique()), edges, regulators


def _human_symbol(uniprot: str) -> tuple[str | None, str | None]:
    """Human gene symbol for a UniProt accession, plus a miss flag.

    Human accessions map directly; mouse accessions map through the MGI
    ortholog table. Returns ``(symbol, None)`` when resolved, or
    ``(None, "uniprot-missing")`` / ``(symbol, "ortholog-missing")`` so a
    stale snapshot surfaces instead of degrading silently.
    """
    rec = _uniprot_symbol().get(uniprot)
    if rec is None:
        return None, "uniprot-missing"
    sym, taxon = rec
    if taxon == "9606":
        return sym.upper(), None
    human = _orthologs().get(sym)
    if human:
        return human.upper(), None
    return sym.upper(), "ortholog-missing"


# ── Ontology resolution over a composite ───────────────────────────


def store_ontology_map(composite) -> dict[str, dict]:
    """``store_path → {namespace: identifier}`` for every annotated path.

    Namespaces are lower-cased so ``GO``/``go`` collapse. The first
    non-empty annotation seen for a path wins (the species owner).
    """
    result: dict[str, dict] = {}
    for pname, proc in composite.processes.items():
        topo = composite.topology[pname]
        for port_name, port in proc.ports_schema().items():
            path = topo.get(port_name, f"{pname}/{port_name}")
            ont = getattr(port, "ontology", None)
            if ont and path not in result:
                result[path] = {k.lower(): v for k, v in ont.items()}
    return result


def resolve_ontology(
    path: str, composite, ontmap: dict[str, dict] | None = None
) -> tuple[dict, str]:
    """Annotation for ``path``, following one observer hop if needed.

    A reporter often reads a derived path (``gz06/x2_integral``) whose
    annotation lives on the source species (``gz06/x``). When ``path``
    itself is unannotated, an observer process writing ``path`` from a
    single INPUT source resolves to that source's annotation. Returns
    ``(annotation, resolved_path)``.
    """
    ontmap = store_ontology_map(composite) if ontmap is None else ontmap
    if ontmap.get(path):
        return ontmap[path], path
    for pname, proc in composite.processes.items():
        topo = composite.topology[pname]
        schema = proc.ports_schema()
        writes = any(
            p.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE, PortRole.LATCHED)
            and topo.get(pn, f"{pname}/{pn}") == path
            for pn, p in schema.items()
        )
        inputs = [
            topo.get(pn, f"{pname}/{pn}")
            for pn, p in schema.items()
            if p.role is PortRole.INPUT
        ]
        if writes and len(inputs) == 1 and ontmap.get(inputs[0]):
            return ontmap[inputs[0]], inputs[0]
    return {}, path


def classify_ontology(ont: dict) -> ObservableKind:
    """Molecular kind from a MIRIAM annotation dict (lower-cased keys)."""
    if not ont:
        return ObservableKind.UNKNOWN
    if "chebi" in ont:
        return ObservableKind.METABOLITE
    if "uniprot" in ont:
        return ObservableKind.PROTEIN
    if "go" in ont:
        aspect = _go_aspect().get(ont["go"])
        if aspect == "cellular_component":
            return ObservableKind.PHYSICAL
        if aspect in ("biological_process", "molecular_function"):
            return ObservableKind.PROCESS
    return ObservableKind.UNKNOWN


# ── Verdict + report ───────────────────────────────────────────────


@dataclass(frozen=True)
class ReporterVerdict:
    gene: str
    observable: str
    resolved_path: str
    kind: ObservableKind
    status: str  # ok | category-error | self-map | sign-conflict | proxy |
    #              tf-target-absent | unannotated
    message: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_warning(self) -> bool:
        return self.status != "ok"


def _branch_hint(gene: str) -> str:
    regs = _collectri()[2].get(gene, {})
    if not regs:
        return f"No CollecTRI regulator of {gene} — annotate the intended TF."
    top = ", ".join(
        f"{tf}({'+' if s > 0 else '−'})" for tf, s in list(regs.items())[:4]
    )
    return (
        f"Model {gene} as transcribed by its regulator via a "
        f"TranscriptionBranch; CollecTRI TFs for {gene}: {top}."
    )


def classify_reporter(reporter, composite, ontmap=None) -> ReporterVerdict:
    """Objective verdict for one reporter against a composite + CollecTRI."""
    tf_set, edges, _ = _collectri()
    obs, gene = reporter.observable, reporter.gene_symbol
    ont, resolved = resolve_ontology(obs, composite, ontmap)
    kind = classify_ontology(ont)
    notes: list[str] = []
    if _PHOSPHO.search(obs.split("/")[-1]):
        notes.append(
            "active/phospho-form inferred from the species name, not the "
            "annotation — confirm or add a PSI-MOD term."
        )

    def verdict(status, msg):
        return ReporterVerdict(
            gene, obs, resolved, kind, status, msg, tuple(notes)
        )

    gene_u = gene.upper()
    if (
        "go" in ont
        and ont["go"] not in _go_aspect()
        and kind is (ObservableKind.UNKNOWN)
    ):
        return verdict(
            "reference-stale",
            f"{gene} ← {resolved}: GO term {ont['go']} is not in the local GO "
            "table — re-run scripts/build_reference_tables.py.",
        )

    if kind in (ObservableKind.METABOLITE, ObservableKind.PHYSICAL):
        what = {
            ObservableKind.METABOLITE: "a metabolite",
            ObservableKind.PHYSICAL: "a physical entity "
            f"({ont.get('go', '')})",
        }[kind]
        return verdict(
            "category-error",
            f"{gene} ← {resolved} is {what}; a transcript cannot track it. "
            + _branch_hint(gene),
        )

    if kind is ObservableKind.PROTEIN:
        sym, miss = _human_symbol(ont["uniprot"])
        if miss == "uniprot-missing":
            return verdict(
                "reference-stale",
                f"{gene} ← {resolved}: UniProt {ont['uniprot']} is annotated "
                "but absent from the local crosswalk — re-run "
                "scripts/build_reference_tables.py.",
            )
        if miss == "ortholog-missing":
            notes.append(
                f"no mouse→human ortholog for {sym}; matched by uppercase — "
                "verify."
            )
        if sym and sym in tf_set:
            if sym == gene_u:
                return verdict(
                    "self-map",
                    f"{gene} ← {resolved}: transcription factor mapped to its "
                    "own gene. Autoregulation is delayed — read a delayed "
                    "TranscriptionBranch, not the bare activity.",
                )
            if (sym, gene_u) in edges:
                want = reporter.sign
                got = edges[(sym, gene_u)]
                if want and want != got:
                    return verdict(
                        "sign-conflict",
                        f"{gene} ← {resolved}: {sym}→{gene} is "
                        f"{'activation' if got > 0 else 'repression'} in "
                        f"CollecTRI (sign {got:+d}), but the reporter asserts "
                        f"{want:+d}.",
                    )
                return verdict(
                    "ok",
                    f"{gene} ← {resolved}: {sym} (TF) → {gene} is a signed "
                    f"CollecTRI target ({got:+d}).",
                )
            return verdict(
                "tf-target-absent",
                f"{gene} ← {resolved}: {sym} is a TF but {gene} is not one of "
                "its CollecTRI targets — mapping unsupported by the regulon.",
            )
        if sym == gene_u:
            return verdict(
                "proxy",
                f"{gene} ← {resolved}: reads the {gene} protein as its own "
                "transcript. Faithful only where the gene is transcriptionally "
                "controlled; a TranscriptionBranch makes it explicit.",
            )
        return verdict(
            "category-error",
            f"{gene} ← {resolved}: a non-TF protein ({sym or ont['uniprot']}) "
            f"mapped to an unrelated transcript {gene}. " + _branch_hint(gene),
        )

    if kind is ObservableKind.PROCESS:
        return verdict(
            "category-error",
            f"{gene} ← {resolved} is a GO process/activity ({ont.get('go')}); "
            "not transcriptionally coupled. " + _branch_hint(gene),
        )

    return verdict(
        "unannotated",
        f"{gene} ← {resolved}: no MIRIAM annotation on the node, so its "
        "molecular kind can't be verified. Annotate the species to enable "
        "the check.",
    )


def validate_reporter_mappings(reporters, composite):
    """Report reporters a transcript cannot faithfully track.

    Returns a :class:`~hallsim.validation.ValidationReport`; every non-``ok``
    verdict is a WARNING. Category errors, self-maps, sign conflicts and
    unannotated nodes each surface with the objective evidence and a
    TranscriptionBranch recommendation where one applies.
    """
    from hallsim.validation import (
        Severity,
        ValidationReport,
        ValidationResult,
    )

    ontmap = store_ontology_map(composite)
    results = []
    for rep in reporters:
        v = classify_reporter(rep, composite, ontmap)
        msg = v.message + (" [" + "; ".join(v.notes) + "]" if v.notes else "")
        results.append(
            ValidationResult(
                Severity.WARNING if v.is_warning else Severity.INFO,
                f"reporter-{v.status}",
                msg,
            )
        )
    return ValidationReport(results=results)


def recommend_reporters(composite, measured_genes) -> list[dict]:
    """Candidate reporters: TF nodes in the composite × their CollecTRI
    targets that are actually measured, sign from the regulon.

    Each candidate is ``{observable, tf, gene, sign, uniprot}`` — the
    objective form of the reporter table, for the user to accept or refine.
    """
    tf_set, edges, _ = _collectri()
    measured = {g.upper() for g in measured_genes}
    ontmap = store_ontology_map(composite)
    seen, out = set(), []
    for path, ont in ontmap.items():
        uni = ont.get("uniprot")
        sym = _human_symbol(uni)[0] if uni else None
        if not sym or sym not in tf_set:
            continue
        for (tf, gene), w in edges.items():
            if tf == sym and gene in measured and (path, gene) not in seen:
                seen.add((path, gene))
                out.append(
                    {
                        "observable": path,
                        "tf": sym,
                        "gene": gene,
                        "sign": w,
                        "uniprot": uni,
                    }
                )
    return sorted(out, key=lambda d: (d["observable"], d["gene"]))
