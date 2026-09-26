"""The data census: every time course the repositories deposit, screened
for what a screened model could be scored on. The data-side mirror of
:mod:`hallsim.census`, and the supply count for a composition benchmark.

Five routes enumerate: ``papers`` (the model census's PubMed ids through
Europe PMC: what each model's own paper deposited), ``geo`` (every series
for the organisms), ``ebi`` (every entry of PRIDE, MetaboLights,
Metabolomics Workbench and ArrayExpress through EBI Search), ``osdr``
(NASA's Open Science Data Repository, where a study's ground-against-flight
contrast is a declared factor) and ``bioimages``. Every row then meets the
same gates, cheapest first and nested:

1. ``contrast``: two sample groups, so a fold change exists — two
   timepoints in an arm, or a perturbed arm beside a control. Three or
   more timepoints is recorded as ``dynamics`` and constrains a rate, but
   is not required: every reader returns a contrast, and gating on it
   discarded nine usable deposits for every one kept;
2. ``measured``: a modality that measures molecules a model integrates —
   every one of them, since a reporter reads a single species and a panel
   is therefore as nameable as a whole proteome;
3. ``matched``: some screened model carries a quantity it measures, by
   ontology, over routes that assume progressively more: a shared
   identifier, a factor's DNA occupancy, a proteome's coverage, a
   transcriptome's regulon, or an unlisted panel resolved on reading;
4. ``loadable``: a reader exists for the deposit's tables.

Arms, a named control and the perturbation labels are recorded, never
gated: an unperturbed time course is data. ``run_census`` streams one row
per dataset to ``rows.jsonl`` and resumes; ``write_report`` writes the
funnel per route and modality, the pair list and the paper census.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from hallsim.datasets import (
    EBI_DOMAINS,
    DatasetCandidate,
    Design,
    Measured,
    geo_by_accession,
    iter_bioimages,
    iter_ebi,
    iter_geo,
    iter_osdr,
    osdr_files,
    paper_data,
    TIME_FACTOR,
    parse_design,
    resolve_perturbation,
    time_values,
)

log = logging.getLogger(__name__)

GATES = ("contrast", "measured", "matched", "loadable")
ROUTES = ("papers", "geo", "ebi", "osdr", "bioimages")
DEFAULT_ORGANISMS = ("Homo sapiens", "Mus musculus")
#: Modalities whose quantities can be named against a model. A modality
#: earns a place here by measuring something a model integrates, not by
#: measuring all of it: a reporter reads one species, so a panel of a few
#: hundred proteins is as nameable as a whole proteome, and the framework
#: already scores single-gene reporters that way.
MEASURABLE = frozenset(
    {
        "expression",
        "proteomics",
        "metabolomics",
        "methylation",
        "binding",
        "ncrna",
        "imaging",
    }
)
#: What the framework reads today, by source and modality.
LOADERS = {
    ("geo", "expression", True): "series-matrix",
    ("geo", "expression", False): "counts-file",
    ("metabolights", "metabolomics", False): "maf",
    ("metabolomics_workbench", "metabolomics", False): "mwtab",
}
#: Routes a reader exists for. A counts file still has to be a
#: well-formed table; NCBI's reprocessed series always are, an
#: author's upload may not be. Sources whose loader the file list decides
#: are absent from :data:`LOADERS`; see :data:`FILE_DECIDED`.
READABLE = frozenset(
    {
        "series-matrix",
        "counts-file",
        "maf",
        "mwtab",
        "mztab",
        "glbulkrnaseq",
    }
)
_COUNTS_FILE = re.compile(
    r"count|tpm|fpkm|rpkm|matrix|expression|abundance|normali[sz]ed", re.I
)
SALVAGE_USABLE = frozenset({"as-is", "cheap-fix"})
#: The measurable modalities whose quantities are molecules, so an unlisted
#: panel can still be resolved by reading its table. Imaging is measurable
#: and not molecular: it clears ``measured`` and matches nothing.
MOLECULAR = MEASURABLE - {"imaging"}


# ── The model side ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelIds:
    """One screened deposit's annotated species, split by what a dataset
    could measure them as."""

    accession: str
    name: str
    uniprot: frozenset
    chebi: frozenset
    tfs: frozenset
    #: The organism the deposit declares, which may be a clade.
    taxon: str = ""


def _human_tfs(uniprots) -> frozenset:
    from hallsim.reporter_wiring import _collectri, _human_symbol

    tf_set, _, _ = _collectri()
    out = set()
    for acc in uniprots:
        sym = _human_symbol(acc)[0]
        if sym and sym in tf_set:
            out.add(acc)
    return frozenset(out)


def load_models(run_dir, usable_only: bool = True) -> list[ModelIds]:
    """The model census's deposits with their species ids; ``usable_only``
    keeps those that clear every gate or need only a cheap fix."""
    from hallsim.census import load_rows

    df = load_rows(run_dir)
    if "species_ids" not in df:
        raise ValueError(
            f"{run_dir} carries no species ids; regenerate its report first"
        )
    if usable_only:
        df = df[(df["stage"] == "pass") | df["salvage"].isin(SALVAGE_USABLE)]
    out = []
    for r in df.itertuples():
        ids = r.species_ids if isinstance(r.species_ids, list) else []
        uni = frozenset(
            i.split(":", 1)[1] for i in ids if i.startswith("uniprot:")
        )
        che = frozenset(i for i in ids if i.startswith("chebi:"))
        if not uni and not che:
            continue
        out.append(
            ModelIds(
                r.accession,
                getattr(r, "name", "") or "",
                uni,
                che,
                _human_tfs(uni),
                str(getattr(r, "taxon", "") or "").strip(),
            )
        )
    return out


# ── One dataset through the gates ───────────────────────────────────


def timed_evidence(cand: DatasetCandidate, design: Design) -> str:
    """Where a *time course* shows: ``titles`` (three or more timepoints
    in some arm), ``factors`` (a declared time factor), ``text`` (three or
    more distinct time tokens in the description), or empty.

    Three timepoints is what constrains a rate, so this is the evidence for
    fitting dynamics — not for whether the deposit is usable at all, which
    is :func:`contrast_of`.
    """
    if design.time_course:
        return "titles"
    if any(TIME_FACTOR.search(f) for f in cand.factors):
        return "factors"
    values = time_values(f"{cand.title} {cand.summary}")
    return "text" if len(values) >= 3 else ""


def contrast_of(design: Design, evidence: str = "") -> str:
    """Which contrast a deposit supports, which is what decides whether it
    can be used at all.

    Every reader returns a fold change between two named sample groups, so
    two groups is the requirement and three timepoints is a bonus:

    ``dynamics``
        three or more timepoints in some arm, read from the sample titles —
        enough to constrain a rate.
    ``declared``
        the repository asserts a time course through a study factor or its
        description, but the sample titles do not show the groups. The
        groups exist; reading them takes the deposit's own metadata files.
    ``course``
        two timepoints in some arm: a change over time, but only an
        endpoint's worth of constraint.
    ``arms``
        two or more arms, so a perturbed group divides by a control at the
        same time. This is the ordinary perturbation experiment and the
        supply most validation actually runs on.
    empty
        one group, so nothing to divide by.
    """
    if design.time_course:
        return "dynamics"
    if evidence:
        return "declared"
    if design.n_timepoints >= 2 and _replicated(design):
        return "course"
    if design.perturbed and _replicated(design):
        return "arms"
    return ""


def _replicated(design: Design) -> bool:
    """Whether the groups could hold replicates: at least two *subjects*
    per arm-and-timepoint cell on average, falling back to samples where
    the deposit names no subject.

    An arm with a single subject cannot be contrasted against anything, and
    a label that gives nearly one arm per sample is a parsed token rather
    than a condition — a plate well, an animal id, or a clinical covariate
    crossed with nine others. Frequency pruning does not catch that case,
    because each token recurs; the count does.

    Subjects rather than samples, because two samples from one organism are
    one observation. A study that assays three brain regions from ten mice
    deposits thirty samples and holds ten independent units, and scoring it
    as thirty is how an effect appears that is not there.
    """
    units = design.n_subjects or design.n_titles
    groups = sum(max(1, len(ts)) for _, ts in design.per_arm) or len(
        design.arms
    )
    return groups > 0 and groups * 2 <= units


def perturbation_labels(cand: DatasetCandidate, design: Design) -> list[str]:
    """The conditions a dataset varied, as labels: the arms other than the
    control, and factor levels declared as ``Treatment:X``."""
    labels = [a for a in design.arms if a and a != design.control]
    for f in cand.factors:
        for part in f.split("|"):
            key, _, value = part.partition(":")
            if value and key.strip().lower() in (
                "treatment",
                "compound",
                "drug",
                "perturbation",
                "genotype",
            ):
                labels.append(value.strip())
    return list(dict.fromkeys(labels))


def nameable(m: Measured) -> bool:
    """Whether the measurement's quantities can be named against a model.

    Completeness buys a *route* — a whole transcriptome reaches any
    TF-annotated model through its regulon — but it is not what makes a
    measurement nameable. A panel that assays a few hundred proteins names
    those proteins, and one of them matching a model species is a closer
    reading than a transcript of a target gene. So every molecular modality
    qualifies; which models it lands on is :func:`match_models`'s question.
    """
    return m.modality in MEASURABLE


def loader_of(cand: DatasetCandidate) -> str:
    """Which reader a deposit's tables need, or why none applies.

    A loader cannot be decided from a modality alone. ArrayExpress deposits
    are labelled processed but mostly ship per-sample raw arrays and
    sequencing files — ``.cel``, ``.idat``, ``.gpr``, ``.bam`` — and a third
    ship no data at all, their raw reads living in ENA; PRIDE deposits are
    labelled with results but only a minority carry a quantification file.
    Both are decided by the file list, the way GEO already is.

    ``unchecked`` is returned where the file list decides and this row does
    not carry one: enumeration metadata does not include it, so it takes a
    per-accession request. That verdict keeps such a row out of the readable
    count instead of asserting a reader that may not apply.
    """
    m = cand.measured
    if cand.source == "geo" and m.modality == "expression":
        if cand.series_matrix_has_values:
            return "series-matrix"
        if any(_COUNTS_FILE.search(f) for f in cand.files):
            return "counts-file"
        return "none"
    if cand.source in FILE_DECIDED:
        if not cand.files:
            return "unchecked"
        return FILE_DECIDED[cand.source](cand.files)
    return LOADERS.get((cand.source, m.modality, m.complete), "none")


def _arrayexpress_loader(files) -> str:
    if any(_COUNTS_FILE.search(f) for f in files):
        return "counts-file"
    return "none"


def _osdr_loader(files) -> str:
    from hallsim.datasets import GL_COUNTS

    if any(GL_COUNTS.search(f) for f in files):
        return "glbulkrnaseq"
    return "none"


def _pride_loader(files) -> str:
    if any(f.lower().endswith(".mztab") for f in files):
        return "mztab"
    return "none"


#: Sources whose loader is decided by the file list rather than by the
#: modality, with the rule that decides it.
FILE_DECIDED = {
    "biostudies-arrayexpress": _arrayexpress_loader,
    "pride": _pride_loader,
    "osdr": _osdr_loader,
}


_MIRRORED = re.compile(r"^E-GEOD-(\d+)$", re.I)


def original_of(
    source: str, accession: str, mirrors: str = ""
) -> tuple[str, str]:
    """``(source, accession)`` of the deposit a row really describes.

    Two repositories re-host other people's studies. ArrayExpress mirrors
    about half of its holdings from GEO and encodes the origin in the
    accession, so ``E-GEOD-12345`` is ``GSE12345`` listed twice; OSDR
    re-hosts from both and states the original outright in ``mirrors``.
    Returning the original collapses the copies without a title comparison,
    and resolves a chain — an OSDR row mirroring ``E-GEOD-12345`` lands on
    the GEO series, not on the ArrayExpress copy of it.
    """
    for candidate in (mirrors.strip(), accession):
        if m := _MIRRORED.match(candidate):
            return "geo", f"GSE{m.group(1)}"
    declared = mirrors.strip().upper()
    if declared.startswith("GSE"):
        return "geo", declared
    if declared.startswith("E-MTAB-"):
        return "biostudies-arrayexpress", declared
    return source, accession


def shared_subjects(designs: dict[str, tuple]) -> list[tuple[str, ...]]:
    """Groups of deposits that assayed the same subjects.

    A repository lists one deposit per tissue, so a single experiment can
    appear as several accessions over one set of organisms. Treating those
    as independent evidence multiplies the apparent sample size without
    adding information. ``designs`` maps an accession to its subject
    identifiers; the result is each group of two or more accessions that
    overlap, largest first.
    """
    names = list(designs)
    seen: set[str] = set()
    groups = []
    for i, a in enumerate(names):
        if a in seen or not designs[a]:
            continue
        group = {a}
        for b in names[i + 1 :]:
            if designs[b] and set(designs[a]) & set(designs[b]):
                group.add(b)
        if len(group) > 1:
            seen |= group
            groups.append(tuple(sorted(group)))
    return sorted(groups, key=len, reverse=True)


#: Taxon strings that name a clade rather than an organism, so a model
#: carrying one cannot be said to agree or disagree with a dataset's.
CLADES = frozenset(
    {
        "mammalia",
        "eukaryota",
        "cellular organisms",
        "vertebrata",
        "metazoa",
        "chordata",
        "",
    }
)


def species_of(organism: str, taxon: str) -> str:
    """How a dataset's organism stands to a model's declared taxon.

    ``same``, ``ortholog`` when they differ, or ``unknown`` when either
    side names a clade or nothing. A cross-species match is not wrong —
    it is how a human-annotated model is scored on mouse data — but it
    carries an ortholog step that the identifiers hide, so it is named
    rather than folded into the others. :func:`match_models` reports this
    for the best-scoring deposit and counts how many of the matched set
    share the dataset's organism.
    """
    a = (organism or "").strip().lower()
    b = (taxon or "").strip().lower()
    if not a or b in CLADES:
        return "unknown"
    return "same" if b in a else "ortholog"


def match_models(
    m: Measured, models: list[ModelIds], organism: str = ""
) -> dict:
    """Which models a measurement lands on, and how.

    ``via`` names the route, ordered by how much it assumes:

    ``direct``
        the deposit lists identifiers and some are a model's species, so
        one accession is the other.
    ``occupancy``
        a DNA-binding or accessibility assay reads a transcription
        factor's engagement with DNA, which is nearer that factor's
        activity than any transcript of its targets.
    ``complete``
        a whole proteome covers a model's proteins without naming them.
    ``regulon``
        a transcriptome reaches a factor's activity through its targets,
        the longest inference of the four.
    ``panel``
        a molecular assay whose quantities are not listed in the metadata;
        which model it lands on is decided by reading the file.
    """
    scored: list[tuple[int, str, tuple]] = []
    via = "none"
    if m.ids:
        ids = {i.lower() for i in m.ids}
        for mod in models:
            carried = mod.chebi | {f"uniprot:{u}".lower() for u in mod.uniprot}
            shared = tuple(sorted(carried & ids))
            if shared:
                scored.append((len(shared), mod.accession, shared))
        via = "direct" if scored else "none"
    elif m.modality == "binding":
        scored = [
            (len(mod.tfs), mod.accession, ()) for mod in models if mod.tfs
        ]
        via = "occupancy" if scored else "none"
    elif m.modality == "proteomics" and m.complete:
        scored = [
            (len(mod.uniprot), mod.accession, ())
            for mod in models
            if mod.uniprot
        ]
        via = "complete" if scored else "none"
    elif m.modality == "expression" and m.complete:
        scored = [
            (len(mod.tfs), mod.accession, ()) for mod in models if mod.tfs
        ]
        via = "regulon" if scored else "none"
    elif m.modality in MOLECULAR:
        scored = [
            (len(mod.uniprot | mod.chebi), mod.accession, ())
            for mod in models
            if mod.uniprot or mod.chebi
        ]
        via = "panel" if scored else "none"
    scored.sort(key=lambda s: (-s[0], s[1]))
    taxa = {mod.accession: mod.taxon for mod in models}
    relations = [
        species_of(organism, taxa.get(acc, "")) for _, acc, _ in scored
    ]
    # A metabolite is the same molecule in every organism, so a ChEBI
    # identity carries no species step; every other route does. Elsewhere
    # the count is what informs: a row matching six hundred deposits of
    # which nine share its organism has not been matched within species,
    # and reporting the best case would say that it had.
    chebi_only = via == "direct" and all(
        i.startswith("chebi:") for _, _, sh in scored for i in sh
    )
    return {
        "species": (
            "n/a" if chebi_only else (relations[0] if relations else "")
        ),
        "n_same_species": 0 if chebi_only else relations.count("same"),
        "via": via,
        "n_models": len(scored),
        "top_models": [
            {"model": acc, "n_shared": n, "shared": list(sh)}
            for n, acc, sh in scored[:5]
        ],
        "direct_pairs": (
            [
                {"model": acc, "n_shared": n, "shared": list(sh)}
                for n, acc, sh in scored
            ]
            if via == "direct"
            else []
        ),
    }


def screen_dataset(
    cand: DatasetCandidate,
    models: list[ModelIds],
    *,
    route: str,
    resolve: bool = False,
) -> dict:
    """One row: the candidate flattened, its design, the gate booleans,
    the stage it stops at and the models it lands on."""
    t0 = time.time()
    design = parse_design(cand.samples)
    evidence = timed_evidence(cand, design)
    labels = perturbation_labels(cand, design)
    m = cand.measured
    row = {
        "route": route,
        "source": cand.source,
        "accession": cand.accession,
        "mirrors": cand.mirrors,
        "title": cand.title[:200],
        "organism": cand.organism,
        "modality": m.modality,
        "complete": m.complete,
        "n_ids": len(m.ids),
        "pubmed": cand.pubmed,
        "n_samples": cand.n_samples,
        "n_arms": len(design.arms),
        "control": design.control or "",
        "n_timepoints": design.n_timepoints,
        "n_subjects": design.n_subjects,
        "time_unit": design.time_unit,
        "perturbed": design.perturbed,
        "perturbations": labels[:12],
        "timed_evidence": evidence,
        "contrast_kind": contrast_of(design, evidence),
        "loader": loader_of(cand),
        # What the gates were judged on, so a row can be screened again
        # without asking the repository twice.
        "raw": {
            "kind": cand.kind,
            # Untruncated: the text route reads it, so a capped copy makes
            # `rescreen` disagree with the run that wrote the row. A third
            # of descriptions were losing their tail at 600 characters.
            "summary": cand.summary,
            "samples": list(cand.samples),
            "files": list(cand.files),
            "factors": list(cand.factors),
            "ids": list(m.ids),
            "url": cand.url,
            "platform": cand.platform,
        },
    }
    # Dynamics is recorded, not gated: a deposit that only supports an
    # endpoint contrast is still data, and discarding it threw away nine
    # rows for every one kept.
    row["dynamics"] = bool(evidence)
    row["timed"] = bool(evidence)
    row["contrast"] = bool(row["contrast_kind"])
    row["measured"] = row["contrast"] and nameable(m)
    match = (
        match_models(m, models, cand.organism)
        if row["measured"]
        else {
            "species": "",
            "n_same_species": 0,
            "via": "none",
            "n_models": 0,
            "top_models": [],
            "direct_pairs": [],
        }
    )
    row.update(match)
    row["matched"] = row["measured"] and match["n_models"] > 0
    row["loadable"] = row["matched"] and row["loader"] in READABLE
    row["stage"] = next((g for g in GATES if not row[g]), "pass")
    if resolve and labels:
        resolved = []
        for label in labels[:6]:
            p = resolve_perturbation(label)
            resolved.append(
                {
                    "label": label,
                    "kind": p.kind,
                    "name": p.name,
                    "chebi": p.chebi,
                    "targets": list(p.targets),
                }
            )
        row["resolved"] = resolved
    row["seconds"] = round(time.time() - t0, 3)
    return row


def candidate_of(row: dict) -> DatasetCandidate:
    """The candidate a row was screened from, rebuilt from its raw part."""
    raw = row.get("raw") or {}
    return DatasetCandidate(
        source=row["source"],
        accession=row["accession"],
        title=row.get("title", ""),
        kind=raw.get("kind", ""),
        organism=row.get("organism", ""),
        n_samples=int(row.get("n_samples") or 0),
        platform=raw.get("platform", ""),
        url=raw.get("url", ""),
        summary=raw.get("summary", ""),
        samples=tuple(raw.get("samples", ())),
        files=tuple(raw.get("files", ())),
        measured=Measured(
            row.get("modality", "unknown"),
            bool(row.get("complete")),
            tuple(raw.get("ids", ())),
        ),
        pubmed=row.get("pubmed", ""),
        factors=tuple(raw.get("factors", ())),
        mirrors=str(row.get("mirrors") or ""),
    )


def rescreen(run_dir, models: list[ModelIds], *, resolve: bool = False) -> int:
    """Screen every stored row again from its raw part, under the current
    gates and models, rewriting ``rows.jsonl`` in place. Returns the count."""
    run = Path(run_dir)
    path = run / "rows.jsonl"
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        if not old.get("raw"):
            out.append(old)
            continue
        new = screen_dataset(
            candidate_of(old), models, route=old["route"], resolve=resolve
        )
        if "resolved" in old and "resolved" not in new:
            new["resolved"] = old["resolved"]
        out.append(new)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in out))
    tmp.replace(path)
    return len(out)


def reason_of(row: dict) -> str:
    """Why a row stopped where it did, in the failure list's words."""
    stage = row["stage"]
    if stage == "pass":
        return ""
    if stage == "contrast":
        if row["n_samples"]:
            return "one sample group, so nothing to divide by"
        return "no sample groups declared"
    if stage == "measured":
        return f"{row['modality']}: quantities not nameable against a model"
    if stage == "matched":
        return (
            f"no screened model carries what it measures ({row['modality']})"
        )
    if row["loader"] == "unchecked":
        return "loader undecided: the deposit's file list is not fetched"
    return f"loader work: {row['loader']}"


# ── Enumeration and the run ─────────────────────────────────────────


def _done(rows_path: Path) -> set[tuple[str, str]]:
    seen = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add((r["source"], r["accession"]))
    return seen


class _Sink:
    """Append-only rows with a progress line, resumable by key."""

    def __init__(self, run: Path):
        self.rows = run / "rows.jsonl"
        self.progress = run / "progress.log"
        self.seen = _done(self.rows)
        self.n = 0
        self.t0 = time.time()
        # Routes may write from several threads; one line at a time.
        self._lock = threading.Lock()

    def has(self, cand: DatasetCandidate) -> bool:
        return (cand.source, cand.accession) in self.seen

    def write(self, row: dict) -> None:
        line = json.dumps(row) + "\n"
        with self._lock:
            with self.rows.open("a") as fh:
                fh.write(line)
            self.seen.add((row["source"], row["accession"]))
            self.n += 1
            n = self.n
        if n % 200 == 0:
            self.note(f"{n} rows in {time.time() - self.t0:.0f} s")

    def note(self, text: str) -> None:
        log.info(text)
        with self._lock:
            with self.progress.open("a") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {text}\n")


def _route_marker(run: Path, route: str) -> Path:
    return run / f"route.{route}.done"


def run_papers(
    run: Path,
    models_run,
    models: list[ModelIds],
    sink: _Sink,
    *,
    workers: int = 4,
    limit: int | None = None,
    with_files: bool = False,
    resolve: bool = False,
) -> None:
    """Every model paper through Europe PMC: flags and links to
    ``papers.jsonl``; the datasets it deposits or cites screened as rows."""
    from hallsim.census import load_rows

    df = load_rows(models_run)
    pubmeds: dict[str, list[str]] = {}
    for r in df.itertuples():
        pm = str(getattr(r, "pubmed", "") or "").split(".")[0]
        if pm and pm != "nan":
            pubmeds.setdefault(pm, []).append(r.accession)
    papers_path = run / "papers.jsonl"
    done = set()
    if papers_path.exists():
        for line in papers_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["pubmed"])
    todo = [p for p in pubmeds if p not in done]
    if limit:
        todo = todo[:limit]
    sink.note(f"papers: {len(todo)} to read ({len(done)} done)")

    def one(pm):
        try:
            return pm, paper_data(pm, with_files=with_files)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one paper never stops the run
            return pm, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, pm) for pm in todo]
        for fut in as_completed(futures):
            pm, paper = fut.result()
            if isinstance(paper, Exception):
                record = {"pubmed": pm, "error": str(paper)[:200]}
            else:
                record = {
                    "pubmed": pm,
                    "models": pubmeds[pm],
                    "pmcid": paper.pmcid,
                    "has_data": paper.has_data,
                    "has_supplement": paper.has_supplement,
                    "datasets": [
                        {
                            "source": d.source,
                            "accession": d.accession,
                            "files": list(d.files),
                        }
                        for d in paper.datasets
                    ],
                    "chemicals": [list(c) for c in paper.chemicals],
                    "biomodels": list(paper.biomodels),
                }
                for d in paper.datasets:
                    if sink.has(d):
                        continue
                    cand = d
                    if d.source == "geo":
                        try:
                            found = geo_by_accession(d.accession)
                            cand = found or d
                        except Exception as exc:  # noqa: BLE001
                            log.info("geo %s: %s", d.accession, exc)
                    sink.write(
                        screen_dataset(
                            cand, models, route="papers", resolve=resolve
                        )
                    )
            with papers_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            time.sleep(0.2)
    _route_marker(run, "papers").touch()


def run_geo(
    run: Path,
    models: list[ModelIds],
    sink: _Sink,
    *,
    organisms=DEFAULT_ORGANISMS,
    limit: int | None = None,
    resolve: bool = False,
) -> None:
    # The offset of the next page is kept so an interrupted route resumes
    # a page before where it stopped rather than from the top.
    offset_path = run / "route.geo.offset"
    start = 0
    if offset_path.exists() and not limit:
        start = max(int(offset_path.read_text() or 0) - 300, 0)
        sink.note(f"geo: resuming at {start}")
    n = 0
    for cand in iter_geo(
        organisms,
        start=start,
        on_page=lambda off: offset_path.write_text(str(off)),
    ):
        if not sink.has(cand):
            sink.write(
                screen_dataset(cand, models, route="geo", resolve=resolve)
            )
        n += 1
        if limit and n >= limit:
            return
    _route_marker(run, "geo").touch()


def run_ebi(
    run: Path,
    models: list[ModelIds],
    sink: _Sink,
    *,
    domains=(
        "pride",
        "metabolights",
        "metabolomics-workbench",
        "arrayexpress",
    ),
    limit: int | None = None,
    resolve: bool = False,
) -> None:
    """The EBI domains, each on its own thread: they are independent
    streams, and one page's latency is most of a domain's cost."""

    def one(name: str) -> None:
        marker = _route_marker(run, f"ebi.{name}")
        if marker.exists():
            return
        dom = EBI_DOMAINS[name]
        offset_path = run / f"route.ebi.{name}.offset"
        start = 0
        if offset_path.exists() and not limit:
            start = max(int(offset_path.read_text() or 0) - 100, 0)
            sink.note(f"ebi {name}: resuming at {start}")
        n = 0
        for cand in iter_ebi(
            dom,
            start=start,
            on_page=lambda off: offset_path.write_text(str(off)),
        ):
            if name == "metabolights" and not cand.accession.startswith(
                "MTBLS"
            ):
                continue
            if not sink.has(cand):
                sink.write(
                    screen_dataset(cand, models, route="ebi", resolve=resolve)
                )
            n += 1
            if limit and n >= limit:
                break
        if not limit:
            marker.touch()
        sink.note(f"ebi {name}: {n} entries")

    with ThreadPoolExecutor(max_workers=len(domains)) as pool:
        for future in [pool.submit(one, name) for name in domains]:
            future.result()
    if not limit:
        _route_marker(run, "ebi").touch()


def run_census(
    models_run,
    *,
    run_dir=None,
    routes=ROUTES,
    organisms=DEFAULT_ORGANISMS,
    limit: int | None = None,
    usable_only: bool = True,
    resolve: bool = False,
    with_files: bool = False,
    workers: int = 4,
) -> Path:
    """Enumerate the routes against the models of ``models_run`` (a model
    census run directory) and stream one row per dataset. ``limit`` caps
    each route for a pilot and leaves no completion marker. Returns the
    run directory."""
    from hallsim.io import make_run_dir, outdir

    run = Path(run_dir) if run_dir else make_run_dir("census-data")
    run.mkdir(parents=True, exist_ok=True)
    latest = run.parent / "latest"
    if run.parent == outdir("census-data") and run.name != "latest":
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run.name)
    models = load_models(models_run, usable_only=usable_only)
    (run / "config.json").write_text(
        json.dumps(
            {
                "models_run": str(models_run),
                "n_models": len(models),
                "usable_only": usable_only,
                "routes": list(routes),
                "organisms": list(organisms),
                "limit": limit,
                "resolve": resolve,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    sink = _Sink(run)
    sink.note(f"{len(models)} models with species ids; routes {list(routes)}")
    for route in routes:
        if _route_marker(run, route).exists():
            sink.note(f"{route}: done earlier")
            continue
        if route == "papers":
            run_papers(
                run,
                models_run,
                models,
                sink,
                workers=workers,
                limit=limit,
                with_files=with_files,
                resolve=resolve,
            )
        elif route == "geo":
            run_geo(
                run,
                models,
                sink,
                organisms=organisms,
                limit=limit,
                resolve=resolve,
            )
        elif route == "ebi":
            run_ebi(run, models, sink, limit=limit, resolve=resolve)
        elif route == "osdr":
            n = 0
            for cand in iter_osdr():
                if with_files and not cand.files:
                    try:
                        cand = replace(cand, files=osdr_files(cand.accession))
                    except Exception as exc:  # noqa: BLE001
                        sink.note(f"{cand.accession}: no file list ({exc})")
                if not sink.has(cand):
                    sink.write(
                        screen_dataset(
                            cand, models, route=route, resolve=resolve
                        )
                    )
                n += 1
                if limit and n >= limit:
                    break
            if not limit:
                _route_marker(run, route).touch()
        elif route == "bioimages":
            n = 0
            for cand in iter_bioimages():
                if not sink.has(cand):
                    sink.write(
                        screen_dataset(
                            cand, models, route=route, resolve=resolve
                        )
                    )
                n += 1
                if limit and n >= limit:
                    break
            if not limit:
                _route_marker(run, route).touch()
        else:
            raise ValueError(f"unknown route {route!r}; have {ROUTES}")
        sink.note(f"{route}: finished, {sink.n} rows written this pass")
    return run


# ── The report ──────────────────────────────────────────────────────


def load_rows(run_dir):
    import pandas as pd

    rows = []
    for line in (Path(run_dir) / "rows.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # An ArrayExpress mirror and its GEO original are one deposit, and the
    # GEO row is the one a reader can open.
    origin = []
    for r in df.itertuples(index=False):
        mirrors = getattr(r, "mirrors", "")
        origin.append(
            original_of(
                r.source,
                r.accession,
                mirrors if isinstance(mirrors, str) else "",
            )
        )
    df["_origin"] = [f"{s}/{a}" for s, a in origin]
    df["_mirror"] = [s != r for (s, _), r in zip(origin, df["source"])]
    df = (
        df.sort_values("_mirror", ascending=False, kind="stable")
        .drop_duplicates("_origin", keep="last")
        .drop(columns=["_origin", "_mirror"])
        .sort_index()
    )
    df["reason"] = [reason_of(r) for r in df.to_dict("records")]
    return df.reset_index(drop=True)


def funnel(df) -> list[tuple[str, int]]:
    """The gate counts, listed first. Rows written under an earlier set of
    gates lack the columns and are refused rather than misread: rescreen
    them and the report follows."""
    missing = [g for g in GATES if g not in df]
    if missing:
        raise ValueError(
            f"rows carry no {', '.join(missing)} gate; they were screened "
            "under earlier rules, so rescreen the run before reporting it"
        )
    out = [("listed", len(df))]
    for g in GATES:
        out.append((g, int(df[g].sum()) if len(df) else 0))
    return out


def _by(df, column) -> dict[str, list[tuple[str, int]]]:
    return {str(k): funnel(g) for k, g in df.groupby(column)}


def _papers_summary(run: Path) -> dict:
    path = run / "papers.jsonl"
    if not path.exists():
        return {}
    papers = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    ok = [p for p in papers if "error" not in p]
    kinds = Counter(d["source"] for p in ok for d in p["datasets"])
    suffixes = Counter(
        Path(f).suffix.lower()
        for p in ok
        for d in p["datasets"]
        for f in d.get("files", [])
    )
    return {
        "n_papers": len(papers),
        "n_read": len(ok),
        # Europe PMC's data flag is on for every model paper, since the
        # BioModels deposit itself is a data link; it is not reported.
        "has_data": sum(p["has_data"] for p in ok),
        "with_pmcid": sum(bool(p.get("pmcid")) for p in ok),
        "has_supplement": sum(p["has_supplement"] for p in ok),
        "with_dataset_link": sum(1 for p in ok if p["datasets"]),
        "with_non_supplement_link": sum(
            1
            for p in ok
            if any(d["source"] != "supplement" for d in p["datasets"])
        ),
        "links_by_source": dict(kinds.most_common()),
        "supplement_suffixes": dict(suffixes.most_common(12)),
        "with_chemicals": sum(1 for p in ok if p["chemicals"]),
    }


def summarize(df, run: Path) -> dict:
    s = {
        "n_rows": int(len(df)),
        "funnel": funnel(df),
        "by_route": _by(df, "route"),
        "by_modality": _by(df, "modality"),
        "timed_evidence": df.loc[df["timed"], "timed_evidence"]
        .value_counts()
        .to_dict(),
        "arms": {
            "one": int((df["n_arms"] == 1).sum()),
            "two": int((df["n_arms"] == 2).sum()),
            "three_or_more": int((df["n_arms"] >= 3).sum()),
            "with_named_control": int((df["control"] != "").sum()),
        },
        "timed_and_perturbed": int((df["timed"] & df["perturbed"]).sum()),
        "timed_unperturbed": int((df["timed"] & ~df["perturbed"]).sum()),
        "contrast": df["contrast_kind"]
        .replace("", "none")
        .value_counts()
        .to_dict(),
        "via": df.loc[df["matched"], "via"].value_counts().to_dict(),
        "loaders": df.loc[df["matched"], "loader"].value_counts().to_dict(),
        "stages": df["stage"].value_counts().to_dict(),
        "perturbation_labels": Counter(
            lb.lower() for labels in df["perturbations"] for lb in labels
        ).most_common(25),
        "papers": _papers_summary(run),
    }
    return s


def _pairs(df):
    import pandas as pd

    rows = []
    for r in df[df["via"] == "direct"].itertuples():
        for p in r.direct_pairs:
            rows.append(
                {
                    "dataset": r.accession,
                    "source": r.source,
                    "title": r.title,
                    "model": p["model"],
                    "n_shared": p["n_shared"],
                    "shared": "; ".join(p["shared"]),
                    "n_timepoints": r.n_timepoints,
                    "perturbed": r.perturbed,
                    "loader": r.loader,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(
            ["n_shared", "n_timepoints"], ascending=False
        )
        if rows
        else pd.DataFrame(
            columns=[
                "dataset",
                "source",
                "title",
                "model",
                "n_shared",
                "shared",
            ]
        )
    )


def _md_funnel(name, rows) -> str:
    from hallsim.census import _md_table

    return f"**{name}**\n\n" + _md_table(["gate", "datasets"], rows)


def write_report(run_dir) -> Path:
    """``report.md``, ``summary.json``, ``funnel.png``, ``pairs.csv``,
    ``failures.csv`` and ``census-data.csv`` for a run."""
    from hallsim.census import _bar_figure, _md_table

    run = Path(run_dir)
    df = load_rows(run)
    s = summarize(df, run)
    (run / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    df.drop(columns=["direct_pairs", "top_models"], errors="ignore").to_csv(
        run / "census-data.csv", index=False
    )
    df[df["stage"] != "pass"][
        [
            "route",
            "source",
            "accession",
            "modality",
            "stage",
            "reason",
            "title",
        ]
    ].to_csv(run / "failures.csv", index=False)
    pairs = _pairs(df)
    pairs.to_csv(run / "pairs.csv", index=False)
    labels = [g for g, _ in s["funnel"]]
    series = {r: [n for _, n in f] for r, f in s["by_route"].items()}
    _bar_figure(
        labels,
        series,
        "Datasets through the gates",
        "",
        run / "funnel.png",
        xlabel="datasets",
    )

    lines = ["# Dataset census", ""]
    cfg = (
        json.loads((run / "config.json").read_text())
        if (run / "config.json").exists()
        else {}
    )
    lines.append(
        f"{s['n_rows']} datasets from {', '.join(cfg.get('routes', []))}; "
        f"{cfg.get('n_models', '?')} screened models with species ids on the other side."
    )
    lines += [
        "",
        "## Funnel",
        "",
        _md_table(["gate", "datasets"], s["funnel"]),
        "",
    ]
    for route, f in s["by_route"].items():
        lines += [_md_funnel(f"route: {route}", f), ""]
    for mod, f in sorted(s["by_modality"].items()):
        lines += [_md_funnel(f"modality: {mod}", f), ""]
    lines += [
        "## Design",
        "",
        _md_table(
            ["", "datasets"],
            [
                ("one arm", s["arms"]["one"]),
                ("two arms", s["arms"]["two"]),
                ("three or more arms", s["arms"]["three_or_more"]),
                ("a named control arm", s["arms"]["with_named_control"]),
                ("time course, perturbed", s["timed_and_perturbed"]),
                ("time course, unperturbed", s["timed_unperturbed"]),
            ],
        ),
        "",
        "Contrast each deposit supports: "
        + ", ".join(f"{k} {v}" for k, v in s["contrast"].items()),
        "",
        "Time course read from: "
        + ", ".join(f"{k} {v}" for k, v in s["timed_evidence"].items()),
        "",
        "## Matches",
        "",
        _md_table(["via", "datasets"], list(s["via"].items())),
        "",
        "Loaders among matched: "
        + ", ".join(f"{k} {v}" for k, v in s["loaders"].items()),
        "",
    ]
    if len(pairs):
        top = pairs.head(30)
        lines += [
            "### Direct pairs (a metabolite the model carries)",
            "",
            _md_table(
                ["dataset", "model", "shared", "timepoints", "perturbed"],
                [
                    (
                        r.dataset,
                        r.model,
                        r.shared[:60],
                        r.n_timepoints,
                        r.perturbed,
                    )
                    for r in top.itertuples()
                ],
            ),
            "",
        ]
    lines += [
        "## Perturbation labels",
        "",
        _md_table(["label", "datasets"], s["perturbation_labels"]),
        "",
    ]
    p = s["papers"]
    if p:
        lines += [
            "## The models' own papers",
            "",
            _md_table(
                ["", "papers"],
                [
                    ("read", f"{p['n_read']} of {p['n_papers']}"),
                    ("in PubMed Central", p["with_pmcid"]),
                    ("has a supplement", p["has_supplement"]),
                    (
                        "links a dataset or supplement bundle",
                        p["with_dataset_link"],
                    ),
                    (
                        "links a repository dataset",
                        p["with_non_supplement_link"],
                    ),
                    ("text-mined chemicals", p["with_chemicals"]),
                ],
            ),
            "",
            "Links by source: "
            + ", ".join(f"{k} {v}" for k, v in p["links_by_source"].items()),
            "",
        ]
        if p["supplement_suffixes"]:
            lines.append(
                "Supplement file types: "
                + ", ".join(
                    f"{k or 'none'} {v}"
                    for k, v in p["supplement_suffixes"].items()
                )
            )
            lines.append("")
    lines += [
        "## Files",
        "",
        "`rows.jsonl`, `census-data.csv`, `failures.csv`, `pairs.csv`, `summary.json`, `funnel.png`"
        + (", `papers.jsonl`" if p else ""),
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return run / "report.md"
