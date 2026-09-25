"""The data census: every time course the repositories deposit, screened
for what a screened model could be scored on. The data-side mirror of
:mod:`hallsim.census`, and the supply count for a composition benchmark.

Three routes enumerate: ``papers`` (the model census's PubMed ids through
Europe PMC: what each model's own paper deposited), ``geo`` (every series
for the organisms) and ``ebi`` (every entry of PRIDE, MetaboLights,
Metabolomics Workbench and ArrayExpress through EBI Search). Every row
then meets the same gates, cheapest first and nested:

1. ``timed``: three or more timepoints, read from the sample titles, a
   declared time factor, or time tokens in the description;
2. ``measured``: a modality whose quantities can be named (a
   transcriptome, a proteome, listed metabolites);
3. ``matched``: some screened model carries a quantity it measures, by
   ontology: a metabolite id in common, a protein in a proteome, a
   transcription factor a transcriptome reads through its regulon;
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
from dataclasses import dataclass
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
    paper_data,
    parse_design,
    resolve_perturbation,
    time_values,
)

log = logging.getLogger(__name__)

GATES = ("timed", "measured", "matched", "loadable")
ROUTES = ("papers", "geo", "ebi", "bioimages")
DEFAULT_ORGANISMS = ("Homo sapiens", "Mus musculus")
#: Modalities whose quantities can be named against a model.
MEASURABLE = frozenset({"expression", "proteomics", "metabolomics"})
#: What the framework reads today, by source and modality.
LOADERS = {
    ("geo", "expression", True): "series-matrix",
    ("geo", "expression", False): "counts-file",
    ("metabolights", "metabolomics", False): "maf",
    ("metabolomics_workbench", "metabolomics", False): "table",
    ("biostudies-arrayexpress", "expression", True): "processed-table",
    ("pride", "proteomics", True): "result-files",
}
#: Routes a reader exists for. A counts file still has to be a
#: well-formed table; NCBI's reprocessed series always are, an
#: author's upload may not be.
READABLE = frozenset({"series-matrix", "counts-file"})
_TIME_FACTOR = re.compile(
    r"\b(time|timepoint|time[- ]?point|time[- ]?course|hour|day|week|"
    r"duration|age)s?\b",
    re.I,
)
_COUNTS_FILE = re.compile(
    r"count|tpm|fpkm|rpkm|matrix|expression|abundance|normali[sz]ed", re.I
)
SALVAGE_USABLE = frozenset({"as-is", "cheap-fix"})


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
            )
        )
    return out


# ── One dataset through the gates ───────────────────────────────────


def timed_evidence(cand: DatasetCandidate, design: Design) -> str:
    """Where the time course shows: ``titles`` (three or more timepoints
    in some arm), ``factors`` (a declared time factor), ``text`` (three or
    more distinct time tokens in the description), or empty."""
    if design.time_course:
        return "titles"
    if any(_TIME_FACTOR.search(f) for f in cand.factors):
        return "factors"
    values = time_values(f"{cand.title} {cand.summary}")
    return "text" if len(values) >= 3 else ""


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
    """Whether the measurement's quantities can be named against a model:
    a whole transcriptome or proteome, or listed metabolite ids."""
    if m.modality in ("expression", "proteomics"):
        return m.complete
    return m.modality == "metabolomics" and bool(m.ids)


def loader_of(cand: DatasetCandidate) -> str:
    m = cand.measured
    key = (cand.source, m.modality, m.complete)
    if cand.source == "geo" and m.modality == "expression":
        if cand.series_matrix_has_values:
            return "series-matrix"
        if any(_COUNTS_FILE.search(f) for f in cand.files):
            return "counts-file"
        return "none"
    return LOADERS.get(key, "none")


def match_models(m: Measured, models: list[ModelIds]) -> dict:
    """Which models a measurement lands on: ``via`` (``direct``,
    ``complete``, ``regulon`` or ``none``), the count, the best five with
    the number of shared quantities, and the shared ids for direct hits."""
    scored: list[tuple[int, str, tuple]] = []
    via = "none"
    if m.modality == "metabolomics" and m.ids:
        ids = {i.lower() for i in m.ids}
        for mod in models:
            shared = tuple(sorted(mod.chebi & ids))
            if shared:
                scored.append((len(shared), mod.accession, shared))
        via = "direct" if scored else "none"
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
    scored.sort(key=lambda s: (-s[0], s[1]))
    return {
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
        "time_unit": design.time_unit,
        "perturbed": design.perturbed,
        "perturbations": labels[:12],
        "timed_evidence": evidence,
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
    row["timed"] = bool(evidence)
    row["measured"] = row["timed"] and nameable(m)
    match = (
        match_models(m, models)
        if row["measured"]
        else {
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
    if stage == "timed":
        if row["n_samples"] and not row["n_timepoints"]:
            return "no timepoints in the sample titles"
        if row["n_timepoints"]:
            return f"{row['n_timepoints']} timepoints, fewer than three"
        return "no time course declared"
    if stage == "measured":
        return f"{row['modality']}: quantities not nameable against a model"
    if stage == "matched":
        return (
            f"no screened model carries what it measures ({row['modality']})"
        )
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
    df = pd.DataFrame(rows).drop_duplicates(
        ["source", "accession"], keep="last"
    )
    df["reason"] = [reason_of(r) for r in df.to_dict("records")]
    return df.reset_index(drop=True)


def funnel(df) -> list[tuple[str, int]]:
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
