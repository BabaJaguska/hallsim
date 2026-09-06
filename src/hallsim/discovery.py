"""Model discovery — search public repositories for a model to import.

The first step an agent takes: turn a mechanism named in prose ("p53
oscillator", "NF-κB signalling") into concrete, fetchable candidates, each of
which :func:`hallsim.sbml_import.process_from_sbml` can turn into a Process.

:func:`search_for_model` fans out over the registered sources and returns a
flat, ranked candidate list. Add a repository by writing one search function
and registering it in :data:`SOURCES` — callers do not change.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

BIOMODELS_SEARCH = "https://www.ebi.ac.uk/biomodels/search"
USER_AGENT = "hallsim-discovery"


@dataclass(frozen=True)
class ModelCandidate:
    """One search hit, enough to decide whether to fetch it."""

    source: str
    id: str
    name: str
    format: str
    url: str
    curated: bool
    submitter: str | None = None
    #: What the file actually is, which ``format`` does not say: SBML-qual is
    #: reported as SBML by BioModels, and no ODE importer can read it.
    kind: str = "unknown"
    #: Free text the source carries — abstract, notes, concept tags. Searched
    #: client-side for repositories with no full-text endpoint.
    description: str = ""
    #: The repository's own curation verdict, verbatim (BioModels:
    #: "CURATED" / "NON_CURATED"). Empty when not fetched. ``curated`` is
    #: inferred from the accession prefix and is only a guess until this is
    #: filled — an uncurated deposit typically carries no ontology
    #: annotations and no declared time unit, which decides whether it can
    #: be composed and checked.
    curation: str = ""
    #: Title of the publication the deposit is attached to, if the record
    #: names one. An empty string on a record that has a paper means the
    #: deposit is not linked to it.
    publication: str = ""
    #: Every filename in the deposit, main and additional. A deposit's
    #: supplementary files often carry the fitting data
    #: :func:`hallsim.intake.published_fit_chi2` needs.
    files: tuple = ()

    def fetch(self) -> Path:
        """Download the main model file to the local cache; return its path."""
        if self.source != "biomodels":
            raise NotImplementedError(
                f"no fetcher for source {self.source!r}; open {self.url} and "
                f"vendor the file under demos/models/ or data/ by hand"
            )
        from hallsim.sbml_import import _download_biomodel_to_cache

        return Path(_download_biomodel_to_cache(self.id))

    def record(self) -> dict:
        """The repository's full record for this model."""
        if self.source != "biomodels":
            raise NotImplementedError(
                f"no record API for source {self.source!r}"
            )
        return biomodels_record(self.id)

    def enriched(self) -> "ModelCandidate":
        """Copy with ``curation``, ``publication`` and ``files`` read from the
        record rather than guessed. One request; search does not do this for
        every hit."""
        return _from_record(self.id, self.record(), fallback=self)

    def fetch_all(self, dest: Path | str | None = None) -> list[Path]:
        """Download every file in the deposit, not only the model, and return
        the local paths.

        A deposit's additional files are where the provenance lives: a README
        saying what was deposited, and often the fitting data that licenses
        every later claim about the model
        (:func:`hallsim.intake.published_fit_chi2`). Fetching only the SBML
        leaves that on the server.
        """
        if self.source != "biomodels":
            raise NotImplementedError(f"no fetcher for source {self.source!r}")
        return download_biomodel_files(self.id, dest=dest)

    def __str__(self) -> str:
        mark = "curated" if self.curated else "uncurated"
        return f"[{self.source}:{self.id}] {self.name} ({self.format}, {mark})"


BIOMODELS_RECORD = "https://www.ebi.ac.uk/biomodels/{model_id}"
BIOMODELS_DOWNLOAD = (
    "https://www.ebi.ac.uk/biomodels/model/download/{model_id}"
)


def biomodels_record(model_id, timeout: float = 30.0) -> dict:
    """The full BioModels record for ``model_id``.

    Carries what the search hit does not: the curation verdict, the linked
    publication, and the deposit's file list. `curationStatus` is the one that
    changes decisions — a `NON_CURATED` deposit has had no ontology
    annotations or unit declarations added, so semantic composition checks are
    blind on it and its clock is a guess.
    """
    return _get_json(
        BIOMODELS_RECORD.format(model_id=_accession(model_id)),
        {"format": "json"},
        timeout,
    )


def _accession(model_id) -> str:
    """``10`` → ``BIOMD0000000010``; a string accession passes through."""
    if isinstance(model_id, int):
        return f"BIOMD{model_id:010d}"
    return str(model_id)


def _record_filenames(record: dict) -> tuple[str, ...]:
    files = record.get("files") or {}
    names = [
        f.get("name", "")
        for group in ("main", "additional")
        for f in (files.get(group) or [])
    ]
    return tuple(n for n in names if n)


def _from_record(model_id, record: dict, fallback=None) -> "ModelCandidate":
    """Build a candidate from a full record, keeping ``fallback``'s search
    fields where the record says nothing."""
    accession = _accession(model_id)
    publication = (record.get("publication") or {}).get("title") or ""
    curation = record.get("curationStatus") or ""
    return ModelCandidate(
        source="biomodels",
        id=accession,
        name=record.get("name") or getattr(fallback, "name", ""),
        format=record.get("format") or getattr(fallback, "format", "SBML"),
        url=getattr(fallback, "url", "")
        or f"https://www.ebi.ac.uk/biomodels/{accession}",
        curated=(
            (curation.upper() == "CURATED")
            if curation
            else accession.startswith("BIOMD")
        ),
        submitter=getattr(fallback, "submitter", None),
        kind=getattr(fallback, "kind", "unknown"),
        description=getattr(fallback, "description", ""),
        curation=curation,
        publication=publication,
        files=_record_filenames(record),
    )


def download_biomodel_files(
    model_id,
    dest: "Path | str | None" = None,
    timeout: float = 60.0,
) -> list["Path"]:
    """Download every file in a BioModels deposit; return the local paths.

    ``_download_biomodel_to_cache`` fetches the model and stops there, but the
    deposit's other files are where the provenance is: a README stating what
    was deposited and under what conditions, and — for the minority of papers
    that deposit it — the fitting data that
    :func:`hallsim.intake.published_fit_chi2` scores against. Defaults to
    ``~/.cache/hallsim/biomodels/<accession>/``.

    A file that comes back empty is not written: the download endpoint 302s,
    and a client that does not follow the redirect gets a silent zero-byte
    body rather than an error.
    """
    import urllib.parse

    accession = _accession(model_id)
    record = biomodels_record(accession, timeout=timeout)
    names = _record_filenames(record)
    out_dir = (
        Path(dest)
        if dest is not None
        else Path.home() / ".cache" / "hallsim" / "biomodels" / accession
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in names:
        target = out_dir / name
        if target.exists() and target.stat().st_size:
            written.append(target)
            continue
        url = BIOMODELS_DOWNLOAD.format(model_id=accession)
        query = urllib.parse.urlencode({"filename": name})
        request = urllib.request.Request(
            f"{url}?{query}", headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except Exception as exc:
            log.warning("%s: could not fetch %r (%s)", accession, name, exc)
            continue
        if not body:
            log.warning("%s: %r came back empty — skipped", accession, name)
            continue
        target.write_bytes(body)
        written.append(target)
    log.info(
        "%s: %d/%d file(s) -> %s", accession, len(written), len(names), out_dir
    )
    return written


def _get_json(url: str, params: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def search_biomodels(
    query: str,
    limit: int = 25,
    curated_only: bool = True,
    sbml_only: bool = True,
    timeout: float = 30.0,
) -> list[ModelCandidate]:
    """BioModels full-text search.

    A ``BIOMD`` accession is the manually curated branch; ``MODEL`` accessions
    are auto-generated or uncurated submissions, excluded unless
    ``curated_only`` is False. ``sbml_only`` drops the MATLAB/R/other-format
    entries the search also returns, since only SBML has an importer.
    """
    payload = _get_json(
        BIOMODELS_SEARCH,
        {"query": query, "format": "json", "numResults": limit},
        timeout,
    )
    candidates = []
    for record in payload.get("models", []):
        model_id = record.get("id", "")
        fmt = record.get("format", "")
        curated = model_id.startswith("BIOMD")
        if curated_only and not curated:
            continue
        if sbml_only and fmt.upper() != "SBML":
            continue
        candidates.append(
            ModelCandidate(
                source="biomodels",
                id=model_id,
                name=record.get("name", ""),
                format=fmt,
                url=record.get("url", ""),
                curated=curated,
                submitter=record.get("submitter"),
            )
        )
    log.info(
        "biomodels '%s': %d hits, %d candidates",
        query,
        payload.get("matches", 0),
        len(candidates),
    )
    return candidates


# ---------------------------------------------------------------------------
# Repositories with no full-text search endpoint
#
# BioModels serves a query directly. ModelDB, BioSimulations and Physiome do
# not: each exposes a full listing and per-record detail, and nothing else. For
# those the index is built once, cached on disk, and matched client-side. The
# first search against such a source pays the build; later ones are local.
# ---------------------------------------------------------------------------

#: Terms longer than this may prefix-match a word; shorter ones, which
#: in practice are acronyms, must match a whole word.
STEM_MIN_CHARS = 3

CACHE_TTL_DAYS = 30.0
INDEX_WORKERS = 16
#: Tries per URL during an index build, backing off between them.
INDEX_ATTEMPTS = 4
#: Gap between straggler refetches, once concurrency has been given up on.
SERIAL_RETRY_PAUSE = 0.5
#: Reject an index build that could not hydrate this fraction of its rows.
#: An empty row is indistinguishable from a model with no annotation, so a
#: partial build caches as a complete one and reads as absence of evidence.
INDEX_MIN_HYDRATED = 0.9


def _cache_dir() -> Path:
    path = Path.home() / ".cache" / "hallsim" / "discovery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_index(
    name: str,
    build: "callable",
    ttl_days: float = CACHE_TTL_DAYS,
    refresh: bool = False,
) -> list[dict]:
    """``build()``'s records, cached on disk under ``~/.cache/hallsim``.

    Rebuilt when older than ``ttl_days`` or when ``refresh``. A build failure
    with a stale cache present returns the stale cache rather than nothing —
    a repository being down should degrade the search, not empty it.
    """
    import time

    path = _cache_dir() / f"{name}.json"
    fresh = (
        path.exists()
        and (time.time() - path.stat().st_mtime) < ttl_days * 86400.0
    )
    if fresh and not refresh:
        return json.loads(path.read_text())
    try:
        records = build()
    except Exception as exc:
        if path.exists():
            log.warning(
                "%s index rebuild failed (%s); using stale cache", name, exc
            )
            return json.loads(path.read_text())
        raise
    path.write_text(json.dumps(records))
    log.info("%s index built: %d records -> %s", name, len(records), path)
    return records


def _fetch_many(
    urls: list[str],
    timeout: float = 30.0,
    workers: int = INDEX_WORKERS,
    attempts: int = INDEX_ATTEMPTS,
) -> list[dict | None]:
    """Fetch JSON from many URLs concurrently, preserving order.

    Index builds are thousands of small requests against a public API; serial
    fetching makes the first search a coffee break. A failed record is None
    rather than an exception — one bad row must not lose the index.

    Overload is retried with backoff. A loaded repository answers a request it
    would otherwise serve with 429 *or* with 500 — JWS returns 500 — so a
    status-code allowlist would drop two thirds of that index on the floor and
    still look like a complete build. Callers must therefore check how many
    rows came back None; :func:`cached_index` writes whatever it is handed.

    Stragglers get a final serial pass. Retrying in place keeps every worker
    hammering a source that is shedding load, and JWS's failures were measured
    to be transient rather than per-URL: the same 40 URLs that failed 29 times
    inside a concurrent batch all succeeded when spaced out.
    """
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor

    def one(url, tries):
        for attempt in range(tries):
            try:
                return _get_json(url, {}, timeout)
            except Exception:
                if attempt == tries - 1:
                    return None
                time.sleep(2.0**attempt * (0.5 + random.random()))
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(lambda u: one(u, attempts), urls))

    missing = [i for i, rec in enumerate(out) if rec is None]
    if missing and len(missing) < len(urls):
        log.info("retrying %d straggler(s) serially", len(missing))
        for i in missing:
            time.sleep(SERIAL_RETRY_PAUSE)
            out[i] = one(urls[i], attempts)
    return out


def _score(query: str, *fields: str) -> int:
    """Match count for ``query``'s terms across ``fields``; 0 means no match.

    Every term must appear somewhere, so a two-word query does not return
    everything matching either word.

    A term longer than ``STEM_MIN_CHARS`` matches at a word boundary and may
    run on, so 'senesc' finds 'senescence' and 'senescent'. A shorter one must
    match a whole word, because short queries are acronyms ('ROS', 'p53') and
    prefix-matching them is indiscriminate: plain substring matching had 'ros'
    hitting 'interossei' and 'cross-bridge', and boundary-anchored prefix
    matching still hits 'Rosenbaum'. The cut is a convention, not a discovery —
    it separates acronyms from stems by the only signal available here.
    """
    haystack = " ".join(f.lower() for f in fields if f)
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return 0
    counts = []
    for t in terms:
        tail = "" if len(t) > STEM_MIN_CHARS else r"\b"
        counts.append(len(re.findall(r"\b" + re.escape(t) + tail, haystack)))
    return sum(counts) if all(counts) else 0


def _ranked(scored: list[tuple[int, ModelCandidate]], limit: int):
    scored.sort(key=lambda sc: -sc[0])
    return [c for _, c in scored[:limit]]


UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def uniprot_accessions(
    symbol: str, taxon: int = 9606, timeout: float = 30.0
) -> tuple[str, ...]:
    """Reviewed UniProt accessions for a gene symbol.

    Tries the repo's own symbol table first — it is offline and instant, but
    covers only the reporter genes — then UniProt's REST API.
    """
    try:
        from hallsim.reporter_wiring import _uniprot_symbol

        local = tuple(
            acc
            for acc, (sym, tax) in _uniprot_symbol().items()
            if sym.upper() == symbol.upper() and str(tax) == str(taxon)
        )
        if local:
            return local
    except Exception:
        pass
    params = {
        "query": (
            f"gene_exact:{symbol} AND organism_id:{taxon} AND reviewed:true"
        ),
        "fields": "accession",
        "format": "tsv",
        "size": "10",
    }
    request = urllib.request.Request(
        f"{UNIPROT_SEARCH}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            rows = response.read().decode("utf-8").splitlines()
    except Exception as exc:
        log.warning("uniprot lookup failed for %r: %s", symbol, exc)
        return ()
    return tuple(r.strip() for r in rows[1:] if r.strip())


def search_by_gene(
    symbol: str, limit: int = 25, taxon: int = 9606, **kwargs
) -> list[ModelCandidate]:
    """BioModels hits for a gene, by symbol **and** by UniProt accession.

    BioModels indexes MIRIAM annotations as well as free text, and the two
    reach different models: a model whose species are annotated but whose
    title and notes never write the symbol is invisible to a symbol search.
    Measured over five genes, accession search found ~60 models a symbol
    search missed — for `TP53`, 4 hits by symbol against 32 by `P04637` —
    and the miss runs both ways, so this returns the union.
    """
    seen: dict[str, ModelCandidate] = {}
    for term in (symbol, *uniprot_accessions(symbol, taxon=taxon)):
        for c in search_biomodels(term, limit=limit, **kwargs):
            seen.setdefault(c.id, c)
    log.info(
        "gene '%s': %d candidates via symbol + accession", symbol, len(seen)
    )
    return list(seen.values())[:limit]


MODELDB_API = "https://modeldb.science/api/v1/models"


def _build_modeldb_index() -> list[dict]:
    ids = _get_json(MODELDB_API, {}, 60.0)
    log.info("modeldb: hydrating %d records (one-time, cached)", len(ids))
    records = _fetch_many([f"{MODELDB_API}/{i}" for i in ids], timeout=30.0)
    out = []
    for rec in records:
        if not rec:
            continue

        def names(field):
            """ModelDB wraps every attribute as ``{"value": ..., "attr_id":
            N}``, and the value is a list of tagged objects for a controlled
            vocabulary but a bare string for free text."""
            value = (rec.get(field) or {}).get("value")
            if isinstance(value, str):
                return value
            return " ".join(
                v.get("object_name", "")
                for v in (value or [])
                if isinstance(v, dict)
            )

        out.append(
            {
                "id": str(rec.get("id")),
                "name": rec.get("name", ""),
                "app": names("modeling_application"),
                "text": " ".join(
                    [
                        names("notes"),
                        names("model_concept"),
                        names("model_type"),
                        names("region"),
                        names("neurons"),
                        names("currents"),
                        names("receptors"),
                    ]
                ),
            }
        )
    return out


def search_modeldb(
    query: str, limit: int = 25, refresh: bool = False, **_
) -> list[ModelCandidate]:
    """ModelDB — computational neuroscience, and the main home of XPP models.

    Indexed by neuron type, ionic current, receptor and brain region, so it is
    the source to reach for a channel, a neuron or a network model and the
    wrong one for anything else. ``format`` is the modelling application
    (NEURON, XPP, MATLAB, …); only XPP has an importer here, and nothing is
    auto-fetchable, so a hit is a pointer to a file to vendor by hand.
    """
    index = cached_index("modeldb", _build_modeldb_index, refresh=refresh)
    scored = []
    for rec in index:
        score = _score(query, rec["name"], rec["text"], rec["app"])
        if not score:
            continue
        scored.append(
            (
                score,
                ModelCandidate(
                    source="modeldb",
                    id=rec["id"],
                    name=rec["name"],
                    format=rec["app"] or "unknown",
                    url=f"https://modeldb.science/{rec['id']}",
                    curated=True,
                    kind="neuronal",
                    description=rec["text"][:2000],
                ),
            )
        )
    log.info("modeldb '%s': %d candidates", query, len(scored))
    return _ranked(scored, limit)


BIOSIM_API = "https://api.biosimulations.org"


def _build_biosimulations_index() -> list[dict]:
    projects = _get_json(f"{BIOSIM_API}/projects", {}, 60.0)
    log.info(
        "biosimulations: hydrating %d projects (one-time, cached)",
        len(projects),
    )
    runs = [p.get("simulationRun", "") for p in projects]
    records = _fetch_many([f"{BIOSIM_API}/metadata/{r}" for r in runs])
    out = []
    for project, record in zip(projects, records):
        meta = ((record or {}).get("metadata") or [{}])[0]
        out.append(
            {
                "id": project.get("id", ""),
                "name": meta.get("title") or project.get("id", ""),
                "text": " ".join(
                    filter(
                        None,
                        [
                            meta.get("abstract") or "",
                            meta.get("description") or "",
                            " ".join(
                                k.get("label", "")
                                for k in (meta.get("keywords") or [])
                            ),
                        ],
                    )
                ),
            }
        )
    return out


def search_biosimulations(
    query: str, limit: int = 25, refresh: bool = False, **_
) -> list[ModelCandidate]:
    """BioSimulations — COMBINE archives with their simulation set-up.

    Broader than BioModels (it carries SBML, CellML, NeuroML, BNGL and SMOLDYN
    projects) and each entry ships a runnable simulation rather than a bare
    model file, so a hit tells you the conditions the authors actually ran.
    """
    index = cached_index(
        "biosimulations", _build_biosimulations_index, refresh=refresh
    )
    scored = []
    for rec in index:
        score = _score(query, rec["name"], rec["text"])
        if not score:
            continue
        scored.append(
            (
                score,
                ModelCandidate(
                    source="biosimulations",
                    id=rec["id"],
                    name=rec["name"],
                    format="combine",
                    url=f"https://biosimulations.org/projects/{rec['id']}",
                    curated=True,
                    description=rec["text"][:2000],
                ),
            )
        )
    log.info("biosimulations '%s': %d candidates", query, len(scored))
    return _ranked(scored, limit)


PHYSIOME_EXPOSURES = "https://models.physiomeproject.org/exposure"


def _build_physiome_index() -> list[dict]:
    payload = _get_json(PHYSIOME_EXPOSURES, {"format": "json"}, 60.0)
    links = (payload.get("collection") or {}).get("links") or []
    return [
        {
            "id": link["href"].rsplit("/", 1)[-1],
            "name": (link.get("prompt") or "").strip(),
            "url": link["href"],
        }
        for link in links
        if link.get("href")
    ]


def search_physiome(
    query: str, limit: int = 25, refresh: bool = False, **_
) -> list[ModelCandidate]:
    """Physiome Model Repository — CellML, across the whole of physiology.

    Its own categories run from calcium dynamics and cardiovascular circulation
    through cell cycle, gene regulation, immunology, metabolism, PKPD and signal
    transduction. Measured over the 1108 cached exposure titles, cardiac and
    electrophysiology account for ~4%, gene regulation is the largest
    identifiable group at ~12%, and 77% of titles match no subject keyword at
    all — so treat any characterisation of its contents, including this one, as
    a summary of titles rather than of models.

    The exposure listing carries titles, so the index is one request. Only the
    title is searchable; there is no abstract in the listing, which makes this
    the shallowest of the four searches. CellML has no importer here yet, so a
    hit is a pointer.
    """
    index = cached_index("physiome", _build_physiome_index, refresh=refresh)
    scored = []
    for rec in index:
        score = _score(query, rec["name"])
        if not score:
            continue
        scored.append(
            (
                score,
                ModelCandidate(
                    source="physiome",
                    id=rec["id"],
                    name=rec["name"],
                    format="cellml",
                    url=rec["url"],
                    curated=True,
                    description=rec["name"],
                ),
            )
        )
    log.info("physiome '%s': %d candidates", query, len(scored))
    return _ranked(scored, limit)


JWS_MODELS = "https://jjj.bio.vu.nl/rest/models/"
JWS_SBML = "https://jjj.bio.vu.nl/models/{slug}/sbml/"
#: JWS sheds load at well under :data:`INDEX_WORKERS`, answering 500 rather
#: than 429 for most of it. Measured: 16 workers hydrated 11 of 40 records,
#: and the same 40 fetched cleanly once spaced out.
JWS_WORKERS = 4


def _build_jws_index() -> list[dict]:
    """Slug, species, reactions and citation for every JWS Online model.

    JWS serves a listing and per-model detail, with the manuscript on a third
    endpoint, so searchable text has to be assembled once and cached — the
    same shape as the ModelDB index.
    """
    listing = _get_json(JWS_MODELS, {}, 60.0)
    slugs = [m["slug"] for m in listing if m.get("slug")]
    log.info("jws: hydrating %d records (one-time, cached)", len(slugs))
    details = _fetch_many(
        [f"{JWS_MODELS}{s}/" for s in slugs], timeout=30.0, workers=JWS_WORKERS
    )
    hydrated = sum(1 for d in details if d)
    if hydrated < INDEX_MIN_HYDRATED * len(slugs):
        raise RuntimeError(
            f"jws: only {hydrated}/{len(slugs)} model records fetched; "
            f"refusing to cache a partial index"
        )
    # A model with no linked paper 404s here, so these are allowed to be
    # sparse in a way the detail fetch is not.
    papers = _fetch_many(
        [f"{JWS_MODELS}{s}/manuscript/" for s in slugs],
        timeout=30.0,
        workers=JWS_WORKERS,
    )
    log.info(
        "jws: %d/%d records, %d with a linked paper",
        hydrated,
        len(slugs),
        sum(1 for p in papers if p),
    )
    out = []
    for base, detail, paper in zip(listing, details, papers):
        d, m = detail or {}, paper or {}
        out.append(
            {
                "slug": base["slug"],
                "name": d.get("name") or base.get("slug", ""),
                "cbm": bool(base.get("cbm") or d.get("cbm")),
                "status": base.get("status", ""),
                "title": m.get("title") or "",
                "authors": " ".join(
                    a.get("family_name", "") for a in (m.get("authors") or [])
                ),
                "year": m.get("year"),
                "pubmed": m.get("pm_id"),
                "doi": m.get("doi") or "",
                "species": " ".join(d.get("species_set") or []),
                "reactions": " ".join(d.get("reaction_set") or []),
            }
        )
    return out


def search_jws(
    query: str,
    limit: int = 25,
    curated_only: bool = True,
    refresh: bool = False,
    **_,
) -> list[ModelCandidate]:
    """JWS Online: curated kinetic models, served as SBML.

    Matches the query against model name, paper title and authors, and the
    species and reaction names — a model whose title never writes a gene is
    still reachable through the species it contains, which is the same miss
    :func:`search_by_gene` exists to close on BioModels.

    ``cbm`` models are constraint-based: stoichiometry with no rate laws,
    solved by linear programming rather than integrated, so they are dropped.
    """
    index = cached_index("jws", _build_jws_index, refresh=refresh)
    scored = []
    for rec in index:
        if rec.get("cbm"):
            continue
        if curated_only and rec.get("status", "").upper() != "CURATED":
            continue
        score = _score(
            query,
            *(
                str(rec.get(k, ""))
                for k in (
                    "slug",
                    "name",
                    "title",
                    "authors",
                    "species",
                    "reactions",
                )
            ),
        )
        if not score:
            continue
        scored.append(
            (
                score,
                ModelCandidate(
                    source="jws",
                    id=rec["slug"],
                    name=rec.get("title") or rec.get("name", ""),
                    format="SBML",
                    url=f"https://jjj.bio.vu.nl/models/{rec['slug']}/",
                    curated=True,
                    submitter=rec.get("authors") or None,
                    description=rec.get("species", ""),
                ),
            )
        )
    log.info(
        "jws '%s': %d candidates of %d indexed", query, len(scored), len(index)
    )
    return _ranked(scored, limit)


SOURCES = {
    "biomodels": search_biomodels,
    "jws": search_jws,
    "modeldb": search_modeldb,
    "biosimulations": search_biosimulations,
    "physiome": search_physiome,
}


def search_for_model(
    query: str,
    limit: int = 25,
    sources: list[str] | None = None,
    **kwargs,
) -> list[ModelCandidate]:
    """Search every registered repository for ``query``.

    A source that errors is logged and skipped, so one repository being down
    does not abort a swarm run. Returns candidates in source-registration
    order; ranking within a source is the repository's own.
    """
    names = sources if sources is not None else list(SOURCES)
    found: list[ModelCandidate] = []
    for name in names:
        search = SOURCES.get(name)
        if search is None:
            raise KeyError(f"unknown source {name!r}; have {list(SOURCES)}")
        try:
            found.extend(
                search(query, limit=limit, **_accepted(search, kwargs))
            )
        except Exception as exc:  # a dead repository is not a failed run
            log.warning("source %r failed for '%s': %s", name, query, exc)
    return found


def _accepted(search, kwargs: dict) -> dict:
    """Drop kwargs a source does not take. ``curated_only`` means something to
    BioModels and nothing to ModelDB, and one source's option must not be an
    error for the others."""
    import inspect

    params = inspect.signature(search).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


@dataclass(frozen=True)
class OutputScreen:
    """Whether a deposit *produces* a quantity, not merely mentions it.

    Annotation search answers "is this model about IL6"; composing needs
    "does this model emit IL6". A module imported to supply an output that it
    only ever consumes contributes nothing — the Ihekwaba 2004 failure.
    """

    model_id: str
    #: ``produces`` | ``no-match`` | ``no-sbml`` | ``unreadable`` |
    #: ``fetch-failed``
    status: str
    produced: tuple[str, ...] = ()
    n_species: int = 0
    n_reactions: int = 0
    #: Why, when the deposit could not be screened.
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "produces"


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


def screen_produced_species(
    model_ids, pattern: str, *, timeout: float = 60.0
) -> list[OutputScreen]:
    """Which of ``model_ids`` synthesise a species matching ``pattern``.

    ``pattern`` is a case-insensitive regex matched against species ids. A
    species counts as produced when it is a *product* of some reaction;
    appearing only as a reactant means the deposit consumes it.

    Every input yields a row, including the ones that could not be read — a
    silent skip hides an unreadable deposit as an uninteresting one.
    """
    import re

    rx = re.compile(pattern, re.I)
    out: list[OutputScreen] = []
    for model_id in model_ids:
        try:
            paths = download_biomodel_files(model_id, timeout=timeout)
        except Exception as exc:  # network, 404, malformed accession
            out.append(
                OutputScreen(
                    str(model_id),
                    "fetch-failed",
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        try:
            model, note = _first_readable_sbml(paths)
        except ImportError as exc:
            out.append(OutputScreen(str(model_id), "no-sbml", note=str(exc)))
            continue
        if model is None:
            status = "no-sbml" if "no .xml" in note else "unreadable"
            out.append(OutputScreen(str(model_id), status, note=note))
            continue
        produced = {
            reaction.getProduct(j).getSpecies()
            for i in range(model.getNumReactions())
            for reaction in (model.getReaction(i),)
            for j in range(reaction.getNumProducts())
            if rx.search(reaction.getProduct(j).getSpecies())
        }
        out.append(
            OutputScreen(
                str(model_id),
                "produces" if produced else "no-match",
                tuple(sorted(produced)),
                model.getNumSpecies(),
                model.getNumReactions(),
            )
        )
    return out


def search_producing(
    query: str, pattern: str, *, limit: int = 40, **kwargs
) -> list[OutputScreen]:
    """Search BioModels for ``query``, keep what *produces* ``pattern``.

    The composable version of a text search: a hit is only useful if the
    quantity you need is something it emits.
    """
    hits = search_biomodels(query, limit=limit, **kwargs)
    return screen_produced_species([h.id for h in hits], pattern)
