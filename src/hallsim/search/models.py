"""Where the models are: search over the repositories, and over the
papers whose supplement carries one.

The first step an agent takes: turn a mechanism named in prose ("p53
oscillator", "NF-κB signalling") into concrete, fetchable candidates, each
of which an importer takes by id (``process_from_sbml(hits[0].id)``).

:func:`search_for_model` fans out over the registered sources and returns a
flat, ranked candidate list. Add a repository by writing one search function
and registering it in :data:`SOURCES` — callers do not change.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from hallsim.search.fetch import (
    USER_AGENT,
    atomic_write,
    cache_dir,
    cached_index,
    cached_json,
    fetch_many,
    get_json,
    is_junk_archive_entry,
)

log = logging.getLogger(__name__)

BIOMODELS_SEARCH = "https://www.ebi.ac.uk/biomodels/search"


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
    #: supplementary files often carry the fitting data a calibration
    #: scores against.
    files: tuple = ()

    def fetch(self) -> Path:
        """Download the model file to the local cache and return its path:
        the main file of a BioModels deposit, the JWS Online file, or the
        first model file in a paper's supplement. ``process_from_sbml`` takes
        the same ids directly."""
        if self.source == "biomodels":
            return download_biomodel_main(self.id)
        if self.source in ("jws", "europepmc"):
            return Path(model_files(self.id, self.source, 120.0)[0])
        raise NotImplementedError(
            f"no fetcher for source {self.source!r}; open {self.url} and "
            f"vendor the file under demos/models/ or data/ by hand"
        )

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
        every later claim about the model. Fetching only the SBML leaves that
        on the server.
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
    return get_json(
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
    main_only: bool = False,
    names=None,
) -> list["Path"]:
    """Download every file in a BioModels deposit; return the local paths.

    ``main_only`` fetches just the deposit's main model file, ``names`` only
    the files named (a deposit's data tables, its COPASI file). A screen that
    only reads the SBML otherwise pulls the PDF, the reaction-diagram PNG and
    SVG, the MATLAB and XPP exports and the curation log with it — megabytes
    per deposit, and across a few hundred candidates that is most of the wall
    clock spent on files nothing opens.

    :func:`download_biomodel_main` fetches the model and stops there, but
    the deposit's other files are where the provenance is: a README stating
    what was deposited and under what conditions, and — for the minority of
    papers that deposit it — the fitting data a calibration scores against.
    Defaults to ``biomodels/<accession>/`` under the cache.

    A file that comes back empty is not written: the download endpoint 302s,
    and a client that does not follow the redirect gets a silent zero-byte
    body rather than an error.
    """
    accession = _accession(model_id)
    record = biomodels_record(accession, timeout=timeout)
    if main_only:
        files = (record.get("files") or {}).get("main") or []
        names = tuple(f.get("name", "") for f in files if f.get("name"))
    elif names is not None:
        wanted = set(names)
        names = tuple(n for n in _record_filenames(record) if n in wanted)
    else:
        names = _record_filenames(record)
    out_dir = (
        Path(dest) if dest is not None else cache_dir("biomodels") / accession
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


#: Over-fetch factor for filters applied client-side. The search truncates
#: server-side, so filtering the returned page silently shrinks the result set
#: — with ``sbml_only`` alone that is 21% of the corpus, and a page that
#: happens to be format-heavy reports "no hits" for a query that had them.
OVERFETCH = 3


def search_biomodels(
    query: str,
    limit: int = 25,
    curated_only: bool = False,
    sbml_only: bool = True,
    timeout: float = 30.0,
) -> list[ModelCandidate]:
    """BioModels full-text search.

    A ``BIOMD`` accession is the manually curated branch, a ``MODEL``
    accession the uncurated one, reported as ``ModelCandidate.curated`` either
    way. **Uncurated deposits are returned by default**: curation status is
    EBI's editorial queue, not a property of the model — it is 53% of the
    corpus (1,692 of 3,212), it holds the only CXCL8 and CCL2 models in the
    repository, and :func:`hallsim.intake.triage_sbml` is a better admission
    test than someone else's backlog. Pass ``curated_only=True`` to restrict.

    ``sbml_only`` drops the MATLAB/R/other-format entries the search also
    returns, since only SBML has an importer.
    """
    # The endpoint answers a quoted phrase with HTTP 400; the words are
    # matched individually, so quotes only ever break the call.
    query = query.replace('"', "")
    payload = get_json(
        BIOMODELS_SEARCH,
        {
            "query": query,
            "format": "json",
            "numResults": limit * OVERFETCH,
        },
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
    kept = candidates[:limit]
    log.info(
        "biomodels '%s': %d hits, %d fetched, %d passed filters, %d returned",
        query,
        payload.get("matches", 0),
        len(payload.get("models", [])),
        len(candidates),
        len(kept),
    )
    return kept


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

#: Reject an index build that could not hydrate this fraction of its rows.
#: An empty row is indistinguishable from a model with no annotation, so a
#: partial build caches as a complete one and reads as absence of evidence.
INDEX_MIN_HYDRATED = 0.9


def term_score(query: str, *fields: str) -> int:
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
    """Reviewed UniProt accessions for a gene symbol, from UniProt's REST
    API. An answer is kept on disk, so a symbol is asked once per machine."""
    params = {
        "query": (
            f"gene_exact:{symbol} AND organism_id:{taxon} AND reviewed:true"
        ),
        "fields": "accession",
        "format": "tsv",
        "size": "10",
    }

    def fetch() -> list[str]:
        request = urllib.request.Request(
            f"{UNIPROT_SEARCH}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            rows = response.read().decode("utf-8").splitlines()
        return [r.strip() for r in rows[1:] if r.strip()]

    try:
        return tuple(cached_json(f"uniprot {symbol.upper()} {taxon}", fetch))
    except Exception as exc:
        log.warning("uniprot lookup failed for %r: %s", symbol, exc)
        return ()


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
    ids = get_json(MODELDB_API, {}, 60.0)
    log.info("modeldb: hydrating %d records (one-time, cached)", len(ids))
    records = fetch_many([f"{MODELDB_API}/{i}" for i in ids], timeout=30.0)
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
        score = term_score(query, rec["name"], rec["text"], rec["app"])
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
    projects = get_json(f"{BIOSIM_API}/projects", {}, 60.0)
    log.info(
        "biosimulations: hydrating %d projects (one-time, cached)",
        len(projects),
    )
    runs = [p.get("simulationRun", "") for p in projects]
    records = fetch_many([f"{BIOSIM_API}/metadata/{r}" for r in runs])
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
        score = term_score(query, rec["name"], rec["text"])
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
    payload = get_json(PHYSIOME_EXPOSURES, {"format": "json"}, 60.0)
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
        score = term_score(query, rec["name"])
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
    listing = get_json(JWS_MODELS, {}, 60.0)
    # The listing repeats a slug once per model version — the beuke* family
    # is 30 models in 180 rows. Fetching per row wastes the duplicates and,
    # worse, counts a broken record once per repeat, which sinks the
    # hydration ratio below the guard on 30 bad models out of 676.
    by_slug = {}
    for entry in listing:
        if entry.get("slug"):
            by_slug.setdefault(entry["slug"], entry)
    listing = list(by_slug.values())
    slugs = list(by_slug)
    log.info("jws: hydrating %d unique records (one-time, cached)", len(slugs))
    details = fetch_many(
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
    papers = fetch_many(
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
    curated_only: bool = False,
    refresh: bool = False,
    **_,
) -> list[ModelCandidate]:
    """JWS Online: kinetic models, served as SBML.

    Matches the query against model name, paper title and authors, and the
    species and reaction names — a model whose title never writes a gene is
    still reachable through the species it contains, which is the same miss
    :func:`search_by_gene` exists to close on BioModels.

    JWS's own ``status`` is reported as ``curated`` / ``curation`` rather than
    filtered on, for the reason it is not filtered on in
    :func:`search_biomodels`. Pass ``curated_only=True`` to restrict.

    ``cbm`` models are constraint-based: stoichiometry with no rate laws,
    solved by linear programming rather than integrated, so they are dropped.
    """
    index = cached_index("jws", _build_jws_index, refresh=refresh)
    scored = []
    for rec in index:
        if rec.get("cbm"):
            continue
        status = rec.get("status", "")
        if curated_only and status.upper() != "CURATED":
            continue
        score = term_score(
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
                    curated=status.upper() == "CURATED",
                    curation=status,
                    submitter=rec.get("authors") or None,
                    description=rec.get("species", ""),
                ),
            )
        )
    log.info(
        "jws '%s': %d candidates of %d indexed", query, len(scored), len(index)
    )
    return _ranked(scored, limit)


def _search_europepmc(*a, **kw):
    """Late import: `literature` imports `ModelCandidate` from here."""
    from hallsim.search.literature import search_europepmc

    return search_europepmc(*a, **kw)


SOURCES = {
    "biomodels": search_biomodels,
    "jws": search_jws,
    # Not a repository: papers whose model was never deposited anywhere. The
    # harvest recovers the file from the supplement, so the screen treats a
    # paper exactly like a deposit.
    "europepmc": _search_europepmc,
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
    import time
    from concurrent.futures import ThreadPoolExecutor

    names = sources if sources is not None else list(SOURCES)
    searches = {}
    for name in names:
        search = SOURCES.get(name)
        if search is None:
            raise KeyError(f"unknown source {name!r}; have {list(SOURCES)}")
        searches[name] = search

    def ask(name):
        search = searches[name]
        started = time.perf_counter()
        try:
            hits = search(query, limit=limit, **_accepted(search, kwargs))
        except Exception as exc:  # a dead repository is not a failed run
            log.warning("source %r failed for '%s': %s", name, query, exc)
            hits = []
        log.info(
            "source %r: %d hits in %.1fs",
            name,
            len(hits),
            time.perf_counter() - started,
        )
        return hits

    # Repositories answer in parallel, so an unreachable one costs its own
    # timeout rather than adding it to every other's; order is preserved.
    with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        results = list(pool.map(ask, names))
    return [c for hits in results for c in hits]


def _accepted(search, kwargs: dict) -> dict:
    """Drop kwargs a source does not take. ``curated_only`` means something to
    BioModels and nothing to ModelDB, and one source's option must not be an
    error for the others."""
    import inspect

    params = inspect.signature(search).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


MODELDB_DOWNLOAD = "https://modeldb.science/download/{model_id}"


def download_modeldb_models(model_id, timeout: float = 120.0) -> list:
    """Model source files from a ModelDB entry, cached on disk.

    ModelDB ships a zip of whatever the authors ran. Only the formats there is
    a reader for are returned — ``.ode`` (XPPAUT) and ``.cps`` — so a NEURON
    or GENESIS entry comes back empty and is reported unscreenable rather than
    silently dropped.
    """
    import io
    import zipfile

    dest = cache_dir("modeldb") / str(model_id)
    if dest.exists():
        return sorted(dest.iterdir())
    url = MODELDB_DOWNLOAD.format(model_id=model_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        blob = response.read()
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise LookupError(f"modeldb {model_id}: not a zip ({exc})") from exc
    for info in archive.infolist():
        name = Path(info.filename).name
        if info.is_dir() or is_junk_archive_entry(info.filename):
            continue
        if not name.lower().endswith((".ode", ".cps")):
            continue
        target = dest / name
        target.write_bytes(archive.read(info))
        out.append(target)
    if not out:
        raise LookupError(
            f"modeldb {model_id}: no .ode or .cps in the deposit — the entry "
            f"is NEURON, MATLAB or another format with no reader here"
        )
    return sorted(out)


def _biomodel_main_text(accession: str, timeout: float = 60.0) -> str:
    """The text of a deposit's main SBML file.

    Older curated deposits name it ``<accession>_url.xml``; newer ones keep
    the author's filename (BIOMD0000001044 is ``Csikasz-Nagy2006.xml``) and
    the conventional name answers HTTP 400. The record says which file is
    main, so that is read when the convention fails.
    """
    import urllib.error

    base = BIOMODELS_DOWNLOAD.format(model_id=accession)

    def fetch(name: str) -> str:
        query = urllib.parse.urlencode({"filename": name})
        request = urllib.request.Request(
            f"{base}?{query}", headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")

    try:
        return fetch(f"{accession}_url.xml")
    except urllib.error.HTTPError as first:
        record = biomodels_record(accession, timeout=timeout)
        main = [
            f.get("name", "")
            for f in ((record.get("files") or {}).get("main") or [])
            if f.get("name")
        ]
        sbml = [n for n in main if n.lower().endswith((".xml", ".sbml"))]
        if not sbml:
            raise first
        return fetch(sbml[0])


def download_biomodel_main(model_id) -> Path:
    """The main SBML file of a BioModels deposit, cached under
    ``biomodels/``. An accession is immutable once deposited, so the first
    call downloads and every later one reads the file."""
    accession = _accession(model_id)
    path = cache_dir("biomodels") / f"{accession}.xml"
    if not path.exists():
        atomic_write(path, _biomodel_main_text(accession).encode("utf-8"))
    return path


def download_jws_model(slug: str) -> Path:
    """A JWS Online model's SBML, cached under ``jws/``."""
    path = cache_dir("jws") / f"{slug}.xml"
    if path.exists():
        return path
    with urllib.request.urlopen(JWS_SBML.format(slug=slug), timeout=60) as fh:
        body = fh.read()
    if b"<sbml" not in body[:4000]:
        raise ValueError(
            f"JWS model {slug!r} did not return SBML — check the slug at "
            f"https://jjj.bio.vu.nl/models/{slug}/"
        )
    atomic_write(path, body)
    return path


def model_files(model_id: str, source: str, timeout: float) -> list:
    """Local model files for one candidate, dispatched on its repository.

    A source missing here is reported unscreenable, never dropped. Adding one
    is a fetcher, not a branch in the screen.
    """
    if source == "biomodels":
        # The screen reads the model and nothing else; fall back to the whole
        # deposit only if the main file turns out not to be readable SBML.
        main = download_biomodel_files(
            model_id, timeout=timeout, main_only=True
        )
        if main:
            return main
        return download_biomodel_files(model_id, timeout=timeout)
    if source == "jws":
        return [download_jws_model(model_id)]
    if source == "modeldb":
        return download_modeldb_models(model_id, timeout=timeout)
    if source == "europepmc":
        from hallsim.search.literature import supplementary_model_files

        files = supplementary_model_files(model_id, timeout=timeout)
        if not files:
            raise LookupError(
                f"{model_id} deposited no model file in its supplement"
            )
        return files
    raise LookupError(f"no SBML fetcher for source {source!r}")
