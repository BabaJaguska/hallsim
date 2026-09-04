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

    def fetch(self) -> Path:
        """Download to the local cache and return the SBML path."""
        if self.source != "biomodels":
            raise NotImplementedError(
                f"no fetcher for source {self.source!r}; open {self.url} and "
                f"vendor the file under demos/models/ or data/ by hand"
            )
        from hallsim.sbml_import import _download_biomodel_to_cache

        return Path(_download_biomodel_to_cache(self.id))

    def __str__(self) -> str:
        mark = "curated" if self.curated else "uncurated"
        return f"[{self.source}:{self.id}] {self.name} ({self.format}, {mark})"


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

CACHE_TTL_DAYS = 30.0
INDEX_WORKERS = 16


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


def _fetch_many(urls: list[str], timeout: float = 30.0) -> list[dict | None]:
    """Fetch JSON from many URLs concurrently, preserving order.

    Index builds are thousands of small requests against a public API; serial
    fetching makes the first search a coffee break. A failed record is None
    rather than an exception — one bad row must not lose the index.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(url):
        try:
            return _get_json(url, {}, timeout)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as pool:
        return list(pool.map(one, urls))


def _score(query: str, *fields: str) -> int:
    """Match count for ``query``'s terms across ``fields``; 0 means no match.

    Every term must appear somewhere, so a two-word query does not return
    everything matching either word. Substring rather than token matching, so
    'senesc' finds 'senescence' and 'senescent'.
    """
    haystack = " ".join(f.lower() for f in fields if f)
    terms = [t for t in query.lower().split() if t]
    if not terms or not all(t in haystack for t in terms):
        return 0
    return sum(haystack.count(t) for t in terms)


def _ranked(scored: list[tuple[int, ModelCandidate]], limit: int):
    scored.sort(key=lambda sc: -sc[0])
    return [c for _, c in scored[:limit]]


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
    """Physiome Model Repository — CellML, mostly cardiac and electrophysiology.

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


SOURCES = {
    "biomodels": search_biomodels,
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
