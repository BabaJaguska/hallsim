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

    def fetch(self) -> Path:
        """Download to the local cache and return the SBML path."""
        if self.source != "biomodels":
            raise NotImplementedError(f"no fetcher for source {self.source!r}")
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


SOURCES = {"biomodels": search_biomodels}


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
            found.extend(search(query, limit=limit, **kwargs))
        except Exception as exc:  # a dead repository is not a failed run
            log.warning("source %r failed for '%s': %s", name, query, exc)
    return found
