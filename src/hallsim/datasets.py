"""Where the data is: dataset search, mirrored on :mod:`hallsim.discovery`.

A calibration needs measurements under a perturbation the composite can
represent, read by :func:`hallsim.gene_reporters.load_gene_expression`.
:func:`search_for_dataset` asks the repositories; each hit says what kind of
data it is, on what platform, and what its samples are called, which is
where the arms show. :func:`platform_columns` reads a platform table's
header from GEO so a hit can be checked against the loader before anything
large is downloaded.
"""

from __future__ import annotations

import gzip
import logging
import urllib.request
from dataclasses import dataclass

import pandas as pd

from hallsim.discovery import _get_json
from hallsim.gene_reporters import choose_annotation, geo_series_urls

log = logging.getLogger(__name__)

GEO_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_ACCESSION_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}"


@dataclass(frozen=True)
class DatasetCandidate:
    """One series, enough to decide whether to fetch it."""

    source: str
    accession: str
    title: str
    #: GEO's data type, e.g. "Expression profiling by array".
    kind: str
    organism: str
    n_samples: int
    platform: str
    url: str
    summary: str = ""
    #: Sample titles as deposited; the arms and timepoints are usually in them.
    samples: tuple[str, ...] = ()

    @property
    def short_kind(self) -> str:
        """GEO's data type in a word: ``array``, ``rna-seq``, ``methylation``,
        else the type as GEO states it."""
        k = self.kind.lower()
        if k.startswith("expression profiling by array"):
            return "array"
        if k.startswith("expression profiling by high throughput"):
            return "rna-seq"
        if k.startswith("methylation"):
            return "methylation"
        return self.kind

    @property
    def series_matrix_has_values(self) -> bool:
        """Array series carry their values in the series matrix; sequencing
        series usually ship counts as supplementary files the loader does
        not read."""
        return self.kind.lower().startswith("expression profiling by array")


def search_geo(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
) -> list[DatasetCandidate]:
    """GEO series matching ``query`` (all fields), newest first, through the
    keyless E-utilities."""
    term = f"({query}) AND gse[EntryType]"
    if organism:
        term += f' AND "{organism}"[Organism]'
    found = _get_json(
        f"{GEO_EUTILS}/esearch.fcgi",
        {"db": "gds", "term": term, "retmax": limit, "retmode": "json"},
        timeout,
    )["esearchresult"]
    ids = found.get("idlist", [])
    log.info("geo '%s': %s hits", query, found.get("count", "?"))
    if not ids:
        return []
    result = _get_json(
        f"{GEO_EUTILS}/esummary.fcgi",
        {"db": "gds", "id": ",".join(ids), "retmode": "json"},
        timeout,
    )["result"]
    out = []
    for uid in result.get("uids", []):
        r = result[uid]
        gpl = str(r.get("gpl", ""))
        out.append(
            DatasetCandidate(
                source="geo",
                accession=r["accession"],
                title=r.get("title", ""),
                kind=r.get("gdstype", ""),
                organism=r.get("taxon", ""),
                n_samples=int(r.get("n_samples", 0) or 0),
                platform=";".join(f"GPL{g}" for g in gpl.split(";") if g),
                url=GEO_ACCESSION_URL.format(r["accession"]),
                summary=r.get("summary", ""),
                samples=tuple(
                    s.get("title", "") for s in r.get("samples", [])
                ),
            )
        )
    return out


SOURCES = {"geo": search_geo}


def search_for_dataset(
    query: str,
    limit: int = 25,
    sources: list[str] | None = None,
    **kwargs,
) -> list[DatasetCandidate]:
    """Search every registered data repository for ``query``. A source that
    errors is logged and skipped."""
    names = sources if sources is not None else list(SOURCES)
    found: list[DatasetCandidate] = []
    for name in names:
        search = SOURCES.get(name)
        if search is None:
            raise KeyError(f"unknown source {name!r}; have {list(SOURCES)}")
        try:
            found.extend(search(query, limit=limit, **kwargs))
        except Exception as exc:  # noqa: BLE001 - a dead repository
            log.warning("source %r failed for '%s': %s", name, query, exc)
    return found


def platform_head(
    accession: str, n_rows: int = 200, *, timeout: float = 60.0
) -> pd.DataFrame:
    """The first ``n_rows`` of the platform table in a series' family SOFT,
    read from the head of the stream so it costs kilobytes, not the table.
    """
    from io import StringIO

    _, soft_url = geo_series_urls(accession)
    lines: list[str] = []
    with urllib.request.urlopen(soft_url, timeout=timeout) as resp:
        with gzip.open(resp, "rt", errors="replace") as fh:
            inside = False
            for line in fh:
                if line.startswith("!platform_table_begin"):
                    inside = True
                elif line.startswith("!platform_table_end"):
                    break
                elif inside:
                    lines.append(line)
                    if len(lines) > n_rows:
                        break
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str)


def loader_route(frame: pd.DataFrame) -> str:
    """How :func:`hallsim.gene_reporters.load_gene_expression` would map
    this platform's probes to genes, or why it cannot."""
    if frame.empty or "ID" not in frame.columns:
        return "no platform table in the series' SOFT"
    try:
        col, route = choose_annotation(frame)
    except ValueError as exc:
        return f"the loader cannot map it: {exc}"
    via = " via MyGene.info" if route == "accession" else ""
    return f"the loader reads {route}s from column {col!r}{via}"
