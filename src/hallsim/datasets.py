"""Where the data is: dataset search, mirrored on :mod:`hallsim.discovery`.

A calibration needs a time course of quantities a composite carries, taken
under conditions it can represent. :func:`search_for_dataset` asks the
repositories: GEO and Zenodo, and through EBI Search PRIDE, MetaboLights,
Metabolomics Workbench and ArrayExpress, with the BioImage Archive through
BioStudies. Each hit says what kind of data it is, what it measured
(:class:`Measured`) and how its samples are arranged (:class:`Design`,
read from the sample titles by :func:`parse_design`: arms, a control arm
when one is named, timepoints). :func:`datasets_of` lists a paper's own
data through Europe PMC, which is the quantity its model was built to
predict. :func:`coverage` says which of a composite's annotated store
paths a hit measures, and :func:`search_measuring` keeps the hits a
composite can be scored on, the way :func:`hallsim.discovery.search_producing`
keeps the deposits that emit a quantity. :func:`platform_head` reads an
expression platform's table head so a hit can be checked against that
loader before anything large is downloaded.
"""

from __future__ import annotations


import gzip
import hashlib
import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from hallsim.discovery import _get_json
from hallsim.gene_reporters import SYMBOL, choose_annotation, geo_series_urls

log = logging.getLogger(__name__)

GEO_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_ACCESSION_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}"
ZENODO_RECORDS = "https://zenodo.org/api/records"
EBI_SEARCH = "https://www.ebi.ac.uk/ebisearch/ws/rest"
BIOSTUDIES = "https://www.ebi.ac.uk/biostudies/api/v1"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OLS_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
MYGENE_QUERY = "https://mygene.info/v3/query"
#: EBI Search pages at most this many entries per call.
EBI_PAGE = 100


# ── What a dataset measured, and how its samples are arranged ──────


@dataclass(frozen=True)
class Measured:
    """What a dataset measured: a modality, whether it covers every
    quantity of its kind (a transcriptome, a proteome), and the quantities
    the deposit itself lists as ``namespace:id`` curies (MetaboLights
    names its ChEBI ids; a platform's genes come later, from its table)."""

    modality: str = "unknown"
    complete: bool = False
    ids: tuple[str, ...] = ()


#: GEO's data types by the quantity they measure.
GEO_MODALITY = (
    ("expression profiling by array", "expression", True),
    ("expression profiling by high throughput", "expression", True),
    ("expression profiling by rt-pcr", "expression", False),
    ("non-coding rna profiling", "ncrna", True),
    ("protein profiling", "proteomics", False),
    ("methylation profiling", "methylation", True),
    ("genome binding/occupancy", "binding", True),
    ("genome variation", "genotype", True),
    ("snp genotyping", "genotype", True),
    ("other", "other", False),
)


def curie(namespace: str, ident: str) -> str:
    """``namespace:id`` in one spelling: the namespace lower-cased and a
    repeated namespace prefix on the id (``chebi:CHEBI:15422``) dropped."""
    ns = namespace.lower()
    ident = str(ident).strip()
    if ident.lower().startswith(ns + ":"):
        ident = ident[len(ns) + 1 :]
    return f"{ns}:{ident}"


def geo_measured(gdstype: str) -> Measured:
    k = gdstype.lower()
    for prefix, modality, complete in GEO_MODALITY:
        if k.startswith(prefix):
            return Measured(modality, complete)
    return Measured("unknown", False)


@dataclass(frozen=True)
class Design:
    """How a series' samples are arranged, read from their titles: the
    arms (the title with time and replicate tokens removed and the common
    prefix dropped), the control arm when one is named, and the timepoints
    per arm in ``time_unit``. Nothing here is a gate: an unperturbed time
    course is data too. ``perturbed`` says whether there is more than one
    arm; ``time_course`` whether some arm has three or more timepoints."""

    arms: tuple[str, ...] = ()
    control: str | None = None
    per_arm: tuple[tuple[str, tuple[float, ...]], ...] = ()
    time_unit: str = ""
    n_titles: int = 0

    @property
    def timepoints(self) -> tuple[float, ...]:
        return tuple(sorted({t for _, ts in self.per_arm for t in ts}))

    @property
    def n_timepoints(self) -> int:
        return max((len(ts) for _, ts in self.per_arm), default=0)

    @property
    def perturbed(self) -> bool:
        return len(self.arms) >= 2

    @property
    def time_course(self) -> bool:
        return self.n_timepoints >= 3

    def summary(self) -> str:
        if not self.n_titles:
            return "no sample titles"
        arms = f"{len(self.arms)} arm" + ("s" if len(self.arms) != 1 else "")
        if self.control:
            arms += f" (control: {self.control})"
        if self.n_timepoints:
            unit = f" {self.time_unit}" if self.time_unit else ""
            return f"{arms}, {self.n_timepoints} timepoints{unit}"
        return f"{arms}, no timepoints in the titles"


_UNIT_HOURS = {
    "s": 1 / 3600,
    "sec": 1 / 3600,
    "min": 1 / 60,
    "h": 1.0,
    "hr": 1.0,
    "hour": 1.0,
    "d": 24.0,
    "day": 24.0,
    "w": 168.0,
    "wk": 168.0,
    "week": 168.0,
    "mo": 720.0,
    "month": 720.0,
    "y": 8760.0,
    "yr": 8760.0,
    "year": 8760.0,
}
_UNIT_NAME = {
    "s": "s",
    "sec": "s",
    "min": "min",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "d": "d",
    "day": "d",
    "w": "wk",
    "wk": "wk",
    "week": "wk",
    "mo": "mo",
    "month": "mo",
    "y": "yr",
    "yr": "yr",
    "year": "yr",
}
# "24h", "24 hr", "7 days", "week 4", "day7", "D07", "T24" (T: ordinal).
_TIME = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?P<v1>\d+(?:[.,]\d+)?)\s*(?P<u1>sec|s|min|hrs?|h|hours?|days?|d|"
    r"wks?|w|weeks?|mo|months?|yrs?|y|years?)"
    r"|(?P<u2>day|hour|week|month|year|min)s?\s*(?P<v2>\d+(?:[.,]\d+)?)"
    r"|(?P<u3>[DHWT])(?P<v3>\d{1,3})"
    r")(?![A-Za-z0-9])",
    re.I,
)
_REPLICATE = re.compile(
    r"(?<![A-Za-z0-9])(?:rep(?:licate)?|biorep|techrep|br|tr|r|n)[ _-]?\d+"
    r"(?![A-Za-z0-9])|[ _-]\d(?![A-Za-z0-9.])$",
    re.I,
)
_SEP = re.compile(r"[\s_\-/,:;|()\[\]]+")
CONTROL_WORDS = frozenset(
    {
        "control",
        "ctrl",
        "ctl",
        "untreated",
        "vehicle",
        "dmso",
        "mock",
        "wt",
        "wildtype",
        "wild",
        "sham",
        "baseline",
        "placebo",
        "unstimulated",
        "uninfected",
        "scramble",
        "scrambled",
        "nc",
        "normal",
        "healthy",
    }
)


def _times(title: str) -> tuple[list[tuple[float, str]], str]:
    """``([(value, unit)], title without the time tokens)``."""
    found = []

    def take(m):
        if m.group("v1"):
            v, u = m.group("v1"), m.group("u1").lower()
            u = u.rstrip("s") if u not in ("s", "hrs") else u
            u = {"hrs": "hr", "sec": "sec", "s": "s"}.get(u, u)
        elif m.group("v2"):
            v, u = m.group("v2"), m.group("u2").lower()
        else:
            v, u = m.group("v3"), m.group("u3").upper()
            u = {"D": "d", "H": "h", "W": "w", "T": "t"}[u]
        found.append((float(v.replace(",", ".")), u))
        return " "

    rest = _TIME.sub(take, title)
    return found, rest


_TIME_LIST = re.compile(
    r"(?<![A-Za-z0-9])((?:\d+(?:[.,]\d+)?\s*(?:,|;|/|and|or|to|-)\s*)+"
    r"\d+(?:[.,]\d+)?)\s*(?:sec|s|min|hrs?|h|hours?|days?|d|wks?|w|"
    r"weeks?|mo|months?|yrs?|y|years?)(?![A-Za-z0-9])",
    re.I,
)


def time_values(text: str) -> set[float]:
    """The distinct time values a description names: ``24 h``, ``day 7``,
    ``D07``, and lists such as ``0, 6 and 24 h`` where the unit closes the
    list."""
    values = set()
    for m in _TIME.finditer(text):
        v = m.group("v1") or m.group("v2") or m.group("v3")
        values.add(float(v.replace(",", ".")))
    for m in _TIME_LIST.finditer(text):
        for v in re.findall(r"\d+(?:[.,]\d+)?", m.group(1)):
            values.add(float(v.replace(",", ".")))
    return values


def _arm_label(rest: str) -> list[str]:
    rest = _REPLICATE.sub(" ", rest)
    return [t for t in _SEP.split(rest) if t]


def _drop_identifiers(labels: list[list[str]]) -> list[list[str]]:
    """Drop tokens that name a sample rather than a condition.

    An arm label has to partition the samples: a token carried by exactly
    one of them is an accession, a well, a plate or an animal id, and
    keeping it makes every sample its own arm. Frequency decides, so this
    works on a cohort nobody has seen and needs no vocabulary."""
    if len(labels) < 3:
        return labels
    seen: dict[str, int] = {}
    for label in labels:
        for token in set(label):
            seen[token] = seen.get(token, 0) + 1
    pruned = [[t for t in label if seen[t] > 1] for label in labels]
    # A title made only of identifiers keeps what it had; dropping
    # everything would merge unrelated samples into one unnamed arm.
    return [new or old for new, old in zip(pruned, labels)]


def _strip_common(labels: list[list[str]]) -> list[list[str]]:
    """Drop the tokens every label shares at its head and its tail."""
    if len(labels) < 2:
        return labels
    head = 0
    while (
        all(len(lb) > head for lb in labels)
        and len({lb[head].lower() for lb in labels}) == 1
    ):
        head += 1
    tail = 0
    while (
        all(len(lb) > head + tail for lb in labels)
        and len({lb[-1 - tail].lower() for lb in labels}) == 1
    ):
        tail += 1
    return [lb[head : len(lb) - tail] for lb in labels]


def parse_design(samples) -> Design:
    """The :class:`Design` behind a list of sample titles."""
    titles = [s for s in samples if s]
    if not titles:
        return Design()
    times, labels, units = [], [], []
    for t in titles:
        found, rest = _times(t)
        times.append(found)
        labels.append(_arm_label(rest))
        units += [u for _, u in found]
    unit_names = {u for u in units if u != "t"}
    if unit_names and len({_UNIT_NAME.get(u, u) for u in unit_names}) > 1:
        # Mixed units in one series: put every time in hours.
        conv = {u: _UNIT_HOURS.get(u, 1.0) for u in unit_names}
        time_unit = "h"
    else:
        conv = {u: 1.0 for u in units}
        time_unit = (
            _UNIT_NAME.get(next(iter(unit_names)), "")
            if (unit_names)
            else ("t" if units else "")
        )
    labels = _drop_identifiers(labels)
    labels = _strip_common(labels)
    per_arm: dict[str, set[float]] = {}
    for found, lb in zip(times, labels):
        arm = " ".join(lb)
        # An ordinal time (T3) beside real units is a label token, not a time.
        for v, u in found:
            if u == "t" and unit_names:
                continue
            per_arm.setdefault(arm, set()).add(round(v * conv.get(u, 1.0), 6))
        per_arm.setdefault(arm, set())
    arms = tuple(sorted(per_arm))
    control = next(
        (
            a
            for a in arms
            if any(tok.lower() in CONTROL_WORDS for tok in a.split())
        ),
        None,
    )
    return Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        time_unit=time_unit,
        n_titles=len(titles),
    )


# ── Candidates ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetCandidate:
    """One deposit, enough to decide whether to fetch it."""

    source: str
    accession: str
    title: str
    #: The repository's data type, e.g. "Expression profiling by array".
    kind: str
    organism: str
    n_samples: int
    platform: str
    url: str
    summary: str = ""
    #: Sample titles as deposited; the arms and timepoints are usually in them.
    samples: tuple[str, ...] = ()
    #: The deposit's file names, for a repository that holds files rather
    #: than a series (Zenodo, a paper's supplement); which of them is a
    #: table decides the loader.
    files: tuple[str, ...] = ()
    measured: Measured = field(default_factory=Measured)
    #: The paper the deposit belongs to, when the repository says.
    pubmed: str = ""
    #: Study factors the repository declares (MetaboLights, Metabolomics
    #: Workbench), the conditions in a form the titles may not carry.
    factors: tuple[str, ...] = ()

    @property
    def short_kind(self) -> str:
        """The data type in a word: ``array``, ``rna-seq``, ``methylation``,
        the modality for a non-GEO source, else the type as stated."""
        k = self.kind.lower()
        if k.startswith("expression profiling by array"):
            return "array"
        if k.startswith("expression profiling by high throughput"):
            return "rna-seq"
        if k.startswith("methylation"):
            return "methylation"
        if self.source != "geo" and self.measured.modality != "unknown":
            return self.measured.modality
        return self.kind

    @property
    def series_matrix_has_values(self) -> bool:
        """Array series carry their values in the series matrix; sequencing
        series usually ship counts as supplementary files the loader does
        not read."""
        return self.kind.lower().startswith("expression profiling by array")

    @property
    def design(self) -> Design:
        return parse_design(self.samples)


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
    return geo_summaries(ids, timeout=timeout)


class PageTooLarge(LookupError):
    """E-utilities will not convert a summary page above 10 MB to JSON."""


def _retrying(fetch, tries: int = 4, pause: float = 2.0, fatal=()):
    """Call ``fetch`` again after a transient failure, waiting longer each
    time; the keyless services answer 429 under load. An exception of a
    ``fatal`` type is raised at once."""
    for attempt in range(tries):
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            if isinstance(exc, fatal) or attempt == tries - 1:
                raise
            log.info("retrying after %s", str(exc)[:160])
            time.sleep(pause * (2**attempt))


def _geo_candidates(result: dict) -> list[DatasetCandidate]:
    out = []
    for uid in result.get("uids", []):
        r = result[uid]
        gpl = str(r.get("gpl", ""))
        pubmed = r.get("pubmedids") or []
        files = tuple(
            f.strip() for f in str(r.get("suppfile") or "").split(",") if f
        )
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
                files=files,
                measured=geo_measured(r.get("gdstype", "")),
                pubmed=str(pubmed[0]) if pubmed else "",
            )
        )
    return out


def geo_summaries(uids, *, timeout: float = 30.0) -> list[DatasetCandidate]:
    """The series behind GEO document ids, one esummary call."""
    result = _retrying(
        lambda: _get_json(
            f"{GEO_EUTILS}/esummary.fcgi",
            {"db": "gds", "id": ",".join(uids), "retmode": "json"},
            timeout,
        )
    )["result"]
    return _geo_candidates(result)


def geo_by_accession(
    accession: str, *, timeout: float = 30.0
) -> DatasetCandidate | None:
    """One GEO series by accession, or ``None``."""
    found = _get_json(
        f"{GEO_EUTILS}/esearch.fcgi",
        {
            "db": "gds",
            "term": f"{accession}[ACCN] AND gse[EntryType]",
            "retmode": "json",
        },
        timeout,
    )["esearchresult"]
    ids = found.get("idlist", [])
    if not ids:
        return None
    hits = geo_summaries(ids[:1], timeout=timeout)
    return hits[0] if hits else None


def iter_geo(
    organisms=("Homo sapiens", "Mus musculus"),
    *,
    page: int = 300,
    start: int = 0,
    timeout: float = 60.0,
    pause: float = 0.35,
    on_page=None,
):
    """Every GEO series for ``organisms`` from offset ``start``, through
    the E-utilities history server: one search, then summaries page by
    page at the keyless rate. ``on_page(offset)`` is called after each
    page, with the offset of the next, so a caller can resume. A history
    the server has let go is searched again."""
    term = (
        "("
        + " OR ".join(f'"{o}"[Organism]' for o in organisms)
        + ") AND gse[EntryType]"
    )

    def history() -> dict:
        return _get_json(
            f"{GEO_EUTILS}/esearch.fcgi",
            {
                "db": "gds",
                "term": term,
                "usehistory": "y",
                "retmax": 0,
                "retmode": "json",
            },
            timeout,
        )["esearchresult"]

    def summaries(search: dict, offset: int, size: int) -> dict:
        payload = _get_json(
            f"{GEO_EUTILS}/esummary.fcgi",
            {
                "db": "gds",
                "query_key": search["querykey"],
                "WebEnv": search["webenv"],
                "retstart": offset,
                "retmax": size,
                "retmode": "json",
            },
            timeout,
        )
        if "result" not in payload:
            text = str(payload)[:200]
            if "max size" in text:
                raise PageTooLarge(text)
            raise LookupError(text)
        return payload["result"]

    def fetch(search: dict, offset: int, size: int) -> list:
        try:
            result = _retrying(
                lambda: summaries(search, offset, size), fatal=PageTooLarge
            )
        except PageTooLarge:
            if size == 1:
                raise
            # A page the converter refuses is fetched in quarters; a
            # series with thousands of samples makes one page that big.
            step = max(size // 4, 1)
            out = []
            for sub in range(offset, offset + size, step):
                time.sleep(pause)
                out += fetch(search, sub, min(step, offset + size - sub))
            return out
        return _geo_candidates(result)

    search = history()
    count = int(search.get("count", 0))
    log.info("geo: %d series for %s", count, ", ".join(organisms))
    offset = start
    while offset < count:
        time.sleep(pause)
        try:
            found = fetch(search, offset, page)
        except Exception as exc:  # noqa: BLE001 - the history expired
            log.info("geo: searching again at %d (%s)", offset, str(exc)[:120])
            search = history()
            found = fetch(search, offset, page)
        yield from found
        offset += page
        if on_page is not None:
            on_page(offset)


def search_zenodo(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
) -> list[DatasetCandidate]:
    """Zenodo records of type dataset matching ``query``, through the
    keyless records API. Zenodo has no organism field, so ``organism`` is
    added to the query as a term. Each hit lists its files; a data table
    among them is what the loader can read."""
    q = f"{query} {organism}" if organism else query
    payload = _get_json(
        ZENODO_RECORDS,
        {"q": q, "type": "dataset", "size": limit},
        timeout,
    )
    hits = payload.get("hits", {}).get("hits", [])
    log.info(
        "zenodo '%s': %s hits", q, payload.get("hits", {}).get("total", "?")
    )
    out = []
    for h in hits:
        md = h.get("metadata", {})
        summary = re.sub(r"<[^>]+>", " ", md.get("description", ""))
        out.append(
            DatasetCandidate(
                source="zenodo",
                accession=str(md.get("doi") or h.get("doi") or h.get("id")),
                title=md.get("title", "") or h.get("title", ""),
                kind=(md.get("resource_type") or {}).get("title", "dataset"),
                organism="",
                n_samples=0,
                platform="",
                url=h.get("doi_url")
                or h.get("links", {}).get("html", "")
                or f"https://zenodo.org/records/{h.get('id', '')}",
                summary=" ".join(summary.split())[:400],
                files=tuple(f.get("key", "") for f in h.get("files", [])),
                measured=Measured("table"),
            )
        )
    return out


# ── EBI Search: PRIDE, MetaboLights, Metabolomics Workbench, ArrayExpress


@dataclass(frozen=True)
class EbiDomain:
    """One EBI Search domain: the fields to ask for and how they map."""

    domain: str
    modality: str
    complete: bool
    fields: tuple[str, ...]
    organism_field: str
    url: str
    id_namespace: str = ""


EBI_DOMAINS = {
    "pride": EbiDomain(
        "pride",
        "proteomics",
        True,
        (
            "name",
            "description",
            "species",
            "omics_type",
            "technology_type",
            "PUBMED",
            "publication_date",
        ),
        "species",
        "https://www.ebi.ac.uk/pride/archive/projects/{}",
    ),
    "metabolights": EbiDomain(
        "metabolights",
        "metabolomics",
        False,
        (
            "name",
            "description",
            "organism",
            "study_factor",
            "study_design",
            "technology_type",
            "CHEBI",
            "PUBMED",
        ),
        "organism",
        "https://www.ebi.ac.uk/metabolights/{}",
        "chebi",
    ),
    "metabolomics-workbench": EbiDomain(
        "metabolomics_workbench",
        "metabolomics",
        False,
        ("name", "species", "study_factor", "technology_type"),
        "species",
        "https://www.metabolomicsworkbench.org/data/DRCCMetadata.php"
        "?Mode=Study&StudyID={}",
    ),
    "arrayexpress": EbiDomain(
        "biostudies-arrayexpress",
        "expression",
        True,
        ("name", "description", "organism", "PUBMED"),
        "organism",
        "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{}",
    ),
}


def _first(fields: dict, key: str) -> str:
    v = fields.get(key) or []
    return str(v[0]) if v else ""


def _ebi_candidate(dom: EbiDomain, entry: dict) -> DatasetCandidate:
    f = entry.get("fields", {})
    factors = tuple(
        x for k in ("study_factor", "study_design") for x in f.get(k) or []
    )
    ids = ()
    if dom.id_namespace:
        ids = tuple(
            curie(dom.id_namespace, v)
            for v in f.get(dom.id_namespace.upper()) or []
            if v
        )
    tech = "; ".join(
        x for k in ("omics_type", "technology_type") for x in f.get(k) or []
    )
    return DatasetCandidate(
        source=dom.domain,
        accession=entry.get("id", ""),
        title=_first(f, "name"),
        kind=tech or dom.modality,
        organism="; ".join(f.get(dom.organism_field) or []),
        n_samples=0,
        platform="",
        url=dom.url.format(entry.get("id", "")),
        summary=_first(f, "description")[:400],
        measured=Measured(dom.modality, dom.complete, ids),
        pubmed=_first(f, "PUBMED"),
        factors=factors,
    )


def ebi_page(
    dom: EbiDomain,
    query: str = "*:*",
    *,
    start: int = 0,
    size: int = EBI_PAGE,
    timeout: float = 30.0,
) -> tuple[int, list[DatasetCandidate]]:
    """``(hit count, candidates)`` for one page of an EBI Search domain.
    ``"*:*"`` is every entry, the census's enumeration."""
    payload = _get_json(
        f"{EBI_SEARCH}/{dom.domain}",
        {
            "query": query,
            "fields": ",".join(dom.fields),
            "start": start,
            "size": min(size, EBI_PAGE),
            "format": "json",
        },
        timeout,
    )
    return int(payload.get("hitCount", 0)), [
        _ebi_candidate(dom, e) for e in payload.get("entries", [])
    ]


def iter_ebi(
    dom: EbiDomain,
    query: str = "*:*",
    *,
    start: int = 0,
    timeout: float = 30.0,
    pause: float = 0.2,
    on_page=None,
):
    """Every candidate of a domain from offset ``start``, page by page,
    one pause per page. ``on_page(offset)`` is called after each page with
    the offset of the next, so a caller can resume."""
    while True:
        total, page = _retrying(
            lambda: ebi_page(dom, query, start=start, timeout=timeout)
        )
        yield from page
        start += len(page)
        if on_page is not None:
            on_page(start)
        if not page or start >= total:
            return
        time.sleep(pause)


def _ebi_source(name: str):
    dom = EBI_DOMAINS[name]

    def search(
        query: str,
        limit: int = 25,
        *,
        organism: str | None = None,
        timeout: float = 30.0,
    ) -> list[DatasetCandidate]:
        q = query
        if organism:
            q = f'({query}) AND {dom.organism_field}:"{organism}"'
        total, page = ebi_page(dom, q, size=limit, timeout=timeout)
        log.info("%s '%s': %d hits", name, q, total)
        return page

    search.__name__ = f"search_{name.replace('-', '_')}"
    search.__doc__ = (
        f"{name} entries matching the query, through EBI Search "
        f"({dom.modality})."
    )
    return search


def search_bioimages(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
) -> list[DatasetCandidate]:
    """BioImage Archive studies matching ``query``, through the BioStudies
    search API: live-cell time-lapse and screens, the modality that
    resolves a single cell's trajectory."""
    q = f"{query} {organism}" if organism else query
    payload = _get_json(
        f"{BIOSTUDIES}/search",
        {"query": q, "facet.collection": "BioImages", "pageSize": limit},
        timeout,
    )
    log.info("bioimages '%s': %s hits", q, payload.get("totalHits", "?"))
    return [
        DatasetCandidate(
            source="bioimages",
            accession=h.get("accession", ""),
            title=h.get("title", ""),
            kind="imaging",
            organism="",
            n_samples=0,
            platform="",
            url=f"https://www.ebi.ac.uk/biostudies/BioImages/studies/"
            f"{h.get('accession', '')}",
            summary=(h.get("content") or "")[:400],
            measured=Measured("imaging"),
        )
        for h in payload.get("hits", [])
    ]


SOURCES = {
    "geo": search_geo,
    "zenodo": search_zenodo,
    "pride": _ebi_source("pride"),
    "metabolights": _ebi_source("metabolights"),
    "metabolomics-workbench": _ebi_source("metabolomics-workbench"),
    "arrayexpress": _ebi_source("arrayexpress"),
    "bioimages": search_bioimages,
}


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


# ── A paper's own data, through Europe PMC ──────────────────────────


@dataclass(frozen=True)
class PaperData:
    """What Europe PMC links to a paper: its flags, the datasets it cites
    or deposits (a supplement bundle counts), the chemicals mined from its
    text as ``chebi:`` curies with their names, and the BioModels deposits
    built from it."""

    pubmed: str
    pmcid: str = ""
    has_data: bool = False
    has_supplement: bool = False
    datasets: tuple[DatasetCandidate, ...] = ()
    chemicals: tuple[tuple[str, str], ...] = ()
    biomodels: tuple[str, ...] = ()


_ACCESSION_SOURCE = (
    (re.compile(r"\bGSE\d+\b"), "geo", "expression", True, GEO_ACCESSION_URL),
    (
        re.compile(r"\bPXD\d+\b"),
        "pride",
        "proteomics",
        True,
        EBI_DOMAINS["pride"].url,
    ),
    (
        re.compile(r"\bE-[A-Z]+-\d+\b"),
        "arrayexpress",
        "expression",
        True,
        EBI_DOMAINS["arrayexpress"].url,
    ),
    (
        re.compile(r"\bMTBLS\d+\b"),
        "metabolights",
        "metabolomics",
        False,
        EBI_DOMAINS["metabolights"].url,
    ),
    (
        re.compile(r"\bS-BIAD\d+\b"),
        "bioimages",
        "imaging",
        False,
        "https://www.ebi.ac.uk/biostudies/BioImages/studies/{}",
    ),
    (
        re.compile(r"\bS-EPMC\d+\b"),
        "supplement",
        "supplement",
        False,
        "https://www.ebi.ac.uk/biostudies/studies/{}",
    ),
)


def _links(payload: dict):
    for cat in payload.get("dataLinkList", {}).get("Category", []):
        for sec in cat.get("Section", []):
            for link in sec.get("Linklist", {}).get("Link", []):
                target = link.get("Target", {})
                ident = target.get("Identifier", {}) or {}
                yield (
                    cat.get("Name", ""),
                    str(ident.get("ID", "")),
                    str(target.get("Title", "")),
                )


def supplement_files(accession: str, *, timeout: float = 30.0) -> tuple:
    """The file names of a BioStudies bundle (a paper's supplement)."""
    study = _get_json(f"{BIOSTUDIES}/studies/{accession}", {}, timeout)
    names: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key != "files":
                    walk(value)
                    continue
                for f in value or []:
                    for g in f if isinstance(f, list) else [f]:
                        if isinstance(g, dict):
                            names.append(g.get("path") or g.get("name") or "")
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(study.get("section", study))
    return tuple(n for n in names if n)


def pmc_supplement_files(pmcid: str, *, timeout: float = 120.0) -> tuple:
    """The file names inside a paper's Europe PMC supplementary archive,
    nested archives opened one level; cached, since the archive is the
    whole supplement."""
    import io
    import zipfile

    def fetch():
        request = urllib.request.Request(
            f"{EUROPEPMC}/{pmcid}/supplementaryFiles",
            headers={"User-Agent": "hallsim"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as fh:
            blob = fh.read()
        names = []
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for info in archive.infolist():
                    names.append(info.filename)
                    if not info.filename.lower().endswith(".zip"):
                        continue
                    try:
                        inner = zipfile.ZipFile(io.BytesIO(archive.read(info)))
                    except zipfile.BadZipFile:
                        continue
                    names += [f"{info.filename}/{n}" for n in inner.namelist()]
        except zipfile.BadZipFile:
            return []
        return names

    return tuple(_cached_json(f"pmc supplement {pmcid}", fetch))


def iter_bioimages(*, page: int = 100, timeout: float = 30.0):
    """Every BioImage Archive study, through the BioStudies collection
    facet, page by page."""
    number = 1
    while True:
        payload = _get_json(
            f"{BIOSTUDIES}/search",
            {
                "facet.collection": "BioImages",
                "pageSize": page,
                "page": number,
            },
            timeout,
        )
        hits = payload.get("hits", [])
        for h in hits:
            yield DatasetCandidate(
                source="bioimages",
                accession=h.get("accession", ""),
                title=h.get("title", ""),
                kind="imaging",
                organism="",
                n_samples=0,
                platform="",
                url=f"https://www.ebi.ac.uk/biostudies/BioImages/studies/"
                f"{h.get('accession', '')}",
                summary=(h.get("content") or "")[:400],
                measured=Measured("imaging"),
            )
        total = int(payload.get("totalHits", 0))
        if not hits or number * page >= total:
            return
        number += 1


def paper_data(
    pubmed, *, with_files: bool = False, timeout: float = 30.0
) -> PaperData:
    """Europe PMC's view of one paper: flags from its record, data links."""
    pmid = str(pubmed).strip()
    core = _get_json(
        f"{EUROPEPMC}/search",
        {
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "resultType": "core",
            "format": "json",
        },
        timeout,
    )
    results = core.get("resultList", {}).get("result", [])
    rec = results[0] if results else {}
    links = _get_json(
        f"{EUROPEPMC}/MED/{pmid}/datalinks", {"format": "json"}, timeout
    )
    datasets: dict[str, DatasetCandidate] = {}
    chemicals, biomodels = [], []
    for category, ident, title in _links(links):
        if category.lower().startswith(
            "chemicals"
        ) and ident.upper().startswith("CHEBI"):
            chemicals.append(
                (ident.lower().replace("chebi:", "chebi:"), title)
            )
            continue
        bm = re.search(r"(BIOMD\d{10}|MODEL\d{10})", ident)
        if bm:
            biomodels.append(bm.group(1))
            continue
        for rx, source, modality, complete, url in _ACCESSION_SOURCE:
            m = rx.search(ident) or rx.search(title)
            if not m:
                continue
            acc = m.group(0)
            if acc in datasets:
                break
            files = ()
            if with_files and source == "supplement":
                try:
                    files = supplement_files(acc, timeout=timeout)
                except Exception as exc:  # noqa: BLE001 - a dead bundle
                    log.info("supplement %s: no file list (%s)", acc, exc)
            datasets[acc] = DatasetCandidate(
                source=source,
                accession=acc,
                title=title or category,
                kind=modality,
                organism="",
                n_samples=0,
                platform="",
                url=url.format(acc),
                summary=category,
                files=files,
                measured=Measured(modality, complete),
                pubmed=pmid,
            )
            break
    pmcid = rec.get("pmcid", "")
    has_supplement = rec.get("hasSuppl") == "Y"
    bundled = any(d.source == "supplement" for d in datasets.values())
    if has_supplement and pmcid and not bundled:
        # The supplement exists but no BioStudies bundle is linked: Europe
        # PMC serves it as one archive.
        files = ()
        if with_files:
            try:
                files = pmc_supplement_files(pmcid, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - a dead archive
                log.info("supplement %s: no file list (%s)", pmcid, exc)
        datasets[pmcid] = DatasetCandidate(
            source="supplement",
            accession=pmcid,
            title="supplementary files",
            kind="supplement",
            organism="",
            n_samples=0,
            platform="",
            url=f"{EUROPEPMC}/{pmcid}/supplementaryFiles",
            summary="Europe PMC supplement",
            files=files,
            measured=Measured("supplement"),
            pubmed=pmid,
        )
    return PaperData(
        pubmed=pmid,
        pmcid=pmcid,
        has_data=rec.get("hasData") == "Y",
        has_supplement=has_supplement,
        datasets=tuple(datasets.values()),
        chemicals=tuple(dict.fromkeys(chemicals)),
        biomodels=tuple(dict.fromkeys(biomodels)),
    )


def datasets_of(pubmed, **kw) -> list[DatasetCandidate]:
    """The datasets a paper deposits or cites, its supplement included."""
    return list(paper_data(pubmed, **kw).datasets)


# ── What was perturbed, resolved to a target ────────────────────────


@dataclass(frozen=True)
class Perturbation:
    """A perturbation label resolved structurally: a compound to its ChEBI
    id and, through ChEMBL's mechanisms, the UniProt targets it acts on; a
    gene to its UniProt accession. ``kind`` is ``compound``, ``gene`` or
    ``unresolved``. Targets are ``uniprot:`` curies."""

    label: str
    kind: str = "unresolved"
    name: str = ""
    chebi: str = ""
    targets: tuple[str, ...] = ()
    mechanism: str = ""


_GENE_EDIT = re.compile(
    r"\b(?:si|sh|KO|KD|OE|del|Δ|crispr|knock(?:out|down)|mutant|null|"
    r"overexpress(?:ion|ing)?|-/-|\+/-)\b",
    re.I,
)


def _cache_dir() -> Path:
    path = Path.home() / ".cache" / "hallsim" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached_json(key: str, fetch):
    path = _cache_dir() / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    path.write_text(json.dumps(value))
    return value


def _chebi(label: str, timeout: float) -> tuple[str, str]:
    payload = _cached_json(
        f"ols chebi {label.lower()}",
        lambda: _get_json(
            OLS_SEARCH,
            {"q": label, "ontology": "chebi", "rows": 1, "type": "class"},
            timeout,
        ),
    )
    docs = payload.get("response", {}).get("docs", [])
    if not docs:
        return "", ""
    return docs[0].get("obo_id", "").lower(), docs[0].get("label", "")


def _chembl_targets(label: str, timeout: float) -> tuple[list[str], str]:
    mols = _cached_json(
        f"chembl molecule {label.lower()}",
        lambda: _get_json(
            f"{CHEMBL}/molecule.json",
            {
                "molecule_synonyms__molecule_synonym__iexact": label,
                "limit": 1,
            },
            timeout,
        ),
    ).get("molecules", [])
    if not mols:
        return [], ""
    chembl_id = mols[0]["molecule_chembl_id"]
    mechs = _cached_json(
        f"chembl mechanism {chembl_id}",
        lambda: _get_json(
            f"{CHEMBL}/mechanism.json",
            {"molecule_chembl_id": chembl_id, "limit": 20},
            timeout,
        ),
    ).get("mechanisms", [])
    targets, names = [], []
    for mech in mechs:
        tid = mech.get("target_chembl_id")
        if mech.get("mechanism_of_action"):
            names.append(mech["mechanism_of_action"])
        if not tid:
            continue
        target = _cached_json(
            f"chembl target {tid}",
            lambda tid=tid: _get_json(
                f"{CHEMBL}/target/{tid}.json", {}, timeout
            ),
        )
        for comp in target.get("target_components", []):
            acc = comp.get("accession")
            if acc:
                targets.append(f"uniprot:{acc}")
    return list(dict.fromkeys(targets)), "; ".join(dict.fromkeys(names))


def _gene_uniprot(symbol: str, timeout: float) -> list[str]:
    payload = _cached_json(
        f"mygene {symbol.upper()}",
        lambda: _get_json(
            MYGENE_QUERY,
            {
                "q": f"symbol:{symbol}",
                "species": "human,mouse",
                "fields": "uniprot.Swiss-Prot",
                "size": 2,
            },
            timeout,
        ),
    )
    out = []
    for hit in payload.get("hits", []):
        sp = (hit.get("uniprot") or {}).get("Swiss-Prot")
        for acc in [sp] if isinstance(sp, str) else sp or []:
            out.append(f"uniprot:{acc}")
    return out


def resolve_perturbation(label: str, *, timeout: float = 30.0) -> Perturbation:
    """Resolve an arm label or a chemical name. Every lookup is keyless
    and cached; a label nothing resolves is returned ``unresolved``."""
    text = label.strip()
    if not text:
        return Perturbation(label)
    tokens = [t for t in _SEP.split(text) if t]
    if _GENE_EDIT.search(text) or len(tokens) == 1:
        for tok in tokens:
            sym = tok.upper()
            if SYMBOL.match(sym) and sym.lower() not in CONTROL_WORDS:
                try:
                    accs = _gene_uniprot(sym, timeout)
                except Exception as exc:  # noqa: BLE001 - a dead service
                    log.info("mygene %s: %s", sym, exc)
                    accs = []
                if accs and _GENE_EDIT.search(text):
                    return Perturbation(
                        label, "gene", sym, targets=tuple(accs)
                    )
    name = " ".join(tokens)
    try:
        chebi, chebi_name = _chebi(name, timeout)
        targets, mechanism = (
            _chembl_targets(name, timeout)
            if chebi
            else (
                [],
                "",
            )
        )
    except Exception as exc:  # noqa: BLE001 - a dead service
        log.info("perturbation %r: %s", label, exc)
        return Perturbation(label)
    if not chebi:
        return Perturbation(label)
    return Perturbation(
        label, "compound", chebi_name, chebi, tuple(targets), mechanism
    )


# ── Coverage of a composite ─────────────────────────────────────────


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
    from hallsim.reporter_wiring import store_ontology_map, tf_observables

    ontmap = store_ontology_map(composite) if ontmap is None else ontmap
    m = candidate.measured
    measured_ids = {i.lower() for i in m.ids}

    def with_ns(ns: str) -> list[str]:
        return sorted(p for p, o in ontmap.items() if ns in o)

    def listed(ns: str) -> list[str]:
        return sorted(
            p
            for p, o in ontmap.items()
            if ns in o and curie(ns, o[ns]).lower() in measured_ids
        )

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


# ── The expression loader's check ───────────────────────────────────


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
