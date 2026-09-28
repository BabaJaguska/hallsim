"""Where the data is: dataset search, mirrored on :mod:`hallsim.search.models`.

A calibration needs a time course of quantities a model carries, taken
under conditions it can represent. :func:`search_for_dataset` asks the
repositories: GEO and Zenodo, and through EBI Search PRIDE, MetaboLights,
Metabolomics Workbench and ArrayExpress, with the BioImage Archive through
BioStudies; NASA's OSDR is enumerated whole. Each hit says what kind of
data it is, what it measured (:class:`Measured`) and how its samples are
arranged (:class:`Design`, read from the sample titles by
:func:`parse_design`: arms, a control arm when one is named, timepoints).
:func:`datasets_of` lists a paper's own data through Europe PMC, which is
the quantity its model was built to predict. :func:`resolve_perturbation`
turns an arm label into a ChEBI id and its targets, or a gene into its
UniProt accession. :func:`platform_head` reads an expression platform's
table head so a hit can be checked against a loader before anything
large is downloaded.
"""

from __future__ import annotations


import gzip
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import ppx

from hallsim.search.fetch import (
    USER_AGENT,
    cache_dir,
    cached_index,
    cached_json,
    get_json,
    is_junk_archive_entry,
    retrying,
)

log = logging.getLogger(__name__)

#: NCBI asks every E-utilities client for a contact address and offers a
#: higher rate to a registered key; both are read from the environment.
NCBI_EMAIL = "hallsim-search@users.noreply.github.com"
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
    #: Distinct subjects behind the samples, where the deposit names them.
    #: Two samples from one animal are one observation, so replication is
    #: counted here and not in the sample count. Zero means unstated.
    n_subjects: int = 0

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

    def to_dict(self) -> dict:
        return {
            "arms": list(self.arms),
            "control": self.control,
            "per_arm": [[a, list(ts)] for a, ts in self.per_arm],
            "time_unit": self.time_unit,
            "n_titles": self.n_titles,
            "n_subjects": self.n_subjects,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        return cls(
            arms=tuple(d.get("arms", ())),
            control=d.get("control"),
            per_arm=tuple(
                (a, tuple(float(t) for t in ts))
                for a, ts in d.get("per_arm", ())
            ),
            time_unit=d.get("time_unit", ""),
            n_titles=int(d.get("n_titles", 0)),
            n_subjects=int(d.get("n_subjects", 0)),
        )


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
_ARM_TOKEN = re.compile(r"[\s/_-]+")

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


def reference_arm(
    arms: tuple[str, ...], values: dict[str, set] | None = None
) -> str | None:
    """Which arm the others are measured against, or ``None``.

    Three signals, strongest first: an arm naming itself a reference, an arm
    holding a zero level of what the others hold, and an arm whose label is
    contained in another's. ``None`` means no arm said — not that there was no
    reference, since an old against young contrast declares neither.
    """
    arms = tuple(arms)
    if len(arms) < 2:
        return None
    parts = {a: [t.lower() for t in _ARM_TOKEN.split(a) if t] for a in arms}
    named = next(
        (a for a in arms if any(t in CONTROL_WORDS for t in parts[a])), None
    )
    if named is not None:
        return named
    if values:
        zeroed = [a for a in arms if values.get(a) == {0.0}]
        if len(zeroed) == 1:
            return zeroed[0]
    tokens = {a: frozenset(parts[a]) for a in arms}
    contained = [
        a
        for a in arms
        if tokens[a] and any(tokens[a] < tokens[b] for b in arms if b != a)
    ]
    return contained[0] if len(contained) == 1 else None


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


#: Whether a declared study factor is a time axis, by its name.
TIME_FACTOR = re.compile(
    r"\b(time|timepoint|time[- ]?point|time[- ]?course|hour|day|week|"
    r"duration|age)s?\b",
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
    control = reference_arm(arms, per_arm)
    return Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        time_unit=time_unit,
        n_titles=len(titles),
    )


#: A unit word inside a column header: ``Time (min)``, ``time_h``.
_UNIT_WORD = re.compile(
    r"(?<![A-Za-z])(sec|s|min|hrs?|h|hours?|days?|d|wks?|w|weeks?|mo|"
    r"months?|yrs?|y|years?)(?![A-Za-z])",
    re.I,
)
#: Files the table reader opens.
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".tab", ".dat", ".xlsx", ".xls")


def read_table(path, *, n_rows: int = 5000) -> "pd.DataFrame | None":
    """A delimited or spreadsheet file as a frame, the delimiter sniffed;
    ``None`` when it is not a table."""
    path = str(path)
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path, nrows=n_rows)
        frame = pd.read_csv(
            path,
            sep=None,
            engine="python",
            nrows=n_rows,
            comment="#",
            encoding_errors="replace",
        )
    except Exception as exc:  # noqa: BLE001 - not a table
        log.info("%s: not read as a table (%s)", path, exc)
        return None
    return frame if frame.shape[1] >= 2 else None


def design_of_table(frame) -> Design | None:
    """The design a long-form table states: its time column gives the
    timepoints, its text columns with a few distinct values give the arms.
    ``None`` when no column reads as time. A wide table — time down the
    first column, one series per column — is one unperturbed arm."""
    if frame is None or frame.empty:
        return None
    n = len(frame)
    time_col = None
    for col in frame.columns:
        # ``time_min`` and ``Time (h)`` both name a time axis; the
        # underscore is a word character, so it is read as a space.
        head = re.sub(r"[_\-]+", " ", str(col)).strip()
        if not (TIME_FACTOR.search(head) or head.lower() == "t"):
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() >= max(2, n // 2) and values.nunique() >= 2:
            time_col = col
            break
    if time_col is None:
        return None
    times = pd.to_numeric(frame[time_col], errors="coerce")
    unit = ""
    if m := _UNIT_WORD.search(re.sub(r"[_\-]+", " ", str(time_col))):
        unit = _UNIT_NAME.get(m.group(1).lower().rstrip("s"), "") or (
            _UNIT_NAME.get(m.group(1).lower(), "")
        )
    arm_cols = [
        c
        for c in frame.columns
        if c != time_col
        and not pd.api.types.is_numeric_dtype(frame[c])
        and 2 <= frame[c].nunique(dropna=True) <= max(2, n // 2)
    ]
    labels = (
        frame[arm_cols].astype(str).agg(" ".join, axis=1)
        if arm_cols
        else pd.Series([""] * n, index=frame.index)
    )
    per_arm: dict[str, set] = {}
    for arm, t_val in zip(labels, times):
        per_arm.setdefault(arm, set())
        if pd.notna(t_val):
            per_arm[arm].add(round(float(t_val), 6))
    arms = tuple(sorted(per_arm))
    control = reference_arm(arms, per_arm)
    return Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        time_unit=unit,
        n_titles=n,
    )


def table_design(paths) -> Design | None:
    """The richest design among ``paths`` read as tables: the one with the
    most timepoints, arms breaking the tie."""
    best = None
    for path in paths:
        design = design_of_table(read_table(path))
        if design is None:
            continue
        key = (design.n_timepoints, len(design.arms))
        if best is None or key > (best.n_timepoints, len(best.arms)):
            best = design
    return best


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
    #: The accession this deposit mirrors, where it declares one. OSDR
    #: re-hosts studies from GEO and ArrayExpress and names the original.
    mirrors: str = ""
    measured: Measured = field(default_factory=Measured)
    #: The paper the deposit belongs to, when the repository says.
    pubmed: str = ""
    #: Study factors the repository declares (MetaboLights, Metabolomics
    #: Workbench), the conditions in a form the titles may not carry.
    factors: tuple[str, ...] = ()
    #: The design as the source states it — a measurement table, a curated
    #: subset listing, an assay-group summary — where a source does not
    #: leave it to the sample titles.
    stated: Design | None = None
    #: The curated record the design was read from, when it is not the
    #: deposit itself: a GEO DataSet over a series, an Expression Atlas
    #: experiment over an ArrayExpress or GEO deposit.
    curated: str = ""

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
        if self.stated is not None:
            return self.stated
        return parse_design(self.samples)


def _entrez():
    """Biopython's E-utilities client, identified as NCBI asks."""
    from Bio import Entrez

    Entrez.email = os.environ.get("NCBI_EMAIL", NCBI_EMAIL)
    Entrez.tool = USER_AGENT
    key = os.environ.get("NCBI_API_KEY")
    if key:
        Entrez.api_key = key
    return Entrez


def esearch(
    term: str, *, db: str = "gds", history: bool = False, retmax: int = 25
) -> dict:
    """An E-utilities search, parsed: ``Count``, ``IdList`` and, with
    ``history``, the ``WebEnv`` and ``QueryKey`` that page it."""
    Entrez = _entrez()

    def call():
        with Entrez.esearch(
            db=db, term=term, retmax=retmax, usehistory="y" if history else "n"
        ) as handle:
            return Entrez.read(handle)

    return retrying(call)


def esummary(
    *,
    db: str = "gds",
    ids=None,
    webenv: str | None = None,
    query_key: str | None = None,
    retstart: int = 0,
    retmax: int = 300,
) -> list[dict]:
    """E-utilities document summaries, by id or by history page. Pacing
    to NCBI's limit and the XML parsing are Biopython's."""
    Entrez = _entrez()
    kw: dict = {"db": db, "retstart": retstart, "retmax": retmax}
    if ids is not None:
        kw["id"] = ",".join(str(i) for i in ids)
    else:
        kw.update(WebEnv=webenv, query_key=query_key)

    def call():
        with Entrez.esummary(**kw) as handle:
            return list(Entrez.read(handle))

    return retrying(call)


def _geo_candidates(records) -> list[DatasetCandidate]:
    out = []
    for r in records:
        gpl = str(r.get("GPL") or "")
        pubmed = [str(int(p)) for p in (r.get("PubMedIds") or [])]
        files = tuple(
            f.strip() for f in str(r.get("suppFile") or "").split(",") if f
        )
        kind = str(r.get("gdsType") or "")
        out.append(
            DatasetCandidate(
                source="geo",
                accession=str(r["Accession"]),
                title=str(r.get("title") or ""),
                kind=kind,
                organism=str(r.get("taxon") or ""),
                n_samples=int(r.get("n_samples") or 0),
                platform=";".join(f"GPL{g}" for g in gpl.split(";") if g),
                url=GEO_ACCESSION_URL.format(r["Accession"]),
                summary=str(r.get("summary") or ""),
                samples=tuple(
                    str(s.get("Title") or "") for s in (r.get("Samples") or [])
                ),
                files=files,
                measured=geo_measured(kind),
                pubmed=pubmed[0] if pubmed else "",
            )
        )
    return out


def search_geo(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
) -> list[DatasetCandidate]:
    """GEO series matching ``query`` (all fields), newest first."""
    term = f"({query}) AND gse[EntryType]"
    if organism:
        term += f' AND "{organism}"[Organism]'
    found = esearch(term, retmax=limit)
    ids = list(found.get("IdList", []))
    log.info("geo '%s': %s hits", query, found.get("Count", "?"))
    if not ids:
        return []
    return geo_summaries(ids, timeout=timeout)


def geo_summaries(uids, *, timeout: float = 30.0) -> list[DatasetCandidate]:
    """The series behind GEO document ids, one summary call."""
    return _geo_candidates(esummary(ids=uids))


def geo_by_accession(
    accession: str, *, timeout: float = 30.0
) -> DatasetCandidate | None:
    """One GEO series by accession, or ``None``."""
    found = esearch(f"{accession}[ACCN] AND gse[EntryType]", retmax=1)
    ids = list(found.get("IdList", []))
    if not ids:
        return None
    hits = geo_summaries(ids[:1], timeout=timeout)
    return hits[0] if hits else None


def iter_geo(
    organisms=("Homo sapiens", "Mus musculus"),
    *,
    entry: str = "gse",
    page: int = 300,
    start: int = 0,
    timeout: float = 60.0,
    on_page=None,
    designs: bool = True,
):
    """Every GEO series (``entry="gse"``) or curated DataSet (``"gds"``)
    for ``organisms`` from offset ``start``, through the E-utilities
    history server: one search, then summaries page by page at the rate
    Biopython keeps. ``on_page(offset)`` is called after each page, with
    the offset of the next, so a caller can resume. A history the server
    has let go is searched again. A DataSet with a time subset gets its
    design read from its SOFT header when ``designs``."""
    term = (
        "("
        + " OR ".join(f'"{o}"[Organism]' for o in organisms)
        + f") AND {entry}[EntryType]"
    )
    shape = (
        _geo_candidates
        if entry == "gse"
        else lambda records: _gds_candidates(
            records, designs=designs, timeout=timeout
        )
    )

    def history() -> dict:
        return esearch(term, history=True, retmax=0)

    def fetch(search: dict, offset: int) -> list:
        return shape(
            esummary(
                webenv=str(search["WebEnv"]),
                query_key=str(search["QueryKey"]),
                retstart=offset,
                retmax=page,
            )
        )

    search = history()
    count = int(search.get("Count", 0))
    log.info("geo: %d %s for %s", count, entry, ", ".join(organisms))
    offset = start
    while offset < count:
        try:
            found = fetch(search, offset)
        except Exception as exc:  # noqa: BLE001 - the history expired
            log.info("geo: searching again at %d (%s)", offset, str(exc)[:120])
            search = history()
            found = fetch(search, offset)
        yield from found
        offset += page
        if on_page is not None:
            on_page(offset)


GEO_DATASETS = "https://ftp.ncbi.nlm.nih.gov/geo/datasets"
GDS_URL = "https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={}"


def gds_subsets(
    accession: str, *, timeout: float = 60.0
) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(type, description, sample ids)`` for every subset a GEO DataSet
    declares, read from the head of its SOFT file and stopped before the
    table, so it costs kilobytes."""
    base = f"{GEO_DATASETS}/{accession[:3]}{accession[3:-3]}nnn/{accession}"
    url = f"{base}/soft/{accession}.soft.gz"
    out: list[tuple[str, str, tuple[str, ...]]] = []
    current: dict = {}

    def flush():
        if current.get("type"):
            out.append(
                (
                    current["type"],
                    current.get("description", ""),
                    tuple(current.get("samples", ())),
                )
            )

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with gzip.open(resp, "rt", errors="replace") as fh:
            for line in fh:
                if line.startswith("!dataset_table_begin"):
                    break
                if line.startswith("^SUBSET"):
                    flush()
                    current = {}
                elif line.startswith("!subset_type"):
                    current["type"] = line.split("=", 1)[1].strip().lower()
                elif line.startswith("!subset_description"):
                    current["description"] = line.split("=", 1)[1].strip()
                elif line.startswith("!subset_sample_id"):
                    current["samples"] = [
                        s.strip()
                        for s in line.split("=", 1)[1].split(",")
                        if s.strip()
                    ]
    flush()
    return out


def gds_design(subsets) -> Design:
    """The design a DataSet's subsets state: time subsets give the
    timepoints, the other subset types give the arms, and sample
    membership joins them. A time-typed subset with no number in it —
    "young", "middle age" — is a group, not a point on an axis."""
    times: dict[str, tuple[float, str]] = {}
    arm_of: dict[str, list[str]] = {}
    for kind, description, samples in subsets:
        found = []
        if TIME_FACTOR.search(kind):
            found, _ = _times(description)
            if not found:
                # A description with no unit: "0", "6", "24" are still an
                # ordered axis, read as unitless ordinals.
                m = re.search(r"\d+(?:[.,]\d+)?", description)
                if m:
                    found = [(float(m.group(0).replace(",", ".")), "")]
        if found:
            for s in samples:
                times[s] = found[0]
            continue
        for s in samples:
            arm_of.setdefault(s, []).append(description)
    units = {u for _, u in times.values() if u}
    conv = {u: 1.0 for u in units}
    time_unit = ""
    if len({_UNIT_NAME.get(u, u) for u in units}) > 1:
        conv = {u: _UNIT_HOURS.get(u, 1.0) for u in units}
        time_unit = "h"
    elif units:
        time_unit = _UNIT_NAME.get(next(iter(units)), "")
    samples = set(times) | set(arm_of)
    per_arm: dict[str, set] = {}
    for s in samples:
        arm = " ".join(arm_of.get(s, []))
        per_arm.setdefault(arm, set())
        if s in times:
            v, u = times[s]
            per_arm[arm].add(round(v * conv.get(u, 1.0), 6))
    arms = tuple(sorted(per_arm))
    control = reference_arm(arms, per_arm)
    return Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        time_unit=time_unit,
        n_titles=len(samples),
    )


def _gds_candidates(
    records, *, designs: bool = True, timeout: float = 60.0
) -> list[DatasetCandidate]:
    """DataSet summaries as candidates for the series they curate. The
    candidate *is* the series — that is where the data lives — with the
    DataSet's subset types as factors and the design its subsets state.

    Subsets are read for every typed DataSet, not only one typed ``time``. A
    curator's grouping is the best statement of a design in the corpus, and on
    an agent or genotype subset it is what names the control arm; skipping
    those left the control to be guessed from sample titles.
    """
    out = []
    for r in records:
        gds = str(r.get("Accession") or "")
        gse = str(r.get("GSE") or "").split(";")[0]
        if not gse:
            continue
        types = tuple(
            dict.fromkeys(
                s.strip().lower()
                for s in str(r.get("SSInfo") or "").split(";")
                if s.strip()
            )
        )
        stated = None
        if designs and types:
            try:
                _geo_throttle()
                stated = gds_design(gds_subsets(gds, timeout=timeout))
            except Exception as exc:  # noqa: BLE001 - the summary stands
                log.info("%s: subsets not read (%s)", gds, exc)
        pubmed = [str(int(p)) for p in (r.get("PubMedIds") or [])]
        gpl = str(r.get("GPL") or "")
        kind = str(r.get("gdsType") or "")
        out.append(
            DatasetCandidate(
                source="geo",
                accession=f"GSE{gse}",
                title=str(r.get("title") or ""),
                kind=kind,
                organism=str(r.get("taxon") or ""),
                n_samples=int(r.get("n_samples") or 0),
                platform=";".join(f"GPL{g}" for g in gpl.split(";") if g),
                url=GDS_URL.format(gds),
                summary=str(r.get("summary") or ""),
                measured=geo_measured(kind),
                pubmed=pubmed[0] if pubmed else "",
                factors=types,
                stated=stated,
                curated=gds,
            )
        )
    return out


def search_geo_datasets(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
) -> list[DatasetCandidate]:
    """GEO's curated DataSets matching ``query``: each is a series whose
    subsets a curator has typed — ``time``, ``agent``, ``infection`` — so
    a time course is declared rather than read from sample titles."""
    term = f"({query}) AND gds[EntryType]"
    if organism:
        term += f' AND "{organism}"[Organism]'
    found = esearch(term, retmax=limit)
    ids = list(found.get("IdList", []))
    log.info("geo datasets '%s': %s hits", query, found.get("Count", "?"))
    if not ids:
        return []
    return _gds_candidates(esummary(ids=ids), timeout=timeout)


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
    payload = get_json(
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
                summary=" ".join(summary.split()),
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
        summary=_first(f, "description"),
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
    payload = get_json(
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
        total, page = retrying(
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


# ── NASA Open Science Data Repository (ex-GeneLab) ───────────────────

OSDR_SEARCH = "https://osdr.nasa.gov/osdr/data/search"
OSDR_FILES = "https://osdr.nasa.gov/osdr/data/osd/files"
OSDR_DOWNLOAD = "https://osdr.nasa.gov/geode-py/ws/studies/{}/download"
#: OSDR's assay measurement types by the quantity they measure. A study
#: running several assays joins their types with a run of spaces.
OSDR_MODALITY = (
    ("transcription profiling", "expression", True),
    ("protein expression profiling", "proteomics", True),
    ("metabolite profiling", "metabolomics", False),
    ("dna methylation profiling", "methylation", True),
    ("genome sequencing", "genotype", True),
    ("amplicon sequencing", "other", False),
    ("metagenomic sequencing", "other", False),
)
#: GeneLab's standardised bulk RNA-seq pipeline writes the same file names
#: for every study it processes, so its output has a fixed layout — unlike
#: an author's upload.
#: GeneLab's standardised bulk RNA-seq pipeline writes the same file names
#: for every study it processes. Whether a study ships normalised counts
#: varies by processing vintage — the brain studies ship only unnormalised —
#: so readability is decided by :data:`GL_COUNTS` and the normalisation the
#: reader applies by :data:`GL_NORMALIZED`.
GL_COUNTS = re.compile(r"_Counts(_[A-Za-z]+)?_GLbulkRNAseq\.csv$", re.I)
#: "Unnormalized" contains "normalized", so the boundary is load-bearing.
GL_NORMALIZED = re.compile(r"(?<![A-Za-z])Normalized_Counts.*\.csv$", re.I)
_OSDR_SPLIT = re.compile(r"\s{2,}")


def osdr_measured(measurement: str) -> Measured:
    """What an OSDR study measured, from its assay measurement type. A
    multi-assay study is named by the first type a reader could use."""
    parts = [p.strip().lower() for p in _OSDR_SPLIT.split(measurement or "")]
    for prefix, modality, complete in OSDR_MODALITY:
        if any(p.startswith(prefix) for p in parts):
            return Measured(modality, complete)
    return Measured("unknown", False)


def _osdr_candidate(entry: dict) -> DatasetCandidate:
    src = entry.get("_source", entry)
    accession = str(src.get("Accession") or "").strip()
    factors = tuple(
        f.strip()
        for f in _OSDR_SPLIT.split(str(src.get("Study Factor Name") or ""))
        if f.strip()
    )
    # Ground against flight is the contrast the programme cares about and
    # the repository states it per study, so it is a factor like any other.
    project = str(src.get("Project Type") or "").strip()
    if project:
        factors = factors + (f"Project:{project}",)
    return DatasetCandidate(
        source="osdr",
        accession=accession,
        title=str(src.get("Study Title") or ""),
        kind=str(src.get("Study Assay Technology Type") or ""),
        organism=str(src.get("organism") or ""),
        n_samples=0,
        platform=str(src.get("Study Assay Technology Platform") or ""),
        url=f"https://osdr.nasa.gov/bio/repo/data/studies/{accession}",
        summary=str(src.get("Study Description") or ""),
        factors=factors,
        mirrors=str(src.get("Data Source Accession") or "").strip(),
        measured=osdr_measured(str(src.get("Study Assay Measurement Type"))),
    )


def osdr_page(
    term: str = "",
    *,
    start: int = 0,
    size: int = 100,
    timeout: float = 90.0,
) -> tuple[int, list[DatasetCandidate]]:
    """``(hit count, candidates)`` for one page of OSDR's study index."""
    payload = get_json(
        OSDR_SEARCH,
        {
            "term": term,
            "from": start,
            "size": size,
            "type": "cgene",
        },
        timeout,
    )
    hits = payload.get("hits", {})
    return int(hits.get("total") or 0), [
        _osdr_candidate(h) for h in hits.get("hits", [])
    ]


def iter_osdr(*, page: int = 100, timeout: float = 90.0):
    """Every OSDR study, page by page. The corpus is small enough to
    enumerate whole, so no query narrows it."""
    start, total = 0, None
    while total is None or start < total:
        total, cands = osdr_page(start=start, size=page, timeout=timeout)
        if not cands:
            break
        for cand in cands:
            yield cand
        start += len(cands)


def osdr_files(accession: str, *, timeout: float = 90.0) -> tuple[str, ...]:
    """Every file name an OSDR study holds."""
    payload = get_json(
        f"{OSDR_FILES}/{accession.replace('OSD-', '')}", {}, timeout
    )
    out: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("file_name"), str):
                out.append(node["file_name"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return tuple(dict.fromkeys(out))


def _walk_paths(node, key: str, out: list) -> None:
    if isinstance(node, dict):
        if isinstance(node.get(key), str):
            out.append(node[key])
        for value in node.values():
            _walk_paths(value, key, out)
    elif isinstance(node, list):
        for item in node:
            _walk_paths(item, key, out)


def biostudies_study(accession: str, *, timeout: float = 60.0) -> dict:
    """A BioStudies study record, kept on disk: the file list and the
    declared factors both come from it."""
    return cached_json(
        f"study biostudies {accession}",
        lambda: get_json(f"{BIOSTUDIES}/studies/{accession}", {}, timeout),
    )


def biostudies_files(
    accession: str, *, timeout: float = 60.0
) -> tuple[str, ...]:
    """Every file an ArrayExpress or BioStudies deposit holds."""
    out: list[str] = []
    _walk_paths(biostudies_study(accession, timeout=timeout), "path", out)
    return tuple(dict.fromkeys(out))


def _walk_attributes(node, out: list) -> None:
    if isinstance(node, dict):
        name = str(node.get("name") or "").lower()
        if name.startswith(("experimental factor", "experimental design")):
            value = str(node.get("value") or "").strip()
            if value:
                out.append(value)
        for value in node.values():
            _walk_attributes(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk_attributes(item, out)


def biostudies_factors(
    accession: str, *, timeout: float = 60.0
) -> tuple[str, ...]:
    """The experimental factors and designs an ArrayExpress study declares
    (``time``, ``compound``, ``time series design``), from the attributes
    of its BioStudies record. The EBI Search index carries none of them,
    so this is one request per study."""
    out: list[str] = []
    _walk_attributes(biostudies_study(accession, timeout=timeout), out)
    return tuple(dict.fromkeys(out))


def pride_files(accession: str, *, timeout: float = 60.0) -> tuple[str, ...]:
    """Every file a PRIDE project holds, listed through ``ppx``."""
    project = ppx.find_project(accession, local=cache_dir("ppx") / accession)
    return tuple(dict.fromkeys(str(f) for f in project.remote_files()))


#: GEO's E-utilities summary names a series' supplementary files by type
#: token alone — ``TXT``, ``CSV``, ``MTX``, ``COUNTS`` — never by name.
GEO_FILE_TOKEN = re.compile(r"^[A-Z0-9_]+$")
_SUPPLEMENTARY_FILE = re.compile(
    r"^!Series_supplementary_file\s*=\s*(\S+)", re.M
)
#: NCBI asks for no more than three requests a second without an API key;
#: the throttle is shared across threads.
_GEO_INTERVAL = 1 / 3
_geo_clock = {"last": 0.0, "lock": threading.Lock()}


def _geo_throttle() -> None:
    with _geo_clock["lock"]:
        wait = _geo_clock["last"] + _GEO_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _geo_clock["last"] = time.monotonic()


def supplementary_names(soft: str) -> tuple[str, ...]:
    """The file names a series' brief SOFT record lists, in order. Anything
    that is not a series record — GEO answers an unknown accession with an
    HTML page and status 200 — is refused rather than read as "no files"."""
    if "^SERIES" not in soft:
        raise LookupError("not a GEO series record")
    names = [u.rsplit("/", 1)[-1] for u in _SUPPLEMENTARY_FILE.findall(soft)]
    return tuple(dict.fromkeys(names))


def geo_files(accession: str, *, timeout: float = 30.0) -> tuple[str, ...]:
    """The supplementary files a GEO series ships, by name.

    The E-utilities summary carries only type tokens, so a counts table is
    indistinguishable from a differential-expression table there. The
    series' own brief SOFT record names every file in a few kilobytes. The
    FTP mirror's ``filelist.txt`` would be lighter still but exists only
    where a series ships an archive.
    """
    url = GEO_ACCESSION_URL.format(accession) + (
        "&targ=self&form=text&view=brief"
    )

    def fetch():
        _geo_throttle()
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout) as fh:
            return supplementary_names(fh.read().decode("utf-8", "replace"))

    return retrying(fetch, tries=3, fatal=(LookupError,))


_FILE_LISTERS = {
    "biostudies-arrayexpress": biostudies_files,
    "pride": pride_files,
    "osdr": osdr_files,
    "geo": geo_files,
}

#: Sources stating each sample's factors as fields rather than in its title.
#: Both spellings of the Workbench appear, by whether the row came through
#: OmicsDI or the source directly. Reading one is
#: :func:`hallsim.metabolites.declared_design`, which lives there because it
#: needs the ISA-Tab and mwTab readers and this package imports nothing from
#: the rest of hallsim.
DESIGN_SOURCES = frozenset(
    {"metabolights", "metabolomics-workbench", "metabolomics_workbench"}
)


def study_files(cand: "DatasetCandidate", *, timeout: float = 15.0) -> tuple:
    """A deposit's file list from its own repository, for the sources
    whose loader the file list decides. Enumeration metadata carries none,
    so this is one request per deposit.

    Each list is written to the on-disk cache as it arrives, so a pass over
    thousands of deposits that is stopped part-way keeps what it fetched
    and a rerun asks only for the rest. The timeout is short on purpose: a
    hung endpoint should fail the one deposit, not stall the pass.
    """
    fetch = _FILE_LISTERS.get(cand.source)
    if fetch is None:
        return cand.files
    listed = cached_json(
        f"files {cand.source} {cand.accession}",
        lambda: list(fetch(cand.accession, timeout=timeout)),
    )
    return tuple(listed)


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
    payload = get_json(
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
            summary=h.get("content") or "",
            measured=Measured("imaging"),
        )
        for h in payload.get("hits", [])
    ]


ATLAS = "https://www.ebi.ac.uk/gxa"
_ATLAS_CONTRAST = re.compile(
    r"^'(?P<test>.+?)' vs '(?P<ref>.+?)'"
    r"(?: in '(?P<context>.+?)')?(?: at '(?P<time>.+?)')?$"
)
_ATLAS_ACCESSION = re.compile(r"^E-GEOD-(\d+)$")


def atlas_experiments(refresh: bool = False) -> list[dict]:
    """Every Expression Atlas experiment — accession, description, species,
    technology, declared factors — from the one listing the site serves,
    cached like a repository index."""
    return cached_index(
        "expression-atlas",
        lambda: get_json(f"{ATLAS}/json/experiments", {}, 120.0).get(
            "experiments", []
        ),
        refresh=refresh,
    )


def atlas_experiment(accession: str, *, timeout: float = 60.0) -> dict:
    """One experiment's record with its assay groups or contrasts."""
    return cached_json(
        f"atlas experiment {accession}",
        lambda: get_json(f"{ATLAS}/json/experiments/{accession}", {}, timeout),
    )


def atlas_design(record: dict) -> Design | None:
    """The design an Atlas experiment states. A baseline experiment lists
    assay groups with their factor values; a differential one lists
    contrasts as ``'test' vs 'reference' in 'context' at 'time'``. Either
    way the time-like factor gives the timepoints and the rest the arms."""
    groups: list[tuple[str, list[tuple[float, str]], int]] = []
    for header in record.get("columnHeaders") or []:
        summary = header.get("assayGroupSummary")
        if summary is not None:
            arm, found = [], []
            for prop in summary.get("properties") or []:
                if prop.get("contrastPropertyType") != "FACTOR":
                    continue
                name = str(prop.get("propertyName") or "")
                value = str(prop.get("testValue") or "")
                if TIME_FACTOR.search(name):
                    times, _ = _times(value)
                    found += times
                else:
                    arm.append(value)
            groups.append(
                (" ".join(arm), found, int(summary.get("replicates") or 1))
            )
            continue
        m = _ATLAS_CONTRAST.match(str(header.get("displayName") or ""))
        if not m:
            continue
        found, _ = _times(m.group("time") or "")
        context = m.group("context") or ""
        for side, key in (
            ("test", "testAssayGroup"),
            ("ref", "referenceAssayGroup"),
        ):
            n = int((header.get(key) or {}).get("replicates") or 1)
            groups.append((f"{m.group(side)} {context}".strip(), found, n))
    if not groups:
        return None
    units = {u for _, found, _ in groups for _, u in found if u}
    conv = {u: 1.0 for u in units}
    time_unit = ""
    if len({_UNIT_NAME.get(u, u) for u in units}) > 1:
        conv = {u: _UNIT_HOURS.get(u, 1.0) for u in units}
        time_unit = "h"
    elif units:
        time_unit = _UNIT_NAME.get(next(iter(units)), "")
    per_arm: dict[str, set] = {}
    n_titles = 0
    for arm, found, n in groups:
        per_arm.setdefault(arm, set())
        for v, u in found:
            per_arm[arm].add(round(v * conv.get(u, 1.0), 6))
        n_titles += n
    arms = tuple(sorted(per_arm))
    control = reference_arm(arms, per_arm)
    return Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        time_unit=time_unit,
        n_titles=n_titles,
    )


def _atlas_candidate(
    rec: dict, *, designs: bool = True, timeout: float = 60.0
) -> DatasetCandidate:
    """An Atlas experiment as a candidate for the deposit it curates: an
    ``E-GEOD`` accession is the GEO series, an ``E-MTAB`` the ArrayExpress
    deposit, anything else stays the Atlas's own."""
    acc = str(rec.get("experimentAccession") or "")
    tech = " ".join(str(x) for x in (rec.get("technologyType") or [])).lower()
    if "proteom" in tech:
        kind, measured = "proteomics", Measured("proteomics", True)
    elif "array" in tech:
        kind = "Expression profiling by array"
        measured = Measured("expression", True)
    else:
        kind = "Expression profiling by high throughput sequencing"
        measured = Measured("expression", True)
    if m := _ATLAS_ACCESSION.match(acc):
        source, accession = "geo", f"GSE{m.group(1)}"
    elif acc.startswith("E-MTAB-"):
        source, accession = "biostudies-arrayexpress", acc
    else:
        source, accession = "expression-atlas", acc
    factors = tuple(str(f) for f in (rec.get("experimentalFactors") or []))
    stated = None
    if designs and any(TIME_FACTOR.search(f) for f in factors):
        try:
            stated = atlas_design(atlas_experiment(acc, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 - the listing stands
            log.info("%s: design not read (%s)", acc, exc)
    return DatasetCandidate(
        source=source,
        accession=accession,
        title=str(rec.get("experimentDescription") or ""),
        kind=kind,
        organism=str(rec.get("species") or ""),
        n_samples=int(rec.get("numberOfAssays") or 0),
        platform="",
        url=f"{ATLAS}/experiments/{acc}",
        summary=str(rec.get("experimentDescription") or ""),
        measured=measured,
        factors=factors,
        stated=stated,
        curated=acc,
    )


def search_expression_atlas(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
) -> list[DatasetCandidate]:
    """Expression Atlas experiments matching ``query`` by description,
    with their declared factors and, where time is one, the design."""
    from hallsim.search.models import term_score

    scored = []
    for rec in atlas_experiments():
        if organism and str(rec.get("species") or "") != organism:
            continue
        score = term_score(
            query,
            str(rec.get("experimentDescription") or ""),
            str(rec.get("experimentAccession") or ""),
        )
        if score:
            scored.append((score, rec))
    scored.sort(key=lambda sr: -sr[0])
    log.info("expression atlas '%s': %d candidates", query, len(scored))
    return [_atlas_candidate(r, timeout=timeout) for _, r in scored[:limit]]


def iter_expression_atlas(
    *, designs: bool = True, timeout: float = 60.0, pause: float = 0.2
):
    """Every Atlas experiment, designs read for those declaring time."""
    for rec in atlas_experiments():
        yield _atlas_candidate(rec, designs=designs, timeout=timeout)
        if designs:
            time.sleep(pause)


OMICSDI = "https://www.omicsdi.org/ws"
#: OmicsDI's omics types by the quantity they measure.
OMICSDI_MODALITY = (
    ("transcriptomics", "expression", True),
    ("proteomics", "proteomics", True),
    ("metabolomics", "metabolomics", False),
    ("lipidomics", "metabolomics", False),
    ("genomics", "genotype", True),
    ("models", "model", False),
)
#: OmicsDI's names for repositories this package names otherwise.
OMICSDI_SOURCE = {
    "metabolights_dataset": "metabolights",
    "atlas-experiments": "expression-atlas",
}


def omicsdi_measured(omics_types) -> Measured:
    for t in omics_types:
        for prefix, modality, complete in OMICSDI_MODALITY:
            if str(t).lower().startswith(prefix):
                return Measured(modality, complete)
    return Measured("unknown", False)


def search_omicsdi(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
    enrich: bool = True,
) -> list[DatasetCandidate]:
    """EBI's OmicsDI, one index over 29 repositories — GEO, ArrayExpress,
    PRIDE, MetaboLights, Metabolomics Workbench, Expression Atlas, the
    BioImage Archive, LINCS, MassIVE and more — searched once. Models are
    left to the model search. A GEO hit is read again from GEO so its
    sample titles, and so its design, come along; the index carries
    neither. ``organism`` filters the hits by the names the index lists."""
    size = limit * 4 if organism else limit
    payload = get_json(
        f"{OMICSDI}/dataset/search",
        {"query": query, "size": size, "start": 0},
        timeout,
    )
    log.info("omicsdi '%s': %s hits", query, payload.get("count", "?"))
    out: list[DatasetCandidate] = []
    for h in payload.get("datasets", []):
        names = [str(o.get("name") or "") for o in (h.get("organisms") or [])]
        if organism and organism not in names:
            continue
        types = [str(x) for x in (h.get("omicsType") or [])]
        measured = omicsdi_measured(types)
        if measured.modality == "model":
            continue
        raw_source = str(h.get("source") or "")
        source = OMICSDI_SOURCE.get(raw_source, raw_source)
        accession = str(h.get("id") or "")
        cand = DatasetCandidate(
            source=source,
            accession=accession,
            title=str(h.get("title") or ""),
            kind="; ".join(types),
            organism="; ".join(n for n in names if n),
            n_samples=0,
            platform="",
            url=f"https://www.omicsdi.org/dataset/{raw_source}/{accession}",
            summary=str(h.get("description") or ""),
            measured=measured,
        )
        if enrich and source == "geo":
            try:
                cand = geo_by_accession(accession, timeout=timeout) or cand
            except Exception as exc:  # noqa: BLE001 - the index stands
                log.info("%s: not read from GEO (%s)", accession, exc)
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def _search_petab(*a, **kw):
    """Late import: `attached` builds on the candidates defined here."""
    from hallsim.search.attached import search_petab

    return search_petab(*a, **kw)


SOURCES = {
    "omicsdi": search_omicsdi,
    "geo": search_geo,
    "geo-datasets": search_geo_datasets,
    "expression-atlas": search_expression_atlas,
    "petab": _search_petab,
    "zenodo": search_zenodo,
    "pride": _ebi_source("pride"),
    "metabolights": _ebi_source("metabolights"),
    "metabolomics-workbench": _ebi_source("metabolomics-workbench"),
    "arrayexpress": _ebi_source("arrayexpress"),
    "bioimages": search_bioimages,
}


#: What a search asks when no source is named: the one index over the
#: repositories, then the sources it does not carry — Zenodo, GEO's
#: curated DataSets, the PEtab collection. ``list(SOURCES)`` asks every
#: repository directly.
DEFAULT_SOURCES = ("omicsdi", "zenodo", "geo-datasets", "petab")


def search_for_dataset(
    query: str,
    limit: int = 25,
    sources: list[str] | None = None,
    **kwargs,
) -> list[DatasetCandidate]:
    """Search the data repositories for ``query``: :data:`DEFAULT_SOURCES`
    unless ``sources`` names others. A source that errors is logged and
    skipped."""
    names = sources if sources is not None else list(DEFAULT_SOURCES)
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
    study = get_json(f"{BIOSTUDIES}/studies/{accession}", {}, timeout)
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


#: A supplement table larger than this is left in the archive.
MAX_TABLE_BYTES = 20 * 1024 * 1024


def pmc_supplement(
    pmcid: str, *, timeout: float = 120.0
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    """A paper's Europe PMC supplementary archive: the file names inside
    it, nested archives opened one level, and the tables it holds,
    extracted beside the cache. Both are kept, since the archive is the
    whole supplement and one download serves both."""
    import io
    import zipfile

    dest = cache_dir("europepmc") / pmcid / "tables"
    marker = dest / ".extracted"

    def keep(name: str, data: bytes) -> None:
        if not name.lower().endswith(TABLE_SUFFIXES):
            return
        if len(data) > MAX_TABLE_BYTES or is_junk_archive_entry(name):
            return
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name.replace("/", "__")).write_bytes(data)

    def fetch():
        request = urllib.request.Request(
            f"{EUROPEPMC}/{pmcid}/supplementaryFiles",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout) as fh:
            blob = fh.read()
        names = []
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for info in archive.infolist():
                    names.append(info.filename)
                    if info.is_dir():
                        continue
                    if not info.filename.lower().endswith(".zip"):
                        keep(info.filename, archive.read(info))
                        continue
                    try:
                        inner = zipfile.ZipFile(io.BytesIO(archive.read(info)))
                    except zipfile.BadZipFile:
                        continue
                    for member in inner.infolist():
                        names.append(f"{info.filename}/{member.filename}")
                        if not member.is_dir():
                            keep(member.filename, inner.read(member))
        except zipfile.BadZipFile:
            names = []
        dest.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return names

    key = f"pmc supplement {pmcid}"
    if marker.exists():
        names = cached_json(key, fetch)
    else:
        # Names cached by a run that did not extract tables: one more
        # download extracts them, and the names it lists are the same.
        names = fetch()
        names = cached_json(key, lambda: names)
    tables = tuple(sorted(p for p in dest.iterdir() if p.name != ".extracted"))
    return tuple(names), tables


def pmc_supplement_files(pmcid: str, *, timeout: float = 120.0) -> tuple:
    """The file names inside a paper's supplementary archive."""
    return pmc_supplement(pmcid, timeout=timeout)[0]


def iter_bioimages(*, page: int = 100, timeout: float = 30.0):
    """Every BioImage Archive study, through the BioStudies collection
    facet, page by page."""
    number = 1
    while True:
        payload = get_json(
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
                summary=h.get("content") or "",
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
    core = get_json(
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
    links = get_json(
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
        files, stated = (), None
        if with_files:
            try:
                files, tables = pmc_supplement(pmcid, timeout=timeout)
                stated = table_design(tables)
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
            stated=stated,
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


#: A gene symbol: HGNC-style, at most 15 characters.
SYMBOL = re.compile(r"^(?:[A-Z][A-Z0-9-]{0,14}|C[0-9XY]+orf[0-9]+)$")
_GENE_EDIT = re.compile(
    r"\b(?:si|sh|KO|KD|OE|del|Δ|crispr|knock(?:out|down)|mutant|null|"
    r"overexpress(?:ion|ing)?|-/-|\+/-)\b",
    re.I,
)


def _chebi(label: str, timeout: float) -> tuple[str, str]:
    payload = cached_json(
        f"ols chebi {label.lower()}",
        lambda: get_json(
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
    mols = cached_json(
        f"chembl molecule {label.lower()}",
        lambda: get_json(
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
    mechs = cached_json(
        f"chembl mechanism {chembl_id}",
        lambda: get_json(
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
        target = cached_json(
            f"chembl target {tid}",
            lambda tid=tid: get_json(
                f"{CHEMBL}/target/{tid}.json", {}, timeout
            ),
        )
        for comp in target.get("target_components", []):
            acc = comp.get("accession")
            if acc:
                targets.append(f"uniprot:{acc}")
    return list(dict.fromkeys(targets)), "; ".join(dict.fromkeys(names))


def _gene_uniprot(symbol: str, timeout: float) -> list[str]:
    payload = cached_json(
        f"mygene {symbol.upper()}",
        lambda: get_json(
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


# ── GEO's files ─────────────────────────────────────────────────────

GEO_SERIES = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def geo_series_urls(accession: str) -> tuple[str, str]:
    """The series-matrix and family-SOFT URLs GEO serves for ``accession``."""
    base = f"{GEO_SERIES}/{accession[:3]}{accession[3:-3]}nnn/{accession}"
    return (
        f"{base}/matrix/{accession}_series_matrix.txt.gz",
        f"{base}/soft/{accession}_family.soft.gz",
    )


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
