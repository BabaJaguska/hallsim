"""Metabolite tables from both repositories — the one route that reaches a
model species by identity.

Every other reader arrives at a model through a proxy. A transcript is two
translation steps and an mRNA half-life away from the protein a model
integrates, and the reporter layer keeps only the sign of the change. A
metabolite assignment file names ChEBI accessions, and a screened deposit's
species carry ChEBI annotations, so a dataset quantity and a model quantity
are joined by one accession being the other — no regulon, no sign
convention, nothing in between.

Each repository's own package parses its own format, so neither
specification is ours to track: ``metabolights-utils`` for MetaboLights
ISA-Tab and its FTP layout, ``mwtab`` for Metabolomics Workbench. What is
here is everything above that: the ChEBI keying, the numeric frame, the
design a study's declared factors imply, and the group contrasts
calibration consumes.

The two differ in where ChEBI comes from. A MetaboLights assignment file
names ChEBI accessions outright. mwTab names compounds and carries PubChem,
InChIKey and KEGG, so :func:`chebi_for` resolves them structurally through
UniChem — which turns out to reach *more* compounds than depositors\' own
ChEBI annotation does.

A study is two files, and the assignment file is only one of them. The
assignment file is one row per measured feature with one column per
sample; the sample file is what each sample *is*, as declared ``Factor
Value[...]`` columns. The time axis lives in the sample file and nowhere in
the assignment file, so a MAF on its own is a matrix with no design — and
it does not say which of its columns are samples either, which is why the
assay file's sample names decide that.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from metabolights_utils.isatab import Reader
from metabolights_utils.provider.ftp_repository import (
    MetabolightsFtpRepository,
)
from metabolights_utils.provider.local_folder_metadata_collector import (
    LocalFolderMetadataCollector,
)
from metabolights_utils.provider.study_provider import (
    MetabolightsStudyProvider,
)

from hallsim.datasets import (
    CONTROL_WORDS,
    TIME_FACTOR,
    Design,
    _UNIT_NAME,
    _cached_json,
    _drop_identifiers,
    _strip_common,
    _times,
    curie,
)
from hallsim.measurements import MeasuredDataset

log = logging.getLogger(__name__)

CHEBI = re.compile(r"CHEBI:\d+", re.I)
_FACTOR = re.compile(r"^Factor Value\s*\[(?P<name>.+?)\]", re.I)
_QUALIFIER = re.compile(
    r"^(Unit|Term Source REF|Term Accession Number)(\.\d+)?$", re.I
)
#: Precision qualifiers on a declared numeric value.
_APPROXIMATE = re.compile(r"^\s*[~\u2248><\u2265\u2264]+\s*")


def isa_frame(table) -> pd.DataFrame:
    """An ISA table from ``metabolights-utils`` as a DataFrame in file
    order. Repeated qualifier headers keep the reader's ``.1``, ``.2``
    suffixes, so a qualifier is found by position rather than by name."""
    return pd.DataFrame(
        {c: table.data.get(c, []) for c in table.columns},
        columns=list(table.columns),
        dtype=str,
    )


# ── The assignment file ─────────────────────────────────────────────


@dataclass(frozen=True)
class MafTable:
    """One assignment file's quantities, with what reading it cost.

    ``quantities`` is ``ChEBI curie × sample`` on the file's own intensity
    scale. Its index repeats where several features were assigned the same
    compound — different adducts, ionisation modes, or NMR resonances of
    one molecule — because those sit on different arbitrary scales and must
    not be added. :class:`MetaboliteDataset` collapses them after taking a
    ratio, where the scales cancel.

    The counts are the denominator worth reporting: annotation coverage is
    a property of the platform, so a targeted panel maps nearly everything
    and an untargeted run maps a minority.
    """

    quantities: pd.DataFrame
    names: dict[str, str]
    features: int
    keyed: int
    unannotated: int
    ambiguous: int
    other_namespace: dict[str, int]

    @property
    def compounds(self) -> int:
        return int(self.quantities.index.nunique())

    @property
    def coverage(self) -> float:
        """Share of the file's measured features that reached a ChEBI
        accession. Not the row count of ``quantities``, which can exceed
        the feature count where one compound carries several accessions."""
        return self.keyed / self.features if self.features else 0.0

    def summary(self) -> str:
        other = ", ".join(
            f"{n} {ns}" for ns, n in sorted(self.other_namespace.items())
        )
        parts = [
            f"{self.features} features",
            f"{self.keyed} keyed to ChEBI ({self.coverage:.0%})",
            f"{self.compounds} compounds",
            f"{self.quantities.shape[1]} samples",
        ]
        if self.unannotated:
            parts.append(f"{self.unannotated} unannotated")
        if self.ambiguous:
            parts.append(f"{self.ambiguous} ambiguous")
        if other:
            parts.append(f"other namespaces: {other}")
        return ", ".join(parts)


def maf_quantities(frame: pd.DataFrame, samples) -> MafTable:
    """A parsed assignment table as ChEBI-keyed quantities.

    ``samples`` names the sample columns, because an assignment file does
    not: its header does not mark which columns are samples, and the
    study's assay file is the specification's own answer.

    Rows without a ChEBI accession are dropped and counted rather than
    resolved: an HMDB or KEGG identifier would need a crosswalk this does
    not carry, and an untargeted feature known only by mass and retention
    time cannot be joined to anything by identity. Non-positive
    intensities become missing, because a zero in an assignment file is
    below detection rather than an abundance of none, and a log of it is
    not a number.
    """
    if "database_identifier" not in frame.columns:
        raise ValueError(
            "no database_identifier column, so this is not an assignment "
            "file"
        )
    wanted = [c for c in frame.columns if c in set(samples)]
    if not wanted:
        raise ValueError(
            f"none of the {len(set(samples))} declared samples names a "
            "column in the assignment file"
        )
    absent = sorted(set(samples) - set(wanted))
    if absent:
        log.info(
            "%d of %d declared samples have no column (%s)",
            len(absent),
            len(set(samples)),
            ", ".join(absent[:5]),
        )

    # The field is pipe-separated where the assignment was not resolved to
    # one compound (``CHEBI:86472|CHEBI:71727``, ``unknown|CHEBI:18089``).
    # A feature standing for either of two compounds belongs to neither, so
    # it is counted rather than attributed to both.
    ids = frame["database_identifier"].fillna("").str.strip()
    candidates = ids.str.count(r"\|") + 1
    single = (ids != "") & (candidates == 1)
    chebi = single & ids.str.fullmatch(CHEBI.pattern, case=False)
    rest = ids[single & ~chebi]
    other = (
        rest.str.split(":")
        .str[0]
        .str.replace(r"\d+$", "", regex=True)
        .str.upper()
        .replace("", "?")
        .value_counts()
        .to_dict()
    )

    kept = frame[chebi]
    values = kept[wanted].apply(pd.to_numeric, errors="coerce")
    values = values.where(values > 0)
    values.index = [curie("chebi", i) for i in ids[chebi]]
    names: dict[str, str] = {}
    if "metabolite_identification" in frame.columns:
        for key, name in zip(
            values.index, kept["metabolite_identification"].fillna("")
        ):
            if name.strip():
                names.setdefault(key, name.strip())

    table = MafTable(
        quantities=values,
        names=names,
        features=len(frame),
        keyed=int(chebi.sum()),
        unannotated=int((ids == "").sum()),
        ambiguous=int((candidates > 1).sum()),
        other_namespace=other,
    )
    log.info("assignment file: %s", table.summary())
    return table


def read_maf(path, *, samples) -> MafTable:
    """One assignment file on disk, keyed by ChEBI."""
    result = Reader.get_assignment_file_reader().read(
        file_buffer_or_path=str(path)
    )
    return maf_quantities(isa_frame(result.isa_table_file.table), samples)


def read_isa_samples(path) -> pd.DataFrame:
    """One ISA-Tab sample file on disk."""
    result = Reader.get_sample_file_reader().read(
        file_buffer_or_path=str(path)
    )
    return isa_frame(result.isa_table_file.table)


# ── The design a study declares ─────────────────────────────────────


def _unit_after(frame: pd.DataFrame, column: str) -> str:
    """The ``Unit`` qualifying a value column: the ISA convention is that a
    qualifier applies to the nearest value column to its left."""
    columns = list(frame.columns)
    for c in columns[columns.index(column) + 1 :]:
        if not _QUALIFIER.match(c):
            break
        if c.lower().startswith("unit"):
            values = frame[c].dropna().unique()
            return str(values[0]).lower() if len(values) else ""
    return ""


def factor_design(
    samples: list[str],
    factors: list[dict[str, str]],
    *,
    time_factor: str | None = None,
    units: dict[str, str] | None = None,
    subjects: list[str] | None = None,
) -> tuple[Design, dict[str, list[str]]]:
    """The :class:`Design` a set of per-sample declared factors implies.

    ``samples[i]`` is a sample's name and ``factors[i]`` its
    ``{factor name: value}``, which is what both repositories that ship
    structured metadata give: ISA-Tab as ``Factor Value[...]`` columns and
    mwTab as a parsed factor dict per sample. An arm is therefore a
    declared condition rather than a string parsed out of a title.

    A factor value is one token, and a token carried by a single sample is
    pruned by :func:`~hallsim.datasets._drop_identifiers`, so a declared
    factor that is really a per-sample identifier does not make every
    sample its own arm; a factor constant across every sample distinguishes
    nothing and is dropped by :func:`~hallsim.datasets._strip_common`.

    The time factor is named explicitly or found by its label. Its value is
    read with the same parser that reads times out of sample titles, so
    ``24``, ``24 h`` and ``day 7`` all work, and ``units`` supplies the unit
    where the format declares it in a separate field.
    """
    names = sorted({k for f in factors for k in f})
    if time_factor is None:
        time_factor = next((n for n in names if TIME_FACTOR.search(n)), None)
    elif time_factor not in names:
        raise ValueError(
            f"no factor named {time_factor!r}; these declare {names}"
        )

    other = [n for n in names if n != time_factor]
    tokens = [[f.get(n, "").strip() for n in other] for f in factors]
    tokens = [[v for v in row if v] for row in tokens]
    labels = _strip_common(_drop_identifiers(tokens))

    unit = ""
    times: list[float | None] = []
    declared = (units or {}).get(time_factor, "") if time_factor else ""
    for f in factors:
        raw = f.get(time_factor, "").strip() if time_factor else ""
        # A declared duration is often approximate — NASA writes a mission
        # length as "~30" days — and the qualifier is about precision, not
        # about a different quantity, so it is dropped before the number is
        # read rather than costing the study its time course.
        raw = _APPROXIMATE.sub("", raw).strip()
        found, _ = _times(raw)
        if found:
            value, u = found[0]
            times.append(value)
            unit = unit or _UNIT_NAME.get(u, u)
            continue
        try:
            times.append(float(raw))
        except ValueError:
            times.append(None)
    if not unit and declared:
        unit = _UNIT_NAME.get(declared.lower().rstrip("s"), declared.lower())

    groups: dict[str, list[str]] = {}
    per_arm: dict[str, set[float]] = {}
    for label, t, sample in zip(labels, times, samples):
        arm = " / ".join(label).strip(" /") or "all"
        per_arm.setdefault(arm, set())
        if t is None:
            key = arm
        else:
            per_arm[arm].add(float(t))
            key = f"{arm} @ {t:g}{unit}"
        groups.setdefault(key, []).append(str(sample))

    arms = tuple(sorted(per_arm))
    control = next(
        (
            a
            for a in arms
            if any(
                tok.lower() in CONTROL_WORDS
                for tok in re.split(r"[\s/_-]+", a)
            )
        ),
        None,
    )
    design = Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        time_unit=unit,
        n_titles=len(samples),
        n_subjects=len({s for s in subjects if s}) if subjects else 0,
    )
    return design, groups


def isa_design(
    frame: pd.DataFrame, *, time_factor: str | None = None
) -> tuple[Design, dict[str, list[str]]]:
    """The :class:`Design` an ISA-Tab sample file declares, and its groups.

    Reads the ``Factor Value[...]`` columns and the ``Unit`` column that
    qualifies each, then hands them to :func:`factor_design`.
    """
    name_column = next(
        (c for c in frame.columns if c.lower() == "sample name"), None
    )
    if name_column is None:
        raise ValueError("the sample file has no Sample Name column")
    columns = {
        m.group("name").strip(): c
        for c in frame.columns
        if (m := _FACTOR.match(c))
    }
    if time_factor is not None and time_factor not in columns:
        raise ValueError(
            f"no Factor Value[{time_factor}] column; this file declares "
            f"{sorted(columns)}"
        )
    factors = [
        {n: str(row[c] or "") for n, c in columns.items()}
        for _, row in frame.fillna("").iterrows()
    ]
    units = {n: _unit_after(frame, c) for n, c in columns.items()}
    source = next(
        (c for c in frame.columns if c.lower() == "source name"), None
    )
    return factor_design(
        [str(s) for s in frame[name_column]],
        factors,
        time_factor=time_factor,
        units=units,
        subjects=(
            [str(s).strip() for s in frame[source].fillna("")]
            if source
            else None
        ),
    )


def assay_samples(model) -> list[str]:
    """Every sample name a study's assay files declare, in order — the
    specification's answer to which assignment-file columns are samples."""
    out: list[str] = []
    for assay in model.assays.values():
        out += [str(s) for s in assay.table.data.get("Sample Name", []) if s]
    return list(dict.fromkeys(out))


# ── The dataset ─────────────────────────────────────────────────────


@dataclass
class MetaboliteDataset(MeasuredDataset):
    """Metabolite quantities as the group contrasts calibration consumes,
    keyed by ChEBI curie so a contrast lands on a model species directly.

    ``quantities`` is on the assignment file's own intensity scale, which
    is arbitrary and usually differs per feature. Nothing here tries to put
    it on a model's concentration scale — a ratio between two groups is
    scale-free, which is why the interface is a fold change and not a
    value.
    """

    quantities: pd.DataFrame
    sample_groups: dict[str, list]
    names: dict[str, str] = field(default_factory=dict)
    design: Design | None = None

    @property
    def log_values(self) -> pd.DataFrame:
        return np.log2(self.quantities)

    def _reduce(self, per_feature: pd.Series) -> pd.Series:
        # Several features can carry one compound on unrelated scales. The
        # scale cancels inside each feature's ratio, so the ratios average
        # and the intensities do not.
        return per_feature.groupby(level=0).mean()

    def _reduce_variance(self, per_feature: pd.Series) -> pd.Series:
        grouped = per_feature.groupby(level=0)
        return grouped.sum() / grouped.count() ** 2

    def features_per_compound(self) -> pd.Series:
        return self.quantities.groupby(level=0).size()

    @classmethod
    def from_study(
        cls, model, *, assay: int = 0, time_factor: str | None = None
    ) -> "MetaboliteDataset":
        """Build from a ``metabolights-utils`` study model.

        A study ships one assignment file per assay — a mass-spec run in
        positive and negative mode is two — and ``assay`` picks which.
        Combining them is a cross-platform decision this does not make.
        """
        if not model.samples:
            raise ValueError("the study model carries no sample file")
        assignments = list(model.metabolite_assignments)
        if not assignments:
            raise ValueError("the study model carries no assignment file")
        if len(assignments) > 1:
            log.info(
                "%d assignment files; reading %s",
                len(assignments),
                assignments[assay],
            )
        samples_frame = isa_frame(next(iter(model.samples.values())).table)
        design, groups = isa_design(samples_frame, time_factor=time_factor)
        declared = assay_samples(model) or [
            s for g in groups.values() for s in g
        ]
        maf = model.metabolite_assignments[assignments[assay]]
        table = maf_quantities(isa_frame(maf.table), declared)

        present = set(table.quantities.columns)
        groups = {k: [s for s in v if s in present] for k, v in groups.items()}
        return cls(
            quantities=table.quantities,
            sample_groups={k: v for k, v in groups.items() if v},
            names=table.names,
            design=design,
        )

    @classmethod
    def from_metabolights(
        cls,
        study,
        *,
        cache: Path | str | None = None,
        assay: int = 0,
        time_factor: str | None = None,
    ) -> "MetaboliteDataset":
        """Build from a MetaboLights accession or a downloaded study
        directory. An accession is fetched over the repository's FTP, into
        ``cache`` when one is given.
        """
        path = Path(study)
        if path.is_dir():
            model = MetabolightsStudyProvider(
                folder_metadata_collector=LocalFolderMetadataCollector()
            ).load_study(
                path.name,
                str(path),
                load_sample_file=True,
                load_assay_files=True,
                load_maf_files=True,
            )
        else:
            repo = MetabolightsFtpRepository(
                local_storage_root_path=str(cache) if cache else None
            )
            model, messages = repo.load_study_model(
                str(study), load_folder_metadata=False
            )
            if model is None:
                raise LookupError(
                    f"{study}: {'; '.join(str(m) for m in messages[:3])}"
                )
        return cls.from_study(model, assay=assay, time_factor=time_factor)

    @classmethod
    def from_workbench(
        cls, study, *, time_factor: str | None = None, timeout: float = 30.0
    ) -> "MetaboliteDataset":
        """Build from a Metabolomics Workbench study or a local mwTab file.

        The second metabolomics repository, and the one that needs a
        crosswalk: mwTab carries PubChem, InChIKey and KEGG identifiers but
        not ChEBI, so :func:`chebi_for` resolves them structurally through
        UniChem. A compound may resolve to several ChEBI accessions, which
        is ChEBI's protonation granularity rather than uncertainty, so all
        of them are kept and only the one a model annotates will match.
        """
        table = read_mwtab(study, timeout=timeout)
        design, groups = mwtab_design(study, time_factor=time_factor)
        present = set(table.quantities.columns)
        groups = {k: [s for s in v if s in present] for k, v in groups.items()}
        return cls(
            quantities=table.quantities,
            sample_groups={k: v for k, v in groups.items() if v},
            names=table.names,
            design=design,
        )

    @classmethod
    def from_maf(
        cls,
        maf_path,
        *,
        sample_groups: dict[str, list],
    ) -> "MetaboliteDataset":
        """Build from an assignment file detached from its study.

        ``sample_groups`` has to name the sample columns, because nothing in
        the file marks them and no assay file is on hand to say.
        """
        samples = [s for g in sample_groups.values() for s in g]
        table = read_maf(maf_path, samples=samples)
        return cls(
            quantities=table.quantities,
            sample_groups=sample_groups,
            names=table.names,
        )


# ── Metabolomics Workbench ──────────────────────────────────────────

WORKBENCH_REST = "https://www.metabolomicsworkbench.org/rest/study/study_id"
UNICHEM = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
#: UniChem's source numbers for the identifiers mwTab carries. KEGG is
#: absent from UniChem, so a compound known only by a KEGG id does not
#: resolve.
UNICHEM_SOURCE = {"pubchem": 22, "hmdb": 18, "lipidmaps": 33}
#: Where mwTab keeps the quantities, by assay.
MWTAB_BLOCKS = ("MS_METABOLITE_DATA", "NMR_METABOLITE_DATA")


def chebi_for(query: str, source: str = "", *, timeout: float = 30.0):
    """Every ChEBI accession UniChem links to one compound.

    ``source`` empty treats ``query`` as an InChIKey, which is a structural
    key and the most reliable route; otherwise it names a source in
    :data:`UNICHEM_SOURCE`.

    Several accessions for one compound is normal and is kept, not treated
    as ambiguity: ChEBI models protonation states as separate entities, so
    L-alanine is both ``CHEBI:16977`` and ``CHEBI:57972`` and a model
    annotates one of them. That is a different thing from a depositor who
    could not tell two compounds apart, which
    :func:`maf_quantities` drops.
    """
    if not query:
        return ()
    body = (
        {"type": "inchikey", "compound": query}
        if not source
        else {
            "type": "sourceID",
            "compound": query,
            "sourceID": UNICHEM_SOURCE[source],
        }
    )

    def fetch():
        import json as _json
        import urllib.request

        request = urllib.request.Request(
            UNICHEM,
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as fh:
            return _json.load(fh)

    # The cache stores whatever the fetch returns, so a timeout must raise
    # rather than answer "no compound": otherwise one bad request records
    # a permanent false negative.
    try:
        payload = _cached_json(
            f"unichem {source or 'inchikey'} {query}", fetch
        )
    except Exception as exc:  # noqa: BLE001 - the id stays unresolved
        log.info("unichem %s %s: %s", source or "inchikey", query, exc)
        return ()
    found = {
        s["compoundId"]
        for c in payload.get("compounds", [])
        for s in c.get("sources", [])
        if s.get("shortName") == "chebi"
    }
    return tuple(sorted(curie("chebi", i) for i in found))


def mwtab_chebi(metabolites, *, timeout: float = 30.0) -> dict:
    """``{metabolite name: (chebi curie, ...)}`` for an mwTab metabolite
    block, preferring the InChIKey and falling back to the source ids it
    carries."""
    out: dict[str, tuple] = {}
    for record in metabolites:
        # mwtab hands back a DuplicatesDict whose .get() answers the
        # default for keys it holds, so normalise before reading.
        row = dict(record)
        name = str(row.get("Metabolite", "")).strip()
        if not name or name in out:
            continue
        found: tuple = ()
        for key, source in (
            ("inchi_key", ""),
            ("pubchem_id", "pubchem"),
            ("hmdb_id", "hmdb"),
            ("lipidmaps_id", "lipidmaps"),
        ):
            value = str(row.get(key, "") or "").strip()
            if value:
                found = chebi_for(value, source, timeout=timeout)
            if found:
                break
        out[name] = found
    return out


def read_mwtab(source, *, timeout: float = 30.0) -> MafTable:
    """A Metabolomics Workbench study as ChEBI-keyed quantities.

    ``source`` is a local mwTab file or a study accession, which is
    fetched from the repository's REST interface.

    mwTab names compounds and carries PubChem, InChIKey and KEGG
    identifiers but not ChEBI, so the join is made through UniChem's
    structural cross-references rather than by matching names. A compound
    that resolves to nothing is counted, and KEGG-only compounds are
    expected among them because UniChem does not index KEGG.
    """
    import mwtab

    path = Path(str(source))
    if not path.exists():
        path = _fetch_mwtab(str(source), timeout=timeout)
    parsed = next(mwtab.read_files(str(path)))
    name = next((k for k in MWTAB_BLOCKS if k in parsed), None)
    if name is None:
        raise ValueError(f"{source} carries no metabolite data block")
    block = dict(parsed[name])
    data = [dict(r) for r in block.get("Data") or []]
    if not data:
        raise ValueError(f"{source} declares no measured values")

    resolved = mwtab_chebi(block.get("Metabolites") or [], timeout=timeout)
    if not resolved:
        raise ValueError(f"{source} declares no metabolite identifiers")
    frame = pd.DataFrame(data).set_index("Metabolite")
    values = frame.apply(pd.to_numeric, errors="coerce")
    values = values.where(values > 0)

    rows, index, names = [], [], {}
    for name, row in values.iterrows():
        for key in resolved.get(str(name).strip(), ()):
            rows.append(row)
            index.append(key)
            names.setdefault(key, str(name).strip())
    quantities = (
        pd.DataFrame(rows, index=index)
        if rows
        else pd.DataFrame(columns=values.columns)
    )

    table = MafTable(
        quantities=quantities,
        names=names,
        features=len(values),
        keyed=sum(1 for v in resolved.values() if v),
        unannotated=sum(1 for v in resolved.values() if not v),
        ambiguous=0,
        other_namespace={},
    )
    log.info("%s: %s", path.name, table.summary())
    return table


def _fetch_mwtab(accession: str, *, timeout: float = 30.0) -> Path:
    import urllib.request

    from hallsim.io import record_checksum

    dest = Path.home() / ".cache" / "hallsim" / "workbench"
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{accession}.mwtab.txt"
    if target.exists() and target.stat().st_size:
        return target
    url = f"{WORKBENCH_REST}/{accession}/mwtab/txt"
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        blob = fh.read()
    if not any(k.encode() in blob for k in MWTAB_BLOCKS):
        raise LookupError(f"{accession}: the repository returned no mwTab")
    target.write_bytes(blob)
    record_checksum(target)
    return target


def mwtab_design(source, *, time_factor: str | None = None):
    """The :class:`Design` a Metabolomics Workbench study declares, and its
    sample groups, from the factor dictionary it carries per sample."""
    import mwtab

    path = Path(str(source))
    if not path.exists():
        path = _fetch_mwtab(str(source))
    parsed = next(mwtab.read_files(str(path)))
    records = parsed.get("SUBJECT_SAMPLE_FACTORS") or []
    if not records:
        raise ValueError(f"{source} declares no SUBJECT_SAMPLE_FACTORS")
    rows = [dict(r) for r in records]
    samples = [str(r.get("Sample ID", "")).strip() for r in rows]
    factors = [dict(r.get("Factors") or {}) for r in rows]
    return factor_design(samples, factors, time_factor=time_factor)
