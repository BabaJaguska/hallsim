"""MetaboLights metabolite tables — the one route that reaches a model
species by identity.

Every other reader arrives at a model through a proxy. A transcript is two
translation steps and an mRNA half-life away from the protein a model
integrates, and the reporter layer keeps only the sign of the change. A
metabolite assignment file names ChEBI accessions, and a screened deposit's
species carry ChEBI annotations, so a dataset quantity and a model quantity
are joined by one accession being the other — no regulon, no sign
convention, nothing in between.

Reading ISA-Tab and fetching studies is ``metabolights-utils``, the
repository's own package, so the assignment specification and the FTP
layout stay its problem. What is here is everything above that: the ChEBI
filter and curie keying, the numeric frame, the design a study's declared
factors imply, and the group contrasts calibration consumes.

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
    _drop_identifiers,
    _strip_common,
    curie,
)
from hallsim.measurements import MeasuredDataset

log = logging.getLogger(__name__)

CHEBI = re.compile(r"CHEBI:\d+", re.I)
_FACTOR = re.compile(r"^Factor Value\s*\[(?P<name>.+?)\]", re.I)
_QUALIFIER = re.compile(
    r"^(Unit|Term Source REF|Term Accession Number)(\.\d+)?$", re.I
)


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
    unannotated: int
    ambiguous: int
    other_namespace: dict[str, int]

    @property
    def compounds(self) -> int:
        return int(self.quantities.index.nunique())

    @property
    def coverage(self) -> float:
        """Share of the file's features that carry a ChEBI accession."""
        return len(self.quantities) / self.features if self.features else 0.0

    def summary(self) -> str:
        other = ", ".join(
            f"{n} {ns}" for ns, n in sorted(self.other_namespace.items())
        )
        parts = [
            f"{self.features} features",
            f"{len(self.quantities)} with a ChEBI id ({self.coverage:.0%})",
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


def isa_design(
    frame: pd.DataFrame, *, time_factor: str | None = None
) -> tuple[Design, dict[str, list[str]]]:
    """The :class:`Design` a sample file declares, and its sample groups.

    Arms come from the factors other than time, so an arm is a declared
    condition rather than a string parsed out of a title — which is the
    reason to prefer a repository that ships ISA-Tab. A factor value is one
    token, and a token carried by a single sample is pruned by
    :func:`~hallsim.datasets._drop_identifiers`, so a declared factor that
    is really a per-sample identifier does not make every sample its own
    arm. The time factor is named explicitly or found by its label; its
    ``Unit`` column sets the clock.

    Returns the design and ``{group_label: [sample name, ...]}``, keyed the
    way :meth:`MeasuredDataset.arm_deltas` expects to look groups up.
    """
    name_column = next(
        (c for c in frame.columns if c.lower() == "sample name"), None
    )
    if name_column is None:
        raise ValueError("the sample file has no Sample Name column")
    factors = {
        m.group("name").strip(): c
        for c in frame.columns
        if (m := _FACTOR.match(c))
    }
    if time_factor is None:
        time_factor = next((n for n in factors if TIME_FACTOR.search(n)), None)
    elif time_factor not in factors:
        raise ValueError(
            f"no Factor Value[{time_factor}] column; this file declares "
            f"{sorted(factors)}"
        )

    other = [factors[n] for n in factors if n != time_factor]
    tokens = [
        [v for v in row if v] for row in frame[other].fillna("").values
    ] or [[] for _ in range(len(frame))]
    labels = _strip_common(_drop_identifiers(tokens))

    unit, times = "", pd.Series([None] * len(frame), index=frame.index)
    if time_factor is not None:
        column = factors[time_factor]
        raw = _unit_after(frame, column)
        unit = _UNIT_NAME.get(raw.rstrip("s"), raw)
        times = pd.to_numeric(frame[column], errors="coerce")

    groups: dict[str, list[str]] = {}
    per_arm: dict[str, set[float]] = {}
    for label, t, sample in zip(labels, times, frame[name_column]):
        arm = " / ".join(label).strip(" /") or "all"
        per_arm.setdefault(arm, set())
        if pd.isna(t):
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
        n_titles=len(frame),
    )
    return design, groups


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
        return per_feature.groupby(level=0).mean().dropna()

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
