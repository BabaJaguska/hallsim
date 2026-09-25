"""PRIDE protein tables — a proteome read against a model's proteins.

A protein abundance is what a mechanistic model integrates, so this joins
one step closer than a transcript does: an mzTab accession is a UniProt
accession, and a screened deposit's species carry UniProt annotations, so
the pairing is an identity. It is still weaker than the metabolite route in
one way — a proteome matches a model because it covers *every* protein
rather than because a particular one was targeted, which is why the census
records it as ``complete`` rather than ``direct``.

Parsing is ``pyteomics.mztab`` and fetching is ``ppx``, both maintained, so
the PSI standard and the PRIDE file layout stay their problem. What is here
is the UniProt keying, the numeric frame, the groups an mzTab's own study
variables declare, and the contrasts calibration consumes.

Coverage is the limit worth stating up front. Measured over 281 PRIDE
projects matching senescence, aging or rapamycin: 6% ship an mzTab, 10% a
MaxQuant ``proteinGroups``, 7% a DIA-NN matrix, 9% identifications with no
quantities at all, and 60% a table of no fixed layout. This module reads
the first of those. Rolling the others up from ion level is what
``directlfq`` exists for, and its 46 declared input formats are the reason
not to hand-write their column conventions here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hallsim.datasets import Design, parse_design
from hallsim.measurements import MeasuredDataset

log = logging.getLogger(__name__)

#: Abundance column families in an mzTab protein section, most specific
#: first: per assay is one column per sample, per study variable is already
#: aggregated over a group.
ABUNDANCE = ("protein_abundance_assay", "protein_abundance_study_variable")


@dataclass(frozen=True)
class MzTabProteins:
    """One mzTab's protein abundances, with what reading it cost.

    ``quantities`` is ``UniProt accession × column`` on the file's own
    scale, and ``level`` says what a column is. ``assay`` is one sample, so
    a group has replicates. ``study_variable`` is a group the depositor
    already averaged, so a contrast is a difference of two single values
    and the deposited ``stdev`` is the only spread there is.
    """

    quantities: pd.DataFrame
    level: str
    stdev: pd.DataFrame | None
    descriptions: dict[str, str]
    proteins: int
    ambiguous: int

    def summary(self) -> str:
        parts = [
            f"{self.proteins} protein rows",
            f"{len(self.quantities)} unambiguous",
            f"{self.quantities.shape[1]} {self.level} columns",
        ]
        if self.ambiguous:
            parts.append(f"{self.ambiguous} ambiguous groups")
        return ", ".join(parts)


def _abundance_columns(columns) -> tuple[str, dict[str, str]]:
    """The most specific abundance family present, and its columns renamed
    to the ``assay[n]`` / ``study_variable[n]`` keys the metadata uses."""
    for family in ABUNDANCE:
        found = {
            c: c[len("protein_abundance_") :]
            for c in columns
            if c.startswith(f"{family}[")
        }
        if found:
            return family[len("protein_abundance_") :], found
    return "", {}


def read_mztab(path) -> MzTabProteins:
    """An mzTab's protein section as UniProt-keyed quantities.

    A protein group whose ``ambiguity_members`` names further accessions is
    counted and dropped: the abundance stands for whichever member is
    present, so attributing it to the leading accession asserts an
    identification the file declined to make. Non-positive abundances
    become missing, because a zero is below detection rather than an
    abundance of none.
    """
    from pyteomics import mztab

    parsed = mztab.MzTab(str(path))
    frame = parsed.protein_table
    if frame is None or frame.empty:
        raise ValueError(f"{path} has no protein section")

    level, renames = _abundance_columns(frame.columns)
    if not renames:
        raise ValueError(
            f"{path} carries no protein abundances, only identifications"
        )

    accession = pd.Series(
        [str(i).strip() for i in frame.index], index=frame.index
    )
    members = (
        frame["ambiguity_members"].fillna("").astype(str).str.strip()
        if "ambiguity_members" in frame.columns
        else pd.Series("", index=frame.index)
    )
    keep = (
        ~members.str.contains(r"[,|]")
        & (members.str.lower() != "null")
        & (accession != "")
        & (accession.str.lower() != "null")
    )
    kept = frame[keep]
    index = list(accession[keep])

    values = kept[list(renames)].apply(pd.to_numeric, errors="coerce")
    values = values.where(values > 0).rename(columns=renames)
    values.index = index

    stdev = None
    spread = {
        c[len("protein_abundance_stdev_") :]: c
        for c in frame.columns
        if c.startswith("protein_abundance_stdev_study_variable[")
    }
    if level == "study_variable" and spread:
        stdev = kept[list(spread.values())].apply(
            pd.to_numeric, errors="coerce"
        )
        stdev.columns = list(spread)
        stdev.index = index

    descriptions = {
        key: str(parsed.metadata.get(f"{key}-description", "")).strip()
        for key in renames.values()
    }
    table = MzTabProteins(
        quantities=values,
        level=level,
        stdev=stdev,
        descriptions=descriptions,
        proteins=len(frame),
        ambiguous=int(members.str.contains(r"[,|]").sum()),
    )
    log.info("%s: %s", Path(path).name, table.summary())
    return table


def mztab_groups(
    table: MzTabProteins, metadata: dict
) -> tuple[Design, dict[str, list[str]]]:
    """The groups an mzTab declares, and the :class:`Design` they imply.

    A study variable is the depositor's own grouping, and its description
    is free text in the same shape as a GEO sample title, so the arms and
    timepoints come from :func:`~hallsim.datasets.parse_design` rather than
    from a second parser. Groups stay keyed by that description, which is
    the name the file gives them.
    """
    columns = list(table.quantities.columns)
    groups: dict[str, list[str]] = {}
    if table.level == "assay":
        for key, description in sorted(metadata_study_variables(metadata)):
            refs = [
                r.strip()
                for r in str(metadata.get(f"{key}-assay_refs", "")).split(",")
                if r.strip() in set(columns)
            ]
            if refs:
                groups.setdefault(description or key, []).extend(refs)
    else:
        for column in columns:
            label = table.descriptions.get(column) or column
            groups.setdefault(label, []).append(column)
    if not groups:
        groups = {c: [c] for c in columns}
    return parse_design(list(groups)), groups


def metadata_study_variables(metadata: dict) -> list[tuple[str, str]]:
    """``(study_variable[n], description)`` pairs in declaration order."""
    keys = sorted(
        {
            k.split("-", 1)[0]
            for k in metadata
            if k.startswith("study_variable[")
        },
        key=lambda k: int(k[len("study_variable[") : -1]),
    )
    return [
        (k, str(metadata.get(f"{k}-description", "")).strip()) for k in keys
    ]


@dataclass
class ProteinDataset(MeasuredDataset):
    """Protein quantities as the group contrasts calibration consumes,
    keyed by UniProt accession so a contrast lands on a model species.

    ``quantities`` is on the deposit's own scale. Nothing here puts it on a
    model's concentration scale — a ratio between two groups is scale-free,
    which is why the interface is a fold change and not a value.
    """

    quantities: pd.DataFrame
    sample_groups: dict[str, list]
    level: str = "assay"
    stdev: pd.DataFrame | None = None
    design: Design | None = None
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def log_values(self) -> pd.DataFrame:
        return np.log2(self.quantities)

    @property
    def has_replicates(self) -> bool:
        """Whether a contrast has replicate spread to weight by. False for
        a summary-level mzTab, where each group is one averaged column and
        :attr:`stdev` is the deposited spread instead."""
        return self.level == "assay" and any(
            len(v) > 1 for v in self.sample_groups.values()
        )

    @classmethod
    def from_mztab(
        cls, path, *, sample_groups: dict[str, list] | None = None
    ) -> "ProteinDataset":
        """Build from an mzTab file, taking the groups it declares unless
        ``sample_groups`` names its own."""
        from pyteomics import mztab

        table = read_mztab(path)
        design = None
        if sample_groups is None:
            design, sample_groups = mztab_groups(
                table, mztab.MzTab(str(path)).metadata
            )
        return cls(
            quantities=table.quantities,
            sample_groups=sample_groups,
            level=table.level,
            stdev=table.stdev,
            design=design,
            descriptions=table.descriptions,
        )

    @classmethod
    def from_pride(
        cls,
        accession: str,
        *,
        cache: Path | str | None = None,
        which: int = 0,
    ) -> "ProteinDataset":
        """Build from a PRIDE project that deposits an mzTab.

        Raises :class:`LookupError` when the project ships none, which is
        the common case — see this module's coverage note.
        """
        import ppx

        project = ppx.find_project(
            accession, local=str(cache) if cache else None
        )
        found = [
            f for f in project.remote_files() if f.lower().endswith(".mztab")
        ]
        if not found:
            raise LookupError(f"{accession} deposits no mzTab")
        if len(found) > 1:
            log.info(
                "%s deposits %d mzTab files; reading %s",
                accession,
                len(found),
                found[which],
            )
        local = project.download(found[which])
        return cls.from_mztab(local[0] if isinstance(local, list) else local)
