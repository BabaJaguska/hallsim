"""The dated census tables shipped with the package, read offline: which
deposits exist for a mechanism and what is wrong with each.

``reference/census/PROVENANCE.md`` describes each table and the ways it
misleads; the readings that cost people most are repeated on the functions
here, since a caller — an agent over MCP in particular — sees the docstring
and not the folder. Nothing requires these files: a missing snapshot reads
as an empty table, and the live search is the fallback.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CENSUS_DIR = Path(__file__).resolve().parent / "reference" / "census"

#: The columns a reader needs to choose a deposit, in the order they help.
MODEL_COLUMNS = (
    "accession",
    "name",
    "stage",
    "how",
    "status",
    "kind",
    "n_species",
    "n_states",
    "n_rate_rules",
    "n_events",
    "time_unit_declared",
    "annotation_coverage",
    "taxon",
    "year",
    "pubmed",
    "go_terms",
    "has_sedml",
    "data_files",
)
DATASET_COLUMNS = (
    "source",
    "accession",
    "title",
    "organism",
    "modality",
    "n_samples",
    "n_arms",
    "control",
    "n_timepoints",
    "time_unit",
    "contrast_kind",
    "design_recovered",
    "perturbations",
    "loader",
    "pubmed",
    "n_shared_ids",
)
#: What a pattern is matched against.
MODEL_TEXT = (
    "accession",
    "name",
    "paper",
    "about",
    "curator_note",
    "go_terms",
    "taxon",
)
DATASET_TEXT = ("accession", "title", "perturbations", "organism")


def snapshot_path(prefix: str) -> Path | None:
    """The latest ``<prefix>_<date>.csv.gz`` under :data:`CENSUS_DIR`, or
    ``None`` where none ships."""
    found = sorted(CENSUS_DIR.glob(f"{prefix}_*.csv.gz"))
    return found[-1] if found else None


@functools.lru_cache(maxsize=4)
def shipped_table(prefix: str) -> pd.DataFrame:
    """One snapshot, every cell a string and an empty one ``""``: the tables
    mix ids, floats and lists, and a typed read shows a PubMed id as
    ``21772264.0``. ``attrs["snapshot"]`` carries the date. Empty where
    none ships."""
    path = snapshot_path(prefix)
    if path is None:
        log.info("no %s snapshot under %s", prefix, CENSUS_DIR)
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.attrs["snapshot"] = path.stem.rsplit("_", 1)[-1].split(".")[0]
    return df


def _select(df, pattern, text_columns, exact: dict, within: dict):
    """Rows whose text columns match ``pattern``, narrowed by ``exact``
    (categorical columns, case-insensitive) and ``within`` (a substring)."""
    if df.empty:
        return df
    cols = [c for c in text_columns if c in df]
    text = (
        df[cols[0]].str.cat(df[cols[1:]], sep=" ")
        if len(cols) > 1
        else df[cols[0]]
    )
    keep = text.str.contains(pattern, flags=re.I, regex=True)
    for col, value in exact.items():
        if value is not None and col in df:
            keep &= df[col].str.lower() == str(value).lower()
    for col, value in within.items():
        if value is not None and col in df:
            keep &= df[col].str.contains(
                re.escape(str(value)), case=False, regex=True
            )
    return df[keep]


def _answer(df, hits, columns, limit: int) -> dict:
    rows = hits[[c for c in columns if c in hits]].head(limit)
    return {
        "snapshot": df.attrs.get("snapshot", ""),
        "n_matching": int(len(hits)),
        "rows": rows.to_dict("records"),
    }


def find_shipped_models(
    pattern: str,
    *,
    stage: str | None = None,
    kind: str | None = None,
    taxon: str | None = None,
    limit: int = 25,
) -> dict:
    """Deposits in the models snapshot whose name, paper title, abstract,
    curator note, GO terms or taxon match ``pattern`` (a regex), each with
    the first gate it failed and what would fix it; ``n_matching`` counts
    every match and ``rows`` holds the first ``limit``. ``stage`` narrows
    to one gate (``pass``, ``clean``, ``at_rest``, ``annotated``,
    ``clock``, ``kinetic``, ``solves``, ``imports``), ``kind`` to a
    formalism (``ode``), ``taxon`` to a substring of the organism. An empty
    ``snapshot`` means no table ships here: search live instead.

    Two readings that cost people time. ``stage`` names the first gate a
    deposit failed, not a quality score: ``pass`` is the best outcome,
    ``clean`` cleared every gate and still carries a note in ``how``, and
    everything down to ``clock`` runs. A GO term is a claim about the
    paper's subject, not about what the model emits, and a deposit with
    zero species keeps its state in rate rules — read ``n_states``, and
    screen the species before trusting either."""
    df = shipped_table("models")
    hits = _select(
        df,
        pattern,
        MODEL_TEXT,
        {"stage": stage, "kind": kind},
        {"taxon": taxon},
    )
    return _answer(df, hits, MODEL_COLUMNS, limit)


def find_shipped_datasets(
    pattern: str,
    *,
    organism: str | None = None,
    modality: str | None = None,
    contrast: str | None = None,
    loader: str | None = None,
    limit: int = 25,
) -> dict:
    """Deposits in the loadable-datasets snapshot — each with a contrast, a
    measured quantity, a model match and a reader that opens it — whose
    title, perturbations, organism or accession match ``pattern`` (a
    regex); ``n_matching`` counts every match and ``rows`` holds the first
    ``limit``. ``organism`` is a substring; ``modality`` (``expression``,
    ``proteomics``, ``metabolomics``), ``contrast`` (``dynamics``,
    ``declared``, ``course``, ``arms``) and ``loader`` are exact. An empty
    ``snapshot`` means no table ships here: search live instead.

    Only the ``direct`` matching route selects on a dataset's own
    identifiers, so ``n_shared_ids`` above zero is the per-dataset strength
    and zero means that route found nothing, not that no overlap exists.
    A ``contrast`` is not always a design you can set up today:
    ``design_recovered`` says whether the arms are actually known."""
    df = shipped_table("datasets_loadable")
    hits = _select(
        df,
        pattern,
        DATASET_TEXT,
        {"modality": modality, "contrast_kind": contrast, "loader": loader},
        {"organism": organism},
    )
    return _answer(df, hits, DATASET_COLUMNS, limit)
