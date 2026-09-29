"""Write the shipped census snapshot from a census-data run.

    python scripts/export_census.py --run outputs/census-data/latest

Selects the rows a reader can open, keeps the columns a consumer needs, and
writes `datasets_loadable_<date>.csv.gz` into the package's reference folder.
`--replace` removes the snapshot it supersedes, so the folder holds one.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

REFERENCE = Path(__file__).resolve().parents[1] / (
    "src/hallsim/reference/census"
)

#: What a consumer needs to choose a dataset, in the order it reads.
COLUMNS = [
    "source",
    "accession",
    "title",
    "organism",
    "species",
    "modality",
    "n_samples",
    "n_arms",
    "control",
    "n_timepoints",
    "time_unit",
    "contrast_kind",
    "design_recovered",
    "perturbed",
    "perturbations",
    "loader",
    "pubmed",
    "n_models",
    "n_shared_ids",
    "n_same_species",
]


def rows_of(run: Path):
    with (run / "rows.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def export_models(run: Path, out: Path, stamp: str, replace: bool) -> Path:
    """Ship the models table `census report` built, which carries the deposit
    metadata the screened rows alone do not."""
    frame = pd.read_csv(run / "census.csv")
    dest = out / f"models_{stamp}.csv.gz"
    frame.to_csv(dest, index=False, compression="gzip")
    if replace:
        for old in sorted(out.glob("models_*.csv.gz")):
            if old != dest:
                old.unlink()
                print(f"removed {old.name}")
    runs = ("clean", "pass", "at_rest", "annotated", "clock")
    print(f"{len(frame)} deposits -> {dest}")
    print(f"  {dest.stat().st_size / 1e6:.2f} MB")
    print(f"  runnable:            {int(frame.stage.isin(runs).sum())}")
    print(f"  composes as-is:      {int((frame.stage == 'pass').sum())}")
    print(
        "  rule-based (no species, has rate rules): "
        f"{int(((frame.n_species.fillna(0) == 0) & (frame.n_rate_rules.fillna(0) > 0)).sum())}"
    )
    return dest


def export(run: Path, out: Path, stamp: str, replace: bool) -> Path:
    kept = []
    total = 0
    for row in rows_of(run):
        total += 1
        if row.get("loadable"):
            kept.append({c: row.get(c) for c in COLUMNS})
    frame = pd.DataFrame(kept, columns=COLUMNS)
    dest = out / f"datasets_loadable_{stamp}.csv.gz"
    frame.to_csv(dest, index=False, compression="gzip")
    if replace:
        for old in sorted(out.glob("datasets_loadable_*.csv.gz")):
            if old != dest:
                old.unlink()
                print(f"removed {old.name}")
    print(f"{len(frame)} loadable of {total} rows -> {dest}")
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")
    # An absent control is an empty string, not a missing value, so counting
    # what is not null counts every row.
    named = frame["control"].fillna("").astype(str).str.strip()
    times = pd.to_numeric(frame["n_timepoints"], errors="coerce")
    print(f"  naming a control:        {int((named != '').sum())}")
    print(f"  design recovered:        {int(frame['design_recovered'].sum())}")
    print(f"  more than one timepoint: {int((times > 1).sum())}")
    print(
        f"  both:                    "
        f"{int(((named != '') & (times > 1)).sum())}"
    )
    print(f"  by loader: {frame['loader'].value_counts().to_dict()}")
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, default=Path("outputs/census-data/latest")
    )
    p.add_argument(
        "--models-run",
        type=Path,
        default=None,
        help="a `simulate census` run whose `report` has been built",
    )
    p.add_argument("--out", type=Path, default=REFERENCE)
    p.add_argument("--stamp", default=date.today().isoformat())
    p.add_argument("--replace", action="store_true")
    a = p.parse_args()
    if a.models_run is not None:
        export_models(a.models_run, a.out, a.stamp, a.replace)
    else:
        export(a.run, a.out, a.stamp, a.replace)
