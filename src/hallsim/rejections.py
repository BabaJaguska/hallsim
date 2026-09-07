"""The rejection registry — which deposits were screened out, and why.

A search that ends in "nothing suitable" is a result only if the reasons are
recorded. `docs/rejections.md` is that record, and this reads it, so the
distribution of failure classes can be counted rather than re-derived from ten
review documents written in prose.

    from hallsim.rejections import load, distribution
    distribution()          # {'wrong-formalism': 9, 'consumes-not-emits': 4, ...}

Reached from the CLI as ``simulate rejections``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The closed vocabulary. A class outside it is a parse error, so a new kind
#: of failure has to be named in `docs/rejections.md` before it can be used —
#: which is the point: an open vocabulary cannot be counted.
CLASSES = (
    "consumes-not-emits",
    "wrong-formalism",
    "no-dynamic-range",
    "not-identifiable",
    "cell-type-mismatch",
    "numerically-unusable",
    "no-provenance",
)

REGISTRY = Path(__file__).resolve().parents[2] / "docs" / "rejections.md"


@dataclass(frozen=True)
class Rejection:
    """One screened-out deposit."""

    id: str
    model: str
    slot: str
    failure_class: str
    reason: str
    evidence: str


def load(path: "Path | str | None" = None) -> list[Rejection]:
    """Every recorded rejection, in file order."""
    text = Path(path or REGISTRY).read_text()
    out: list[Rejection] = []
    for line in text.splitlines():
        if not line.startswith("| BIOMD") and not line.startswith("| MODEL"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            raise ValueError(
                f"rejections.md: expected 6 cells, got {len(cells)}: {line}"
            )
        if cells[3] not in CLASSES:
            raise ValueError(
                f"rejections.md: {cells[0]} has class {cells[3]!r}, which is "
                f"not one of {CLASSES}. Name it in the file's class table "
                f"first."
            )
        out.append(Rejection(*cells))
    return out


def distribution(path=None) -> dict[str, int]:
    """``{failure class: count}``, commonest first."""
    from collections import Counter

    counts = Counter(r.failure_class for r in load(path))
    return dict(counts.most_common())


def summary(path=None) -> str:
    """The registry as a table, for a report or the CLI."""
    rows = load(path)
    dist = distribution(path)
    width = max(len(c) for c in dist) if dist else 0
    lines = [f"{len(rows)} deposits screened out"]
    for cls, n in dist.items():
        bar = "#" * n
        lines.append(f"  {cls:<{width}}  {n:3d}  {bar}")
    slots = sorted({r.slot for r in rows})
    lines.append(f"  slots: {', '.join(slots)}")
    return "\n".join(lines)
