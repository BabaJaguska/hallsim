"""``docs/known-problems.md`` is the defect registry, and its ids are cited
from 28 files. Two failure modes have both happened: parallel branches
allocating the same next number (five colliding pairs at 33dc98a, reconciled
by hand at dd26a7f), and renumbering during that reconciliation leaving cited
ids pointing at nothing. Neither is visible by reading a diff.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "known-problems.md"

#: An entry heading. The checkbox has three states: open, done, and ``~`` for
#: partially fixed — a validator that knows only two silently ignores entries.
ENTRY = re.compile(r"^- \[[ x~]\] \*\*(P\d+\.\d+)\b", re.M)
REFERENCE = re.compile(r"\bP\d+\.\d+\b")

SEARCH_DIRS = ("docs", "src", "demos", "tests")
SUFFIXES = (".md", ".py")


def _entry_ids() -> list[str]:
    return ENTRY.findall(REGISTRY.read_text(encoding="utf-8"))


def _cited() -> dict[str, list[str]]:
    """``{id: [file, ...]}`` for every P-number cited anywhere."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for d in SEARCH_DIRS:
        for path in (ROOT / d).rglob("*"):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for ref in set(REFERENCE.findall(text)):
                out[ref].append(str(path.relative_to(ROOT)))
    return out


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not present")
def test_entry_ids_are_unique():
    """Two branches reaching for the same next free number is invisible in a
    diff and only shows up as an ambiguous cross-reference much later."""
    dupes = {
        pid: n for pid, n in collections.Counter(_entry_ids()).items() if n > 1
    }
    assert not dupes, (
        f"duplicate ids in {REGISTRY.name}: {dupes}. Renumber the entry added "
        "later; grep the tree for the old id first, because cited ids move "
        "with it."
    )


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not present")
def test_every_cited_id_resolves():
    """A renumber that misses a citation leaves a pointer to nothing."""
    ids = set(_entry_ids())
    dangling = {
        ref: files for ref, files in _cited().items() if ref not in ids
    }
    assert (
        not dangling
    ), "cited but not an entry in known-problems.md: " + "; ".join(
        f"{ref} ({', '.join(sorted(files))})"
        for ref, files in sorted(dangling.items())
    )
