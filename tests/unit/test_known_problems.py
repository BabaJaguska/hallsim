"""``docs/known-problems.md`` is the defect registry, and its ids are cited
from 28 files. Two failure modes have both happened: parallel branches
allocating the same next number (five colliding pairs at 33dc98a, reconciled
by hand at dd26a7f), and renumbering during that reconciliation leaving cited
ids pointing at nothing. Neither is visible by reading a diff.
"""

from __future__ import annotations

import collections
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: The registry is two files: the worklist of what is still wrong, and the
#: archive of what is fixed. Ids are allocated and cited across both, so both
#: are read wherever an id has to be unique or has to resolve.
REGISTRY = ROOT / "docs" / "known-problems.md"
ARCHIVE = ROOT / "docs" / "fixed-problems.md"

#: An entry heading. The checkbox has three states: open, done, and ``~`` for
#: partially fixed — a validator that knows only two silently ignores entries.
ENTRY = re.compile(r"^- \[[ x~]\] \*\*(P\d+\.\d+)\b", re.M)
REFERENCE = re.compile(r"\bP\d+\.\d+\b")

#: Ids are prose, so they are cited from prose. Code says why a line exists
#: in words; :func:`test_code_cites_no_registry_ids` keeps it that way.
SEARCH_DIRS = ("docs",)
CODE_DIRS = ("src", "demos", "tests")
SUFFIXES = (".md", ".py")


def _entry_ids() -> list[str]:
    return [
        pid
        for path in (REGISTRY, ARCHIVE)
        if path.exists()
        for pid in ENTRY.findall(path.read_text(encoding="utf-8"))
    ]


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
    ), "cited but not an entry in either registry file: " + "; ".join(
        f"{ref} ({', '.join(sorted(files))})"
        for ref, files in sorted(dangling.items())
    )


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not present")
def test_worklist_holds_no_closed_entries():
    """The worklist is what is still wrong. A closed entry left in it reads
    as outstanding work and pads the file the next session has to triage."""
    closed = re.findall(
        r"^- \[x\] \*\*(P\d+\.\d+)\b",
        REGISTRY.read_text(encoding="utf-8"),
        re.M,
    )
    assert not closed, (
        f"closed entries still in {REGISTRY.name}: {closed}. Move the whole "
        f"block to {ARCHIVE.name}; the id stays cited from the code it "
        "changed, which is why the archive exists rather than a deletion."
    )


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not present")
def test_code_cites_no_registry_ids():
    """A defect id in a comment dates the line to an incident and sends the
    reader out of the file. The reason belongs in the code, the id in the
    registry. Shares its implementation with the pre-commit hook."""
    spec = importlib.util.spec_from_file_location(
        "check_no_registry_ids", ROOT / "scripts" / "check_no_registry_ids.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = module.violations([ROOT / d for d in CODE_DIRS])
    assert not found, "registry ids cited from code:\n" + "\n".join(found)
