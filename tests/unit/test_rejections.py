"""The rejection registry parses, and every row means something.

The registry is evidence, so its failure modes are silent-wrong ones: a
mistyped accession, a class invented in one row, a reason that says nothing.
An id that names the wrong deposit is the worst of them — `BIOMD0000000140`
was recorded as Ihekwaba 2004 in a draft of this file and is Hoffmann 2002.
"""

import re

import pytest

from hallsim.rejections import CLASSES, REGISTRY, distribution, load


@pytest.fixture(scope="module")
def rows():
    return load()


def test_the_registry_parses_and_is_not_empty(rows):
    assert rows


def test_every_class_is_in_the_closed_vocabulary(rows):
    assert {r.failure_class for r in rows} <= set(CLASSES)


def test_every_declared_class_is_documented(rows):
    """A class nothing uses is fine; a class the table never defines is not."""
    table = REGISTRY.read_text()
    for cls in CLASSES:
        assert f"`{cls}`" in table, f"{cls} is undocumented"


def test_ids_are_unique(rows):
    ids = [r.id for r in rows]
    assert len(ids) == len(set(ids)), "a deposit is recorded twice"


def test_ids_are_well_formed_accessions(rows):
    for r in rows:
        assert re.fullmatch(r"(BIOMD|MODEL)\d{10}", r.id), r.id


def test_every_row_carries_a_reason_and_evidence(rows):
    for r in rows:
        assert len(r.reason) > 25, f"{r.id}: reason is too thin to act on"
        assert r.model and r.slot and r.evidence, r.id


def test_evidence_paths_resolve(rows):
    """A row pointing at a document is only useful if the document exists."""
    root = REGISTRY.parent.parent
    for r in rows:
        if "/" in r.evidence and r.evidence.endswith(".md"):
            # scratch/ and docs/review-*.md are gitignored, so a progress log
            # or a reviewer report is valid evidence on the machine that
            # produced it and absent on a fresh clone.
            if r.evidence.startswith(("scratch/", "docs/review-")):
                continue
            assert (root / r.evidence).exists(), f"{r.id}: {r.evidence}"


def test_distribution_counts_every_row(rows):
    assert sum(distribution().values()) == len(rows)
