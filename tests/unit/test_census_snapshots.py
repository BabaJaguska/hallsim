"""The shipped census tables, read offline (hallsim.census_snapshots)."""

from hallsim import census_snapshots
from hallsim.census_snapshots import (
    find_shipped_datasets,
    find_shipped_models,
    shipped_table,
)


def test_the_shipped_models_snapshot_answers_offline():
    out = find_shipped_models(r"\bp53\b", limit=5)
    assert out["snapshot"] and out["n_matching"] > 5
    assert len(out["rows"]) == 5
    row = out["rows"][0]
    assert {"accession", "stage", "how", "n_states"} <= set(row)
    assert row["accession"].startswith("BIOMD")
    # Every cell is text, so an id never reads as a float.
    assert all(isinstance(v, str) for v in row.values())
    gated = find_shipped_models(r"\bp53\b", stage="clock")
    assert 0 < gated["n_matching"] < out["n_matching"]
    assert {r["stage"] for r in gated["rows"]} == {"clock"}


def test_the_shipped_datasets_snapshot_narrows_by_organism():
    out = find_shipped_datasets("rapamycin", organism="homo", limit=10)
    assert out["n_matching"] > 0
    assert all("Homo" in r["organism"] for r in out["rows"])
    assert "n_models" not in out["rows"][0]


def test_a_missing_snapshot_reads_as_nothing(monkeypatch, tmp_path):
    shipped_table.cache_clear()
    monkeypatch.setattr(census_snapshots, "CENSUS_DIR", tmp_path)
    try:
        assert find_shipped_datasets("x") == {
            "snapshot": "",
            "n_matching": 0,
            "rows": [],
        }
    finally:
        shipped_table.cache_clear()
