"""The shared network and cache layer under every search source."""

import json
import time

import pytest

from hallsim.search import fetch


def test_an_empty_build_is_not_cached(tmp_path, monkeypatch):
    """A repository serves an empty listing while warming up; cached, it
    would read as the repository holding nothing for a month."""
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="no records"):
        fetch.cached_index("x", lambda: [])
    assert not fetch.index_path("x").exists()
    assert fetch.cached_index("x", lambda: [{"id": 1}]) == [{"id": 1}]
    assert fetch.cached_index("x", lambda: []) == [{"id": 1}]  # stale kept


def test_a_stale_index_outlives_a_failed_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    path = fetch.index_path("y")
    path.write_text(json.dumps([{"id": 2}]))
    old = time.time() - 40 * 86400
    import os

    os.utime(path, (old, old))

    def boom():
        raise OSError("down")

    assert fetch.cached_index("y", boom) == [{"id": 2}]


def test_a_failed_fetch_leaves_no_cached_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)

    def boom():
        raise OSError("down")

    with pytest.raises(OSError):
        fetch.cached_json("k", boom)
    assert list(tmp_path.rglob("*.json")) == []
    assert fetch.cached_json("k", lambda: {"a": 1}) == {"a": 1}
    assert fetch.cached_json("k", boom) == {"a": 1}
