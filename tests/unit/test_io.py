"""Output conventions: where runs and tracked results go, and the stamp
that says what produced them."""

import re

from hallsim import io


def test_versions_names_the_libraries_and_the_commit():
    v = io.versions()
    for key in ("jax", "diffrax", "equinox", "optax", "hallsim"):
        assert v[key], key
    # A git checkout: `git describe`, which is the tag alone on a release
    # commit ("v0.1.0"), the tag plus distance and short hash after it, or
    # the short hash before any tag; "-dirty" when the tree has edits.
    assert re.fullmatch(
        r"(v\d[\w.]*(-\d+-g[0-9a-f]{7,})?|[0-9a-f]{7,})(-dirty)?",
        v["hallsim_commit"],
    ), v["hallsim_commit"]
    assert v["python"].count(".") == 2
    assert v["platform"]


def test_results_dir_is_tracked_and_outputs_are_not(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "_ROOT", tmp_path)
    assert io.results_dir("census") == tmp_path / "results" / "census"
    assert io.outdir("census") == tmp_path / "outputs" / "census"
    assert (tmp_path / "results" / "census").is_dir()
