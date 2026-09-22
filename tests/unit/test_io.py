"""Output conventions: where runs and tracked results go, and the stamp
that says what produced them."""

from hallsim import io


def test_versions_names_the_libraries_and_the_commit():
    v = io.versions()
    for key in ("jax", "diffrax", "equinox", "optax", "hallsim"):
        assert v[key], key
    # A git checkout: the short hash, "-dirty" when the tree has edits.
    assert v["hallsim_commit"] and len(v["hallsim_commit"]) >= 7
    assert v["python"].count(".") == 2
    assert v["platform"]


def test_results_dir_is_tracked_and_outputs_are_not(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "_ROOT", tmp_path)
    assert io.results_dir("census") == tmp_path / "results" / "census"
    assert io.outdir("census") == tmp_path / "outputs" / "census"
    assert (tmp_path / "results" / "census").is_dir()
