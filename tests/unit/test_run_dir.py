"""Two runs started in the same second get two folders."""

from hallsim import io


def test_a_run_dir_is_never_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "outdir", lambda name: tmp_path / name)
    a = io.make_run_dir("demo", stamp="same")
    b = io.make_run_dir("demo", stamp="same")
    assert a != b and a.is_dir() and b.is_dir()
    assert (tmp_path / "demo" / "latest").resolve() == b.resolve()
