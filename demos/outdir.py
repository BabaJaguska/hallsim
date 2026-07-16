"""Shared output-directory convention for demo figures.

Every demo writes its figures under ``<repo>/outputs/<name>/`` so generated
plots live in one predictable place, one folder per demo, instead of
scattering across ``demos/``, ``one_off_scripts/``, and the repo root.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def outdir(name: str) -> Path:
    """Return ``<repo>/outputs/<name>/``, creating it if needed."""
    d = _ROOT / "outputs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d
