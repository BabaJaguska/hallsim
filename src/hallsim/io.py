"""Filesystem conventions for HallSim outputs.

One predictable place for generated artifacts: ``<repo>/outputs/<name>/``,
one folder per run/demo, instead of scattering plots across the tree.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def outdir(name: str) -> Path:
    """Return ``<repo>/outputs/<name>/``, creating it if needed."""
    d = _ROOT / "outputs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_run_dir(name: str, stamp: str | None = None) -> Path:
    """Timestamped subfolder of :func:`outdir`, with ``latest`` symlinked to
    it — so a run never overwrites the last one and figure scripts can follow
    ``<name>/latest``. ``stamp`` overrides the generated timestamp.
    """
    base = outdir(name)
    base.mkdir(parents=True, exist_ok=True)
    label = stamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = base / label
    # Two runs started in the same second must not share a folder.
    n = 2
    while run.exists():
        run = base / f"{label}-{n}"
        n += 1
    run.mkdir(parents=True, exist_ok=False)
    latest = base / "latest"
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists():
        shutil.rmtree(latest)
    latest.symlink_to(run.name)
    return run
