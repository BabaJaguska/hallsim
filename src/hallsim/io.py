"""Filesystem conventions for HallSim outputs.

One predictable place for generated artifacts: ``<repo>/outputs/<name>/``,
one folder per run/demo, instead of scattering plots across the tree.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def outdir(name: str) -> Path:
    """Return ``<repo>/outputs/<name>/``, creating it if needed."""
    d = _ROOT / "outputs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(name: str) -> Path:
    """Return ``<repo>/results/<name>/``, creating it if needed. Unlike
    :func:`outdir` this folder is tracked: it holds the small, stamped
    products of a run that the repository carries (a census table, a
    benchmark score), so git history is their time series."""
    d = _ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def versions() -> dict:
    """What a result was produced under: ``jax``, ``diffrax``, ``equinox``,
    ``optax`` and ``hallsim`` from installed metadata, ``hallsim_commit``
    from ``git describe`` on the checkout (nearest tag, distance and short
    hash, ``-dirty`` when the tree has uncommitted changes), ``python`` and
    ``platform``. ``None`` where a package is not installed or the tree is
    not a git checkout. Written beside every run's numbers so a table can
    be traced to the code that made it."""
    import importlib.metadata as md
    import platform
    import subprocess

    out = {}
    for dist in ("jax", "diffrax", "equinox", "optax", "hallsim"):
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = None
    try:
        out["hallsim_commit"] = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(_ROOT),
                    "describe",
                    "--tags",
                    "--always",
                    "--dirty",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001 - no git, no commit
        out["hallsim_commit"] = None
    out["python"] = platform.python_version()
    out["platform"] = platform.platform()
    return out


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


CHECKSUM_FILE = "SHA256SUMS"


def file_sha256(path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sums(sums: Path) -> dict[str, str]:
    if not sums.exists():
        return {}
    out = {}
    for line in sums.read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        if digest and name:
            out[name] = digest
    return out


def record_checksum(path) -> str:
    """Write ``path``'s SHA-256 into the ``SHA256SUMS`` beside it (one
    ``<hex>  <name>`` line per file, as ``sha256sum`` writes it) and return
    the digest."""
    path = Path(path)
    sums = path.parent / CHECKSUM_FILE
    digest = file_sha256(path)
    entries = {**_read_sums(sums), path.name: digest}
    sums.write_text("".join(f"{d}  {n}\n" for n, d in sorted(entries.items())))
    return digest


def verify_checksum(path) -> str | None:
    """Check ``path`` against the ``SHA256SUMS`` beside it. Returns the
    digest when the file is listed there and matches, ``None`` when it is
    not listed; a mismatch raises."""
    path = Path(path)
    sums = path.parent / CHECKSUM_FILE
    expected = _read_sums(sums).get(path.name)
    if expected is None:
        return None
    digest = file_sha256(path)
    if digest != expected:
        raise ValueError(
            f"{path} does not match the SHA-256 recorded in {sums} "
            f"({digest} != {expected}); delete it and fetch again."
        )
    return digest
