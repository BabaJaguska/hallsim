"""Vendored SBML sources for the bundled models.

Each published model lives under ``<first-author><year>/`` beside its BioModels
id. :func:`sbml_source` returns a path when the file is vendored and the plain
BioModels id otherwise, so ``process_from_sbml`` downloads and caches it — a
checkout missing a file still builds, just with network on first import.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def sbml_source(directory: str, filename: str, biomodel_id: str) -> str:
    """Vendored SBML path if present, else ``biomodel_id`` to fetch by."""
    path = _ROOT / directory / filename
    return str(path) if path.is_file() else biomodel_id


def sbml_dir(directory: str) -> Path:
    """Directory holding one publication's vendored SBML files."""
    return _ROOT / directory
