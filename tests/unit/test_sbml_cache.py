"""The converted-SBML cache: reuse on a hit, and safe under concurrency.

``_preprocess_sbml`` / ``_strip_events`` write into ``~/.cache/hallsim`` and
are hit by every import. Two properties matter and neither is visible from a
single-process run: a second import must not reconvert, and N processes
racing a cold cache must not read each other's half-written documents.

Subprocesses get a throwaway ``HOME`` so these never touch the developer's
real cache.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hallsim.models.multi_hallmark import GZ06_SBML_PATH
from hallsim.sbml_import import _preprocess_sbml, _strip_events

N_RACERS = 12  # enough to reliably overlap the write window


def _run(script: str, home: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": home, "HALLSIM_COMPILATION_CACHE_DIR": "off"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
    )


IMPORT_ONE = f"""
    import logging; logging.disable(logging.WARNING)
    from hallsim.sbml_import import _preprocess_sbml, _strip_events
    p = _preprocess_sbml({str(GZ06_SBML_PATH)!r})
    q = _strip_events(p)
    import libsbml
    doc = libsbml.readSBMLFromFile(q)
    assert doc.getNumErrors() == 0, f"{{q}} has {{doc.getNumErrors()}} errors"
    assert doc.getModel() is not None
    print("ok")
"""


def test_conversion_is_reused_not_rewritten(tmp_path):
    """A second call with an unchanged source must not rewrite the file."""
    home = str(tmp_path)
    assert _run(IMPORT_ONE, home).returncode == 0

    converted = tmp_path / ".cache" / "hallsim" / "converted"
    before = {p.name: p.stat().st_mtime_ns for p in converted.glob("*.xml")}
    assert before, "no converted file was produced"

    assert _run(IMPORT_ONE, home).returncode == 0
    after = {p.name: p.stat().st_mtime_ns for p in converted.glob("*.xml")}
    assert after == before


def test_edited_source_reconverts(tmp_path, monkeypatch):
    """The key is the source's size and mtime, so an edit must invalidate."""
    monkeypatch.setenv("HOME", str(tmp_path))
    src = tmp_path / "model.xml"
    src.write_bytes(Path(GZ06_SBML_PATH).read_bytes())

    first = _preprocess_sbml(str(src))
    first_mtime = os.stat(first).st_mtime_ns

    os.utime(src, ns=(0, 0))  # a different stamp, same bytes
    second = _preprocess_sbml(str(src))
    assert os.stat(second).st_mtime_ns != first_mtime


@pytest.mark.slow
def test_cold_cache_survives_concurrent_importers(tmp_path):
    """N processes racing an empty cache must all get a parseable document.

    Discriminates: against a non-atomic write this fails with ``libsbml could
    not parse`` on most runs.
    """
    home = str(tmp_path)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(IMPORT_ONE)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "HOME": home,
                "HALLSIM_COMPILATION_CACHE_DIR": "off",
            },
        )
        for _ in range(N_RACERS)
    ]
    results = [(p.wait(), *p.communicate()) for p in procs]
    failures = [(code, err[-400:]) for code, _out, err in results if code != 0]
    assert (
        not failures
    ), f"{len(failures)}/{N_RACERS} racers failed: {failures}"


def test_strip_events_returns_source_when_there_are_no_events(
    tmp_path, monkeypatch
):
    """An event-free model needs no copy — the source path passes through."""
    monkeypatch.setenv("HOME", str(tmp_path))
    expanded = _preprocess_sbml(str(GZ06_SBML_PATH))
    assert _strip_events(expanded) == expanded
