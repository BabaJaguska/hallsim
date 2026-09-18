"""Every Python block in the user-facing pages runs as written.

The blocks of one page run top to bottom in one namespace, the way a reader
follows them, so a later block may use what an earlier one defined. Minutes
on CPU, needs the network (BioModels, the GEO dataset) and the multi-hallmark
demo, so ``make test`` skips it; ``make test-docs`` runs it before a release.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("README.md", "docs/architecture.md", "docs/calibration.md")

pytestmark = [pytest.mark.slow, pytest.mark.network, pytest.mark.demo]


def python_blocks(page: str) -> list[tuple[int, str]]:
    """``(line, source)`` for each fenced ``python`` block on the page."""
    text = (ROOT / page).read_text(encoding="utf-8")
    return [
        (text[: m.start()].count("\n") + 1, m.group(1))
        for m in re.finditer(r"```python\n(.*?)```", text, re.S)
    ]


@pytest.mark.parametrize("page", PAGES)
def test_page_examples_run_as_written(page, monkeypatch):
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": f"doc_{Path(page).stem}"}
    blocks = python_blocks(page)
    assert blocks, f"{page} has no python blocks"
    for line, source in blocks:
        try:
            exec(compile(source, f"{page}:{line}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - any failure is the finding
            pytest.fail(f"{page}:{line} raised {type(exc).__name__}: {exc}")
