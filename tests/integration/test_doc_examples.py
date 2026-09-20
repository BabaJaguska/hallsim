"""Every Python block in the user-facing pages runs as written.

The blocks of one page run top to bottom in one namespace, the way a reader
follows them, so a later block may use what an earlier one defined. Minutes
on CPU, needs the network (BioModels, the GEO dataset) and the multi-hallmark
demo, so ``make test`` skips it; ``make test-docs`` runs it before a release.
A repository that is down is not a finding about the page: a connection
failure or a 5xx skips the page and names the service, while a 4xx, which
is a request of ours the service refused, still fails it.
"""

import re
import urllib.error
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


_OUTAGE = re.compile(r"HTTP Error (5\d\d|429)|Service Unavailable")


@pytest.mark.parametrize("page", PAGES)
def test_page_examples_run_as_written(page, monkeypatch, caplog):
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": f"doc_{Path(page).stem}"}
    blocks = python_blocks(page)
    assert blocks, f"{page} has no python blocks"
    for line, source in blocks:
        caplog.clear()
        try:
            exec(compile(source, f"{page}:{line}", "exec"), namespace)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                pytest.skip(f"{page}:{line}: {exc.url} answered {exc.code}")
            pytest.fail(f"{page}:{line} raised HTTPError: {exc}")
        except urllib.error.URLError as exc:
            pytest.skip(f"{page}:{line}: no connection: {exc.reason}")
        except Exception as exc:  # noqa: BLE001 - any failure is the finding
            # A search skips a repository that is down and carries on, so
            # the block then fails downstream of the outage, not on itself.
            outage = _OUTAGE.search(caplog.text)
            if outage:
                pytest.skip(
                    f"{page}:{line}: a repository was down during this "
                    f"block ({outage.group(0)}); it then raised "
                    f"{type(exc).__name__}: {exc}"
                )
            pytest.fail(f"{page}:{line} raised {type(exc).__name__}: {exc}")
