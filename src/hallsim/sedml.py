"""A deposit's own simulated horizon, read from its SED-ML.

Every screen has to run a model over some window, and a window the caller
picked is a number about nobody's biology: a model whose dynamics occupy
144,000 time units screened over 10 has barely left its initial condition. A
deposit that ships SED-ML states the horizon its author simulated over, in the
same time its rate laws are written in — so this settles the window **without**
settling the clock, which is a separate and harder problem. A deposit that
declares no time unit still has a usable horizon here.

The catch is that BioModels auto-generates a SED-ML stub for deposits that
shipped none, whose single time course is ``auto_ten_seconds`` ending at 10. So
the presence of a ``.sedml`` file asserts nothing, and a census column counting
files cannot tell the two apart: of 172 cached files, 57 are that stub and every
one of them re-asserts the 10 this module exists to replace. Only an authored
time course is evidence, which is why the predicate here is *authored*.

    from hallsim.sedml import authored_horizon
    authored_horizon("BIOMD0000000490")   # 144000.0
    authored_horizon("BIOMD0000000151")   # None — the generated stub
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger(__name__)

#: A time course whose id carries this prefix was written by the repository's
#: exporter rather than by the model's author. One convention of one source,
#: not a list of names: discarding it is safe in both directions, because no
#: authored horizon in the cached corpus is 10, and a caller who loses one
#: falls back to 10 and gets the same answer anyway.
GENERATED_PREFIX = "auto_"


def horizon_in(text: str) -> float | None:
    """The authored horizon in one SED-ML document, or ``None``.

    The largest ``outputEndTime`` over the time courses the author wrote,
    ignoring generated ones. Several time courses means several simulated
    protocols and the longest is the one that contains the others.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        log.info("SED-ML did not parse: %s", str(exc)[:120])
        return None

    ends = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "uniformTimeCourse":
            continue
        if (element.get("id") or "").startswith(GENERATED_PREFIX):
            continue
        raw = element.get("outputEndTime")
        if raw is None:
            continue
        try:
            end = float(raw)
        except ValueError:
            continue
        start = element.get("outputStartTime") or element.get("initialTime")
        try:
            begin = float(start) if start is not None else 0.0
        except ValueError:
            begin = 0.0
        # The horizon is the window the author looked at, so a course that
        # starts late still has to be integrated from the initial condition.
        if end > begin:
            ends.append(end)
    return max(ends) if ends else None


def sedml_files(accession: str) -> list[Path]:
    """Cached SED-ML beside a BioModels deposit, newest name order."""
    from hallsim.search.fetch import cache_dir

    root = cache_dir("biomodels")
    return sorted(
        list((root / accession).glob("*.sedml"))
        + list(root.glob(f"{accession}*.sedml"))
    )


def authored_horizon(source) -> float | None:
    """The horizon a deposit's author simulated over, or ``None``.

    ``source`` is a ``.sedml`` path, a directory holding one, or a BioModels
    accession whose files are already cached. Returns ``None`` when nothing is
    cached, when the only time courses are generated, or when the document does
    not parse — in every one of those cases the deposit has not stated a
    horizon, and the caller must decide rather than be handed a default here.
    """
    candidates: list[Path] = []
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.glob("*.sedml"))
        else:
            candidates = sedml_files(str(source))

    horizons = [
        h
        for h in (
            horizon_in(p.read_text(errors="replace"))
            for p in candidates
            if p.is_file()
        )
        if h is not None
    ]
    return max(horizons) if horizons else None
