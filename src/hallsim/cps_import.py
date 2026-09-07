"""Import a COPASI ``.cps`` model by converting it to SBML on the way in.

A paper's supplement is often a ``.cps`` rather than a deposited SBML file —
COPASI is what much of the field models in, and those files are complete,
parameterised and already validated by their authors. They are the cheapest
uncollected models there are, and nothing here has been able to read them.

Conversion is COPASI's own SBML exporter, not a reimplementation: the ``.cps``
schema is large, and translating it by hand would be reinventing something the
tool that defines the format already does correctly.

    from hallsim.sbml_import import process_from_sbml
    proc = process_from_sbml("supplement/model.cps", name="paper")

``process_from_sbml`` routes a ``.cps`` path here automatically, so every
downstream check — triage, the numerical screen, reporters, calibration —
sees an ordinary SBML import and needs no special case.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: SBML level/version to export. L3V2 carries the constructs the importer
#: already understands; COPASI down-converts what it must and reports it.
SBML_LEVEL, SBML_VERSION = 3, 2


class CopasiUnavailableError(ImportError):
    """The COPASI bindings are not installed."""


def _copasi():
    try:
        import COPASI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CopasiUnavailableError(
            "reading a .cps needs COPASI's own bindings: "
            "`uv pip install python-copasi`, or install the 'copasi' extra. "
            "They are an extra rather than a base dependency because the "
            "wheel is a native build."
        ) from exc
    return COPASI


def cps_to_sbml(
    cps_path: str | Path, out_path: str | Path | None = None
) -> str:
    """Convert a ``.cps`` to SBML and return the path written.

    Caches under ``~/.cache/hallsim/copasi/`` keyed on the file's own contents,
    so re-importing an unchanged model is a local read and an edited one is
    converted again.
    """
    COPASI = _copasi()
    cps_path = Path(cps_path)
    if not cps_path.is_file():
        raise FileNotFoundError(f"no such .cps file: {cps_path}")

    if out_path is None:
        digest = hashlib.sha256(cps_path.read_bytes()).hexdigest()[:16]
        out = Path.home() / ".cache" / "hallsim" / "copasi"
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{cps_path.stem}_{digest}.xml"
        if out_path.exists():
            log.info("Using cached SBML for '%s'.", cps_path.name)
            return str(out_path)
    out_path = Path(out_path)

    dm = COPASI.CRootContainer.addDatamodel()
    if not dm.loadModel(str(cps_path)):
        raise ValueError(
            f"COPASI could not load {cps_path}: "
            f"{COPASI.CCopasiMessage.getAllMessageText()}"
        )
    if not dm.exportSBML(str(out_path), True, SBML_LEVEL, SBML_VERSION):
        raise ValueError(
            f"COPASI loaded {cps_path} but could not export it as SBML "
            f"L{SBML_LEVEL}V{SBML_VERSION}: "
            f"{COPASI.CCopasiMessage.getAllMessageText()}"
        )
    _drop_model_history(out_path)
    log.info(
        "Converted COPASI '%s' to SBML L%dV%d.",
        cps_path.name,
        SBML_LEVEL,
        SBML_VERSION,
    )
    return str(out_path)


def _drop_model_history(path: Path) -> None:
    """Remove the RDF creation history COPASI writes on export.

    libsbml reads it back as a *warning* — the annotation is missing
    attributes libsbml wants — and ``sbmltoodejax.parse`` refuses any document
    with ``getNumErrors() > 0``, warnings included. So every COPASI export
    fails to import over provenance metadata that carries no dynamics. Dropped
    here rather than worked around downstream, because the file this writes is
    a build artefact, not the deposit.
    """
    import libsbml

    doc = libsbml.readSBML(str(path))
    model = doc.getModel()
    if model is None:
        return
    model.unsetModelHistory()
    for getter, count in (
        (model.getSpecies, model.getNumSpecies()),
        (model.getReaction, model.getNumReactions()),
        (model.getCompartment, model.getNumCompartments()),
        (model.getParameter, model.getNumParameters()),
    ):
        for i in range(count):
            getter(i).unsetModelHistory()
    libsbml.writeSBMLToFile(doc, str(path))


def is_cps(path) -> bool:
    """Whether ``path`` looks like a COPASI file, by extension."""
    return isinstance(path, (str, Path)) and str(path).lower().endswith(".cps")
