"""SBML auto-import — convert BioModels SBML files into Process instances.

Uses ``sbmltoodejax`` to convert SBML species and reactions into a
JAX-compatible RHS function, then wraps it as a :class:`Process` with
auto-generated ports and metadata.

Example
-------
>>> reed_gsh = process_from_sbml(268, name="reed_gsh")
>>> reed_gsh.ports_schema()  # auto-generated from SBML species
>>> reed_gsh.metadata()      # SBML annotations, species descriptions

This is a Phase 3 feature — the stub is provided now so that the
import path exists and tests can verify the interface.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def process_from_sbml(model_id: int, name: str | None = None):
    """Load a BioModels SBML model and wrap it as a Process.

    Parameters
    ----------
    model_id:
        BioModels numeric ID, e.g. ``268`` for Reed2008 Glutathione.
    name:
        Human-readable name for the process.  Defaults to
        ``"biomodel_{model_id}"``.

    Returns
    -------
    Process instance with auto-generated ports from SBML species.

    Raises
    ------
    NotImplementedError
        This is a Phase 3 stub.
    """
    raise NotImplementedError(
        f"SBML import for BioModels #{model_id} is not yet implemented. "
        "This is planned for Phase 3 of the architecture redesign."
    )
