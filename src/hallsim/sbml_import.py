"""SBML auto-import — convert BioModels SBML files into Process instances.

Uses ``sbmltoodejax`` to convert SBML species and reactions into a
JAX-compatible RHS function, then wraps it as a :class:`Process` with
auto-generated ports and metadata.

Example
-------
>>> proc = process_from_sbml(10, name="mapk_cascade")
>>> proc.ports_schema()    # auto-generated from SBML species
>>> proc.metadata()        # SBML annotations

Requires ``sbmltoodejax`` to be installed::

    pip install sbmltoodejax
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process

log = logging.getLogger(__name__)


class SBMLProcess(Process):
    """Process auto-generated from an SBML model via sbmltoodejax.

    This wraps the JAX-compiled model step function from sbmltoodejax,
    exposing SBML species as EVOLVED ports.

    Not constructed directly — use :func:`process_from_sbml`.
    """

    _species_names: tuple[str, ...] = ()
    _species_y0: tuple[float, ...] = ()
    _model: Any = None  # sbmltoodejax model object
    _w0: Any = None
    _c: Any = None
    _name: str = ""

    def ports_schema(self):
        return {
            name: Port(
                role=PortRole.EVOLVED,
                default=float(y0),
                units="dimensionless",
                description=f"SBML species: {name}",
            )
            for name, y0 in zip(self._species_names, self._species_y0)
        }

    def derivative(self, t, state):
        # Build y vector in the order sbmltoodejax expects
        y = jnp.array([state[name] for name in self._species_names])
        dydt = self._model.modelstepfunc.ratefunc(y, t, self._w0, self._c)
        return {
            name: dydt[i]
            for i, name in enumerate(self._species_names)
        }

    def metadata(self):
        base = super().metadata()
        base["sbml_name"] = self._name
        base["n_species"] = len(self._species_names)
        return base


def process_from_sbml(
    model_id: int,
    name: str | None = None,
) -> SBMLProcess:
    """Load a BioModels SBML model and wrap it as a Process.

    Parameters
    ----------
    model_id:
        BioModels numeric ID, e.g. ``10`` for Kholodenko2000 MAPK.
    name:
        Human-readable name for the process. Defaults to
        ``"biomodel_{model_id}"``.

    Returns
    -------
    SBMLProcess instance with auto-generated ports from SBML species.

    Raises
    ------
    ImportError
        If ``sbmltoodejax`` is not installed.
    """
    try:
        from sbmltoodejax.utils import load_biomodel
    except ImportError:
        raise ImportError(
            "sbmltoodejax is required for SBML import. "
            "Install it with: pip install sbmltoodejax"
        )

    name = name or f"biomodel_{model_id}"
    log.info(f"Loading BioModels #{model_id} as '{name}'...")

    model, y0, w0, c = load_biomodel(model_id)

    # Extract species names from the model's index mapping
    y_indexes = model.modelstepfunc.y_indexes
    # y_indexes maps species_name -> index; sort by index to get ordered names
    species_ordered = sorted(y_indexes.items(), key=lambda x: x[1])
    species_names = tuple(name for name, _ in species_ordered)

    log.info(f"Loaded {len(species_names)} species: {species_names}")

    proc = object.__new__(SBMLProcess)
    # Set fields directly (bypassing __init__ since this is a dynamic construction)
    object.__setattr__(proc, '_species_names', species_names)
    object.__setattr__(proc, '_species_y0', tuple(float(y0[i]) for i in range(len(species_names))))
    object.__setattr__(proc, '_model', model)
    object.__setattr__(proc, '_w0', w0)
    object.__setattr__(proc, '_c', c)
    object.__setattr__(proc, '_name', name)

    return proc
