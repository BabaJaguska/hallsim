"""Reusable Process primitives.

Building blocks with no domain content — coupling edges, clamps, events,
integrators, and the neural block — that any composite wires in via topology.
Re-exported here so ``hallsim.models.<name>`` is discoverable without knowing
each module path; importing the submodule directly stays the canonical form.

Domain models (ERiQ, the multi-hallmark composite, mitochondrial ageing, the
stem-cell niche, and the vendored SBML papers) live in ``demos/models/``. They
are worked examples, not part of the framework, and they are transient-response
models — the wrong shape for sustained-perturbation endpoint data.
"""

from hallsim.models.bistable_latch import BistableLatch
from hallsim.models.clamp_edge import (
    ClampEdge,
    clamp_species,
    measure_unclamped_flux,
    place_clamp_rate,
)
from hallsim.models.hill_edge import HillActivationEdge
from hallsim.models.kick_event import KickEvent
from hallsim.models.running_integral import RunningIntegral
from hallsim.models.saturating_removal import SaturatingRemoval

__all__ = [
    "BistableLatch",
    "ClampEdge",
    "HillActivationEdge",
    "KickEvent",
    "RunningIntegral",
    "SaturatingRemoval",
    "clamp_species",
    "measure_unclamped_flux",
    "place_clamp_rate",
]
