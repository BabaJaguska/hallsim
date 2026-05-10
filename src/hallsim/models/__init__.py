"""HallSim composable process models.

Each module defines one or more Process subclasses that can be wired
into Composites via topology maps. The most useful builders and Process
subclasses are re-exported here so AI agents and humans can discover
them via ``hallsim.models.<name>`` without knowing each module path.

Lazy import: importing a submodule explicitly (e.g.
``from hallsim.models.eriq import build_eriq_composite``) is still the
canonical path; the re-exports below are convenience aliases.
"""

from hallsim.models.damage_p53_eriq import build_damage_p53_eriq_composite
from hallsim.models.eriq import (
    ERiQEnergyMetabolism,
    ERiQOxidativeStress,
    ERiQSignaling,
    ERiQSignalingNoP53,
    build_eriq_composite,
)
from hallsim.models.kick_event import KickEvent
from hallsim.models.p53_bridge import P53Bridge
from hallsim.models.saturating_removal import SaturatingRemoval
from hallsim.models.stem_cell_niche import build_niche_crosstalk

__all__ = [
    "ERiQEnergyMetabolism",
    "ERiQOxidativeStress",
    "ERiQSignaling",
    "ERiQSignalingNoP53",
    "KickEvent",
    "P53Bridge",
    "SaturatingRemoval",
    "build_damage_p53_eriq_composite",
    "build_eriq_composite",
    "build_niche_crosstalk",
]
