"""Stem Cell Niche — age-dependent decline in niche signaling.

Models the deterioration of the stem cell niche by contributing
negative (decay) derivatives to key ligand/receptor species in the
Sivakumar2011 crosstalk model (BIOMD0000000398).

The niche process composes additively with the SBML crosstalk model:
both declare EVOLVED ports on the same species, and the Composite sums
their derivatives.  At severity=0 the niche contributes nothing; at
severity=1 it drives strong ligand depletion → stem cell exhaustion.

Biological rationale
--------------------
With aging, the stem cell niche deteriorates:
- Wnt ligand availability declines → reduced self-renewal
- EGF/growth factor signaling drops → less proliferative capacity
- Shh signaling weakens → niche structural deterioration
- Notch lateral inhibition is disrupted → impaired asymmetric division

These map directly to the "Stem Cell Exhaustion" hallmark
(Lopez-Otin et al., Cell 2023).

Usage
-----
>>> from hallsim.models.stem_cell_niche import StemCellNiche, build_niche_crosstalk
>>> comp = build_niche_crosstalk(severity=0.5)
>>> result = Simulator().run(comp, t_span=(0.0, 100.0), dt=0.5)
"""

from __future__ import annotations

from pathlib import Path

from hallsim.process import Port, PortRole, Process

# Sivakumar2011 crosstalk model (BIOMD0000000398) species IDs
# for the four niche ligands / receptors.
CROSSTALK_WNT = "s107"  # Wnt (extracellular)
CROSSTALK_EGF = "s96"  # EGF
CROSSTALK_SHH = "s81"  # Shh (Sonic Hedgehog)
CROSSTALK_NOTCH = "s57"  # Notch receptor


class StemCellNiche(Process):
    """Age-dependent niche deterioration for Sivakumar2011 crosstalk model.

    Contributes decay derivatives to niche ligand species, scaled by
    ``severity``.  Composes additively with the SBML crosstalk process.

    Parameters
    ----------
    severity:
        Niche deterioration level in [0, 1].
        0 = healthy niche (no effect), 1 = severely deteriorated.
    wnt_decay:
        Decay rate constant for Wnt ligand.
    egf_decay:
        Decay rate constant for EGF.
    shh_decay:
        Decay rate constant for Shh.
    notch_decay:
        Decay rate constant for Notch receptor availability.
    """

    severity: float = 0.0
    wnt_decay: float = 0.08
    egf_decay: float = 0.08
    shh_decay: float = 0.06
    notch_decay: float = 0.04

    def ports_schema(self):
        return {
            CROSSTALK_WNT: Port(
                role=PortRole.EVOLVED,
                default=5.0,
                units="dimensionless",
                description="Wnt ligand — niche self-renewal signal",
            ),
            CROSSTALK_EGF: Port(
                role=PortRole.EVOLVED,
                default=5.0,
                units="dimensionless",
                description="EGF — niche proliferative signal",
            ),
            CROSSTALK_SHH: Port(
                role=PortRole.EVOLVED,
                default=5.0,
                units="dimensionless",
                description="Shh — niche structural signal",
            ),
            CROSSTALK_NOTCH: Port(
                role=PortRole.EVOLVED,
                default=5.0,
                units="dimensionless",
                description="Notch receptor — lateral inhibition / asymmetric division",
            ),
        }

    def derivative(self, t, state):
        s = self.severity
        return {
            CROSSTALK_WNT: -s * self.wnt_decay * state[CROSSTALK_WNT],
            CROSSTALK_EGF: -s * self.egf_decay * state[CROSSTALK_EGF],
            CROSSTALK_SHH: -s * self.shh_decay * state[CROSSTALK_SHH],
            CROSSTALK_NOTCH: -s * self.notch_decay * state[CROSSTALK_NOTCH],
        }

    def metadata(self):
        base = super().metadata()
        base["hallmark"] = "Stem Cell Exhaustion"
        base["reference"] = "Sivakumar et al. 2011 (BIOMD0000000398)"
        base["description"] = (
            "Niche deterioration: severity-dependent decay of Wnt, EGF, "
            "Shh, and Notch signaling ligands/receptors."
        )
        return base


def build_niche_crosstalk(
    severity: float = 0.0,
    sbml_path: str | None = None,
):
    """Build a Composite wiring StemCellNiche + Sivakumar crosstalk model.

    Parameters
    ----------
    severity:
        Niche deterioration severity (0-1).
    sbml_path:
        Path to the crosstalk SBML file.  Defaults to the bundled
        ``models/sivakumar2011/crosstalk_BIOMD0000000398.xml``.

    Returns
    -------
    Composite with two processes: ``"crosstalk"`` and ``"niche"``.
    """
    from hallsim.composite import Composite
    from hallsim.sbml_import import process_from_sbml

    if sbml_path is None:
        sbml_path = str(
            Path(__file__).parent.parent.parent.parent
            / "models"
            / "sivakumar2011"
            / "crosstalk_BIOMD0000000398.xml"
        )

    crosstalk = process_from_sbml(sbml_path, name="crosstalk")
    niche = StemCellNiche(severity=severity)

    # All crosstalk species map to their own store paths
    species = list(crosstalk.ports_schema().keys())
    crosstalk_topo = {s: s for s in species}

    # Niche ports wire to the same store paths as the ligand species
    niche_topo = {
        CROSSTALK_WNT: CROSSTALK_WNT,
        CROSSTALK_EGF: CROSSTALK_EGF,
        CROSSTALK_SHH: CROSSTALK_SHH,
        CROSSTALK_NOTCH: CROSSTALK_NOTCH,
    }

    return Composite(
        processes={"crosstalk": crosstalk, "niche": niche},
        topology={"crosstalk": crosstalk_topo, "niche": niche_topo},
        validate=False,
    )
