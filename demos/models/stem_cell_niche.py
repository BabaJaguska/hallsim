"""Stem Cell Niche — age-dependent decline in niche signaling.

Contributes decay derivatives to the Wnt / EGF / Shh / Notch species of the
Sivakumar 2011 crosstalk model (BIOMD0000000398): declining self-renewal,
proliferative capacity, niche structure, and asymmetric division, i.e. the
Stem Cell Exhaustion hallmark (Lopez-Otin 2023).

A demonstration of additive composition — both this and the SBML model declare
EVOLVED ports on the same species, so the Composite sums their derivatives. At
severity=0 the niche contributes nothing.

>>> comp = build_niche_crosstalk(severity=0.5)
>>> result = Scheduler().run(comp, t_span=(0.0, 100.0), macro_dt=0.5)
"""

from __future__ import annotations


from hallsim.process import Port, PortRole, Process

# Sivakumar2011 crosstalk model (BIOMD0000000398) species IDs
# for the four niche ligands / receptors.
CROSSTALK_WNT = "s107"  # Wnt (extracellular)
CROSSTALK_EGF = "s96"  # EGF
CROSSTALK_SHH = "s81"  # Shh (Sonic Hedgehog)
CROSSTALK_NOTCH = "s57"  # Notch receptor


class StemCellNiche(Process):
    """Age-dependent niche deterioration, as decay derivatives on the
    Sivakumar 2011 ligand species scaled by ``severity`` (0 healthy → 1
    severely deteriorated). ``*_decay`` are the per-ligand rate constants."""

    hallmark = "Stem Cell Exhaustion"
    reference = "Sivakumar et al. 2011 (BIOMD0000000398)"
    description = (
        "Niche deterioration: severity-dependent decay of Wnt, EGF, Shh, "
        "and Notch signaling ligands/receptors."
    )

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


def build_niche_crosstalk(
    severity: float = 0.0,
    sbml_path: str | None = None,
):
    """Composite of ``"crosstalk"`` (Sivakumar 2011, bundled unless
    ``sbml_path`` says otherwise) and ``"niche"`` at ``severity``."""
    from hallsim.composite import Composite
    from hallsim.sbml_import import process_from_sbml

    if sbml_path is None:
        from demos.models.sbml import sbml_source

        sbml_path = sbml_source(
            "sivakumar2011",
            "crosstalk_BIOMD0000000398.xml",
            "BIOMD0000000398",
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
