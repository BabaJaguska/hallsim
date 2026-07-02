"""DamageNFkBActivator — genotoxic-stress-driven NF-κB activation edge.

Couples DallePezze 2014's DNA-damage state to the Ihekwaba 2004 NF-κB
module: accumulated DNA damage activates IKK, raising NF-κB signalling.
This is the composite's genomic-instability → inflammaging edge — the
channel through which the Genomic Instability hallmark reaches the NF-κB
readout, complementary to the nutrient-sensing channel
(:class:`hallsim.models.mtor_nfkb.MtorNFkBActivator`).

Biology
-------
Persistent DNA double-strand breaks activate NF-κB through the ATM →
NEMO → IKK axis: ATM phosphorylates NEMO (IKKγ) on damage, driving its
nuclear export and IKK activation.

- **ATM → NEMO → IKK.** DSB-induced ATM activation links to IKK via
  regulated NEMO shuttling (Wu et al. 2006, Science — DOI
  10.1126/science.1121513; Miyamoto 2011, Cell Res — DOI
  10.1038/cr.2010.179).
- **In senescence.** Persistent DDR/ATM signalling is required to
  establish the NF-κB-dependent SASP; ATM is a key driver of
  NF-κB-dependent DNA-damage-induced senescence (Salminen et al. 2012,
  Cell Signal — DOI 10.1016/j.cellsig.2011.12.006).

The sign is **activating**: DNA damage ↑ → NF-κB ↑. The edge encodes the
proximal damage → IKK step; the downstream NFKBIA readout emerges from
Ihekwaba 2004's own dynamics. It models the early, IKK-dependent phase of
genotoxic NF-κB activation.

Equation
--------
Additive contribution to the Ihekwaba IKK pool, Hill-gated on the DP14
DNA-damage state so it is near-zero at baseline damage and saturates in
the DDIS regime::

    d(IKK)/dt += k_act · H_act(DNA_damage; K_dmg, n)

Composes additively with the NF-κB module via EVOLVED port-role
semantics: Ihekwaba's intrinsic IKK dynamics are preserved and this term
is summed in alongside the mTORC1-driven term.
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, calibratable
from hallsim.utils import h_act


class DamageNFkBActivator(Process):
    """DNA-damage → IKK activation edge (DallePezze 2014 → Ihekwaba 2004).

    Reads the DP14 DNA-damage state and contributes a Hill-gated positive
    derivative to the NF-κB module's IKK pool. At baseline damage the
    contribution is small; as damage accumulates (DDIS) it engages,
    raising IKK and downstream NF-κB activity so the Genomic Instability
    hallmark reaches the NFKBIA readout.

    Parameters
    ----------
    k_act:
        Rate constant for the damage-driven IKK source term. Starting
        value only — fit against the NFKBIA reporter via
        :class:`hallsim.calibration.CalibrationProblem`. 0.02 places the
        saturated contribution at ~20% of the IKK baseline (0.1) per
        time unit, matching the scale of the mTORC1-driven edge so
        neither channel dominates the IKK pool a priori.
    K_dmg:
        Hill half-saturation threshold on ``DNA_damage``. Set to 500,
        the geometric midpoint of DP14's settled DNA_damage operating
        range across the arms (~10 unstressed to ~2.8e4 at full DDIS), so
        control sits well below the inflection (contribution ≈ 0) and
        DDIS well above it (contribution ≈ saturated) — a clean
        genomic-instability differential.
    n:
        Hill cooperativity. Default 2.0 — graded engagement across the
        operating range rather than a hard switch.
    """

    timescale: float | None = None  # Joins the composite's default group

    k_act: float = calibratable(
        0.02,
        description=(
            "DNA damage → IKK edge strength; fit against the NFKBIA "
            "reporter across the DDIS/control arms."
        ),
    )
    K_dmg: float = 500.0  # measurement-grounded (DP14 DNA_damage range)
    n: float = 2.0  # Hill cooperativity — fixed

    def ports_schema(self):
        return {
            # EVOLVED additive contribution to Ihekwaba 2004's IKK pool.
            "IKK": Port(
                role=PortRole.EVOLVED,
                default=0.1,
                units="dimensionless",
                description="genomic-instability drive summed into IKK",
                ontology={"go": "GO:0008384"},  # IkappaB kinase activity
                reads_value=False,  # pure source: term depends on damage, not IKK
            ),
            # INPUT: DNA-damage state from DallePezze 2014.
            "DNA_damage": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="DP14 accumulated DNA damage",
                ontology={"go": "GO:0006974"},  # DNA damage response
            ),
        }

    def derivative(self, t, state):
        K = jnp.asarray(self.K_dmg)
        n = jnp.asarray(self.n)
        drive = h_act(state["DNA_damage"], K, n)
        return {"IKK": self.k_act * drive}

    def metadata(self):
        base = super().metadata()
        base["hallmark"] = "Genomic Instability"
        base["reference"] = (
            "Wu et al. 2006; Miyamoto 2011; Salminen et al. 2012"
        )
        base["description"] = (
            "DNA damage → IKK activation edge coupling DallePezze 2014's "
            "genomic-instability state to the Ihekwaba 2004 NF-κB module."
        )
        return base
