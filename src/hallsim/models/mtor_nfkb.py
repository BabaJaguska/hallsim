"""MtorNFkBActivator — mTORC1-driven NF-κB activation crosstalk edge.

Couples DallePezze 2014's nutrient-sensing network to the Ihekwaba 2004
NF-κB module: active mTORC1 promotes IKK activity, raising NF-κB
signalling. This is the first cross-publication mechanistic edge in the
multi-hallmark composite — the two SBML models were previously
co-simulated under a shared hallmark knob but never coupled.

Biology
-------
mTORC1 is a positive regulator of NF-κB, established in two
complementary contexts that bracket our validation experiment (senescent
human fibroblasts ± rapamycin):

- **mTORC1 → IKK.** mTORC1/Raptor is required for IKK catalytic
  activity downstream of Akt, directly promoting NF-κB activation
  (Dan et al. 2008, Genes Dev — DOI 10.1101/gad.1662308).
- **mTORC1 → SASP.** In senescence, mTORC1 promotes IL1A translation;
  IL1A drives NF-κB transcriptional activity and the SASP, and
  rapamycin lowers it (Laberge et al. 2015, Nat Cell Biol —
  DOI 10.1038/ncb3195).

Both place the sign as **activating**: mTOR ↑ → NF-κB ↑, so rapamycin
(mTOR ↓) suppresses NF-κB. The edge encodes the proximal mTORC1 → IKK
step; the downstream NFKBIA readout emerges from Ihekwaba 2004's own
dynamics.

Equation
--------
Additive contribution to the Ihekwaba IKK pool, Hill-gated on active
mTORC1 so it is near-zero when mTOR is suppressed and ramps as mTOR
rises into the senescent regime::

    d(IKK)/dt += k_act · H_act(mTORC1_pS2448; K_mtor, n)

Composes additively with the NF-κB module via EVOLVED port-role
semantics: Ihekwaba's intrinsic IKK dynamics are preserved and this
term is summed in.
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, calibratable
from hallsim.utils import h_act


class MtorNFkBActivator(Process):
    """mTORC1 → IKK activation edge (DallePezze 2014 → Ihekwaba 2004).

    Reads active mTORC1 from DP14 and contributes a Hill-gated positive
    derivative to the NF-κB module's IKK pool. At suppressed mTOR
    (rapamycin) the contribution is small; as mTOR rises (DDIS) it
    engages, raising IKK and downstream NF-κB activity.

    Parameters
    ----------
    k_act:
        Rate constant for the mTORC1-driven IKK source term. Starting
        value only — fit against the NFKBIA reporter via
        :class:`hallsim.calibration.CalibrationProblem`. 0.02 places the
        saturated contribution at ~20% of the IKK baseline (0.1) per
        time unit, the order at which it can compete with Ihekwaba's
        intrinsic IKK turnover without dominating it.
    K_mtor:
        Hill half-saturation threshold on ``mTORC1_pS2448``. Set to 4.0,
        the midpoint of DP14's settled mTORC1 operating range across the
        rapamycin→DDIS arms (~2.8 full-rapa to ~5.7 untreated DDIS), so
        the two arms straddle the inflection and produce a differential
        IKK drive.
    n:
        Hill cooperativity. Default 2.0 — graded engagement across the
        operating range rather than a hard switch.
    """

    timescale: float | None = None  # Joins the composite's default group

    k_act: float = calibratable(
        0.02,
        description=(
            "mTORC1 → IKK edge strength; fit against the NFKBIA reporter "
            "across the rapamycin/DDIS arms."
        ),
    )
    K_mtor: float = 4.0  # measurement-grounded (DP14 mTORC1 range) — fixed
    n: float = 2.0  # Hill cooperativity — fixed

    def ports_schema(self):
        return {
            # EVOLVED additive contribution to Ihekwaba 2004's IKK pool.
            "IKK": Port(
                role=PortRole.EVOLVED,
                default=0.1,
                units="dimensionless",
                description="Ihekwaba IKK pool, receives mTORC1 activation",
                ontology={"go": "GO:0008384"},  # IkappaB kinase activity
                reads_value=False,  # pure source: term depends on mTORC1, not IKK
            ),
            # INPUT: active mTORC1 from DallePezze 2014.
            "mTORC1_pS2448": Port(
                role=PortRole.INPUT,
                default=10.0,
                units="dimensionless",
                description="DP14 active mTORC1 (Ser2448-phosphorylated)",
                ontology={"go": "GO:0031931"},  # TORC1 complex
            ),
        }

    def derivative(self, t, state):
        K = jnp.asarray(self.K_mtor)
        n = jnp.asarray(self.n)
        drive = h_act(state["mTORC1_pS2448"], K, n)
        return {"IKK": self.k_act * drive}

    def metadata(self):
        base = super().metadata()
        base["hallmark"] = "Deregulated Nutrient Sensing"
        base["reference"] = "Dan et al. 2008; Laberge et al. 2015"
        base["description"] = (
            "mTORC1 → IKK activation edge coupling DallePezze 2014's "
            "nutrient-sensing network to the Ihekwaba 2004 NF-κB module."
        )
        return base
