"""P53CDKN1AActivator — the canonical p53 → p21 transcription edge.

Couples the Geva-Zatorsky 2006 p53 oscillator to DallePezze 2014's CDKN1A
(p21) state: p53 transcriptionally induces p21, the first and archetypal p53
target gene. Without this edge the composite's p21 responds to DNA damage only
through DP14's FoxO3a path, and the p53 oscillator (GZ06) drives just one
readout (DDB2) — p21 and p53 answer the same insult through two parallel paths
that never meet. This edge closes that gap and gives GZ06 a second observer.

Biology
-------
p21 (CDKN1A) was the first identified p53 transcriptional target; p53 binds the
CDKN1A promoter and induces it as the proximal effector of p53-dependent
cell-cycle arrest (el-Deiry et al. 1993). Pulsed p53 preferentially drives
arrest genes including p21 (Purvis et al. 2012, Science 336:1440).

Rate law
--------
Additive Hill-gated transcription flux onto the DP14 CDKN1A state::

    d(CDKN1A)/dt += k_act · H_act(p53; K_p53, n)

The Hill form for p53 → target transcription is validated for p21 specifically
by Shi et al. 2021 (FEBS Open Bio 11:1799), who fit a Hill coefficient n ≈ 1.8
to p21 expression driven by p53 pulsing; ``n`` is pinned to that value.
CDKN1A's own turnover (in DP14) integrates this pulsatile flux, so p21 responds
to p53 *activity* — the same amplitude signal DDB2's RMS reads — and the
instantaneous drive is correct without a separate integral (protein-stability
argument: Hanson et al. 2019; Porter et al. 2016). Composes additively via
EVOLVED port semantics alongside DP14's intrinsic FoxO3a-driven transcription.
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, calibratable
from hallsim.utils import h_act


class P53CDKN1AActivator(Process):
    """p53 → CDKN1A (p21) transcription edge (Geva-Zatorsky 2006 → DallePezze
    2014).

    Reads the GZ06 p53 state and contributes a Hill-gated positive derivative
    to DP14's CDKN1A (p21) — the canonical p53 → p21 induction. Lives in the
    p53/DP14 group so it reads p53 live (resolving the oscillation), and its
    pulsatile flux is integrated by CDKN1A's own turnover.

    Parameters
    ----------
    k_act:
        Rate constant for the p53-driven CDKN1A transcription term. Starting
        value only — fit against the CDKN1A reporter via
        :class:`hallsim.calibration.CalibrationProblem`; a MAP prior keeps the
        p53 path from swamping DP14's FoxO3a path a priori.
    K_p53:
        Hill half-saturation on p53. GZ06's p53 (``x``) runs ~0→1 over the
        oscillatory/sustained range, so 0.3 places the inflection where the
        oscillator is meaningfully active — a modelling choice on the
        normalised p53 range, not a fitted value.
    n:
        Hill cooperativity, pinned to 1.8 (Shi et al. 2021's fitted p21 Hill
        coefficient).
    """

    timescale: float | None = None  # joins the p53/DP14 default group

    k_act: float = calibratable(
        10.0,
        description=(
            "p53 → CDKN1A edge strength; fit against the CDKN1A reporter "
            "across the DDIS/control arms. Bridges GZ06's p53 (O(1)) to DP14's "
            "CDKN1A units (O(30) in the calibrated etoposide regime): ~10 places "
            "the saturated p53 contribution at ~25% of DP14's CDKN1A a priori "
            "(measured), so neither the p53 nor the FoxO3a path dominates before "
            "the fit rebalances them."
        ),
    )
    K_p53: float = 0.3  # normalised p53 operating-range midpoint
    n: float = 1.8  # Hill coefficient — Shi et al. 2021 (fixed)

    def ports_schema(self):
        return {
            # EVOLVED additive contribution to DallePezze 2014's CDKN1A state.
            "CDKN1A": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="dimensionless",
                description="p53-driven transcription summed into CDKN1A",
                ontology={"go": "GO:0006357"},  # regulation of transcription
                reads_value=False,  # source term depends on p53, not CDKN1A
            ),
            # INPUT: p53 (x) state from Geva-Zatorsky 2006.
            "p53": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="GZ06 p53 level",
                ontology={"go": "GO:0006977"},  # DNA damage response, p53
            ),
        }

    def derivative(self, t, state):
        K = jnp.asarray(self.K_p53)
        n = jnp.asarray(self.n)
        # Clip p53 ≥ 0: a NeuralODE surrogate can dip slightly negative in the
        # fixed-point regime, and a non-integer Hill power of a negative value
        # is undefined. p53 is a non-negative concentration.
        p53 = jnp.maximum(state["p53"], 0.0)
        return {"CDKN1A": self.k_act * h_act(p53, K, n)}

    def metadata(self):
        base = super().metadata()
        base["hallmark"] = "Genomic Instability"
        base["reference"] = (
            "el-Deiry et al. 1993; Purvis et al. 2012; Shi et al. 2021"
        )
        base["description"] = (
            "p53 → CDKN1A (p21) transcription edge coupling the Geva-Zatorsky "
            "2006 p53 oscillator to DallePezze 2014's CDKN1A state."
        )
        return base
