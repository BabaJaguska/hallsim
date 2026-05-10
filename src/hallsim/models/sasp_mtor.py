"""SASPmTORActivator — persistent mTORC1 activation in chronic-stress senescence.

In senescent cells, mTORC1 stays anomalously high despite low ATP, high
p53, and the AMPK pressure that should suppress it. This module adds the
chronic-stress-driven mTOR activation term that overrides AMPK's
inhibitory control.

Biology
-------
The persistence of mTORC1 in DNA-damage-induced senescence (DDIS) is
multi-pronged:

- **DDR → Akt → mTORC1.** Sustained DNA-damage response activates ATM,
  which (via PIDD/RIP1 and several reported mechanisms) drives Akt
  phosphorylation independent of growth-factor PI3K — Akt then
  activates mTORC1 by inhibiting TSC1/2.
- **SASP autocrine loop.** Senescence-associated secretory phenotype
  cytokines (IL-1α, IL-6) drive JAK/STAT signaling, which feeds back
  to maintain mTORC1 in a chronic, growth-factor-independent fashion.
- **ROS → PTEN inactivation.** Chronic oxidative stress oxidizes the
  catalytic cysteine of PTEN, removing its lipid-phosphatase activity
  and generating a constitutive PI3K → Akt signal. ERiQ already
  encodes this in its algebraic PTEN(ROS) Hill term.

Net effect: the *acute* AMPK-inhibits-mTORC1 axis (which canonical ERiQ
captures) is overridden in *chronic* senescence by these stress-driven
activation channels.

References
----------
- Carroll et al. 2017, eLife — *Persistent mTORC1 signaling in cell
  senescence results from defects in amino acid and growth factor
  sensing*. DOI: 10.7554/eLife.31268
- Laberge et al. 2015, Nat Cell Biol — *mTOR regulates the
  pro-tumorigenic senescence-associated secretory phenotype by
  promoting IL1A translation*. DOI: 10.1038/ncb3195
- Herranz et al. 2015, Nat Cell Biol — *mTOR regulates MAPKAPK2
  translation to control the senescence-associated secretory
  phenotype*. DOI: 10.1038/ncb3225
- Houssaini et al. 2018, JCI Insight — *mTOR pathway activation drives
  lung cell senescence and emphysema*. DOI: 10.1172/jci.insight.93203
- Fielder et al. 2017 — DDR-driven SASP and bystander mTORC1 effects
  (cited in kosmos discovery report).

Equation
--------
This Process contributes an additive term to the ERiQ ``mTOR_activity``
state variable (Ay)::

    d(mTOR_activity)/dt += k_sasp · H_act(mito_damage; K_d, n)
                                  · H_act(p53_activity; K_p, n)

Both factors are Hill activations gated on the chronic-stress signature
(damage and p53 both rise persistently in DDIS). At low stress both
factors are near zero and the canonical AMPK-mTOR balance dominates; as
stress rises the term ramps up and pushes Ay toward the senescent
high-mTOR steady state. p53 is used (not NF-κB directly) because it is
a state variable in our ERiQ — NF-κB is algebraic — and DDR-p53 acts
upstream of NF-κB in the SASP cascade (Acosta et al. 2013, Chien et al.
2011).

This is composed *additively* with ERiQSignaling: the existing AMPK-
driven integrator dynamics for ``mTOR_activity`` are preserved, and the
SASP term is summed in via the EVOLVED port-role semantics of HallSim.
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.pathway_mapper import h_act
from hallsim.process import Port, PortRole, Process


class SASPmTORActivator(Process):
    """Chronic-stress mTORC1 activation term, gated on damage and p53.

    Composes additively with ERiQSignaling's canonical mTOR dynamics
    (which use AMPK-mediated inhibition). When damage and p53 are low,
    this Process contributes ~0 derivative; when both rise persistently
    (DDIS regime), it pushes ``mTOR_activity`` upward, recovering the
    paradoxical high-mTOR senescence phenotype documented in the
    literature.

    Parameters
    ----------
    k_sasp:
        Rate constant for the chronic-stress activation term. Default
        0.05 places the saturated contribution at the same order as
        ERiQSignaling's gy=0.1 integrator gain — the two terms can
        compete near steady state.
    K_damage, K_p53:
        Hill half-saturation thresholds for the damage and p53 inputs.
        Defaults (1.0, 0.7) place the inflection just above each
        variable's homeostatic value (mito_damage_ss ≈ 0.07,
        p53_activity_ss ≈ 0.87) so the term is near-zero at baseline
        and engages as either rises into the senescent regime.
    n:
        Hill cooperativity. Default 4.0 gives switch-like engagement
        consistent with multi-step DDR → Akt activation cascades.
    """

    timescale: float | None = None  # Joins ERiQ's default group

    k_sasp: float = 0.05
    K_damage: float = 1.0
    K_p53: float = 0.7
    n: float = 4.0

    def ports_schema(self):
        return {
            # EVOLVED additive contribution to ERiQ's mTOR_activity (Ay).
            # ERiQSignaling must declare the same port as EVOLVED for
            # this composition to validate.
            "mTOR_activity": Port(
                role=PortRole.EVOLVED,
                default=-0.1936,
                units="dimensionless",
                description="ERiQ mTOR_activity (Ay), receives SASP add-on",
                ontology={"go": "GO:0031929"},
            ),
            "mito_damage": Port(
                role=PortRole.INPUT,
                default=0.0724,
                description="Persistent DDR signal — chronic-stress proxy",
                ontology={"go": "GO:0006974"},  # cellular response to DNA dmg
            ),
            "p53_activity": Port(
                role=PortRole.INPUT,
                default=0.8734,
                description="p53 transcriptional activity (DDR-p53 axis)",
                ontology={"go": "GO:0030330"},
            ),
        }

    def derivative(self, t, state):
        K_d = jnp.asarray(self.K_damage)
        K_p = jnp.asarray(self.K_p53)
        n = jnp.asarray(self.n)
        H_dmg = h_act(state["mito_damage"], K_d, n)
        H_p53 = h_act(state["p53_activity"], K_p, n)
        return {"mTOR_activity": self.k_sasp * H_dmg * H_p53}


class mTORInhibitor(Process):
    """Rapamycin-like mTORC1 inhibitor — pharmacological perturbation.

    Models a small-molecule inhibitor (rapamycin or analog) that
    suppresses mTORC1 catalytic activity. In our framework this is an
    additive negative driver on the ``mTOR_activity`` state, composing
    with both ``ERiQSignaling`` (canonical AMPK-driven dynamics) and
    ``SASPmTORActivator`` (chronic-stress activation) via the EVOLVED
    port-role semantics: the three contributions sum at the RHS.

    Why this is a Process and not a hallmark severity:
    Rapamycin is a *pharmacological intervention*, not an emergent
    aging hallmark. It enters the system as an external perturbation
    of variable strength (dose). Treating it as an EVOLVED-writer
    process keeps the hallmark interface clean (one knob per hallmark)
    and makes intervention dose a separate, composable parameter.

    Equation
    --------
    ::

        d(mTOR_activity)/dt += -strength

    The constant negative driver shifts the integrator equilibrium of
    mTOR_activity downward by approximately ``strength / gy_eff``
    (where gy_eff is the effective integrator gain of ERiQSignaling).
    Default ``strength=0.05`` matches the order of ERiQSignaling's
    integrator gain (gy = 0.1) and produces a moderate mTOR
    suppression consistent with rapamycin pharmacology
    (Sehgal 1998, Sabatini 2017).

    Validation use
    --------------
    Used in the DDIS+rapamycin rescue arm of the GSE248823 concordance
    demo (``demos/concordance_ddis.py``). The model predicts a
    pathway-wide suppression that should match the measured ssGSEA
    deltas for ``DDIS_RAPA_vs_DDIS_D14``.

    References
    ----------
    - Sehgal 1998, *Rapamycin (sirolimus, Rapamune)*. Clin Biochem 31.
      The original pharmacology.
    - Sabatini 2017, *Twenty-five years of mTOR: uncovering the link
      from nutrients to growth*. PNAS.
    """

    timescale: float | None = None  # Joins ERiQ's default group

    strength: float = 0.0  # 0 = no rapamycin, 0.05+ = significant inhibition

    def ports_schema(self):
        return {
            "mTOR_activity": Port(
                role=PortRole.EVOLVED,
                default=-0.1936,
                units="dimensionless",
                description="ERiQ mTOR_activity (Ay), receives rapamycin negative driver",
                ontology={"go": "GO:0031929"},
            ),
        }

    def derivative(self, t, state):
        return {"mTOR_activity": -jnp.asarray(self.strength)}
