"""SaturatingRemoval — Uri Alon's damage accumulation model.

Generic ODE for damage (D) with constant induction, age-dependent
ramping, and saturating Michaelis-Menten repair:

    dD/dt = alpha + eta * tau - beta * D / (K + D)

Where:
- alpha: constant damage induction rate (e.g. genotoxic exposure rate,
  basal endogenous damage). Acts as the **Genomic Instability** hallmark
  lever: alpha=0.01 is baseline, alpha=0.03–0.05 is acute exposure.
- eta: age-scaled damage production coefficient (Uri Alon's original
  formulation, models the lifetime ramp of damage rate).
- beta: maximum repair capacity (Vmax of the MM repair).
- K: Michaelis-Menten constant — DSB level at half-max repair.
- tau: age-scaled time = ``t * tau_scale``. Converts simulation time to
  whatever unit ``eta`` expects (years by default).

This single Process covers two distinct biological framings via the
``alpha`` / ``eta`` split:

1. **Karin-Alon senescent-cell turnover** (eta>0, alpha=0): basal damage
   that ramps with organismal age. Reference: Karin & Alon (2019).

2. **DDR / Genomic Instability** (alpha>0, eta=0): constant damage
   induction representing genotoxic exposure or repair-pathway
   dysfunction. Reference: Reinhardt & Yaffe (2009).

When used as the upstream module for an SBML-imported p53 model (e.g.
Geva-Zatorsky 2006 BIOMD0000000157), the ``damage`` port is wired
through topology to the SBML model's damage-input parameter exposed via
``process_from_sbml(..., tunable_params=("psi",))``. With defaults
``alpha=0.01, eta=0, beta=0.10, K=9.0`` the steady state ``D_ss = 1.0``
matches GZ06's ``psi`` baseline — no rescaling needed in topology.

References
----------
- Uri Alon, *An Introduction to Systems Biology* (2006). Chapter on
  saturating production-removal motifs.
- Karin O, Alon U (2019). "Senescent cell turnover slows with age
  providing an explanation for the Gompertz law." Nat Commun 10:5495.
- Reinhardt HC, Yaffe MB (2009). "Kinases that control the cell cycle
  in response to DNA damage: Chk1, Chk2, and MK2." Curr Opin Cell Biol
  21(2):245-255.
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


class SaturatingRemoval(Process):
    """Damage accumulation with saturating (Michaelis-Menten) repair.

    Parameters
    ----------
    alpha:
        Constant damage induction rate. Default 0 (preserves the original
        Karin-Alon age-ramp formulation). Set positive to use this Process
        as a DDR / Genomic Instability hallmark lever.
    eta:
        Age-scaled damage production coefficient. Default 0 (DDR mode).
        Set positive for the Karin-Alon age-ramp formulation.
    beta:
        Maximum repair capacity.
    K:
        Michaelis-Menten constant for repair.
    tau_scale:
        Time-to-years conversion factor (default 1, leaving t in input
        units; set to 1/(24*365) for hours→years etc.).
    """

    alpha: float = 0.0
    eta: float = 0.0
    beta: float = 1.0
    K: float = 0.1
    tau_scale: float = 1.0

    def ports_schema(self):
        return {
            "damage": Port(
                role=PortRole.EXCLUSIVE,
                default=0.0,
                units="dimensionless",
                description="Accumulated cellular damage (DSB pool)",
                ontology={
                    "go": "GO:0006974"
                },  # cellular response to DNA damage
            ),
        }

    def derivative(self, t, state):
        D = state["damage"]
        D_pos = jnp.maximum(D, 0.0)
        tau = t * self.tau_scale
        production = self.alpha + self.eta * tau
        repair = self.beta * D_pos / (self.K + D_pos)
        return {"damage": production - repair}

    def metadata(self):
        base = super().metadata()
        # Hallmark mapping depends on which mode is active. Report both
        # for downstream introspection (e.g. LLM-assisted composition).
        if self.alpha > 0 and self.eta == 0:
            base["hallmark"] = "Genomic Instability"
            base["mode"] = "ddr"
        elif self.eta > 0 and self.alpha == 0:
            base["hallmark"] = "Genomic Instability (age-ramp)"
            base["mode"] = "age_ramp"
        else:
            base["hallmark"] = "Genomic Instability"
            base["mode"] = "mixed"
        base["reference"] = (
            "Alon 2006; Karin & Alon 2019; Reinhardt & Yaffe 2009"
        )
        return base
