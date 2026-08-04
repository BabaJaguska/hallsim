"""SaturatingRemoval — Uri Alon's damage accumulation model.

    dD/dt = alpha + eta·tau - beta·D / (K + D)

with ``tau = t · tau_scale`` (years by default). The ``alpha``/``eta`` split
covers two framings in one Process: Karin-Alon senescent-cell turnover
(``eta>0, alpha=0``, damage ramping with organismal age) and DDR / Genomic
Instability (``alpha>0, eta=0``, constant genotoxic exposure).

As the upstream module for an SBML p53 model, wire ``damage`` to that model's
damage-input parameter. The defaults put ``D_ss = 1.0``, matching GZ06's
``psi`` baseline, so no rescaling is needed in topology.

References: Alon, *An Introduction to Systems Biology* (2006); Karin & Alon,
Nat Commun 10:5495 (2019); Reinhardt & Yaffe, Curr Opin Cell Biol 21:245 (2009).
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


class SaturatingRemoval(Process):
    """Damage accumulation with saturating (Michaelis-Menten) repair.

    ``alpha`` is the constant induction rate (the DDR / Genomic Instability
    lever), ``eta`` the age-scaled production coefficient (Karin-Alon age
    ramp); both default to 0, so pick a mode by setting one. ``beta`` is max
    repair capacity, ``K`` its Michaelis constant, and ``tau_scale`` converts
    ``t`` to the unit ``eta`` expects (e.g. 1/(24*365) for hours→years).
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
