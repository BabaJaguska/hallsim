"""SaturatingRemoval — Uri Alon's damage accumulation model.

Simple ODE for generic damage (D) with saturating repair:

    dD/dt = eta * tau - beta * D / (K + D)

Where:
- eta: damage production rate
- beta: maximum repair capacity
- K: Michaelis-Menten constant (damage level at half-max repair)
- tau: age-scaled time (converts simulation time to years)

Reference: Uri Alon, "An Introduction to Systems Biology" (2006).
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


class SaturatingRemoval(Process):
    """Damage accumulation with saturating (Michaelis-Menten) repair.

    Parameters
    ----------
    eta:
        Damage production rate.
    beta:
        Maximum repair capacity.
    K:
        Michaelis-Menten constant for repair.
    tau_scale:
        Time-to-years conversion factor (default: 1/(24*365) for hours→years).
    """

    eta: float = 0.5
    beta: float = 1.0
    K: float = 0.1
    tau_scale: float = 1.0 / (24.0 * 365.0)

    def ports_schema(self):
        return {
            "damage": Port(
                role=PortRole.EXCLUSIVE,
                default=0.0,
                units="dimensionless",
                description="Accumulated cellular damage",
            ),
        }

    def derivative(self, t, state):
        D = state["damage"]
        tau = t * self.tau_scale
        dD = self.eta * tau - (self.beta * D) / (self.K + D)
        return {"damage": dD}
