"""GatedRemoval — first-order removal of a state while a condition holds.

    d(target)/dt += -k_remove · hill_gate(trigger; K, n) · target

The primitive for a *commitment*: once the trigger crosses ``K`` the target is
cleared at rate ``k_remove``, and while it does not the edge contributes
nothing. Removal is proportional to the target, so it decays rather than
crossing zero.

``HillEdge(mode="flux")`` cannot express this — its target is a pure source
and cannot scale by the value it writes — and ``ClampEdge`` cannot either,
because its rate is a constant rather than a gated signal.
"""

from __future__ import annotations

import equinox as eqx

from hallsim.kinetics import hill_gate
from hallsim.process import Port, PortRole, Process, calibratable


class GatedRemoval(Process):
    """Gated first-order removal; see module docstring for the rate law."""

    timescale: float | None = None

    k_remove: float = calibratable(
        1.0, description="clearance rate while the gate is open (1/time)."
    )
    K: float = 1.0  # trigger level at which the gate is half open
    n: float = 4.0  # gate steepness

    target_default: float | None = eqx.field(static=True, default=None)
    target_ontology: dict | None = eqx.field(static=True, default=None)
    target_description: str = eqx.field(static=True, default="")
    trigger_ontology: dict | None = eqx.field(static=True, default=None)
    trigger_description: str = eqx.field(static=True, default="")
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "target": Port(
                role=PortRole.EVOLVED,
                default=self.target_default,
                units="dimensionless",
                description=self.target_description,
                ontology=self.target_ontology or {},
                reads_value=True,
            ),
            "trigger": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=self.trigger_description,
                ontology=self.trigger_ontology or {},
            ),
        }

    def derivative(self, t, state):
        gate = hill_gate(state["trigger"], self.K, self.n)
        return {"target": -self.k_remove * gate * state["target"]}
