"""BistableLatch — an autocatalytic self-sustaining state that commits.

A single state variable that a *transient* trigger can push past a threshold,
after which positive feedback holds it in a high fixed point indefinitely —
the minimal primitive for a committed cell-state transition (senescence,
differentiation, transformation) whose persistence outlives its trigger.

    d(latch)/dt = k_trigger · hill(trigger; K_trig, n_trig)      # transient kick
                + k_feedback · hill(latch;   K_fb,  n_fb) · (1 − latch)   # autocatalysis
                − k_decay   · latch

With ``trigger = 0`` the latch has a stable OFF root at 0 and — for a steep
enough feedback — a stable ON root near ``k_feedback / (k_feedback + k_decay)``,
separated by an unstable middle root. A trigger pulse that lifts the state past
that middle root latches it ON; a sub-threshold trigger relaxes back to OFF.
The committed state drives a downstream pool as a pure source:

    d(target)/dt += k_output · latch

Ports are generic (``latch`` state, ``trigger`` input, ``target`` source); the
store paths they connect live in the composite topology, so a commitment node
is added by *instantiating* this with its rate constants and semantic metadata
rather than authoring a new Process. Place ``(k_feedback, K_fb, n_fb, k_decay)``
in the bistable regime and confirm latch-not-relax with the bifurcation tools
(:mod:`hallsim.bifurcation`); place the trigger Hill at the driver's real
operating point (:meth:`CalibrationProblem.operating_ranges`).
"""

from __future__ import annotations

import equinox as eqx

from hallsim.kinetics import hill_gate
from hallsim.process import Port, PortRole, Process, calibratable


class BistableLatch(Process):
    """Autocatalytic bistable state; see module docstring for the rate law."""

    timescale: float | None = None

    k_trigger: float = calibratable(
        1.0,
        description="trigger→latch induction gain (kicks the state past threshold).",
    )
    # Plain defaults, so __check_init__ traces them: static would bake each
    # value into the solve and hide it from jax.grad.
    K_trig: float = 1.0
    n_trig: float = 2.0

    k_feedback: float = 1.0
    K_fb: float = 0.4
    n_fb: float = 4.0
    k_decay: float = 0.3

    k_output: float = calibratable(
        1.0,
        description="latch→target source gain; fit against the target reporter.",
    )

    latch_default: float = eqx.field(static=True, default=0.0)
    target_default: float = eqx.field(static=True, default=0.0)
    latch_ontology: dict | None = eqx.field(static=True, default=None)
    latch_description: str = eqx.field(static=True, default="")
    trigger_ontology: dict | None = eqx.field(static=True, default=None)
    trigger_description: str = eqx.field(static=True, default="")
    target_ontology: dict | None = eqx.field(static=True, default=None)
    target_description: str = eqx.field(static=True, default="")
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "latch": Port(
                role=PortRole.EVOLVED,
                default=self.latch_default,
                units="dimensionless",
                description=self.latch_description,
                ontology=self.latch_ontology or {},
                reads_value=True,
            ),
            "trigger": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=self.trigger_description,
                ontology=self.trigger_ontology or {},
            ),
            "target": Port(
                role=PortRole.EVOLVED,
                default=self.target_default,
                units="dimensionless",
                description=self.target_description,
                ontology=self.target_ontology or {},
                reads_value=False,
            ),
        }

    def derivative(self, t, state):
        latch = state["latch"]
        kick = hill_gate(state["trigger"], self.K_trig, self.n_trig)
        auto = hill_gate(latch, self.K_fb, self.n_fb)
        d_latch = (
            self.k_trigger * kick
            + self.k_feedback * auto * (1.0 - latch)
            - self.k_decay * latch
        )
        return {"latch": d_latch, "target": self.k_output * latch}
