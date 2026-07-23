"""HillActivationEdge — generic Hill-gated additive coupling edge.

One primitive for every cross-model activation edge:

    d(target)/dt += k_act · ∏ᵢ hill_gate(sourceᵢ; Kᵢ, nᵢ)

Ports are generic (``target`` + one ``source`` per driver); the store
paths they connect live in the composite topology, so an agent adds a
coupling by *instantiating* this edge with ``(k_act, K, n)`` and its
metadata rather than authoring a new Process class. Single-source is the
common case; pass ``sources=(...)`` with matching ``K``/``n`` tuples for an
AND of drivers (the gates multiply).

``target`` is an EVOLVED pure source (``reads_value=False``): the term
depends on the sources, not on the path it writes, so it composes
additively with the target module's intrinsic dynamics without creating a
spurious feedback cycle.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from hallsim.kinetics import hill_gate
from hallsim.process import Port, PortRole, Process, calibratable


class HillActivationEdge(Process):
    """Hill-gated additive edge; see module docstring for the rate law."""

    timescale: float | None = None

    k_act: float = calibratable(
        1.0, description="Hill-edge strength; fit against the target reporter."
    )
    K: tuple = (1.0,)  # per-source half-saturation threshold
    n: tuple = (2.0,)  # per-source Hill cooperativity

    sources: tuple = eqx.field(static=True, default=("source",))
    target_default: float = eqx.field(static=True, default=0.0)
    target_ontology: dict | None = eqx.field(static=True, default=None)
    target_description: str = eqx.field(static=True, default="")
    source_ontology: tuple | None = eqx.field(static=True, default=None)
    source_descriptions: tuple | None = eqx.field(static=True, default=None)
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        ont = self.source_ontology or ((None,) * len(self.sources))
        descs = self.source_descriptions or (("",) * len(self.sources))
        ports = {
            "target": Port(
                role=PortRole.EVOLVED,
                default=self.target_default,
                units="dimensionless",
                description=self.target_description,
                ontology=self.target_ontology or {},
                reads_value=False,
            )
        }
        for name, o, d in zip(self.sources, ont, descs):
            ports[name] = Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=d,
                ontology=o or {},
            )
        return ports

    def derivative(self, t, state):
        drive = jnp.asarray(1.0)
        for name, K, n in zip(self.sources, self.K, self.n):
            drive = drive * hill_gate(
                state[name], jnp.asarray(K), jnp.asarray(n)
            )
        return {"target": self.k_act * drive}


class HillSignalEdge(Process):
    """Assigns a signal store path from a source via a Hill — the algebraic
    (ASSIGNED) sibling of :class:`HillActivationEdge`. Each step:

        signal = basal + (hi − basal) · hill_gate(source; K, n)

    computed as a cross-process assignment rule (no integration, no timescale
    lag). An imported model reads the ``signal`` path through a plain
    parameter INPUT (``ImportedODEProcess.with_param_input``), so the Hill
    transform is a first-class composable edge rather than baked into a
    driver. ``basal`` is the fittable floor; ``hi``/``K``/``n`` are structural.
    """

    timescale: float | None = None
    basal: float = calibratable(
        0.3, description="signal floor at source→0; fit against the reporter."
    )
    hi: float = eqx.field(static=True, default=1.0)
    K: float = eqx.field(static=True, default=1.0)
    n: float = eqx.field(static=True, default=2.0)

    source_ontology: dict | None = eqx.field(static=True, default=None)
    source_description: str = eqx.field(static=True, default="")
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "signal": Port(
                role=PortRole.ASSIGNED,
                default=self.basal,
                units="dimensionless",
                description="Hill-bridged algebraic signal.",
            ),
            "source": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=self.source_description,
                ontology=self.source_ontology or {},
            ),
        }

    def assign(self, t, state):
        gate = hill_gate(
            state["source"], jnp.asarray(self.K), jnp.asarray(self.n)
        )
        return {"signal": self.basal + (self.hi - self.basal) * gate}
