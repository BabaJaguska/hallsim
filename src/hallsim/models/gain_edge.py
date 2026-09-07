"""GainEdge — a linear coupling edge, as a flux or as a level.

    value = offset + gain · source

    mode="flux"    d(target)/dt += value    EVOLVED, summed with the target
    mode="level"   signal        = value    ASSIGNED, sole owner

The units bridge between two models that both carry a quantity on their own
arbitrary scale: a level read back through ``with_param_input`` rescales the
source onto the target model's scale without a threshold or a saturation.
``gain`` is placed once from the two models' homeostatic operating points
(:func:`place_gain`) and is calibratable from there; ``gain < 0`` inhibits.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, calibratable


class GainEdge(Process):
    """Linear coupling edge; see module docstring for the rate law."""

    timescale: float | None = None

    offset: float = calibratable(0.0, description="value at source→0.")
    gain: float = calibratable(1.0, description="slope in target per source.")

    mode: str = eqx.field(static=True, default="level")
    source: str = eqx.field(static=True, default="source")
    target_default: float | None = eqx.field(static=True, default=None)
    target_ontology: dict | None = eqx.field(static=True, default=None)
    target_description: str = eqx.field(static=True, default="")
    source_ontology: dict | None = eqx.field(static=True, default=None)
    source_description: str = eqx.field(static=True, default="")
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    @property
    def out_port(self) -> str:
        return "target" if self.mode == "flux" else "signal"

    def __check_init__(self):
        super().__check_init__()
        if self.mode not in ("flux", "level"):
            raise ValueError(
                f"GainEdge mode must be 'flux' or 'level', got {self.mode!r}"
            )

    def ports_schema(self):
        if self.mode == "flux":
            out = Port(
                role=PortRole.EVOLVED,
                default=self.target_default,
                units="dimensionless",
                description=self.target_description,
                ontology=self.target_ontology or {},
                reads_value=False,
            )
        else:
            out = Port(
                role=PortRole.ASSIGNED,
                default=self.offset,
                units="dimensionless",
                description=self.target_description
                or "Linearly bridged algebraic signal.",
                ontology=self.target_ontology or {},
            )
        return {
            self.out_port: out,
            self.source: Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=self.source_description,
                ontology=self.source_ontology or {},
            ),
        }

    def _value(self, state):
        return self.offset + self.gain * jnp.asarray(state[self.source])

    def derivative(self, t, state):
        if self.mode != "flux":
            return {}
        return {"target": self._value(state)}

    def assign(self, t, state):
        if self.mode != "level":
            return {}
        return {"signal": self._value(state)}


def place_gain(source_rest: float, target_rest: float) -> float:
    """The slope that maps the source model's homeostatic level onto the
    target model's, ``target_rest / source_rest``: both deposits are at rest
    at their own published values, so the line through the origin and that
    point is the only placement that needs no fitted number."""
    source_rest = float(source_rest)
    if source_rest <= 0.0:
        raise ValueError(
            f"source rest level must be positive, got {source_rest:g}"
        )
    return float(target_rest) / source_rest
