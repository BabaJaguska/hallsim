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

from typing import NamedTuple

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

    def frozen_at(self, levels: dict) -> "GainEdge":
        """A copy that emits the constant this edge would emit at ``levels``.

        The null for "what does this coupling buy": the target keeps receiving
        the value the edge was placed to deliver, and stops receiving the
        source's variation. See :func:`hallsim.ablation.freeze_coupling`.
        """
        return self.with_param("gain", 0.0).with_param(
            "offset", float(self.offset + self.gain * levels[self.source])
        )

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


class GainPlacement(NamedTuple):
    """A placed line: ``value = offset + gain * source``."""

    gain: float
    offset: float


def place_gain_from_ranges(source, target) -> GainPlacement:
    """The line mapping one model's operating envelope onto the other's.

    ``source`` and ``target`` are each ``(lo, hi)`` or anything carrying
    ``.lo`` / ``.hi`` — :class:`~hallsim.calibration.OperatingRange` fits.
    Both come from running the deposits, so a placement needs no experimental
    data; :func:`hallsim.diagnostics.operating_range` produces them.

    Two points determine the line, so unlike :func:`place_gain` this makes no
    assumption that it passes through the origin, and it is defined when
    either model rests at zero. It needs the two source levels to differ:
    a model that does not move over its own perturbation carries no
    information about what it drives, and that is a fact about the model
    rather than something a placement rule can supply.
    """

    def ends(r):
        return (
            (float(r.lo), float(r.hi))
            if hasattr(r, "lo")
            else (float(r[0]), float(r[-1]))
        )

    s_lo, s_hi = ends(source)
    t_lo, t_hi = ends(target)
    span = s_hi - s_lo
    if span == 0.0:
        raise ValueError(
            f"source range is a single level ({s_lo:g}), so no line through "
            "it is determined. Widen the perturbation the range was measured "
            "over, or place the edge from a reported response instead."
        )
    gain = (t_hi - t_lo) / span
    return GainPlacement(gain=gain, offset=t_lo - gain * s_lo)


def place_gain(source_rest: float, target_rest: float) -> float:
    """The slope through the origin and one shared operating point,
    ``target_rest / source_rest``.

    One point determines a line only under the assumption that the line passes
    through the origin — that the target is zero where the source is zero.
    Where that is not something you would assert, place from two points with
    :func:`place_gain_from_ranges` instead, which also handles a model that
    rests at zero.
    """
    source_rest = float(source_rest)
    if source_rest <= 0.0:
        raise ValueError(
            f"source rest level must be positive, got {source_rest:g}"
        )
    return float(target_rest) / source_rest
