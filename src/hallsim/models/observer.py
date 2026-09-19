"""SumObserver — one store path holding the sum of several others.

    total = Σ parts

A readout that spans species — the aggregates a model keeps as three
pools, a family of conjugates, a bound and a free form — becomes one path
that a reporter, a figure or a coupling edge can read like any species.
The output is ASSIGNED: algebraic, materialised at every save point, never
integrated. Reading a species lifts the import-time freeze on it, so a
model's terminal products count here even when nothing else reads them.
"""

from __future__ import annotations

import equinox as eqx
import sympy

from hallsim.process import Port, PortRole, Process


class SumObserver(Process):
    """``total`` = the sum over the ``parts`` block; ``elements`` names the
    parts, and the topology binds one store path to each."""

    description = "Sum of several store paths as one algebraic path."

    timescale: float | None = eqx.field(static=True, default=None)
    elements: tuple = eqx.field(static=True, default=())
    units: str = eqx.field(static=True, default="")
    what: str = eqx.field(static=True, default="")

    def ports_schema(self):
        return {
            "parts": Port(
                role=PortRole.INPUT,
                default=0.0,
                units=self.units,
                elements=tuple(self.elements),
                description=f"Summands of {self.what or 'the total'}",
            ),
            "total": Port(
                role=PortRole.ASSIGNED,
                default=0.0,
                units=self.units,
                description=self.what or "Sum of the parts",
            ),
        }

    def assign(self, t, state):
        return {"total": state["parts"].sum(axis=-1)}

    def assignment_rules(self):
        """``total = parts_0 + parts_1 + ...``, one symbol per element of the
        block, which the SBML exporter binds to the block's paths in order."""
        terms = [sympy.Symbol(f"parts_{i}") for i in range(len(self.elements))]
        return (("total", sympy.Add(*terms) if terms else sympy.Integer(0)),)

    def derivative(self, t, state):
        return {}
