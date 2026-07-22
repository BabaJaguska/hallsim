"""Newton steady state: analytic fixed point, moiety conservation, IFT
gradient, and the COPASI-style time-dependence guard."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.steady_state import (
    conservation_laws,
    steady_state,
    warn_if_time_dependent,
)


class Reversible(Process):
    """A ⇌ B mass-action. Conserves A+B; fixed point A*/B* = k2/k1."""

    k1: float = 2.0
    k2: float = 1.0

    def ports_schema(self):
        return {
            "A": Port(role=PortRole.EVOLVED, default=1.0),
            "B": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def derivative(self, t, state):
        v = self.k1 * state["A"] - self.k2 * state["B"]
        return {"A": -v, "B": v}


class TimedSource(Process):
    """Explicitly time-dependent: dA/dt = sin(t) — no fixed point."""

    def ports_schema(self):
        return {"A": Port(role=PortRole.EVOLVED, default=0.0)}

    def derivative(self, t, state):
        return {"A": jnp.sin(t)}


def _rev(k1=2.0, k2=1.0):
    return Composite(
        processes={"r": Reversible(k1=k1, k2=k2)},
        topology={"r": {"A": "A", "B": "B"}},
        validate=False,
        semantic_validation={"check_semantics": False},
    )


def test_finds_analytic_fixed_point():
    comp = _rev(k1=2.0, k2=1.0)  # A+B=1 → A*=1/3, B*=2/3
    keys = comp.store_keys()
    yss = steady_state(comp)
    assert float(yss[keys.index("A")]) == pytest.approx(1 / 3, abs=1e-6)
    assert float(yss[keys.index("B")]) == pytest.approx(2 / 3, abs=1e-6)


def test_conservation_preserved():
    comp = _rev()
    yss = steady_state(comp)
    assert float(jnp.sum(yss)) == pytest.approx(1.0, abs=1e-9)


def test_one_conservation_law():
    comp = _rev()
    laws = conservation_laws(comp, comp.initial_state_vec())
    assert laws.shape[0] == 1  # A+B conserved


def test_ift_gradient_matches_analytic():
    # A* = k2/(k1+k2) · total; dA*/dk1 = -k2·total/(k1+k2)^2 = -1/9 at k1=2,k2=1
    comp0 = _rev()
    keys = comp0.store_keys()
    laws = conservation_laws(comp0, comp0.initial_state_vec())

    def a_star(k1):
        comp = _rev(k1=k1, k2=1.0)
        return steady_state(comp, laws=laws)[keys.index("A")]

    g = jax.grad(a_star)(2.0)
    assert float(g) == pytest.approx(-1 / 9, abs=1e-4)


def test_time_dependent_warns(caplog):
    comp = Composite(
        processes={"s": TimedSource()},
        topology={"s": {"A": "A"}},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="hallsim.steady_state"):
        auton = warn_if_time_dependent(comp, comp.initial_state_vec())
    assert not auton
    assert any("time-dependent" in r.message for r in caplog.records)
