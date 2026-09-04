"""Newton steady state: analytic fixed point, moiety conservation, IFT
gradient, and the COPASI-style time-dependence guard."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.steady_state import (
    conservation_laws,
    conserved_moieties,
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


class Binding(Process):
    """E + S ⇌ ES mass-action, stoichiometry declared.

    Two moieties that *share* a species — E+ES and S+ES — so the exact left
    null space is a basis whose rows are neither unit-length nor mutually
    orthogonal. The case that separates orthonormalising from normalising.
    """

    kon: float = 1.0
    koff: float = 0.5

    def ports_schema(self):
        return {
            "E": Port(role=PortRole.EVOLVED, default=1.0),
            "S": Port(role=PortRole.EVOLVED, default=2.0),
            "ES": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def stoichiometry(self):
        return {
            "species": ("E", "S", "ES"),
            "reactions": ("bind",),
            "matrix": ((-1,), (-1,), (1,)),
        }

    def derivative(self, t, state):
        v = self.kon * state["E"] * state["S"] - self.koff * state["ES"]
        return {"E": -v, "S": -v, "ES": v}


def _binding():
    return Composite(
        processes={"b": Binding()},
        topology={"b": {"E": "E", "S": "S", "ES": "ES"}},
        validate=False,
        semantic_validation={"check_semantics": False},
    )


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


def test_laws_are_orthonormal():
    """``L.T @ L`` is only a projector when the rows are orthonormal. A
    null-space basis is neither, and a caller projecting with a non-normalised
    ``L`` silently leaves the conservation leaf."""
    for comp in (_rev(), _binding()):
        laws = conservation_laws(comp, comp.initial_state_vec())
        gram = np.asarray(laws @ laws.T)
        assert np.allclose(gram, np.eye(laws.shape[0]), atol=1e-10), gram


def test_projector_step_stays_on_the_leaf():
    """A tangent step taken with ``I - L.T @ L`` must not move any conserved
    total — the property the projector exists for."""
    comp = _binding()
    y0 = comp.initial_state_vec()
    laws = conservation_laws(comp, y0)
    tangent = jnp.eye(y0.size) - laws.T @ laws
    rng = np.random.default_rng(0)
    step = tangent @ jnp.asarray(rng.normal(size=y0.size))
    drift = np.abs(np.asarray(laws @ (y0 + step) - laws @ y0))
    assert drift.max() < 1e-12, drift


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


# ═══════════════════════════════════════════════════════════════════════════
# Conserved moieties from stoichiometry
# ═══════════════════════════════════════════════════════════════════════════


def test_moieties_are_integer_and_named():
    """A ⇌ B conserves A+B, with coefficients 1 and 1 — not 0.707."""
    n = {
        "species": ("A", "B"),
        "reactions": ("forward",),
        "matrix": ((-1.0,), (1.0,)),
    }
    assert conserved_moieties(n) == [{"A": 1, "B": 1}]


def test_no_moiety_when_species_is_produced_and_degraded():
    n = {
        "species": ("X",),
        "reactions": ("synth", "degrade"),
        "matrix": ((1.0, -1.0),),
    }
    assert conserved_moieties(n) == []


def test_moieties_do_not_depend_on_rate_constants():
    """The property the Jacobian null space cannot hold: a conservation law
    is a fact about the wiring, so no rate constant can change it."""
    n = {
        "species": ("A", "B", "C"),
        "reactions": ("exchange", "decay"),
        "matrix": ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0)),
    }
    assert conserved_moieties(n) == [{"A": 1, "B": 1}]


def test_slow_decay_is_not_conserved(caplog):
    """C decays at k=1e-12. Declared stoichiometry says it is not conserved
    however slow it gets; the Jacobian at a point cannot tell."""

    class Exchange(Process):
        def ports_schema(self):
            return {
                "A": Port(role=PortRole.EVOLVED, default=1.0),
                "B": Port(role=PortRole.EVOLVED, default=1.0),
            }

        def derivative(self, t, state):
            flux = state["A"] - state["B"]
            return {"A": -flux, "B": flux}

        def stoichiometry(self):
            return {
                "species": ("A", "B"),
                "reactions": ("exchange",),
                "matrix": ((-1.0,), (1.0,)),
            }

    class SlowDecay(Process):
        k: float = 1e-12

        def ports_schema(self):
            return {"C": Port(role=PortRole.EVOLVED, default=1.0)}

        def derivative(self, t, state):
            return {"C": -self.k * state["C"]}

        def stoichiometry(self):
            return {
                "species": ("C",),
                "reactions": ("decay",),
                "matrix": ((-1.0,),),
            }

    comp = Composite(
        processes={"ex": Exchange(), "dec": SlowDecay()},
        topology={"ex": {"A": "p/A", "B": "p/B"}, "dec": {"C": "p/C"}},
        semantic_validation=False,
    )
    keys = comp.store_keys()
    laws = conservation_laws(comp, comp.initial_state_vec())
    assert laws.shape[0] == 1
    # Rows are orthonormal, so the law reads as a direction: A+B, not C.
    row = {keys[i]: float(v) for i, v in enumerate(laws[0]) if abs(v) > 1e-12}
    assert set(row) == {"p/A", "p/B"}
    assert row["p/A"] == pytest.approx(row["p/B"])
    assert row["p/A"] == pytest.approx(1 / np.sqrt(2))


def test_undeclared_process_falls_back():
    """One process without a stoichiometry means N cannot describe the
    composite, so the exact path must not be used."""

    class Opaque(Process):
        def ports_schema(self):
            return {"C": Port(role=PortRole.EVOLVED, default=1.0)}

        def derivative(self, t, state):
            return {"C": -0.1 * state["C"]}

    from hallsim.steady_state import composite_stoichiometry

    comp = Composite(
        processes={"op": Opaque()},
        topology={"op": {"C": "p/C"}},
        semantic_validation=False,
    )
    assert composite_stoichiometry(comp) is None


class Degrading(Process):
    """dx/dt = b − k·x + w·up. Nothing is conserved."""

    k: float = 0.4
    w: float = 0.15
    b: float = 1.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=0.5),
            "up": Port(role=PortRole.INPUT, default=0.5),
        }

    def derivative(self, t, state):
        return {"x": self.b - self.k * state["x"] + self.w * state["up"]}


def _ring(cls, n):
    paths = [f"net/x{i}" for i in range(n)]
    return Composite(
        processes={f"p{i}": cls() for i in range(n)},
        topology={
            f"p{i}": {"x": paths[i], "up": paths[(i - 1) % n]}
            for i in range(n)
        },
        semantic_validation=False,
    )


def _sampling_calls(monkeypatch):
    """Count infer_conservation_laws calls without changing what it returns."""
    import hallsim.steady_state as ss

    calls = []
    real = ss.infer_conservation_laws

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ss, "infer_conservation_laws", spy)
    return calls


def test_law_free_composite_skips_parameter_sampling(monkeypatch):
    """A Jacobian that admits no conserved combination rules them all out, so
    the eight sampled Jacobians would be spent returning nothing."""
    calls = _sampling_calls(monkeypatch)
    comp = _ring(Degrading, 6)
    laws = conservation_laws(comp, comp.initial_state_vec())
    assert laws.shape[0] == 0
    assert not calls, "sampled despite the Jacobian ruling every law out"


def test_moiety_composite_still_samples_and_finds_its_law(monkeypatch):
    """The guard is one-directional: a rank-deficient Jacobian still pays for
    the search, because a slow mode looks the same at a single point."""
    calls = _sampling_calls(monkeypatch)
    comp = Composite(
        processes={"r": Reversible()},
        topology={"r": {"A": "c/A", "B": "c/B"}},
        semantic_validation=False,
    )
    laws = conservation_laws(comp, comp.initial_state_vec())
    assert laws.shape[0] == 1
    assert calls, "guard rejected a composite that has a conservation law"
