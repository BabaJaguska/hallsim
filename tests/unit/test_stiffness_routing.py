"""Tests for per-group solver routing (``Scheduler(auto_stiffness=...)``).

Covers:
- Routing is on by default: stiff groups get the implicit solver, non-stiff
  groups keep the explicit one
- Pinning ``solver=`` turns routing off, with a warning; ``auto_stiffness=False``
  turns it off without one
- Cold-trace behaviour: routing degrades to the explicit solver and says so,
  and ``warm_up`` resolves the verdict outside the trace
"""

from __future__ import annotations

import logging

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler

_LOOSE_STEPS = 2**20  # explicit solver on a stiff group needs the headroom


class StiffPair(Process):
    """Two decoupled decays five orders of magnitude apart — Jacobian
    eigenvalues ``{-k_fast, -k_slow}``. At ``macro_dt=1`` the fast mode needs
    ~``k_fast`` stability-limited substeps, well past the analyzer's default
    threshold of 100, so the group is unambiguously stiff."""

    timescale: float = 1.0
    k_fast: float = 1.0e4
    k_slow: float = 1.0e-1

    def ports_schema(self):
        return {
            "f": Port(
                role=PortRole.EVOLVED, default=1.0, units="dimensionless"
            ),
            "s": Port(
                role=PortRole.EVOLVED, default=1.0, units="dimensionless"
            ),
        }

    def derivative(self, t, state):
        return {
            "f": -self.k_fast * state["f"],
            "s": -self.k_slow * state["s"],
        }


class MildDecay(Process):
    """Single slow decay — nothing for the analyzer to flag."""

    timescale: float = 1.0
    rate: float = 0.1

    def ports_schema(self):
        return {
            "x": Port(
                role=PortRole.EVOLVED, default=1.0, units="dimensionless"
            )
        }

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


def _composite(proc, name):
    return Composite(
        processes={name: proc},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )


def _solvers(result):
    return {
        v["solver"]
        for v in result.stats.values()
        if isinstance(v, dict) and "solver" in v
    }


class TestRoutingDefault:
    """Routing is on by default and picks per group."""

    def test_stiff_group_routed_to_implicit(self):
        res = Scheduler().run(
            _composite(StiffPair(), "stiff"), t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(res) == {"Kvaerno5"}
        assert bool(res.ok)

    def test_non_stiff_group_stays_explicit(self):
        res = Scheduler().run(
            _composite(MildDecay(), "mild"), t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(res) == {"Tsit5"}

    def test_routing_changes_the_solver_not_the_answer(self):
        """Both routes solve the same system to their stated tolerance.

        Not bit-identical, and shouldn't be: the stiff route carries the
        magnitude-scaled vector ``atol`` (``max(atol, atol_scale·|y0|)``,
        1e-6 here) rather than the scalar 1e-9, which is the whole point —
        it buys the step-count collapse. So the agreement to check is
        solver-tolerance-level, not exactness.
        """
        comp = _composite(StiffPair(), "stiff")
        routed = Scheduler().run(comp, t_span=(0.0, 5.0), macro_dt=1.0)
        explicit = Scheduler(auto_stiffness=False, max_steps=_LOOSE_STEPS).run(
            comp, t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(routed) != _solvers(explicit)
        assert jnp.allclose(routed.ys, explicit.ys, rtol=1e-3, atol=1e-6)


class TestRoutingOptOut:
    """Two ways to turn routing off: pin a solver, or say so."""

    def test_pinned_solver_disables_routing(self, caplog):
        """A pinned solver means that solver everywhere, stiff or not — and
        the Scheduler says so rather than quietly ignoring one argument."""
        with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
            sched = Scheduler(solver=dfx.Tsit5(), max_steps=_LOOSE_STEPS)
        assert any("routing is off" in r.message for r in caplog.records)
        res = sched.run(
            _composite(StiffPair(), "stiff"), t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(res) == {"Tsit5"}

    def test_auto_stiffness_false_disables_routing_silently(self, caplog):
        """Explicitly off is not a conflict, so it warns about nothing."""
        with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
            sched = Scheduler(auto_stiffness=False, max_steps=_LOOSE_STEPS)
        assert not any("routing is off" in r.message for r in caplog.records)
        res = sched.run(
            _composite(StiffPair(), "stiff"), t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(res) == {"Tsit5"}

    def test_pinned_solver_wins_over_explicit_auto_stiffness(self, caplog):
        """solver= and auto_stiffness=True together is not an error: the pin
        wins, loudly."""
        with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
            sched = Scheduler(
                solver=dfx.Tsit5(), auto_stiffness=True, max_steps=_LOOSE_STEPS
            )
        assert any("routing is off" in r.message for r in caplog.records)
        res = sched.run(
            _composite(StiffPair(), "stiff"), t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(res) == {"Tsit5"}


class TestRoutingUnderTracing:
    """A cold cache under tracing cannot measure a Jacobian — the eigenvalues
    are tracers — so routing degrades to the explicit solver and says so.
    ``warm_up`` resolves the verdict outside the trace instead."""

    def _loss(self, sched, k):
        comp = _composite(StiffPair(), "stiff")
        c = eqx.tree_at(lambda c: c.processes["stiff"].k_slow, comp, k)
        return sched.run(c, t_span=(0.0, 5.0), macro_dt=1.0).get("stiff/s")[-1]

    def test_default_degrades_and_still_differentiates(self, caplog):
        sched = Scheduler(max_steps=_LOOSE_STEPS)
        with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
            grad = jax.grad(lambda k: self._loss(sched, k))(0.1)
        assert jnp.isfinite(grad) and abs(float(grad)) > 0.0
        assert any(
            "cannot measure group stiffness" in r.message
            for r in caplog.records
        )

    def test_warm_up_routes_under_autodiff(self, caplog):
        """warm_up resolves the verdict eagerly, so the traced run reuses it
        and keeps the implicit solver — no fallback, no warning."""
        sched = Scheduler()
        comp = _composite(StiffPair(), "stiff")
        sched.warm_up(comp, (0.0, 5.0), macro_dt=1.0)
        with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
            grad = jax.grad(lambda k: self._loss(sched, k))(0.1)
        assert jnp.isfinite(grad) and abs(float(grad)) > 0.0
        assert not any(
            "cannot measure group stiffness" in r.message
            for r in caplog.records
        )
        assert _solvers(sched.run(comp, (0.0, 5.0), macro_dt=1.0)) == {
            "Kvaerno5"
        }

    def test_degraded_run_does_not_poison_the_cache(self):
        """The traced fallback must not be cached — a later eager run still
        gets to measure and route."""
        sched = Scheduler(max_steps=_LOOSE_STEPS)
        jax.grad(lambda k: self._loss(sched, k))(0.1)
        res = sched.run(
            _composite(StiffPair(), "stiff"), t_span=(0.0, 5.0), macro_dt=1.0
        )
        assert _solvers(res) == {"Kvaerno5"}
