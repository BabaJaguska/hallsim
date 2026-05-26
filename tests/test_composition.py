"""Tests for the composable architecture (Process, Composite, Scheduler).

Covers:
- Port schema declaration and roles
- Store construction from port defaults
- Topology validation (exclusive conflicts, missing mappings)
- Additive composition (two processes writing to the same store path)
- Exclusive composition (one process owns a variable)
- Mixed additive + exclusive in the same composite
- Full ODE solve via Scheduler + Diffrax
- Perturbation (kick) mid-simulation
- Differentiability through the full simulation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler, SchedulerResult
from hallsim.store import build_initial_store, validate_topology


# ═══════════════════════════════════════════════════════════════════════════
# Toy processes for testing
# ═══════════════════════════════════════════════════════════════════════════


class Production(Process):
    """Constant production: dx/dt = +rate."""

    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"x": jnp.asarray(self.rate)}


class Decay(Process):
    """First-order decay: dx/dt = -rate * x."""

    rate: float = 0.05

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


class ExclusiveGrowth(Process):
    """Logistic growth — exclusively owns x."""

    rate: float = 0.1
    capacity: float = 10.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EXCLUSIVE, default=1.0, units="cells"),
        }

    def derivative(self, t, state):
        x = state["x"]
        return {"x": self.rate * x * (1.0 - x / self.capacity)}


class TwoPort(Process):
    """Reads y (input), writes derivative to x (evolved)."""

    coupling: float = 0.5

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=0.0),
            "y": Port(role=PortRole.INPUT, default=1.0),
        }

    def derivative(self, t, state):
        return {"x": self.coupling * state["y"]}


class Oscillator(Process):
    """Simple harmonic oscillator: dx/dt = v, dv/dt = -k*x."""

    k: float = 1.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=1.0, units="m"),
            "v": Port(role=PortRole.EVOLVED, default=0.0, units="m/s"),
        }

    def derivative(self, t, state):
        return {
            "x": state["v"],
            "v": -self.k * state["x"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Port and Process basics
# ═══════════════════════════════════════════════════════════════════════════


class TestPortSchema:

    def test_port_roles(self):
        p = Production()
        assert set(p.ports_with_role(PortRole.EVOLVED).keys()) == {"x"}
        assert (
            p.ports_with_role(PortRole.EVOLVED)["x"].role == PortRole.EVOLVED
        )
        assert p.ports_with_role(PortRole.EXCLUSIVE) == {}
        assert p.ports_with_role(PortRole.INPUT) == {}

    def test_exclusive_role(self):
        g = ExclusiveGrowth()
        assert g.ports_with_role(PortRole.EXCLUSIVE).keys() == {"x"}
        assert g.ports_with_role(PortRole.EVOLVED) == {}

    def test_input_role(self):
        tp = TwoPort()
        assert tp.ports_with_role(PortRole.INPUT).keys() == {"y"}
        assert tp.ports_with_role(PortRole.EVOLVED).keys() == {"x"}

    def test_output_port_names(self):
        tp = TwoPort()
        assert tp.output_port_names() == {"x"}

    def test_metadata(self):
        d = Decay()
        meta = d.metadata()
        assert meta["name"] == "Decay"
        assert "x" in meta["ports"]
        assert meta["ports"]["x"]["role"] == "evolved"


# ═══════════════════════════════════════════════════════════════════════════
# Store utilities
# ═══════════════════════════════════════════════════════════════════════════


class TestStore:

    def test_build_initial_store(self):
        procs = {"prod": Production(), "decay": Decay()}
        topo = {
            "prod": {"x": "pool/x"},
            "decay": {"x": "pool/x"},
        }
        store = build_initial_store(procs, topo)
        # Production's default=0.0 wins (first in dict order)
        assert "pool/x" in store
        assert store["pool/x"].shape == ()

    def test_build_store_separate_paths(self):
        procs = {"prod": Production(), "decay": Decay()}
        topo = {
            "prod": {"x": "pool/a"},
            "decay": {"x": "pool/b"},
        }
        store = build_initial_store(procs, topo)
        assert "pool/a" in store
        assert "pool/b" in store


# ═══════════════════════════════════════════════════════════════════════════
# Topology validation
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:

    def test_valid_topology(self):
        procs = {"prod": Production(), "decay": Decay()}
        topo = {"prod": {"x": "pool/x"}, "decay": {"x": "pool/x"}}
        assert validate_topology(procs, topo) == []

    def test_missing_port_mapping(self):
        procs = {"tp": TwoPort()}
        topo = {"tp": {"x": "pool/x"}}  # missing "y"
        errors = validate_topology(procs, topo)
        assert any("y" in e for e in errors)

    def test_extra_topology_entry(self):
        procs = {"prod": Production()}
        topo = {"prod": {"x": "pool/x", "z": "pool/z"}}  # z not in schema
        errors = validate_topology(procs, topo)
        assert any("z" in e for e in errors)

    def test_exclusive_conflict(self):
        """Two processes both claim EXCLUSIVE on the same store path."""
        procs = {"a": ExclusiveGrowth(), "b": ExclusiveGrowth()}
        topo = {"a": {"x": "pool/x"}, "b": {"x": "pool/x"}}
        errors = validate_topology(procs, topo)
        assert any("Exclusive conflict" in e for e in errors)

    def test_exclusive_vs_evolved_conflict(self):
        """One process EXCLUSIVE, another EVOLVED on same path."""
        procs = {"excl": ExclusiveGrowth(), "prod": Production()}
        topo = {"excl": {"x": "pool/x"}, "prod": {"x": "pool/x"}}
        errors = validate_topology(procs, topo)
        assert any("Exclusive conflict" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════
# Composite construction
# ═══════════════════════════════════════════════════════════════════════════


class TestComposite:

    def test_build_composite(self):
        composite = Composite(
            processes={"prod": Production(), "decay": Decay()},
            topology={"prod": {"x": "pool/x"}, "decay": {"x": "pool/x"}},
        )
        assert "pool/x" in composite.store_paths()

    def test_initial_state(self):
        composite = Composite(
            processes={"prod": Production(), "decay": Decay()},
            topology={"prod": {"x": "pool/x"}, "decay": {"x": "pool/x"}},
        )
        y0 = composite.initial_state()
        assert "pool/x" in y0

    def test_rhs_additive(self):
        """Two processes with EVOLVED ports on the same path → additive."""
        prod = Production(rate=0.1)
        decay = Decay(rate=0.05)
        composite = Composite(
            processes={"prod": prod, "decay": decay},
            topology={"prod": {"x": "pool/x"}, "decay": {"x": "pool/x"}},
        )
        rhs, keys = composite.build_rhs()
        y_vec = composite.flatten({"pool/x": jnp.array(2.0)}, keys)
        dydt = composite.unflatten(rhs(0.0, y_vec), keys)
        # Production: +0.1, Decay: -0.05 * 2.0 = -0.1 → net = 0.0
        assert jnp.allclose(dydt["pool/x"], jnp.array(0.0), atol=1e-6)

    def test_rhs_exclusive(self):
        """Single process with EXCLUSIVE port."""
        growth = ExclusiveGrowth(rate=0.1, capacity=10.0)
        composite = Composite(
            processes={"growth": growth},
            topology={"growth": {"x": "pop/cells"}},
        )
        rhs, keys = composite.build_rhs()
        y_vec = composite.flatten({"pop/cells": jnp.array(5.0)}, keys)
        dydt = composite.unflatten(rhs(0.0, y_vec), keys)
        expected = 0.1 * 5.0 * (1.0 - 5.0 / 10.0)
        assert jnp.allclose(dydt["pop/cells"], jnp.array(expected), atol=1e-6)

    def test_validation_raises(self):
        """Invalid topology raises ValueError."""
        with pytest.raises(ValueError, match="Topology validation failed"):
            Composite(
                processes={"a": ExclusiveGrowth(), "b": ExclusiveGrowth()},
                topology={"a": {"x": "pool/x"}, "b": {"x": "pool/x"}},
            )

    def test_composite_metadata(self):
        composite = Composite(
            processes={"prod": Production()},
            topology={"prod": {"x": "pool/x"}},
        )
        meta = composite.metadata()
        assert "prod" in meta
        assert meta["prod"]["name"] == "Production"

    def test_input_port_wiring(self):
        """A process with an INPUT port reads from another process's output."""
        prod = Production(rate=0.5)
        tp = TwoPort(coupling=2.0)
        composite = Composite(
            processes={"prod": prod, "tp": tp},
            topology={
                "prod": {"x": "pool/a"},
                "tp": {"x": "pool/b", "y": "pool/a"},
            },
        )
        rhs, keys = composite.build_rhs()
        y_vec = composite.flatten(
            {"pool/a": jnp.array(3.0), "pool/b": jnp.array(0.0)}, keys
        )
        dydt = composite.unflatten(rhs(0.0, y_vec), keys)
        # prod: d(pool/a)/dt = 0.5
        # tp: d(pool/b)/dt = 2.0 * pool/a = 2.0 * 3.0 = 6.0
        assert jnp.allclose(dydt["pool/a"], jnp.array(0.5), atol=1e-6)
        assert jnp.allclose(dydt["pool/b"], jnp.array(6.0), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Scheduler (full ODE solve)
# ═══════════════════════════════════════════════════════════════════════════


class TestScheduler:

    def test_simple_decay(self):
        """Exponential decay: x(t) = x0 * exp(-rate * t)."""
        decay = Decay(rate=0.1)
        composite = Composite(
            processes={"decay": decay},
            topology={"decay": {"x": "x"}},
        )
        sched = Scheduler()
        result = sched.run(
            composite, t_span=(0.0, 10.0), macro_dt=0.1, save_dt=0.1
        )

        assert isinstance(result, SchedulerResult)
        assert result.ts.shape[0] > 1
        # Check exponential decay at t=10
        x_final = float(result.get("x")[-1])
        x_expected = 1.0 * jnp.exp(-0.1 * 10.0)
        assert abs(x_final - float(x_expected)) < 0.01

    def test_production_plus_decay_steady_state(self):
        """Production + decay reaches steady state at rate_prod / rate_decay."""
        prod = Production(rate=0.5)
        decay = Decay(rate=0.1)
        composite = Composite(
            processes={"prod": prod, "decay": decay},
            topology={"prod": {"x": "x"}, "decay": {"x": "x"}},
        )
        sched = Scheduler()
        # IC: x = 0 (override the port default via the trailing-axis tensor).
        y0 = composite.initial_state_vec()
        y0 = y0.at[composite.store_keys().index("x")].set(0.0)
        result = sched.run(
            composite, t_span=(0.0, 200.0), macro_dt=1.0, save_dt=1.0, y0=y0
        )
        x_final = float(result.get("x")[-1])
        # Steady state: production / decay = 0.5 / 0.1 = 5.0
        assert abs(x_final - 5.0) < 0.1

    def test_oscillator(self):
        """Simple harmonic oscillator conserves energy."""
        osc = Oscillator(k=1.0)
        composite = Composite(
            processes={"osc": osc},
            topology={"osc": {"x": "pos", "v": "vel"}},
        )
        sched = Scheduler(rtol=1e-6, atol=1e-8)
        result = sched.run(
            composite, t_span=(0.0, 20.0), macro_dt=0.1, save_dt=0.1
        )
        x = result.get("pos")
        v = result.get("vel")
        # Energy = 0.5*k*x^2 + 0.5*v^2 should be ~constant
        energy = 0.5 * 1.0 * x**2 + 0.5 * v**2
        assert jnp.allclose(energy, energy[0], atol=1e-3)

    def test_save_final_only(self):
        """save_dt covering the full t_span saves only start + end."""
        decay = Decay(rate=0.1)
        composite = Composite(
            processes={"decay": decay},
            topology={"decay": {"x": "x"}},
        )
        sched = Scheduler()
        # macro_dt = save_dt = full span → 2 save points (start + end).
        result = sched.run(
            composite, t_span=(0.0, 10.0), macro_dt=10.0, save_dt=10.0
        )
        assert result.ts.shape[0] == 2

    def test_perturbation(self):
        """Mid-simulation kick changes trajectory — kick is a KickEvent
        Process composed via topology, dispatched by the Scheduler."""
        from hallsim.models.kick_event import KickEvent
        from hallsim.scheduler import Scheduler

        decay = Decay(rate=0.1)
        kick = KickEvent(kick_time=10.0, deltas={"x": 5.0}, units={"x": "uM"})
        composite = Composite(
            processes={"decay": decay, "kick": kick},
            topology={"decay": {"x": "x"}, "kick": {"x": "x"}},
        )
        result = Scheduler().run(
            composite,
            t_span=(0.0, 20.0),
            macro_dt=1.0,
            save_dt=1.0,
        )
        # At t=10, x should have been boosted
        x = result.get("x")
        t10_idx = int(jnp.argmin(jnp.abs(result.ts - 10.0)))
        t11_idx = t10_idx + 1
        # After kick, x should be higher than exponential decay alone
        x_no_kick = 1.0 * jnp.exp(-0.1 * 11.0)
        assert float(x[t11_idx]) > float(x_no_kick)
        # The kick fired exactly once.
        assert len(result.events) == 1
        assert result.events[0].process == "kick"

    def test_multi_process_simulation(self):
        """Run a composite with input ports through the Scheduler."""
        prod = Production(rate=1.0)
        tp = TwoPort(coupling=0.5)
        composite = Composite(
            processes={"prod": prod, "tp": tp},
            topology={
                "prod": {"x": "source"},
                "tp": {"x": "sink", "y": "source"},
            },
        )
        result = Scheduler().run(
            composite, t_span=(0.0, 10.0), macro_dt=0.5, save_dt=0.5
        )
        # Source grows linearly: source(t) ≈ t
        # Sink grows quadratically: sink(t) ≈ 0.5 * 0.5 * t^2 = 0.25*t^2
        source_final = float(result.get("source")[-1])
        assert source_final > 9.0  # should be ~10
        sink_final = float(result.get("sink")[-1])
        assert sink_final > 20.0  # should be ~25


# ═══════════════════════════════════════════════════════════════════════════
# Differentiability
# ═══════════════════════════════════════════════════════════════════════════


class TestDifferentiability:

    def test_grad_through_simulation(self):
        """jax.grad through a full ODE solve w.r.t. a process parameter."""

        def loss_fn(rate):
            decay = Decay(rate=rate)
            composite = Composite(
                processes={"decay": decay},
                topology={"decay": {"x": "x"}},
                validate=False,  # skip validation for speed in grad
            )
            result = Scheduler(max_steps=10_000).run(
                composite,
                t_span=(0.0, 5.0),
                macro_dt=5.0,
                save_dt=5.0,
            )
            return jnp.squeeze(result.get("x")[-1])

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(0.1))
        # Gradient should be negative: higher decay rate → lower final x
        assert float(g) < 0.0

    def test_grad_through_severity_to_smooth_event_proxy(self):
        """Gradients flow through a severity-like parameter to a smooth
        sigmoid surrogate for an event indicator.

        Hard events (``ProcessKind.EVENT``) cast a boolean condition and are
        therefore non-differentiable. The standard engineering workaround is
        a sigmoid surrogate ``σ((q - θ) / τ)`` for "event has fired by T",
        where ``q`` is a continuous quantity, ``θ`` a threshold, ``τ`` a
        temperature. This test pins down the contract: gradients of such a
        surrogate w.r.t. an upstream continuous parameter (here played by a
        hallmark-severity-style ``rate``) are well-defined and match finite
        differences, which is the differentiability claim used in lieu of
        (or as a stepping stone to) IFT-based event-time gradients.
        """
        threshold = 0.5
        temperature = 0.05
        T = 1.0

        def event_proxy(severity):
            # Severity drives a linear accumulator: dx/dt = severity
            # so x(T) = severity * T (clean analytic case).
            prod = Production(rate=severity)
            composite = Composite(
                processes={"damage": prod},
                topology={"damage": {"x": "x"}},
                validate=False,
            )
            result = Scheduler(max_steps=10_000).run(
                composite,
                t_span=(0.0, T),
                macro_dt=T,
                save_dt=T,
            )
            x_T = jnp.squeeze(result.get("x")[-1])
            # Smooth surrogate for "event fired by T".
            return jax.nn.sigmoid((x_T - threshold) / temperature)

        # severity*T == threshold puts us at the sigmoid inflection,
        # where the surrogate is most sensitive — gradient should be large
        # and unambiguously positive.
        severity_0 = jnp.array(threshold / T)
        g = jax.grad(event_proxy)(severity_0)
        assert float(g) > 0.0

        # Central finite-difference sanity check.
        eps = 1e-3
        fd = (
            event_proxy(severity_0 + eps) - event_proxy(severity_0 - eps)
        ) / (2 * eps)
        rel_err = abs(float(g) - float(fd)) / (abs(float(fd)) + 1e-8)
        assert rel_err < 1e-2, (
            f"autodiff grad={float(g):.4f} disagrees with FD={float(fd):.4f} "
            f"(rel_err={rel_err:.2e})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_single_process_no_topology_sugar(self):
        """A process with identity topology (port name = store path)."""
        decay = Decay(rate=0.1)
        composite = Composite(
            processes={"decay": decay},
            topology={"decay": {"x": "x"}},
        )
        y0 = composite.initial_state()
        assert "x" in y0

    def test_empty_derivative(self):
        """A process that returns no derivatives (input-only) is valid."""

        class Observer(Process):
            def ports_schema(self):
                return {"x": Port(role=PortRole.INPUT, default=0.0)}

            def derivative(self, t, state):
                return {}

        composite = Composite(
            processes={"obs": Observer(), "prod": Production()},
            topology={
                "obs": {"x": "val"},
                "prod": {"x": "val"},
            },
        )
        rhs, keys = composite.build_rhs()
        y_vec = composite.flatten({"val": jnp.array(1.0)}, keys)
        dydt = composite.unflatten(rhs(0.0, y_vec), keys)
        assert jnp.allclose(dydt["val"], jnp.array(0.1), atol=1e-6)
