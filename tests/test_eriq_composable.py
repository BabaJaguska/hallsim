"""Tests for the composable ERiQ model (decomposed into 3 Processes)."""

import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.models.eriq import (
    ERIQ_HOMEOSTATIC_IC,
    ERiQEnergyMetabolism,
    ERiQOxidativeStress,
    ERiQSignaling,
    _compute_algebraic,
    build_eriq_composite,
)
from hallsim.process import PortRole
from hallsim.simulator import Simulator


# ── Algebraic layer ─────────────────────────────────────────────────────


class TestAlgebraicComputation:
    def test_compute_at_homeostasis(self):
        """Algebraic nodes should produce finite values at homeostatic IC."""
        obs = _compute_algebraic(ERIQ_HOMEOSTATIC_IC)
        for key, val in obs.items():
            assert jnp.isfinite(val), f"{key} is not finite: {val}"

    def test_ros_scales_with_activity(self):
        state = dict(ERIQ_HOMEOSTATIC_IC)
        obs1 = _compute_algebraic(state)

        state["ROS_activity"] = 0.2
        obs2 = _compute_algebraic(state)

        assert obs2["ROS"] > obs1["ROS"]

    def test_ampk_inversely_related_to_atp(self):
        state = dict(ERIQ_HOMEOSTATIC_IC)
        obs1 = _compute_algebraic(state)

        # Increase mito_function → more ATP → less AMPK
        state["mito_function"] = 10.0
        obs2 = _compute_algebraic(state)

        assert obs2["AMPK"] < obs1["AMPK"]

    def test_params_override(self):
        """_SA parameters can be passed via _params key."""
        state = dict(ERIQ_HOMEOSTATIC_IC)
        obs1 = _compute_algebraic(state)

        state["_params"] = {"ROS_SA": 2.0}
        obs2 = _compute_algebraic(state)

        assert obs2["ROS"] == pytest.approx(2.0 * obs1["ROS"], rel=1e-6)


# ── Individual Process tests ────────────────────────────────────────────


class TestERiQEnergyMetabolism:
    def test_ports(self):
        proc = ERiQEnergyMetabolism()
        schema = proc.ports_schema()
        exclusive = {k for k, v in schema.items() if v.role == PortRole.EXCLUSIVE}
        inputs = {k for k, v in schema.items() if v.role == PortRole.INPUT}
        assert exclusive == {"mito_function", "mito_enzymes", "glycolysis", "glycolytic_enzymes"}
        assert "mito_damage" in inputs
        assert "ROS_activity" in inputs

    def test_derivative_finite(self):
        proc = ERiQEnergyMetabolism()
        derivs = proc.derivative(0.0, ERIQ_HOMEOSTATIC_IC)
        for key, val in derivs.items():
            assert jnp.isfinite(val), f"d({key})/dt is not finite: {val}"

    def test_glycol_sa_scaling(self):
        proc1 = ERiQEnergyMetabolism(GLYCOL_SA=1.0)
        proc2 = ERiQEnergyMetabolism(GLYCOL_SA=2.0)
        d1 = proc1.derivative(0.0, ERIQ_HOMEOSTATIC_IC)
        d2 = proc2.derivative(0.0, ERIQ_HOMEOSTATIC_IC)
        # Glycolysis derivative should scale with GLYCOL_SA
        assert abs(float(d2["glycolysis"])) > abs(float(d1["glycolysis"]))


class TestERiQOxidativeStress:
    def test_ports(self):
        proc = ERiQOxidativeStress()
        schema = proc.ports_schema()
        exclusive = {k for k, v in schema.items() if v.role == PortRole.EXCLUSIVE}
        assert exclusive == {"mito_damage", "ROS_integrator_c", "ROS_activity"}

    def test_derivative_finite(self):
        proc = ERiQOxidativeStress()
        derivs = proc.derivative(0.0, ERIQ_HOMEOSTATIC_IC)
        for key, val in derivs.items():
            assert jnp.isfinite(val), f"d({key})/dt is not finite: {val}"


class TestERiQSignaling:
    def test_ports(self):
        proc = ERiQSignaling()
        schema = proc.ports_schema()
        exclusive = {k for k, v in schema.items() if v.role == PortRole.EXCLUSIVE}
        assert exclusive == {"mTOR_integrator_c", "mTOR_activity",
                             "p53_integrator_c", "p53_activity"}

    def test_derivative_finite(self):
        proc = ERiQSignaling()
        derivs = proc.derivative(0.0, ERIQ_HOMEOSTATIC_IC)
        for key, val in derivs.items():
            assert jnp.isfinite(val), f"d({key})/dt is not finite: {val}"


# ── Composite wiring ────────────────────────────────────────────────────


class TestBuildERiQComposite:
    def test_builds_without_error(self):
        comp = build_eriq_composite()
        assert len(comp.processes) == 3

    def test_store_paths(self):
        comp = build_eriq_composite()
        paths = comp.store_paths()
        assert "eriq/mito_function" in paths
        assert "eriq/ROS_activity" in paths
        assert "eriq/p53_activity" in paths
        assert len(paths) == 11  # 11 unique state variables

    def test_initial_state_matches_homeostatic(self):
        comp = build_eriq_composite()
        y0 = comp.initial_state()
        assert float(y0["eriq/mito_function"]) == pytest.approx(3.6239)
        assert float(y0["eriq/mito_damage"]) == pytest.approx(0.0724)

    def test_build_rhs_returns_callable(self):
        comp = build_eriq_composite()
        rhs = comp.build_rhs()
        y0 = comp.initial_state()
        dy = rhs(0.0, y0)
        for key, val in dy.items():
            assert jnp.isfinite(val), f"RHS({key}) is not finite"

    def test_custom_prefix(self):
        comp = build_eriq_composite(prefix="cell")
        assert "cell/mito_function" in comp.store_paths()


# ── Integration: short simulation ───────────────────────────────────────


class TestERiQSimulation:
    def test_short_simulation_stable(self):
        """Run ERiQ for a short time and verify no NaN/Inf."""
        comp = build_eriq_composite()
        sim = Simulator()
        result = sim.run(comp, t_span=(0.0, 100.0), dt=1.0)

        assert len(result.ts) > 0
        for key, traj in result.ys.items():
            assert jnp.all(jnp.isfinite(traj)), f"{key} has non-finite values"

    def test_damage_increases_over_time(self):
        """Mitochondrial damage should accumulate from homeostasis."""
        comp = build_eriq_composite()
        sim = Simulator()
        result = sim.run(comp, t_span=(0.0, 500.0), dt=10.0)

        damage = result.ys["eriq/mito_damage"]
        # Damage should increase from initial value
        assert float(damage[-1]) > float(damage[0])

    def test_homeostatic_derivatives_small(self):
        """At homeostatic IC, derivatives should be near-zero (quasi-steady state)."""
        comp = build_eriq_composite()
        rhs = comp.build_rhs()
        y0 = comp.initial_state()
        dy = rhs(0.0, y0)

        for key, val in dy.items():
            # Derivatives should be small at homeostasis (not exactly zero
            # because decomposition introduces splitting error)
            assert abs(float(val)) < 1.0, f"d({key})/dt = {float(val)} is too large at homeostasis"

    def test_differentiable(self):
        """Can compute gradients through ERiQ composite RHS."""
        import jax

        def loss(glycol_sa):
            comp = build_eriq_composite(GLYCOL_SA=glycol_sa, validate=False)
            rhs = comp.build_rhs()
            y0 = comp.initial_state()
            dy = rhs(0.0, y0)
            return dy["eriq/glycolysis"] ** 2

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_three_processes_all_continuous(self):
        """All 3 ERiQ processes are CONTINUOUS kind."""
        comp = build_eriq_composite()
        assert len(comp.continuous_processes()) == 3
        assert len(comp.discrete_processes()) == 0
        assert len(comp.event_processes()) == 0
