"""Tests for composable models: SaturatingRemoval, NeuralODE, Hallmarks."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.models.eriq import ERiQOxidativeStress
from hallsim.process import PortRole
from hallsim.scheduler import Scheduler


# ── SaturatingRemoval ───────────────────────────────────────────────────


class TestSaturatingRemoval:
    def test_ports(self):
        from hallsim.models.saturating_removal import SaturatingRemoval

        proc = SaturatingRemoval()
        schema = proc.ports_schema()
        assert "damage" in schema
        assert schema["damage"].role == PortRole.EXCLUSIVE

    def test_derivative_at_zero(self):
        from hallsim.models.saturating_removal import SaturatingRemoval

        proc = SaturatingRemoval(eta=0.5, beta=1.0, K=0.1)
        dy = proc.derivative(0.0, {"damage": jnp.array(0.0)})
        # At t=0, tau=0, so production=0, repair=0 → dD=0
        assert float(dy["damage"]) == pytest.approx(0.0, abs=1e-6)

    def test_damage_accumulates(self):
        from hallsim.models.saturating_removal import SaturatingRemoval

        proc = SaturatingRemoval(eta=0.5, beta=1.0, K=0.1, tau_scale=0.01)
        comp = Composite(
            processes={"damage": proc},
            topology={"damage": {"damage": "cell/damage"}},
            validate=False,
        )
        sched = Scheduler()
        result = sched.run(
            comp, t_span=(0.0, 100.0), macro_dt=1.0, save_dt=1.0
        )
        damage = result.get("cell/damage")
        assert float(damage[-1]) > float(damage[0])

    def test_differentiable(self):
        from hallsim.models.saturating_removal import SaturatingRemoval

        def loss(eta):
            proc = SaturatingRemoval(eta=eta)
            comp = Composite(
                processes={"d": proc},
                topology={"d": {"damage": "pool/d"}},
                validate=False,
            )
            rhs, keys = comp.build_rhs()
            y0_vec = comp.flatten(comp.initial_state(), keys)
            dy_vec = rhs(100.0, y0_vec)
            return dy_vec[keys.index("pool/d")] ** 2

        grad = jax.grad(loss)(0.5)
        assert jnp.isfinite(grad)


# ── NeuralODE Process ───────────────────────────────────────────────────


class TestNeuralODEProcess:
    def test_construction(self):
        from hallsim.models.neuralode import NeuralODEProcess

        proc = NeuralODEProcess(fields=["a", "b", "c"], width=16, depth=1)
        schema = proc.ports_schema()
        assert len(schema) == 3
        assert all(v.role == PortRole.EVOLVED for v in schema.values())

    def test_derivative_shape(self):
        from hallsim.models.neuralode import NeuralODEProcess

        proc = NeuralODEProcess(fields=["x", "y"], width=16, depth=1)
        state = {"x": jnp.array(0.5), "y": jnp.array(-0.3)}
        dy = proc.derivative(0.0, state)
        assert "x" in dy and "y" in dy
        assert jnp.isfinite(dy["x"]) and jnp.isfinite(dy["y"])

    def test_in_composite(self):
        from hallsim.models.neuralode import NeuralODEProcess

        proc = NeuralODEProcess(fields=["x", "y"], width=16, depth=1)
        comp = Composite(
            processes={"neural": proc},
            topology={"neural": {"x": "pool/x", "y": "pool/y"}},
            validate=False,
        )
        sched = Scheduler()
        result = sched.run(comp, t_span=(0.0, 5.0), macro_dt=0.5, save_dt=0.5)
        assert jnp.all(jnp.isfinite(result.get("pool/x")))

    def test_differentiable(self):
        import equinox as eqx
        from hallsim.models.neuralode import NeuralODEProcess

        proc = NeuralODEProcess(fields=["x"], width=8, depth=1)

        @eqx.filter_grad
        def grad_fn(proc):
            state = {"x": jnp.array(1.0)}
            dy = proc.derivative(0.0, state)
            return dy["x"] ** 2

        grad = grad_fn(proc)
        # Should return gradient w.r.t. MLP weights
        assert grad is not None


# ── Hallmark Handles ────────────────────────────────────────────────────


class TestHallmarkHandles:
    def test_apply_modifies_parameter(self):
        from hallsim.hallmarks import HALLMARK_REGISTRY
        from hallsim.models.eriq import ERiQOxidativeStress

        handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
        procs = {"oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=1.0)}

        modified = handle.apply(procs, severity=0.5)
        # MDAMAGE_SA should be 1.0 + 0.5 * 2.0 = 2.0
        assert float(modified["oxidative_stress"].MDAMAGE_SA) == pytest.approx(
            2.0
        )

    def test_severity_zero_no_change(self):
        from hallsim.hallmarks import HALLMARK_REGISTRY
        from hallsim.models.eriq import ERiQOxidativeStress

        handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
        procs = {"oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=1.0)}

        modified = handle.apply(procs, severity=0.0)
        assert float(modified["oxidative_stress"].MDAMAGE_SA) == pytest.approx(
            1.0
        )

    def test_severity_one_max_effect(self):
        from hallsim.hallmarks import HALLMARK_REGISTRY
        from hallsim.models.eriq import ERiQOxidativeStress

        handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
        procs = {"oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=1.0)}

        modified = handle.apply(procs, severity=1.0)
        # 1.0 + 1.0 * 2.0 = 3.0
        assert float(modified["oxidative_stress"].MDAMAGE_SA) == pytest.approx(
            3.0
        )

    def test_apply_hallmarks_multiple(self):
        from hallsim.hallmarks import apply_hallmarks
        from hallsim.models.eriq import (
            ERiQEnergyMetabolism,
            ERiQOxidativeStress,
        )

        procs = {
            "oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=1.0),
            "energy": ERiQEnergyMetabolism(GLYCOL_SA=1.0),
        }

        modified = apply_hallmarks(
            procs,
            {
                "Mitochondrial Dysfunction": 0.8,
                "Deregulated Nutrient Sensing": 0.5,
            },
        )

        assert float(modified["oxidative_stress"].MDAMAGE_SA) == pytest.approx(
            1.0 + 0.8 * 2.0
        )
        assert float(modified["energy"].GLYCOL_SA) == pytest.approx(
            1.0 + 0.5 * 0.5
        )

    def test_missing_process_ignored(self):
        from hallsim.hallmarks import HALLMARK_REGISTRY

        handle = HALLMARK_REGISTRY["Genomic Instability"]
        # "damage_repair" process doesn't exist → should not crash
        procs = {"something_else": ERiQOxidativeStress()}
        modified = handle.apply(procs, severity=0.5)
        assert "something_else" in modified

    def test_summary(self):
        from hallsim.hallmarks import HALLMARK_REGISTRY

        handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
        summary = handle.summary(severity=0.5)
        assert "oxidative_stress.MDAMAGE_SA" in summary
        assert summary["oxidative_stress.MDAMAGE_SA"] == pytest.approx(2.0)

    def test_transform_reads_calibrated_base_not_hardcoded_constant(self):
        """The multiplicative refactor's core invariant: if you change
        the base value, the transform's output scales with it.
        """
        from hallsim.hallmarks import HALLMARK_REGISTRY
        from hallsim.models.eriq import ERiQOxidativeStress

        handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
        # base=1 → severity=1 → 3
        out1 = handle.apply(
            {"oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=1.0)},
            severity=1.0,
        )
        # base=5 → severity=1 → 15 (5x scaling preserved)
        out5 = handle.apply(
            {"oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=5.0)},
            severity=1.0,
        )
        assert float(out1["oxidative_stress"].MDAMAGE_SA) == pytest.approx(3.0)
        assert float(out5["oxidative_stress"].MDAMAGE_SA) == pytest.approx(
            15.0
        )

    def test_grad_through_severity(self):
        """Severity-differentiability claim — pass a jnp.ndarray severity
        and confirm jax.grad through apply_hallmarks returns finite.
        """
        import jax
        import jax.numpy as jnp
        from hallsim.hallmarks import HALLMARK_REGISTRY
        from hallsim.models.eriq import ERiQOxidativeStress

        handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
        procs = {"oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=1.0)}

        def loss(sev):
            modified = handle.apply(procs, severity=sev)
            return modified["oxidative_stress"].MDAMAGE_SA ** 2

        g = jax.grad(loss)(jnp.asarray(0.5))
        assert jnp.isfinite(g)
        # d/dh ((1 + 2h)^2) at h=0.5 = 2 * 2 * 2 = 8 → check magnitude
        assert float(g) == pytest.approx(8.0)
