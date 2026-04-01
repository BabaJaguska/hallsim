"""Tests for composable models: SaturatingRemoval, NeuralODE, Hallmarks, DataValidation."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import PortRole
from hallsim.simulator import Simulator


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
        sim = Simulator()
        result = sim.run(comp, t_span=(0.0, 100.0), dt=1.0)
        damage = result.ys["cell/damage"]
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
            rhs = comp.build_rhs()
            y0 = comp.initial_state()
            dy = rhs(100.0, y0)
            return dy["pool/d"] ** 2

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
        sim = Simulator()
        result = sim.run(comp, t_span=(0.0, 5.0), dt=0.5)
        assert jnp.all(jnp.isfinite(result.ys["pool/x"]))

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


# ── Data Validation Layer ───────────────────────────────────────────────


class TestDataValidation:
    def test_perfect_concordance(self):
        from hallsim.data_validation import (
            MeasuredScores,
            PathwayMapping,
            validate_against_data,
        )

        baseline = {"eriq/mTOR_activity": jnp.array(1.0)}
        perturbed = {"eriq/mTOR_activity": jnp.array(0.5)}  # decreased

        measured = MeasuredScores(
            condition="rapamycin",
            pathway_scores={"MTOR_SIGNALING": -0.5},  # also decreased
        )

        mappings = [
            PathwayMapping(
                state_var="eriq/mTOR_activity",
                pathway_name="MTOR_SIGNALING",
            ),
        ]

        result = validate_against_data(baseline, perturbed, measured, mappings)
        assert result.concordance == 1.0
        assert result.n_pathways == 1

    def test_mismatch(self):
        from hallsim.data_validation import (
            MeasuredScores,
            PathwayMapping,
            validate_against_data,
        )

        baseline = {"eriq/mTOR_activity": jnp.array(1.0)}
        perturbed = {"eriq/mTOR_activity": jnp.array(1.5)}  # increased

        measured = MeasuredScores(
            condition="rapamycin",
            pathway_scores={"MTOR_SIGNALING": -0.5},  # decreased
        )

        mappings = [
            PathwayMapping(
                state_var="eriq/mTOR_activity",
                pathway_name="MTOR_SIGNALING",
            ),
        ]

        result = validate_against_data(baseline, perturbed, measured, mappings)
        assert result.concordance == 0.0

    def test_inverse_direction_mapping(self):
        from hallsim.data_validation import (
            MeasuredScores,
            PathwayMapping,
            validate_against_data,
        )

        baseline = {"x": jnp.array(1.0)}
        perturbed = {"x": jnp.array(2.0)}  # x increases

        measured = MeasuredScores(
            condition="test",
            pathway_scores={"pathway": -0.3},  # pathway decreases
        )

        # direction=-1: x increase should correspond to pathway decrease
        mappings = [
            PathwayMapping(
                state_var="x", pathway_name="pathway", direction=-1.0
            )
        ]

        result = validate_against_data(baseline, perturbed, measured, mappings)
        assert result.concordance == 1.0

    def test_missing_pathway_skipped(self):
        from hallsim.data_validation import (
            MeasuredScores,
            PathwayMapping,
            validate_against_data,
        )

        baseline = {"x": jnp.array(1.0)}
        perturbed = {"x": jnp.array(2.0)}
        measured = MeasuredScores(condition="test", pathway_scores={})

        mappings = [PathwayMapping(state_var="x", pathway_name="missing")]
        result = validate_against_data(baseline, perturbed, measured, mappings)
        assert result.n_pathways == 0

    def test_str_output(self):
        from hallsim.data_validation import ValidationResult

        result = ValidationResult(
            n_pathways=2,
            concordance=0.5,
            per_pathway=[
                {
                    "pathway": "A",
                    "state_var": "x",
                    "sim_delta": 0.1,
                    "meas_delta": 0.2,
                    "concordant": True,
                },
                {
                    "pathway": "B",
                    "state_var": "y",
                    "sim_delta": 0.1,
                    "meas_delta": -0.1,
                    "concordant": False,
                },
            ],
        )
        s = str(result)
        assert "concordance = 50.0%" in s
        assert "OK" in s
        assert "MISMATCH" in s

    def test_eriq_pathway_mappings_exist(self):
        from hallsim.data_validation import ERIQ_PATHWAY_MAPPINGS

        assert len(ERIQ_PATHWAY_MAPPINGS) >= 5
        for m in ERIQ_PATHWAY_MAPPINGS:
            assert m.state_var.startswith("eriq/")


# Import needed for test_missing_process_ignored
from hallsim.models.eriq import ERiQOxidativeStress  # noqa: E402
