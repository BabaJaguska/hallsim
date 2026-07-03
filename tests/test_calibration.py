"""Tests for hallsim.calibration.Calibrator.

Covers both autodiff modes on simple synthetic problems where the
optimal parameters are known analytically.
"""

from __future__ import annotations

import jax.numpy as jnp
import optax
import pytest

from hallsim.calibration import Calibrator, CalibrationHistory


# ═══════════════════════════════════════════════════════════════════════════
# Forward-mode autodiff on a synthetic problem
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardMode:
    """Synthetic loss with a known optimum: parabola at (a*, b*)."""

    def test_converges_to_known_optimum(self):
        def loss(p):
            return (p["a"] - 1.5) ** 2 + (p["b"] + 0.7) ** 2

        cal = Calibrator(
            loss_fn=loss,
            init_params={"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)},
            mode="forward",
            learning_rate=0.1,
            verbose=False,
        )
        history = cal.fit(steps=200)
        assert isinstance(history, CalibrationHistory)
        assert history.losses[-1] < 1e-3
        assert float(history.final_params["a"]) == pytest.approx(1.5, abs=0.05)
        assert float(history.final_params["b"]) == pytest.approx(
            -0.7, abs=0.05
        )

    def test_clamping_respected(self):
        """Clamps should hold parameters inside the box even if the
        unconstrained optimum is outside."""

        def loss(p):
            return (p["a"] - 10.0) ** 2  # optimum at a=10

        cal = Calibrator(
            loss_fn=loss,
            init_params={"a": jnp.asarray(0.0)},
            clamps={"a": (0.0, 2.0)},  # but we clamp to [0, 2]
            mode="forward",
            learning_rate=0.5,
            verbose=False,
        )
        history = cal.fit(steps=50)
        # Final param should saturate at the clamp upper bound.
        assert float(history.final_params["a"]) == pytest.approx(2.0, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# Reverse-mode autodiff on the same synthetic problem
# ═══════════════════════════════════════════════════════════════════════════


class TestReverseMode:

    def test_converges_to_known_optimum(self):
        def loss(p):
            return (p["a"] - 0.3) ** 2 + (p["b"] - 2.0) ** 2

        cal = Calibrator(
            loss_fn=loss,
            init_params={"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)},
            mode="reverse",
            learning_rate=0.1,
            verbose=False,
        )
        history = cal.fit(steps=200)
        assert history.losses[-1] < 1e-3
        assert float(history.final_params["a"]) == pytest.approx(0.3, abs=0.05)
        assert float(history.final_params["b"]) == pytest.approx(2.0, abs=0.05)

    def test_forward_and_reverse_agree(self):
        """The two autodiff modes should converge to the same optimum."""

        def loss(p):
            return (p["a"] - 1.0) ** 2 + (p["b"] + 0.5) ** 2

        init = {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}
        cal_f = Calibrator(
            loss_fn=loss,
            init_params=init,
            mode="forward",
            learning_rate=0.1,
            verbose=False,
        )
        cal_r = Calibrator(
            loss_fn=loss,
            init_params=init,
            mode="reverse",
            learning_rate=0.1,
            verbose=False,
        )
        h_f = cal_f.fit(steps=100)
        h_r = cal_r.fit(steps=100)
        # Should converge to the same params within numerical noise.
        for k in init.keys():
            assert jnp.isclose(
                h_f.final_params[k], h_r.final_params[k], atol=1e-3
            )


# ═══════════════════════════════════════════════════════════════════════════
# Custom optimizer
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomOptimizer:

    def test_custom_optimizer_accepted(self):
        def loss(p):
            return p["a"] ** 2

        cal = Calibrator(
            loss_fn=loss,
            init_params={"a": jnp.asarray(2.0)},
            optimizer=optax.sgd(0.1),
            mode="forward",
            verbose=False,
        )
        history = cal.fit(steps=50)
        # SGD on a quadratic converges geometrically; should be near 0.
        assert float(history.final_params["a"]) == pytest.approx(0.0, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Invalid configurations
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidConfig:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            Calibrator(
                loss_fn=lambda p: 0.0,
                init_params={"a": 0.0},
                mode="invalid",  # type: ignore
            )


# ═══════════════════════════════════════════════════════════════════════════
# CalibrationProblem — high-level framework
# ═══════════════════════════════════════════════════════════════════════════


class TestConditionAndParameterRef:

    def test_condition_construction(self):
        from hallsim.calibration import Condition

        c = Condition("DDIS", {"Genomic Instability": 1.0})
        assert c.name == "DDIS"
        assert c.hallmarks["Genomic Instability"] == 1.0

    def test_parameter_ref_construction(self):
        from hallsim.calibration import ParameterRef

        p = ParameterRef(
            process_name="dp14",
            field="parameters.k",
            init=1.0,
            clamp=(0.0, 10.0),
        )
        assert p.process_name == "dp14"
        assert p.field == "parameters.k"
        assert p.init == 1.0
        assert p.clamp == (0.0, 10.0)


class TestCalibrationProblemValidation:
    """Construction-time validation: typos are caught early."""

    def _toy_setup(self):
        """A 1-process composite with a single tunable scalar attribute.

        Using a real Composite (not a mock) verifies the wiring against
        the actual framework. Single-process keeps integration fast.
        """
        import pandas as pd

        from hallsim.calibration import Condition, ParameterRef
        from hallsim.composite import Composite
        from hallsim.gene_reporters import GeneReporter
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay()},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )

        reporters = [
            GeneReporter(
                observable="pool/x",
                gene_symbol="GENE_X",
                sign=+1,
            ),
        ]
        conditions = {
            "ctrl": Condition("ctrl", {}),
            "DDIS": Condition("DDIS", {}),
        }
        arm_pairs = {"DDIS_vs_ctrl": ("DDIS", "ctrl")}
        data = {"DDIS_vs_ctrl": pd.Series({"GENE_X": -0.5})}
        params = {
            "rate": ParameterRef(process_name="decay", field="rate", init=0.1),
        }
        return comp, reporters, conditions, data, arm_pairs, params

    def test_arm_pairs_reference_unknown_condition_raises(self):
        from hallsim.calibration import CalibrationProblem

        comp, reporters, conds, data, arm_pairs, params = self._toy_setup()
        with pytest.raises(KeyError, match="unknown condition"):
            CalibrationProblem(
                composite=comp,
                reporters=reporters,
                conditions=conds,
                data=data,
                arm_pairs={"bad": ("DDIS", "NONEXISTENT")},
                params=params,
                fit_arms=[],
            )

    def test_fit_arms_must_be_in_arm_pairs(self):
        from hallsim.calibration import CalibrationProblem

        comp, reporters, conds, data, arm_pairs, params = self._toy_setup()
        with pytest.raises(KeyError, match="not in arm_pairs"):
            CalibrationProblem(
                composite=comp,
                reporters=reporters,
                conditions=conds,
                data=data,
                arm_pairs=arm_pairs,
                params=params,
                fit_arms=["NONEXISTENT"],
            )

    def test_pure_dial_param_is_blocked(self):
        """Guard rail: fitting a parameter whose hallmark transform IGNORES
        the base value (severity replaces it — a pure dial, e.g. an
        exposure level set directly to the severity) is degenerate and
        raises, naming the hallmark."""
        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            ParameterRef,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import GeneReporter
        from hallsim.hallmarks import HallmarkHandle, ParameterMapping
        from hallsim.process import Port, PortRole, Process
        import pandas as pd

        class Knob(Process):
            knob: float = 1.0

            def ports_schema(self):
                return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

            def derivative(self, t, state):
                return {"x": -self.knob * state["x"]}

        comp = Composite(
            processes={"k": Knob()},
            topology={"k": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )

        # Transform ignores `base` — severity IS the value (a pure dial).
        custom_reg = {
            "Test Hallmark": HallmarkHandle(
                name="Test Hallmark",
                mappings=[
                    ParameterMapping(
                        process_name="k",
                        param_name="knob",
                        transform=lambda h, base: h,
                    ),
                ],
            ),
        }
        with pytest.raises(ValueError, match="'Test Hallmark'"):
            CalibrationProblem(
                composite=comp,
                reporters=[
                    GeneReporter(observable="pool/x", gene_symbol="GX")
                ],
                conditions={"a": Condition("a", {})},
                data={"a_vs_a": pd.Series({"GX": 0.0})},
                arm_pairs={"a_vs_a": ("a", "a")},
                params={
                    "dial": ParameterRef(
                        process_name="k", field="knob", init=1.0
                    ),
                },
                fit_arms=["a_vs_a"],
                hallmark_registry=custom_reg,
            )

    def test_scaled_magnitude_param_is_fittable(self):
        """A parameter scaled by a multiplicative hallmark transform
        (``base * f(severity)``) is the magnitude that full severity maps
        to — legitimately fittable (severity keeps its 0→1 meaning), so
        construction does NOT raise. This is the case the dial-only guard
        rail must let through."""
        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            ParameterRef,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import GeneReporter
        from hallsim.hallmarks import HallmarkHandle, ParameterMapping
        from hallsim.process import Port, PortRole, Process
        import pandas as pd

        class Knob(Process):
            knob: float = 1.0

            def ports_schema(self):
                return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

            def derivative(self, t, state):
                return {"x": -self.knob * state["x"]}

        comp = Composite(
            processes={"k": Knob()},
            topology={"k": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        # Transform depends on `base` — fitting it calibrates the magnitude.
        custom_reg = {
            "Test": HallmarkHandle(
                name="Test",
                mappings=[
                    ParameterMapping(
                        process_name="k",
                        param_name="knob",
                        transform=lambda h, base: base * h,
                    )
                ],
            ),
        }
        # Should construct without raising:
        CalibrationProblem(
            composite=comp,
            reporters=[GeneReporter(observable="pool/x", gene_symbol="GX")],
            conditions={"a": Condition("a", {})},
            data={"a_vs_a": pd.Series({"GX": 0.0})},
            arm_pairs={"a_vs_a": ("a", "a")},
            params={
                "magnitude": ParameterRef(
                    process_name="k", field="knob", init=1.0
                ),
            },
            fit_arms=["a_vs_a"],
            hallmark_registry=custom_reg,
        )

    def test_params_reference_unknown_process_raises(self):
        from hallsim.calibration import (
            CalibrationProblem,
            ParameterRef,
        )

        comp, reporters, conds, data, arm_pairs, _params = self._toy_setup()
        with pytest.raises(KeyError, match="not in composite.processes"):
            CalibrationProblem(
                composite=comp,
                reporters=reporters,
                conditions=conds,
                data=data,
                arm_pairs=arm_pairs,
                params={
                    "bad": ParameterRef(
                        process_name="nonexistent",
                        field="rate",
                        init=0.1,
                    ),
                },
                fit_arms=[],
            )


class TestCalibrationProblemEndToEnd:
    """Runs .loss(), .fit(steps=2), .evaluate() on a toy composite."""

    def _setup(self):
        """One-process composite with a tunable rate parameter; data
        prescribes a Δ_data sign that the loss can chase."""
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            ParameterRef,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import GeneReporter
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay()},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        reporters = [
            GeneReporter(
                observable="pool/x",
                gene_symbol="GENE_X",
                sign=+1,
            ),
            GeneReporter(
                observable="pool/x",
                gene_symbol="GENE_Y",
                sign=-1,
            ),
        ]
        return CalibrationProblem(
            composite=comp,
            reporters=reporters,
            conditions={
                "ctrl": Condition("ctrl", {}),
                "high": Condition("high", {}),
            },
            data={
                "high_vs_ctrl": pd.Series({"GENE_X": -0.5, "GENE_Y": +0.5}),
            },
            arm_pairs={"high_vs_ctrl": ("high", "ctrl")},
            params={
                "rate": ParameterRef(
                    process_name="decay",
                    field="rate",
                    init=0.2,
                    clamp=(0.001, 5.0),
                ),
            },
            fit_arms=["high_vs_ctrl"],
            t_end=5.0,
            macro_dt=1.0,
            n_save=3,
        )

    def test_loss_returns_finite_scalar(self):
        problem = self._setup()
        v = problem.loss({"rate": jnp.asarray(0.2)})
        assert jnp.isfinite(v)
        assert v.shape == ()

    def test_fit_decreases_loss_or_stays(self):
        problem = self._setup()
        # 3 steps is enough to confirm machinery runs; not testing
        # convergence on a contrived toy problem.
        history = problem.fit(steps=3, learning_rate=0.05, verbose=False)
        assert len(history.losses) == 3
        for v in history.losses:
            assert jnp.isfinite(v)

    def test_evaluate_returns_per_arm_per_timepoint_concordance(self):
        problem = self._setup()
        params = {"rate": jnp.asarray(0.2)}
        results = problem.evaluate(params)
        assert "high_vs_ctrl" in results
        # A plain Series is the degenerate single-timepoint case, normalized
        # to {t_end: series}; evaluate returns {arm: {timepoint: result}}.
        per_t = results["high_vs_ctrl"]
        assert set(per_t) == {5.0}
        r = per_t[5.0]
        assert r.n_compared == 2  # 2 reporters, both have data

    def test_trajectory_data_fits_multiple_timepoints(self):
        """A {timepoint: Δseries} arm makes the loss a trajectory fit: the
        loss stays a finite scalar and evaluate reports every timepoint."""
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            ParameterRef,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import GeneReporter
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay()},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        reporters = [GeneReporter(observable="pool/x", gene_symbol="GENE_X")]
        problem = CalibrationProblem(
            composite=comp,
            reporters=reporters,
            conditions={
                "ctrl": Condition("ctrl", {}),
                "high": Condition("high", {}),
            },
            data={
                "high_vs_ctrl": {
                    2.0: pd.Series({"GENE_X": -0.2}),
                    5.0: pd.Series({"GENE_X": -0.5}),
                },
            },
            arm_pairs={"high_vs_ctrl": ("high", "ctrl")},
            params={
                "rate": ParameterRef(
                    process_name="decay", field="rate", init=0.2
                ),
            },
            fit_arms=["high_vs_ctrl"],
            t_end=5.0,
            macro_dt=1.0,
            n_save=6,
        )
        v = problem.loss({"rate": jnp.asarray(0.2)})
        assert jnp.isfinite(v) and v.shape == ()
        results = problem.evaluate({"rate": jnp.asarray(0.2)})
        assert set(results["high_vs_ctrl"]) == {2.0, 5.0}
