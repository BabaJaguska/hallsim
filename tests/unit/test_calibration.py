"""Tests for hallsim.calibration.Calibrator.

Covers both autodiff modes on simple synthetic problems where the
optimal parameters are known analytically.
"""

from __future__ import annotations

import equinox as eqx
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
        assert float(history.best_params["a"]) == pytest.approx(1.5, abs=0.05)
        assert float(history.best_params["b"]) == pytest.approx(-0.7, abs=0.05)

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
        assert float(history.best_params["a"]) == pytest.approx(2.0, abs=1e-3)


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
        assert float(history.best_params["a"]) == pytest.approx(0.3, abs=0.05)
        assert float(history.best_params["b"]) == pytest.approx(2.0, abs=0.05)

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
                h_f.best_params[k], h_r.best_params[k], atol=1e-3
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
        assert float(history.best_params["a"]) == pytest.approx(0.0, abs=0.02)


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


class TestCalibrationProblemValidation:
    """Construction-time validation: typos are caught early."""

    def _toy_setup(self):
        """A 1-process composite with a single tunable scalar attribute.

        Using a real Composite (not a mock) verifies the wiring against
        the actual framework. Single-process keeps integration fast.
        """
        import pandas as pd

        from hallsim.calibration import Condition, FitParam
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
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
            Readout(
                path="pool/x",
                key="GENE_X",
                sign=+1,
            ),
        ]
        conditions = {
            "ctrl": Condition("ctrl", {}),
            "DDIS": Condition("DDIS", {}),
        }
        arms = {"DDIS_vs_ctrl": "DDIS"}
        data = {"DDIS_vs_ctrl": pd.Series({"GENE_X": -0.5})}
        params = {
            "rate": FitParam(process_name="decay", field="rate"),
        }
        return comp, reporters, conditions, data, arms, params

    def test_arm_reference_unknown_condition_raises(self):
        from hallsim.calibration import Arm, CalibrationProblem

        comp, reporters, conds, data, arms, params = self._toy_setup()
        with pytest.raises(KeyError, match="unknown condition"):
            CalibrationProblem(
                composite=comp,
                readouts=reporters,
                conditions=conds,
                data=data,
                arms={"bad": Arm("DDIS", reference="NONEXISTENT")},
                params=params,
                fit_arms=[],
            )

    def test_fit_arms_must_be_in_arms(self):
        from hallsim.calibration import CalibrationProblem

        comp, reporters, conds, data, arms, params = self._toy_setup()
        with pytest.raises(KeyError, match="not in arms"):
            CalibrationProblem(
                composite=comp,
                readouts=reporters,
                conditions=conds,
                data=data,
                arms=arms,
                params=params,
                fit_arms=["NONEXISTENT"],
            )

    def test_pure_dial_param_is_blocked(self):
        """Guard rail: fitting an input level a handle sets (severity
        replaces it — an exposure level) is degenerate and raises, naming
        the handle."""
        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
        from hallsim.handles import Handle, ParameterMapping
        from hallsim.process import Port, PortRole, Process, calibratable
        import pandas as pd

        class Knob(Process):
            knob: float = calibratable(1.0, level=True)

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

        # A level: severity IS the value.
        custom_reg = {
            "Test Hallmark": Handle(
                name="Test Hallmark",
                mappings=[
                    ParameterMapping(
                        process_name="k",
                        param_name="knob",
                        floor=0.0,
                        slope=1.0,
                    ),
                ],
            ),
        }
        with pytest.raises(ValueError, match="'Test Hallmark'"):
            CalibrationProblem(
                composite=comp,
                readouts=[Readout(path="pool/x", key="GX")],
                conditions={"a": Condition("a", {})},
                data={"a_vs_a": pd.Series({"GX": 0.0})},
                arms={"a_vs_a": "a"},
                params={
                    "dial": FitParam(process_name="k", field="knob"),
                },
                fit_arms=["a_vs_a"],
                registry=custom_reg,
            )

    def test_scaled_magnitude_param_is_fittable(self):
        """A rate a handle scales (``base * (floor + slope * severity)``) is
        the magnitude that full severity maps to — legitimately fittable
        (severity keeps its 0→1 meaning), so construction does NOT raise.
        This is the case the level-only guard rail must let through."""
        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
        from hallsim.handles import Handle, ParameterMapping
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
        # A rate: the mapping scales `base`, so fitting it calibrates the
        # magnitude.
        custom_reg = {
            "Test": Handle(
                name="Test",
                mappings=[
                    ParameterMapping(
                        process_name="k",
                        param_name="knob",
                        floor=0.0,
                        slope=1.0,
                    )
                ],
            ),
        }
        # Should construct without raising:
        CalibrationProblem(
            composite=comp,
            readouts=[Readout(path="pool/x", key="GX")],
            conditions={"a": Condition("a", {})},
            data={"a_vs_a": pd.Series({"GX": 0.0})},
            arms={"a_vs_a": "a"},
            params={
                "magnitude": FitParam(process_name="k", field="knob"),
            },
            fit_arms=["a_vs_a"],
            registry=custom_reg,
        )

    def test_params_reference_unknown_process_raises(self):
        from hallsim.calibration import (
            CalibrationProblem,
            FitParam,
        )

        comp, reporters, conds, data, arms, _params = self._toy_setup()
        with pytest.raises(KeyError, match="not in composite.processes"):
            CalibrationProblem(
                composite=comp,
                readouts=reporters,
                conditions=conds,
                data=data,
                arms=arms,
                params={
                    "bad": FitParam(
                        process_name="nonexistent",
                        field="rate",
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
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
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
            Readout(
                path="pool/x",
                key="GENE_X",
                sign=+1,
            ),
            Readout(
                path="pool/x",
                key="GENE_Y",
                sign=-1,
            ),
        ]
        return CalibrationProblem(
            composite=comp,
            readouts=reporters,
            conditions={
                "ctrl": Condition("ctrl", {}),
                "high": Condition("high", {}),
            },
            data={
                "high_vs_ctrl": pd.Series({"GENE_X": -0.5, "GENE_Y": +0.5}),
            },
            arms={"high_vs_ctrl": "high"},
            params={
                "rate": FitParam(
                    process_name="decay",
                    field="rate",
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
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
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
        reporters = [Readout(path="pool/x", key="GENE_X")]
        problem = CalibrationProblem(
            composite=comp,
            readouts=reporters,
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
            arms={"high_vs_ctrl": "high"},
            params={
                "rate": FitParam(process_name="decay", field="rate"),
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


class TestParameterOverrides:
    """``with_overrides`` is the one route for changing a parameter for a run,
    fitted or not. Editing a fitted field in the pytree instead cannot survive
    — every evaluation substitutes the current iterate over it — and an inert
    ablation reads as an edge that carries no influence, so it raises."""

    def _problem(self):
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1
            scale: float = 1.0

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
                }

            def derivative(self, t, state):
                return {"x": -self.scale * self.rate * state["x"]}

        return CalibrationProblem(
            composite=Composite(
                processes={"decay": Decay()},
                topology={"decay": {"x": "pool/x"}},
                validate=False,
                semantic_validation=False,
            ),
            readouts=[Readout(path="pool/x", key="GENE_X", sign=1)],
            conditions={
                "ctrl": Condition("ctrl", {}),
                "high": Condition("high", {}),
            },
            data={"high_vs_ctrl": pd.Series({"GENE_X": -0.5})},
            arms={"high_vs_ctrl": "high"},
            params={"rate": FitParam(process_name="decay", field="rate")},
            fit_arms=["high_vs_ctrl"],
            t_end=5.0,
            macro_dt=1.0,
            n_save=3,
        )

    def test_editing_a_fitted_field_raises(self):
        problem = self._problem()
        problem.composite = eqx.tree_at(
            lambda c: c.processes["decay"].rate,
            problem.composite,
            jnp.asarray(0.0),
        )
        with pytest.raises(ValueError, match="no effect"):
            problem.readout_trajectories({"rate": jnp.asarray(0.2)}, "ctrl")

    def test_override_pins_a_fitted_parameter(self):
        """The route the raise points at: same answer as passing the value in
        by hand, without the caller knowing the field is fitted."""
        problem = self._problem()
        by_hand = problem.readout_trajectories(
            {"rate": jnp.asarray(0.0)}, "ctrl"
        )
        pinned = problem.with_overrides({"rate": 0.0})
        by_override = pinned.readout_trajectories(
            {"rate": jnp.asarray(0.2)}, "ctrl"
        )
        assert jnp.array_equal(by_hand[1], by_override[1])

    def test_override_addresses_a_fitted_field_either_way(self):
        """``"rate"`` (what params calls it) and ``"decay.rate"`` (where it
        lives) name the same thing, so neither spelling has to be remembered.
        """
        problem = self._problem()
        params = {"rate": jnp.asarray(0.2)}
        by_name = problem.with_overrides({"rate": 0.0})
        by_address = problem.with_overrides({"decay.rate": 0.0})
        assert jnp.array_equal(
            by_name.readout_trajectories(params, "ctrl")[1],
            by_address.readout_trajectories(params, "ctrl")[1],
        )

    def test_override_pins_an_unfitted_field(self):
        """Same call for a field nobody fits — equivalent to editing it."""
        problem = self._problem()
        params = {"rate": jnp.asarray(0.2)}
        edited = eqx.tree_at(
            lambda c: c.processes["decay"].scale,
            problem.composite,
            jnp.asarray(3.0),
        )
        by_edit = self._problem()
        by_edit.composite = edited
        pinned = problem.with_overrides({"decay.scale": 3.0})
        assert jnp.array_equal(
            by_edit.readout_trajectories(params, "ctrl")[1],
            pinned.readout_trajectories(params, "ctrl")[1],
        )

    def test_unknown_override_key_raises(self):
        problem = self._problem()
        with pytest.raises(KeyError, match="neither a fittable"):
            problem.with_overrides({"nonsense": 0.0})
        with pytest.raises(KeyError, match="names no field"):
            problem.with_overrides({"decay.nonsense": 0.0})

    def test_overrides_compose_and_leave_the_original_alone(self):
        problem = self._problem()
        params = {"rate": jnp.asarray(0.2)}
        both = problem.with_overrides({"rate": 0.0}).with_overrides(
            {"decay.scale": 3.0}
        )
        assert both._override_params and both._override_fields
        assert not problem._override_params and not problem._override_fields
        assert not jnp.array_equal(
            problem.readout_trajectories(params, "ctrl")[1],
            both.readout_trajectories(params, "ctrl")[1],
        )

    def test_editing_an_unfitted_field_reaches_the_solver(self):
        problem = self._problem()
        params = {"rate": jnp.asarray(0.2)}
        _, base = problem.readout_trajectories(params, "ctrl")
        problem.composite = eqx.tree_at(
            lambda c: c.processes["decay"].scale,
            problem.composite,
            jnp.asarray(3.0),
        )
        _, edited = problem.readout_trajectories(params, "ctrl")
        assert not jnp.allclose(base, edited)


class TestArmReferences:
    """An arm reads against its own start (X_t/X_0), another condition at
    the matched time (X_cond,t/X_ref,t), or nothing (the value itself, in
    the data's units through the reporter's scale)."""

    def _problem(self, reference, scale=1.0):
        import pandas as pd

        from hallsim.calibration import (
            Arm,
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                # x(0)=2 (≠1) so baseline (÷x₀) and raw (÷1) differ — at x₀=1
                # log2(x₀)=0 collapses the two modes.
                return {
                    "x": Port(role=PortRole.EVOLVED, default=2.0, units="uM")
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay()},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        reporters = [Readout(path="pool/x", key="GENE_X", scale=scale)]
        return CalibrationProblem(
            composite=comp,
            readouts=reporters,
            conditions={
                "ctrl": Condition("ctrl", {}),
                "high": Condition("high", {}),
            },
            data={"high_vs_ctrl": {5.0: pd.Series({"GENE_X": -0.5})}},
            arms={"high_vs_ctrl": Arm("high", reference=reference)},
            params={"rate": FitParam(process_name="decay", field="rate")},
            fit_arms=["high_vs_ctrl"],
            t_end=5.0,
            macro_dt=1.0,
            n_save=6,
        )

    def test_references_are_distinct(self):
        # ctrl and high share dynamics (no hallmarks), so the contrast
        # against ctrl is exactly 1 (readout 0) while own-start (÷ t=0) and
        # no reference are not — the three must produce different losses.
        p = {"rate": jnp.asarray(0.2)}
        vals = {
            ref: float(self._problem(ref).loss(p))
            for ref in ("t0", "ctrl", None)
        }
        assert vals["ctrl"] != vals["t0"]
        assert vals["ctrl"] != vals[None]
        assert vals["t0"] != vals[None]

    def test_no_reference_is_the_value_through_the_scale(self):
        import numpy as np

        p = {"rate": jnp.asarray(0.2)}
        # x(0) = 2 decaying at 0.2: the value at t = 5 is 2e^-1, and with no
        # reference the readout is that value times the reporter's scale.
        one = self._problem(None).predicted(p, "high_vs_ctrl", [5.0])
        two = self._problem(None, scale=2.0).predicted(
            p, "high_vs_ctrl", [5.0]
        )
        assert float(one[0, 0]) == pytest.approx(2 * np.exp(-1.0), rel=1e-3)
        assert float(two[0, 0]) == pytest.approx(2 * float(one[0, 0]))
        assert self._problem(None).arm_pairs == {
            "high_vs_ctrl": ("high", "high")
        }
        assert self._problem("ctrl").arm_pairs == {
            "high_vs_ctrl": ("high", "ctrl")
        }

    def test_unknown_reference_rejected(self):
        with pytest.raises(KeyError, match="unknown condition"):
            self._problem("cross_arm")


class TestEquilibrationBaselineMatchesReadout:
    """The equilibration baseline ``summ_b`` (read off the fixed point without a
    run) must equal each reporter's readout evaluated on the control condition
    run to steady state — for *every* readout kind. A power=2 RMS reporter reads
    √⟨x²⟩ = x_fp at the fixed point, NOT x_fp², so summ_b must not be raised to
    the integral power."""

    def _problem(self):
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout, oscillating_readout
        from hallsim.models.running_integral import RunningIntegral
        from hallsim.process import Port, PortRole, Process

        class SetPoint(Process):
            """dx/dt = k(target - x): a nonzero stable fixed point at x=target
            (target != 1 so x_fp and x_fp**2 are distinguishable)."""

            k: float = 1.0
            target: float = 2.0

            def ports_schema(self):
                return {"x": Port(role=PortRole.EVOLVED, default=2.0)}

            def derivative(self, t, state):
                return {"x": self.k * (self.target - state["x"])}

        comp = Composite(
            processes={
                "sp": SetPoint(),
                "xi": RunningIntegral(power=2.0, tau=2.0),  # leaky ∫x² → RMS
            },
            topology={
                "sp": {"x": "s/x"},
                "xi": {"source": "s/x", "integral": "s/x2"},
            },
            validate=False,
            semantic_validation=False,
        )
        reporters = [
            oscillating_readout(  # RMS √⟨x²⟩ over ∫x² → x_fp at steady state
                path="s/x2",
                key="RMS_GENE",
                readout="zerophase_rms",
                tau=2.0,
                sign=+1,
            ),
            Readout(path="s/x", key="LEVEL_GENE", sign=+1),
        ]
        conditions = {
            "ctrl": Condition("ctrl", {}),
            "DDIS": Condition("DDIS", {}),
        }
        data = {
            "DDIS_vs_ctrl": pd.Series({"RMS_GENE": 0.0, "LEVEL_GENE": 0.0})
        }
        return CalibrationProblem(
            composite=comp,
            readouts=reporters,
            conditions=conditions,
            data=data,
            arms={"DDIS_vs_ctrl": "DDIS"},
            params={"k": FitParam(process_name="sp", field="k")},
            fit_arms=["DDIS_vs_ctrl"],
            equilibrate=True,
            equilibration_condition="ctrl",
            t_end=30.0,
            macro_dt=5.0,
        )

    def test_baseline_readout_equals_control_readout(self):
        prob = self._problem()
        init = {"k": jnp.asarray(1.0)}
        prob.warm_up(init)
        subst = prob._substitute(prob.composite.processes, init)
        y0, ref_readout = prob._equilibrate(subst)

        # Run the control from the fixed point and read each reporter late.
        ts, trajs = prob._simulate_condition(
            subst, prob.conditions["ctrl"], y0=y0
        )
        readout = prob._reporter_summaries(ts, trajs, jnp.asarray([27.0]))

        # The force-linked baseline applies the SAME summary as the run, so the
        # two match exactly (not an assumption — the identical transform). The
        # leaky RMS reporter reads √(τ·x_fp²)=√8 (τ=2), NOT x_fp²=4; the level
        # reporter reads x_fp=2. The √τ constant cancels in every fold-change.
        assert jnp.allclose(ref_readout[:, 0], readout[:, 0], rtol=1e-3)
        assert jnp.allclose(
            ref_readout[:, 0], jnp.asarray([jnp.sqrt(8.0), 2.0]), rtol=1e-2
        )


class TestPriorStrength:
    """A prior_sigma given in the parameter's own units instead of log10
    decades is not a weak prior, it is no prior — and it is invisible."""

    def _problem(self, sigma, clamp):
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
        from hallsim.process import Port, PortRole, Process

        class Knob(Process):
            knob: float = 1.0

            def ports_schema(self):
                return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

            def derivative(self, t, state):
                return {"x": -self.knob * state["x"]}

        return CalibrationProblem(
            composite=Composite(
                processes={"k": Knob()},
                topology={"k": {"x": "pool/x"}},
                validate=False,
                semantic_validation=False,
            ),
            readouts=[Readout(path="pool/x", key="GX")],
            conditions={"a": Condition("a", {})},
            data={"a_vs_a": pd.Series({"GX": 0.0})},
            arms={"a_vs_a": "a"},
            params={
                "knob": FitParam(
                    process_name="k",
                    field="knob",
                    clamp=clamp,
                    prior=1.0,
                    prior_sigma=sigma,
                )
            },
            fit_arms=["a_vs_a"],
        )

    def test_sane_sigma_carries_real_precision(self):
        report = self._problem(0.5, (0.01, 100.0)).prior_report()
        assert report[0]["precision"] == pytest.approx(2.0 / 0.25)
        assert report[0]["share"] is None
        assert report[0]["operative"] is None

    def test_a_prior_is_judged_against_the_data(self):
        problem = self._problem(0.5, (0.01, 100.0))
        name = next(iter(problem.fittables))
        assert problem.prior_report({name: 1e9})[0]["operative"] is False
        assert problem.prior_report({name: 1e-9})[0]["operative"] is True

    def test_linear_units_sigma_is_flagged(self, caplog):
        import logging

        problem = self._problem(9000.0, (0.01, 100.0))
        name = next(iter(problem.fittables))
        with caplog.at_level(logging.WARNING, logger="hallsim.calibration"):
            problem._warn_inoperative_priors({name: 1.0})
        assert problem.prior_report({name: 1.0})[0]["operative"] is False
        assert any("log10 decades" in r.message for r in caplog.records)


class TestStartingValueComesFromTheModel:
    """A fitted parameter starts at the composite's own value, always, so a
    declaration cannot contradict the model."""

    def _problem(self, rate):
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import Readout
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay(rate=rate)},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        return CalibrationProblem(
            composite=comp,
            readouts=[Readout(path="pool/x", key="GX", sign=+1)],
            conditions={"ctrl": Condition("ctrl", {})},
            data={"ctrl_vs_ctrl": pd.Series({"GX": 0.0})},
            arms={"ctrl_vs_ctrl": "ctrl"},
            params={"rate": FitParam(process_name="decay", field="rate")},
            fit_arms=["ctrl_vs_ctrl"],
        )

    @pytest.mark.parametrize("rate", [0.1, 0.37, 15.6])
    def test_start_tracks_the_composite(self, rate):
        from hallsim.process import read_param

        problem = self._problem(rate)
        start = float(problem.initial_params()["rate"])
        assert start == pytest.approx(rate)
        assert start == pytest.approx(
            float(read_param(problem.composite.processes["decay"], "rate"))
        )

    def test_no_declared_starting_value_exists(self):
        """There is nowhere to write a start that could contradict the model."""
        import dataclasses

        from hallsim.calibration import FitCoefficient, FitParam

        for cls in (FitParam, FitCoefficient):
            assert "init" not in {
                f.name for f in dataclasses.fields(cls)
            }, f"{cls.__name__} regained a declared starting value"


class TestConditionStartAndWindow:
    """A condition may start from its own state, over its own window, and a
    batched start runs one member per initial state; an arm with no
    reference then fits the observed trajectory directly."""

    RATE = 0.2

    def _problem(self, start, window, data, reference=None, **kw):
        from hallsim.calibration import (
            Arm,
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import trajectory_readouts
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=2.0, units="uM")
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay()},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        return CalibrationProblem(
            composite=comp,
            readouts=trajectory_readouts("pool/x"),
            conditions={
                "obs": Condition("obs", {}, start=start, window=window),
                "ctrl": Condition("ctrl", {}),
            },
            data={"obs": data},
            arms={"obs": Arm("obs", reference=reference)},
            params={"rate": FitParam(process_name="decay", field="rate")},
            fit_arms=["obs"],
            t_end=5.0,
            macro_dt=1.0,
            n_save=11,
            **kw,
        )

    def _batched(self):
        import numpy as np
        import pandas as pd

        x0 = np.array([1.0, 2.0, 4.0])
        data = {
            t: pd.DataFrame({"pool/x": x0 * np.exp(-self.RATE * (t - 2.0))})
            for t in (3.0, 4.0)
        }
        return self._problem({"pool/x": x0}, (2.0, 4.0), data), x0

    def test_batched_start_reads_each_member(self):
        import numpy as np

        problem, x0 = self._batched()
        out = problem.predicted(
            {"rate": jnp.asarray(self.RATE)}, "obs", [3.0, 4.0]
        )
        assert out.shape == (1, 2, 3)
        expected = x0[None, :] * np.exp(-self.RATE * np.array([[1.0], [2.0]]))
        np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-3)

    def test_loss_is_zero_at_the_generating_rate(self):
        problem, _ = self._batched()
        at_truth = float(problem.loss({"rate": jnp.asarray(self.RATE)}))
        off = float(problem.loss({"rate": jnp.asarray(0.5)}))
        assert at_truth < 1e-6
        assert off > 1e-2

    def test_forward_fit_recovers_the_rate(self):
        from hallsim.calibration import Calibrator

        problem, _ = self._batched()
        cal = Calibrator(
            loss_fn=problem.loss,
            init_params={"rate": jnp.asarray(0.35)},
            mode="forward",
            learning_rate=0.05,
            verbose=False,
        )
        hist = cal.fit(steps=40)
        assert float(hist.best_params["rate"]) == pytest.approx(
            self.RATE, abs=0.03
        )

    def test_own_start_is_the_t0_reference(self):
        import numpy as np
        import pandas as pd

        # x = 4 at the window's opening t = 1, so log2(x(3)/x(1)) = -0.4/ln 2
        # whatever the shared start holds at time 0.
        data = {3.0: pd.Series({"pool/x": -0.4 / np.log(2.0)})}
        problem = self._problem(
            {"pool/x": 4.0}, (1.0, 3.0), data, reference="t0"
        )
        assert float(problem.loss({"rate": jnp.asarray(self.RATE)})) < 1e-6

    def test_series_data_broadcasts_over_members(self):
        import numpy as np
        import pandas as pd

        # One target for every member: only the member starting at 2 hits it.
        problem = self._problem(
            {"pool/x": np.array([1.0, 2.0, 4.0])},
            (2.0, 4.0),
            {4.0: pd.Series({"pool/x": 2.0 * np.exp(-self.RATE * 2.0)})},
        )
        assert problem._arm_data_matrix["obs"].shape == (1, 1, 3)
        assert float(problem.loss({"rate": jnp.asarray(self.RATE)})) > 1e-3

    def test_describe_records_start_and_window(self):
        import json

        problem, _ = self._batched()
        cond = problem.describe()["conditions"]["obs"]
        assert cond["start"] == {"pool/x": [1.0, 2.0, 4.0]}
        assert cond["window"] == [2.0, 4.0]
        json.dumps(problem.describe())

    def test_wiring_errors(self):
        import numpy as np
        import pandas as pd

        series = {4.0: pd.Series({"pool/x": 1.0})}
        with pytest.raises(KeyError, match="not a store path"):
            self._problem({"pool/y": 1.0}, None, series)
        with pytest.raises(ValueError, match="batch lengths"):
            self._problem_two_paths(series)
        with pytest.raises(ValueError, match="t_end > t_start"):
            self._problem({"pool/x": 1.0}, (3.0, 3.0), series)
        with pytest.raises(ValueError, match="no batched start"):
            self._problem(
                None, None, {4.0: pd.DataFrame({"pool/x": [1.0, 2.0]})}
            )
        with pytest.raises(ValueError, match="rows"):
            self._problem(
                {"pool/x": np.array([1.0, 2.0, 4.0])},
                None,
                {4.0: pd.DataFrame({"pool/x": [1.0, 2.0]})},
            )

    def _problem_two_paths(self, data):
        # A second store path to give a mismatched batch length to.
        from hallsim.calibration import (
            Arm,
            CalibrationProblem,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import trajectory_readouts
        from hallsim.process import Port, PortRole, Process

        class TwoDecay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=2.0, units="uM"),
                    "y": Port(role=PortRole.EVOLVED, default=2.0, units="uM"),
                }

            def derivative(self, t, state):
                return {
                    "x": -self.rate * state["x"],
                    "y": -self.rate * state["y"],
                }

        comp = Composite(
            processes={"decay": TwoDecay()},
            topology={"decay": {"x": "pool/x", "y": "pool/y"}},
            validate=False,
            semantic_validation=False,
        )
        return CalibrationProblem(
            composite=comp,
            readouts=trajectory_readouts("pool/x"),
            conditions={
                "obs": Condition(
                    "obs",
                    {},
                    start={
                        "pool/x": jnp.asarray([1.0, 2.0]),
                        "pool/y": jnp.asarray([1.0, 2.0, 3.0]),
                    },
                )
            },
            data={"obs": data},
            arms={"obs": Arm("obs", reference=None)},
            params={"rate": FitParam(process_name="decay", field="rate")},
            fit_arms=["obs"],
            t_end=5.0,
            macro_dt=1.0,
        )

    def test_evaluate_reads_the_member_mean(self):
        problem, _ = self._batched()
        results = problem.evaluate({"rate": jnp.asarray(self.RATE)})
        assert set(results["obs"]) == {3.0, 4.0}
        (row,) = results["obs"][4.0].rows
        member_mean = float(problem._arm_data_matrix["obs"][0, 1].mean())
        assert row.delta_data == pytest.approx(member_mean)
        assert row.delta_sim == pytest.approx(member_mean, rel=1e-3)


class TestShootingConditions:
    """A trajectory set cut into shooting windows is a set of batched
    conditions whose consecutive windows share their boundary sample."""

    RATE = 0.2

    def _trajectories(self):
        import numpy as np

        ts = np.linspace(0.0, 5.0, 11)
        x0 = np.array([1.0, 2.0, 4.0])
        ys = (x0[:, None] * np.exp(-self.RATE * ts[None, :]))[..., None]
        return ts, ys

    def _problem(self, conditions, data, arms):
        from hallsim.calibration import CalibrationProblem, FitParam
        from hallsim.composite import Composite
        from hallsim.gene_reporters import trajectory_readouts
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=2.0, units="uM")
                }

            def derivative(self, t, state):
                return {"x": -self.rate * state["x"]}

        comp = Composite(
            processes={"decay": Decay()},
            topology={"decay": {"x": "pool/x"}},
            validate=False,
            semantic_validation=False,
        )
        return CalibrationProblem(
            composite=comp,
            readouts=trajectory_readouts("pool/x"),
            conditions=conditions,
            data=data,
            arms=arms,
            params={"rate": FitParam(process_name="decay", field="rate")},
            fit_arms=list(arms),
            macro_dt=1.0,
            n_save=11,
        )

    def test_windows_share_their_boundary_sample(self):
        from hallsim.calibration import shooting_conditions

        ts, ys = self._trajectories()
        conds, data, arms = shooting_conditions(ts, ys, ["pool/x"], segments=2)
        assert list(conds) == ["shoot0", "shoot1"]
        assert conds["shoot0"].window == (0.0, 2.5)
        assert conds["shoot1"].window == (2.5, 5.0)
        assert list(conds["shoot1"].start["pool/x"]) == list(ys[:, 5, 0])
        # The first window's last sample is the second window's start.
        assert max(data["shoot0"]) == 2.5
        assert list(data["shoot0"][2.5]["pool/x"]) == list(ys[:, 5, 0])
        assert all(a.reference is None for a in arms.values())

    def test_loss_is_zero_at_the_generating_rate(self):
        from hallsim.calibration import shooting_conditions

        ts, ys = self._trajectories()
        problem = self._problem(
            *shooting_conditions(ts, ys, ["pool/x"], segments=2)
        )
        assert float(problem.loss({"rate": jnp.asarray(self.RATE)})) < 1e-6
        assert float(problem.loss({"rate": jnp.asarray(0.4)})) > 1e-3
        # A curriculum stage is a subset of the windows.
        one = problem.data_loss({"rate": jnp.asarray(0.4)}, ["shoot0"])
        assert float(one) > 1e-3

    def test_match_keeps_a_prefix_of_each_window(self):
        from hallsim.calibration import shooting_conditions

        ts, ys = self._trajectories()
        conds, data, _ = shooting_conditions(
            ts, ys, ["pool/x"], segments=2, match=0.4
        )
        assert conds["shoot0"].window == (0.0, 1.0)
        assert sorted(data["shoot0"]) == [0.5, 1.0]
        assert conds["shoot1"].window == (2.5, 3.5)

    def test_single_trajectory_and_bad_shapes(self):
        from hallsim.calibration import shooting_conditions

        ts, ys = self._trajectories()
        conds, _, _ = shooting_conditions(ts, ys[0], ["pool/x"])
        assert conds["shoot0"].start["pool/x"].shape == (1,)
        with pytest.raises(ValueError, match="paths"):
            shooting_conditions(ts, ys, ["pool/x", "pool/y"])
        with pytest.raises(ValueError, match="samples"):
            shooting_conditions(ts[:-1], ys, ["pool/x"])


class TestCollocation:
    """The composite's field at observed states against their slopes: a
    term without a solve, on its own as a pretraining stage or weighted
    into the loss."""

    RATE = 0.2

    def _problem(self, weight=1.0, **collocation_kw):
        import numpy as np

        from hallsim.calibration import Collocation, shooting_conditions

        ts = np.linspace(0.0, 5.0, 21)
        x0 = np.array([1.0, 2.0, 4.0])
        ys = (x0[:, None] * np.exp(-self.RATE * ts[None, :]))[..., None]
        conds, data, arms = shooting_conditions(ts, ys, ["pool/x"])
        colloc = Collocation(
            ts,
            ys,
            collocation_kw.pop("paths", ("pool/x",)),
            weight=weight,
            **collocation_kw,
        )
        return TestShootingConditions()._problem(conds, data, arms), colloc

    def _with(self, weight=1.0, **collocation_kw):
        from hallsim.calibration import CalibrationProblem

        base, colloc = self._problem(weight, **collocation_kw)
        return CalibrationProblem(
            **{**base._ctor_kwargs, "collocation": colloc}
        )

    def test_residual_vanishes_at_the_generating_rate(self):
        problem = self._with()
        near = float(
            problem.collocation_loss({"rate": jnp.asarray(self.RATE)})
        )
        far = float(problem.collocation_loss({"rate": jnp.asarray(0.4)}))
        assert near < 1e-3
        assert far > 1e-1

    def test_loss_adds_the_weighted_term(self):
        p = {"rate": jnp.asarray(0.3)}
        plain = self._with(weight=0.0)
        weighted = self._with(weight=2.0)
        expected = float(plain.loss(p)) + 2.0 * float(
            weighted.collocation_loss(p)
        )
        assert float(weighted.loss(p)) == pytest.approx(expected, rel=1e-6)

    def test_pretraining_stage_recovers_the_rate_without_a_solve(self):
        from hallsim.calibration import Calibrator

        problem = self._with(weight=0.0)
        hist = Calibrator(
            loss_fn=problem.collocation_loss,
            init_params={"rate": jnp.asarray(0.35)},
            mode="forward",
            learning_rate=0.05,
            verbose=False,
        ).fit(steps=60)
        assert float(hist.best_params["rate"]) == pytest.approx(
            self.RATE, abs=0.02
        )

    def test_describe_and_wiring_errors(self):
        import json

        problem = self._with(weight=0.5)
        entry = problem.describe()["collocation"]
        assert entry["paths"] == ["pool/x"]
        assert entry["n_samples"] == 63
        json.dumps(problem.describe())
        with pytest.raises(KeyError, match="not store paths"):
            self._with(paths=("pool/y",))
        with pytest.raises(KeyError, match="not in conditions"):
            self._with(condition="nope")


class TestFitBlock:
    """A learned block's trainable partition is one flat fittable: linear
    space, outside the identifiability report, fitted in reverse mode."""

    def _problem(self):
        import numpy as np

        from hallsim.calibration import (
            CalibrationProblem,
            FitBlock,
            shooting_conditions,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import trajectory_readouts
        from hallsim.models.neuralode import NeuralODEProcess

        ts = np.linspace(0.0, 5.0, 11)
        x0 = np.array([1.0, 2.0, 4.0])
        ys = (x0[:, None] * np.exp(-0.2 * ts[None, :]))[..., None]
        conds, data, arms = shooting_conditions(ts, ys, ["pool/x"], segments=2)
        comp = Composite(
            processes={"m": NeuralODEProcess(fields=["x"], width=8, depth=1)},
            topology={"m": {"state": ["pool/x"]}},
            validate=False,
            semantic_validation=False,
        )
        return CalibrationProblem(
            composite=comp,
            readouts=trajectory_readouts("pool/x"),
            conditions=conds,
            data=data,
            arms=arms,
            params={"net": FitBlock("m")},
            fit_arms=list(arms),
            macro_dt=1.0,
            n_save=11,
        )

    def test_flat_partition_round_trips(self):
        import jax

        problem = self._problem()
        init = problem.initial_params()
        assert init["net"].ndim == 1 and init["net"].size > 8
        assert problem.scalar_fittables == {}
        block = problem.composite.processes["m"]
        same = problem._substitute(problem.composite.processes, init)["m"]
        shifted = problem._substitute(
            problem.composite.processes, {"net": init["net"] + 0.1}
        )["m"]
        assert jnp.array_equal(shifted.in_mean, block.in_mean)  # frozen

        def leaves(mlp):
            return jax.tree_util.tree_leaves(eqx.filter(mlp, eqx.is_array))

        for a, b in zip(leaves(block.mlp), leaves(same.mlp)):
            assert jnp.array_equal(a, b)
        assert any(
            not jnp.allclose(a, b)
            for a, b in zip(leaves(block.mlp), leaves(shifted.mlp))
        )

    def test_fit_is_reverse_mode_and_keeps_the_best_iterate(self):
        problem = self._problem()
        init = problem.initial_params()
        before = float(problem.loss(init))
        hist = problem.fit(steps=8, mode="forward", learning_rate=0.01)
        assert hist.settings["mode"] == "reverse"
        assert hist.settings["log_params"] == []
        assert hist.best_params["net"].shape == init["net"].shape
        assert hist.best_loss <= before
        assert float(problem.loss(hist.best_params)) == pytest.approx(
            hist.best_loss, rel=1e-6
        )

    def test_unknown_process_rejected(self):
        from hallsim.calibration import CalibrationProblem, FitBlock

        problem = self._problem()
        with pytest.raises(KeyError, match="not in composite.processes"):
            CalibrationProblem(
                **{**problem._ctor_kwargs, "params": {"net": FitBlock("z")}}
            )


class TestMinibatchKey:
    """A loss that draws a minibatch from a key is stepped on fresh draws
    and ranked on one fixed draw."""

    def test_keys_reach_the_loss_and_the_best_is_ranked_on_a_fixed_draw(self):
        import jax

        from hallsim.calibration import Calibrator

        seen = []

        def loss(params, key=None):
            seen.append(key is not None)
            draw = 0.0 if key is None else jax.random.uniform(key)
            return (params["a"] - 1.0) ** 2 + 0.1 * draw

        hist = Calibrator(
            loss_fn=loss,
            init_params={"a": jnp.asarray(0.0)},
            mode="reverse",
            learning_rate=0.2,
            minibatch_seed=3,
            verbose=False,
        ).fit(steps=15)
        assert True in seen and False in seen  # drawn steps, exact ranking
        assert float(hist.best_params["a"]) == pytest.approx(1.0, abs=0.15)
        # Ranked on the whole objective: the best loss is the exact value at
        # the best point, without the draw.
        assert hist.best_loss == pytest.approx(
            float(loss(hist.best_params)), rel=1e-6
        )

    def test_lbfgs_refuses_a_minibatch(self):
        from hallsim.calibration import Calibrator

        with pytest.raises(ValueError, match="L-BFGS"):
            Calibrator(
                loss_fn=lambda p, k: p["a"] ** 2,
                init_params={"a": jnp.asarray(1.0)},
                method="lbfgs",
                minibatch_seed=0,
                verbose=False,
            ).fit(steps=2)


class TestCollocationMatchedAndBatch:
    def _problem(self, **kw):
        import numpy as np

        from hallsim.calibration import (
            CalibrationProblem,
            Collocation,
            Condition,
            FitParam,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import trajectory_readouts
        from hallsim.process import Port, PortRole, Process

        class Driven(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(role=PortRole.EVOLVED, default=2.0, units="uM"),
                    "u": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
                }

            def derivative(self, t, state):
                return {
                    "x": -self.rate * state["u"] * state["x"],
                    "u": 0.0 * state["u"],
                }

        comp = Composite(
            processes={"d": Driven()},
            topology={"d": {"x": "pool/x", "u": "pool/u"}},
            validate=False,
            semantic_validation=False,
        )
        ts = np.linspace(0.0, 5.0, 21)
        u = np.array([0.5, 1.0, 2.0])
        x = 2.0 * np.exp(-0.2 * u[:, None] * ts[None, :])
        ys = np.stack([x, np.broadcast_to(u[:, None], x.shape)], axis=-1)
        return CalibrationProblem(
            composite=comp,
            readouts=trajectory_readouts("pool/x"),
            conditions={"c": Condition("c", {})},
            data={},
            arms={},
            params={"rate": FitParam(process_name="d", field="rate")},
            fit_arms=[],
            collocation=Collocation(
                ts,
                ys,
                ("pool/x", "pool/u"),
                matched=kw.pop("matched", ("pool/x",)),
                **kw,
            ),
            t_end=5.0,
        )

    def test_unmatched_path_sets_the_state_only(self):
        import jax

        problem = self._problem()
        assert problem._colloc_out.shape == (1,)
        # Central differences on a 0.25 grid at decay rates up to 0.4.
        assert (
            float(problem.collocation_loss({"rate": jnp.asarray(0.2)})) < 5e-3
        )
        assert (
            float(problem.collocation_loss({"rate": jnp.asarray(0.5)})) > 0.1
        )
        # No arms: the loss is the collocation term alone.
        assert float(
            problem.loss({"rate": jnp.asarray(0.5)})
        ) == pytest.approx(
            float(problem.collocation_loss({"rate": jnp.asarray(0.5)}))
        )
        batched = self._problem(batch=8)
        k = jax.random.PRNGKey(0)
        a = float(batched.collocation_loss({"rate": jnp.asarray(0.5)}, k))
        b = float(batched.collocation_loss({"rate": jnp.asarray(0.5)}))
        assert a != b  # a draw of 8 against every sample

    def test_matched_must_be_a_path(self):
        with pytest.raises(KeyError, match="collocation.matched"):
            self._problem(matched=("pool/y",))


class TestMemberMinibatch(TestConditionStartAndWindow):
    """``member_batch`` draws that many members of a batched condition per
    evaluation from the loss's key; without a key every member runs."""

    def test_members_are_drawn_from_the_key(self):
        import jax

        from hallsim.calibration import CalibrationProblem

        problem, _ = self._batched()
        mini = CalibrationProblem(
            **{**problem._ctor_kwargs, "member_batch": 2}
        )
        p = {"rate": jnp.asarray(0.5)}
        full = float(mini.data_loss(p, ["obs"]))
        assert full == pytest.approx(float(problem.data_loss(p, ["obs"])))
        key = jax.random.PRNGKey(0)
        drawn = float(mini.data_loss(p, ["obs"], key))
        assert drawn != full
        k_data, _ = jax.random.split(key)
        assert float(mini.loss(p, key)) == pytest.approx(
            float(mini.data_loss(p, ["obs"], k_data))
        )
        assert mini.describe()["member_batch"] == 2

    def test_a_separate_ranking_function_selects_the_best(self):
        from hallsim.calibration import Calibrator

        hist = Calibrator(
            loss_fn=lambda p, key=None: (p["a"] - 1.0) ** 2,
            eval_loss_fn=lambda p: (p["a"] - 0.5) ** 2,
            init_params={"a": jnp.asarray(0.0)},
            mode="reverse",
            learning_rate=0.1,
            minibatch_seed=1,
            verbose=False,
        ).fit(steps=20)
        # Descent goes to 1; the ranking function prefers the pass by 0.5.
        assert float(hist.best_params["a"]) == pytest.approx(0.5, abs=0.1)
        assert hist.best_loss == pytest.approx(
            (float(hist.best_params["a"]) - 0.5) ** 2
        )
