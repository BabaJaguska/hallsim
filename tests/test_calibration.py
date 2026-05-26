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
        assert float(history.final_params["b"]) == pytest.approx(-0.7, abs=0.05)

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
            loss_fn=loss, init_params=init, mode="forward",
            learning_rate=0.1, verbose=False,
        )
        cal_r = Calibrator(
            loss_fn=loss, init_params=init, mode="reverse",
            learning_rate=0.1, verbose=False,
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
