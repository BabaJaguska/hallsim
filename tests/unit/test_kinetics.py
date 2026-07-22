"""Rate-law primitives: values, bounds, clamping, and differentiability."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.kinetics import (
    hill_gate,
    hill_inhibition,
    mass_action,
    michaelis_menten,
)

_a = jnp.asarray


class TestHill:
    def test_half_saturation(self):
        assert float(hill_gate(_a(4.0), _a(4.0), _a(2.0))) == pytest.approx(
            0.5, abs=1e-6
        )

    def test_bounds_and_monotonic(self):
        lo = float(hill_gate(_a(1.0), _a(4.0), _a(2.0)))
        hi = float(hill_gate(_a(16.0), _a(4.0), _a(2.0)))
        assert 0.0 < lo < hi < 1.0

    def test_negative_input_clamps_to_zero(self):
        assert float(hill_gate(_a(-5.0), _a(4.0), _a(2.0))) == 0.0

    def test_inhibition_is_complement(self):
        x, K, n = _a(7.0), _a(4.0), _a(2.0)
        assert float(hill_inhibition(x, K, n)) == pytest.approx(
            1.0 - float(hill_gate(x, K, n)), abs=1e-9
        )

    def test_differentiable(self):
        g = jax.grad(lambda x: hill_gate(x, _a(4.0), _a(2.0)))(_a(4.0))
        assert jnp.isfinite(g) and g > 0


class TestMichaelisMenten:
    def test_half_vmax_at_km(self):
        v = michaelis_menten(_a(2.0), vmax=_a(10.0), km=_a(2.0))
        assert float(v) == pytest.approx(5.0, abs=1e-4)

    def test_saturates_toward_vmax(self):
        v = michaelis_menten(_a(1e6), vmax=_a(10.0), km=_a(2.0))
        assert float(v) == pytest.approx(10.0, rel=1e-3)

    def test_negative_substrate_clamps(self):
        assert float(michaelis_menten(_a(-1.0), _a(10.0), _a(2.0))) == 0.0

    def test_differentiable(self):
        g = jax.grad(lambda s: michaelis_menten(s, _a(10.0), _a(2.0)))(_a(2.0))
        assert jnp.isfinite(g) and g > 0


class TestMassAction:
    def test_orders(self):
        assert float(mass_action(_a(2.0))) == 2.0  # zeroth order
        assert float(mass_action(_a(2.0), _a(3.0))) == 6.0  # first order
        assert float(mass_action(_a(2.0), _a(3.0), _a(4.0))) == 24.0  # second

    def test_negative_reactant_clamps(self):
        assert float(mass_action(_a(2.0), _a(-3.0), _a(4.0))) == 0.0

    def test_differentiable(self):
        g = jax.grad(lambda a: mass_action(_a(2.0), a, _a(4.0)))(_a(3.0))
        assert float(g) == pytest.approx(8.0, abs=1e-6)
