"""GatedRemoval: the edge is inert below its gate, clears above it, and the
decay is proportional so the target never crosses zero."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.models.gated_removal import GatedRemoval
from hallsim.process import PortRole


def _edge(**kw):
    kw.setdefault("k_remove", 1.0)
    kw.setdefault("K", 1.0)
    kw.setdefault("n", 4.0)
    return GatedRemoval(**kw)


def _dy(edge, trigger, target):
    return float(
        edge.derivative(
            0.0,
            {
                "trigger": jnp.asarray(trigger),
                "target": jnp.asarray(target),
            },
        )["target"]
    )


class TestGatedRemoval:
    def test_target_reads_its_own_value(self):
        schema = _edge().ports_schema()
        assert schema["target"].role is PortRole.EVOLVED
        assert schema["target"].reads_value is True
        assert schema["trigger"].role is PortRole.INPUT

    def test_inert_below_the_gate(self):
        assert _dy(_edge(), 0.0, 10.0) == pytest.approx(0.0, abs=1e-12)

    def test_clears_above_the_gate(self):
        assert _dy(_edge(), 50.0, 10.0) == pytest.approx(-10.0, rel=1e-6)

    def test_removal_is_proportional_so_zero_is_a_fixed_point(self):
        assert _dy(_edge(), 50.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_half_open_at_K(self):
        assert _dy(_edge(), 1.0, 8.0) == pytest.approx(-4.0, rel=1e-6)

    def test_differentiable_through_trigger_and_rate(self):
        g = jax.grad(
            lambda x: _edge().derivative(
                0.0, {"trigger": x, "target": jnp.asarray(5.0)}
            )["target"]
        )(jnp.asarray(2.0))
        assert jnp.isfinite(g)
        gk = jax.grad(
            lambda k: _edge(k_remove=k).derivative(
                0.0, {"trigger": jnp.asarray(5.0), "target": jnp.asarray(5.0)}
            )["target"]
        )(jnp.asarray(1.0))
        assert jnp.isfinite(gk) and gk < 0

    def test_k_remove_is_the_calibratable_surface(self):
        assert {p.field for p in _edge().calibratable_params()} == {"k_remove"}
