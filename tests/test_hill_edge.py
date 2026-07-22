"""Tests for HillActivationEdge and its wiring into the multi-hallmark
composite (the mTORC1→IKK, DNA-damage→IKK, and p53→CDKN1A edges)."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.models.hill_edge import HillActivationEdge
from hallsim.process import PortRole


def _edge(**kw):
    kw.setdefault("k_act", 0.02)
    kw.setdefault("K", (4.0,))
    kw.setdefault("n", (2.0,))
    return HillActivationEdge(**kw)


class TestHillActivationEdge:
    def test_ports_are_generic(self):
        schema = _edge().ports_schema()
        assert schema["target"].role == PortRole.EVOLVED
        assert schema["target"].reads_value is False  # pure source
        assert schema["source"].role == PortRole.INPUT

    def test_only_target_in_derivative(self):
        dy = _edge().derivative(0.0, {"source": jnp.array(4.0)})
        assert set(dy.keys()) == {"target"}

    def test_activating_monotonic_and_saturating(self):
        e = _edge()
        low = float(e.derivative(0.0, {"source": jnp.array(2.8)})["target"])
        high = float(e.derivative(0.0, {"source": jnp.array(5.7)})["target"])
        assert 0.0 < low < high
        sat = float(e.derivative(0.0, {"source": jnp.array(1e4)})["target"])
        assert sat <= e.k_act + 1e-9  # Hill saturates at 1

    def test_half_saturation_at_K(self):
        # source == K -> drive == 0.5 -> target == k_act/2.
        d = _edge().derivative(0.0, {"source": jnp.array(4.0)})["target"]
        assert float(d) == pytest.approx(0.01, abs=1e-6)

    def test_differentiable_through_source_and_rate(self):
        g = jax.grad(
            lambda m: _edge().derivative(0.0, {"source": m})["target"]
        )(jnp.array(4.0))
        assert jnp.isfinite(g) and g > 0
        gk = jax.grad(
            lambda k: _edge(k_act=k).derivative(
                0.0, {"source": jnp.array(4.0)}
            )["target"]
        )(jnp.array(0.02))
        assert jnp.isfinite(gk) and gk > 0

    def test_k_act_is_the_calibratable_surface(self):
        assert {p.field for p in _edge().calibratable_params()} == {"k_act"}

    def test_declarative_metadata_folds_in(self):
        m = _edge(hallmark="H", reference="R", description="D").metadata()
        assert (m["hallmark"], m["reference"], m["description"]) == (
            "H",
            "R",
            "D",
        )

    def test_multi_source_gates_multiply(self):
        e = _edge(k_act=1.0, K=(1.0, 1.0), n=(2.0, 2.0), sources=("a", "b"))
        assert set(e.ports_schema()) == {"target", "a", "b"}
        d = e.derivative(0.0, {"a": jnp.array(1.0), "b": jnp.array(1.0)})
        assert float(d["target"]) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.network
class TestMultiHallmarkWiring:
    # Builds the full composite (downloads DP14/GZ06/Ihekwaba SBML on a
    # clean checkout); deselected from `make test` via `-m "not network"`.
    def test_edges_present_and_wired_generically(self):
        from hallsim.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )

        comp = build_multi_hallmark_composite()
        for name in ("mtor_nfkb", "damage_nfkb", "p53_cdkn1a"):
            assert isinstance(comp.processes[name], HillActivationEdge), name
        assert comp.topology["mtor_nfkb"] == {
            "source": "dp14/mTORC1_pS2448",
            "target": "nfkb/IKK",
        }
        assert comp.topology["damage_nfkb"]["target"] == "nfkb/IKK"
        assert comp.topology["p53_cdkn1a"]["target"] == "dp14/CDKN1A"

    def test_ikk_receives_additive_contribution(self):
        from hallsim.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )

        comp = build_multi_hallmark_composite(validate=False)
        rhs, keys = comp.build_rhs()
        dy = rhs(0.0, comp.initial_state_vec(), None)
        assert jnp.isfinite(dy[keys.index("nfkb/IKK")])
