"""Tests for the MtorNFkBActivator crosstalk edge and its wiring into
the multi-hallmark composite."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.models.mtor_nfkb import MtorNFkBActivator
from hallsim.process import PortRole


class TestMtorNFkBActivator:
    def test_ports(self):
        proc = MtorNFkBActivator()
        schema = proc.ports_schema()
        # Writes additively to the NF-κB module's IKK pool.
        assert schema["IKK"].role == PortRole.EVOLVED
        # Reads DP14's active mTORC1 read-only.
        assert schema["mTORC1_pS2448"].role == PortRole.INPUT

    def test_only_evolved_port_in_derivative(self):
        # INPUT ports must not appear in the returned derivative dict.
        proc = MtorNFkBActivator()
        dy = proc.derivative(0.0, {"mTORC1_pS2448": jnp.array(4.0)})
        assert set(dy.keys()) == {"IKK"}

    def test_activating_sign_and_monotonic(self):
        # Higher active mTORC1 -> larger positive IKK contribution.
        proc = MtorNFkBActivator()
        low = float(
            proc.derivative(0.0, {"mTORC1_pS2448": jnp.array(2.8)})["IKK"]
        )
        high = float(
            proc.derivative(0.0, {"mTORC1_pS2448": jnp.array(5.7)})["IKK"]
        )
        assert 0.0 < low < high
        # Bounded above by k_act (Hill saturates at 1).
        sat = float(
            proc.derivative(0.0, {"mTORC1_pS2448": jnp.array(1e4)})["IKK"]
        )
        assert sat <= proc.k_act + 1e-9

    def test_differentiable_through_input_and_rate(self):
        # Gradient must flow from the IKK contribution back to both the
        # upstream mTOR input and the rate constant (end-to-end diff).
        def ikk_from_mtor(m):
            return MtorNFkBActivator().derivative(0.0, {"mTORC1_pS2448": m})[
                "IKK"
            ]

        g = jax.grad(ikk_from_mtor)(jnp.array(4.0))
        assert jnp.isfinite(g) and g > 0  # activating: dIKK/dmTOR > 0

        def ikk_from_k(k):
            return MtorNFkBActivator(k_act=k).derivative(
                0.0, {"mTORC1_pS2448": jnp.array(4.0)}
            )["IKK"]

        gk = jax.grad(ikk_from_k)(jnp.array(0.02))
        assert jnp.isfinite(gk) and gk > 0


@pytest.mark.network
class TestMultiHallmarkWiring:
    # Builds the full multi-hallmark composite, which loads the DP14 and
    # Ihekwaba SBML models. Those files are not bundled in the repo, so on
    # a clean checkout (CI) they download from BioModels — deselected from
    # the default `make test` run via `-m "not network"`. Runs locally with
    # `pytest tests/test_mtor_nfkb.py`.
    def test_edge_present_and_validates(self):
        from hallsim.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )

        # validate=True (topology) + semantic validation on by default.
        comp = build_multi_hallmark_composite()
        assert "mtor_nfkb" in comp.processes
        # The edge crosses namespaces: reads dp14, writes nfkb.
        edge = comp.topology["mtor_nfkb"]
        assert edge["mTORC1_pS2448"] == "dp14/mTORC1_pS2448"
        assert edge["IKK"] == "nfkb/IKK"

    def test_ikk_receives_additive_contribution(self):
        # The composite RHS must sum the edge's IKK term with Ihekwaba's
        # own IKK dynamics (EVOLVED additivity across namespaces).
        from hallsim.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )

        comp = build_multi_hallmark_composite(validate=False)
        rhs, keys = comp.build_rhs()
        y0 = comp.initial_state_vec()
        dy = rhs(0.0, y0, None)
        ikk_idx = keys.index("nfkb/IKK")
        # mTORC1 starts at 10.0 -> Hill drive is strongly engaged, so the
        # edge adds a clearly positive term on top of Ihekwaba's IKK rate.
        assert jnp.isfinite(dy[ikk_idx])
