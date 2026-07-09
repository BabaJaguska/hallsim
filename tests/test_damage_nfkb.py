"""Tests for the DamageNFkBActivator crosstalk edge and its wiring into
the multi-hallmark composite (the genomic-instability → NF-κB channel)."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.models.damage_nfkb import DamageNFkBActivator
from hallsim.process import PortRole


class TestDamageNFkBActivator:
    def test_ports(self):
        proc = DamageNFkBActivator()
        schema = proc.ports_schema()
        # Writes additively to the NF-κB module's IKK pool.
        assert schema["IKK"].role == PortRole.EVOLVED
        # Reads DP14's DNA-damage state read-only.
        assert schema["DNA_damage"].role == PortRole.INPUT

    def test_only_evolved_port_in_derivative(self):
        proc = DamageNFkBActivator()
        dy = proc.derivative(0.0, {"DNA_damage": jnp.array(500.0)})
        assert set(dy.keys()) == {"IKK"}

    def test_activating_sign_and_saturates(self):
        # K_dmg=19 is the geometric midpoint of the etoposide-regime
        # DNA_damage range (~9.6 control -> ~37 DDIS at the fitted potency
        # ~10). Control sits below the Hill half-saturation, DDIS above it,
        # and the term saturates toward k_act at high damage.
        proc = DamageNFkBActivator()
        ctrl = float(
            proc.derivative(0.0, {"DNA_damage": jnp.array(9.6)})["IKK"]
        )
        ddis = float(
            proc.derivative(0.0, {"DNA_damage": jnp.array(37.0)})["IKK"]
        )
        sat = float(
            proc.derivative(0.0, {"DNA_damage": jnp.array(1e4)})["IKK"]
        )
        # Activating: more damage -> larger contribution.
        assert 0.0 <= ctrl < ddis < sat
        # K straddles the range: control below half-saturation, DDIS above.
        assert ctrl < 0.5 * proc.k_act < ddis
        # Saturates toward k_act at high damage.
        assert sat == pytest.approx(proc.k_act, rel=1e-3)

    def test_differentiable_through_input_and_rate(self):
        def ikk_from_dmg(d):
            return DamageNFkBActivator().derivative(0.0, {"DNA_damage": d})[
                "IKK"
            ]

        g = jax.grad(ikk_from_dmg)(jnp.array(500.0))
        assert jnp.isfinite(g) and g > 0  # activating: dIKK/dDamage > 0

        def ikk_from_k(k):
            return DamageNFkBActivator(k_act=k).derivative(
                0.0, {"DNA_damage": jnp.array(28000.0)}
            )["IKK"]

        gk = jax.grad(ikk_from_k)(jnp.array(0.02))
        assert jnp.isfinite(gk) and gk > 0

    def test_k_act_is_calibratable(self):
        names = {p.field for p in DamageNFkBActivator().calibratable_params()}
        assert "k_act" in names


@pytest.mark.network
class TestDamageEdgeWiring:
    # Builds the full multi-hallmark composite (downloads DP14 + Ihekwaba
    # on a clean checkout); deselected from default `make test` via
    # `-m "not network"`. Runs locally with `pytest tests/test_damage_nfkb.py`.
    def test_edge_present_and_wired(self):
        from hallsim.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )

        comp = build_multi_hallmark_composite()
        assert "damage_nfkb" in comp.processes
        edge = comp.topology["damage_nfkb"]
        assert edge["DNA_damage"] == "dp14/DNA_damage"
        assert edge["IKK"] == "nfkb/IKK"

    @pytest.mark.slow
    def test_damage_edge_flips_nfkbia_direction(self):
        """The genomic-instability channel must make the NFKBIA reporter
        rise with DDIS. Without it NF-κB tracks only mTOR (which falls in
        DDIS), so NFKBIA runs backwards (ctrl > DDIS); with it, DDIS > ctrl."""
        from hallsim import apply_hallmarks
        from hallsim.composite import Composite
        from hallsim.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )
        from hallsim.scheduler import Scheduler

        base = build_multi_hallmark_composite(validate=False)
        sched = Scheduler()
        T = 14.0

        def nfkbia(hallmarks):
            procs = apply_hallmarks(base.processes, hallmarks)
            comp = Composite(
                processes=procs,
                topology=base.topology,
                validate=False,
                semantic_validation={"check_semantics": False},
            )
            r = sched.run(
                comp,
                t_span=(0.0, T),
                macro_dt=T / 20,
                y0=comp.initial_state_vec(),
                save_dt=T / 20,
            )
            return float(r.get("nfkb/IkBat")[-1])

        ctrl = nfkbia(
            {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}
        )
        ddis = nfkbia(
            {"Genomic Instability": 1.0, "Deregulated Nutrient Sensing": 1.0}
        )
        assert ddis > ctrl
