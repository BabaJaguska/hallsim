"""Tests for SASPmTORActivator: chronic-stress mTORC1 activation.

Invariants:
- At baseline (low damage, low p53), the contribution is near zero.
- Under DDIS-like stress (high damage + high p53), the contribution is
  positive and substantial.
- When composed with ERiQSignaling on the same store path, both
  processes contribute additively (EVOLVED port semantics).
- The full damage_p53_eriq composite with ``with_sasp_mtor=True`` shows
  higher late-time mTOR than without, in the high-damage regime —
  recovering the senescence mTORC1 paradox.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hallsim import Composite, Scheduler
from hallsim.models.damage_p53_eriq import build_damage_p53_eriq_composite
from hallsim.models.sasp_mtor import SASPmTORActivator, mTORInhibitor
from hallsim.process import PortRole, ProcessKind


class TestSASPmTORActivatorContract:

    def test_kind_is_continuous(self):
        proc = SASPmTORActivator()
        assert proc.kind == ProcessKind.CONTINUOUS

    def test_ports_schema(self):
        proc = SASPmTORActivator()
        schema = proc.ports_schema()
        assert "mTOR_activity" in schema
        assert "mito_damage" in schema
        assert "p53_activity" in schema
        # mTOR_activity is EVOLVED so ERiQSignaling can also write
        assert schema["mTOR_activity"].role == PortRole.EVOLVED
        assert schema["mito_damage"].role == PortRole.INPUT
        assert schema["p53_activity"].role == PortRole.INPUT


class TestSASPmTORDerivative:

    def test_zero_at_baseline(self):
        """Low damage, low p53 → near-zero SASP contribution."""
        proc = SASPmTORActivator(K_damage=1.0, K_p53=0.7, n=4.0)
        d = proc.derivative(
            0.0,
            {
                "mito_damage": jnp.asarray(0.05),
                "p53_activity": jnp.asarray(0.1),
            },
        )
        assert float(d["mTOR_activity"]) < 1e-3

    def test_positive_under_chronic_stress(self):
        """High damage + high p53 → positive SASP contribution."""
        proc = SASPmTORActivator(k_sasp=0.05, K_damage=1.0, K_p53=0.7, n=4.0)
        d = proc.derivative(
            0.0,
            {
                "mito_damage": jnp.asarray(2.0),
                "p53_activity": jnp.asarray(1.5),
            },
        )
        assert float(d["mTOR_activity"]) > 0.04
        # Bounded above by k_sasp (since H_act * H_act <= 1)
        assert float(d["mTOR_activity"]) <= proc.k_sasp + 1e-6

    def test_monotone_in_damage(self):
        """At fixed p53, SASP contribution increases with damage."""
        proc = SASPmTORActivator()
        damages = jnp.linspace(0.05, 5.0, 20)
        contribs = [
            float(
                proc.derivative(
                    0.0,
                    {
                        "mito_damage": d,
                        "p53_activity": jnp.asarray(1.0),
                    },
                )["mTOR_activity"]
            )
            for d in damages
        ]
        # Monotone non-decreasing
        diffs = jnp.diff(jnp.asarray(contribs))
        assert jnp.all(diffs >= -1e-7)

    def test_monotone_in_p53(self):
        proc = SASPmTORActivator()
        p53s = jnp.linspace(0.05, 3.0, 20)
        contribs = [
            float(
                proc.derivative(
                    0.0,
                    {
                        "mito_damage": jnp.asarray(1.5),
                        "p53_activity": p,
                    },
                )["mTOR_activity"]
            )
            for p in p53s
        ]
        diffs = jnp.diff(jnp.asarray(contribs))
        assert jnp.all(diffs >= -1e-7)


class TestAdditiveComposition:
    """When SASPmTORActivator and an ERiQSignaling-like writer both
    target the same store path with EVOLVED, the RHS sums their
    contributions. This test verifies the additive port-role contract."""

    def test_two_evolved_writers_sum(self):
        """Compose SASP + a toy constant-rate writer; check sum."""
        from hallsim.process import Port, Process

        class ConstantMTORDriver(Process):
            rate: float = 0.02

            def ports_schema(self):
                return {
                    "mTOR_activity": Port(role=PortRole.EVOLVED, default=0.0),
                }

            def derivative(self, t, state):
                return {"mTOR_activity": jnp.asarray(self.rate)}

        sasp = SASPmTORActivator(k_sasp=0.05, K_damage=0.5, K_p53=0.5, n=4.0)
        const = ConstantMTORDriver(rate=0.02)
        comp = Composite(
            processes={"sasp": sasp, "const": const},
            topology={
                "sasp": {
                    "mTOR_activity": "cell/mTOR",
                    "mito_damage": "cell/damage",
                    "p53_activity": "cell/p53",
                },
                "const": {"mTOR_activity": "cell/mTOR"},
            },
        )
        rhs, keys = comp.build_rhs()
        state = comp.flatten(
            {
                "cell/mTOR": jnp.asarray(0.0),
                "cell/damage": jnp.asarray(2.0),  # high
                "cell/p53": jnp.asarray(2.0),  # high
            },
            keys,
        )
        dy = comp.unflatten(rhs(0.0, state), keys)
        # SASP at saturated stress ≈ k_sasp = 0.05; const = 0.02; sum ≈ 0.07
        assert 0.05 < float(dy["cell/mTOR"]) < 0.075


class TestmTORInhibitor:
    """Rapamycin-like Process: additive negative driver on mTOR_activity."""

    def test_zero_strength_is_zero_derivative(self):
        proc = mTORInhibitor(strength=0.0)
        d = proc.derivative(0.0, {"mTOR_activity": jnp.asarray(0.5)})
        assert float(d["mTOR_activity"]) == 0.0

    def test_positive_strength_gives_negative_driver(self):
        proc = mTORInhibitor(strength=0.05)
        d = proc.derivative(0.0, {"mTOR_activity": jnp.asarray(0.5)})
        assert float(d["mTOR_activity"]) == pytest.approx(-0.05)

    def test_independent_of_state_value(self):
        """Constant negative driver — same magnitude regardless of mTOR."""
        proc = mTORInhibitor(strength=0.1)
        for v in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            d = proc.derivative(0.0, {"mTOR_activity": jnp.asarray(v)})
            assert float(d["mTOR_activity"]) == pytest.approx(-0.1)

    def test_composes_with_sasp_additively(self):
        """SASP pushes mTOR up, rapamycin pulls down — sum is the net."""
        sasp = SASPmTORActivator(k_sasp=0.05, K_damage=0.5, K_p53=0.5, n=4.0)
        rapa = mTORInhibitor(strength=0.05)
        comp = Composite(
            processes={"sasp": sasp, "rapa": rapa},
            topology={
                "sasp": {
                    "mTOR_activity": "cell/mTOR",
                    "mito_damage": "cell/damage",
                    "p53_activity": "cell/p53",
                },
                "rapa": {"mTOR_activity": "cell/mTOR"},
            },
        )
        rhs, keys = comp.build_rhs()
        state = comp.flatten(
            {
                "cell/mTOR": jnp.asarray(0.0),
                "cell/damage": jnp.asarray(2.0),  # high
                "cell/p53": jnp.asarray(2.0),  # high
            },
            keys,
        )
        dy = comp.unflatten(rhs(0.0, state), keys)
        # SASP +0.05 (saturated) and rapa -0.05 → net ≈ 0
        assert abs(float(dy["cell/mTOR"])) < 1e-3


@pytest.mark.slow
class TestCompositeWithRapamycin:
    """End-to-end: DDIS+rapamycin should suppress late-time mTOR vs DDIS."""

    def test_rapamycin_suppresses_mtor_in_ddis(self):
        sched = Scheduler(max_steps=2_000_000)
        ddis = build_damage_p53_eriq_composite(
            alpha=0.05, with_sasp_mtor=True, rapamycin_strength=0.0
        )
        ddis_rapa = build_damage_p53_eriq_composite(
            alpha=0.05, with_sasp_mtor=True, rapamycin_strength=0.05
        )
        res_ddis = sched.run(
            ddis, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0
        )
        res_rapa = sched.run(
            ddis_rapa, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0
        )
        mtor_ddis = float(res_ddis.get("eriq/mTOR_activity")[-1])
        mtor_rapa = float(res_rapa.get("eriq/mTOR_activity")[-1])
        # Rapamycin should pull mTOR below the DDIS-only level
        assert mtor_rapa < mtor_ddis


@pytest.mark.slow
class TestCompositeWithSASP:
    """Full integration: with_sasp_mtor=True should raise late-time mTOR
    in the high-damage regime, recovering the DDIS mTORC1 paradox."""

    def test_high_damage_mtor_higher_with_sasp(self):
        sched = Scheduler(max_steps=2_000_000)
        # Without SASP — canonical ERiQ
        comp_off = build_damage_p53_eriq_composite(
            alpha=0.05, with_sasp_mtor=False
        )
        # With SASP — the corrected variant
        comp_on = build_damage_p53_eriq_composite(
            alpha=0.05, with_sasp_mtor=True
        )

        res_off = sched.run(
            comp_off, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0
        )
        res_on = sched.run(
            comp_on, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0
        )

        mtor_off = float(res_off.get("eriq/mTOR_activity")[-1])
        mtor_on = float(res_on.get("eriq/mTOR_activity")[-1])
        # SASP module should push mTOR higher under chronic stress
        assert mtor_on > mtor_off
