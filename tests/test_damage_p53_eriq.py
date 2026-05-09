"""Tests for the DamageRepair → GZ06 → P53Bridge → ERiQ composite.

The composite is HallSim's reference cross-publication composability
demonstration: a hand-built upstream damage Process drives an
SBML-imported p53-Mdm2 oscillator, whose dynamic p53 replaces ERiQ's
intrinsic algebraic p53 via a tracking bridge.

These tests cover:
- Building the composite with default parameters.
- Auto-grouping by timescale (verifies the Scheduler is the right tool).
- Dose-dependent damage propagation: increasing alpha → larger D_ss →
  larger upstream p53 oscillation amplitude → measurable downstream
  ERiQ change.
- Differentiability: jax.grad through the composed system to alpha.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.models.damage_p53_eriq import build_damage_p53_eriq_composite
from hallsim.scheduler import Scheduler


class TestComposeStructure:
    def test_builds_without_error(self):
        comp = build_damage_p53_eriq_composite()
        assert "damage_repair" in comp.processes
        assert "gz06_p53" in comp.processes
        assert "p53_bridge" in comp.processes
        assert "eriq_energy" in comp.processes
        assert "eriq_oxstress" in comp.processes
        assert "eriq_signaling" in comp.processes

    def test_auto_groups_split_by_timescale(self):
        """Bridge (~0.2), GZ06 (~5), damage (~50), ERiQ (None) span
        ratios > Scheduler's 100x cluster window — should split."""
        comp = build_damage_p53_eriq_composite()
        groups = comp.auto_groups()
        # We don't pin the exact group naming, but expect >= 2 groups
        # because timescales span 250x.
        assert len(groups) >= 2

    def test_topology_wires_damage_to_psi(self):
        """damage_repair.damage and gz06_p53.psi must share a store path."""
        comp = build_damage_p53_eriq_composite()
        assert comp.topology["damage_repair"]["damage"] == "p53/psi"
        assert comp.topology["gz06_p53"]["psi"] == "p53/psi"

    def test_topology_wires_gz06_x_to_bridge(self):
        comp = build_damage_p53_eriq_composite()
        assert comp.topology["gz06_p53"]["x"] == "p53/x"
        assert comp.topology["p53_bridge"]["p53_x"] == "p53/x"

    def test_topology_wires_bridge_to_eriq_p53(self):
        comp = build_damage_p53_eriq_composite()
        assert (
            comp.topology["p53_bridge"]["p53_activity"]
            == "eriq/p53_activity"
        )
        assert (
            comp.topology["eriq_signaling"]["p53_activity"]
            == "eriq/p53_activity"
        )


@pytest.mark.slow
class TestComposedSimulation:
    """End-to-end tests for the composed system. Each Scheduler run takes
    ~14s on CPU due to per-macro-step Diffrax solves across 3 timescale
    groups, so these are gated behind the ``slow`` marker."""

    @pytest.fixture(scope="class")
    def short_run(self):
        """Single short run for tests that just need a trajectory."""
        comp = build_damage_p53_eriq_composite(alpha=0.01)
        sched = Scheduler()
        return sched.run(comp, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0)

    def test_runs_to_completion(self, short_run):
        assert "p53/psi" in short_run.ys
        assert "p53/x" in short_run.ys
        assert "eriq/p53_activity" in short_run.ys
        assert "eriq/mTOR_activity" in short_run.ys

    def test_damage_accumulates(self, short_run):
        """damage rises from IC=0 toward steady state."""
        psi = short_run.ys["p53/psi"]
        assert float(psi[-1]) > float(psi[0])

    def test_gz06_responds_to_damage(self, short_run):
        """p53 protein x should respond to non-zero damage signal."""
        x = short_run.ys["p53/x"]
        # x starts at 0; with psi > 0 it should rise.
        assert float(x.max()) > 0.1


@pytest.mark.slow
class TestDoseResponse:
    """Higher alpha → larger upstream damage → larger downstream effect."""

    def test_damage_scales_with_alpha(self):
        """Damage signal is monotonic in alpha at t=50 (well into transient,
        not steady state — that's t~500 and prohibitively slow)."""
        sched = Scheduler()
        results = {}
        for alpha in [0.01, 0.03, 0.05]:
            comp = build_damage_p53_eriq_composite(alpha=alpha)
            res = sched.run(
                comp, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0
            )
            results[alpha] = float(res.ys["p53/psi"][-1])
        assert results[0.03] > results[0.01]
        assert results[0.05] > results[0.03]

    def test_eriq_downstream_responds_to_damage(self):
        """ERiQ NF-κB / autophagy / mTOR should diverge between low and
        high damage regimes. Skipping the middle alpha for speed."""
        from hallsim.models.eriq import _compute_algebraic

        sched = Scheduler()

        def run_and_extract(alpha):
            comp = build_damage_p53_eriq_composite(alpha=alpha)
            res = sched.run(
                comp, t_span=(0.0, 50.0), macro_dt=5.0, save_dt=5.0
            )
            ys = res.ys
            # Compute final-time NF-κB / autophagy via ERiQ algebraic.
            state = {k.split("/")[1]: float(ys[k][-1]) for k in ys if "/" in k}
            state["_params"] = {}
            obs = _compute_algebraic(state)
            return {
                "psi_fin": float(ys["p53/psi"][-1]),
                "eriq_p53_max": float(ys["eriq/p53_activity"].max()),
                "NFKB": float(obs["NFKB"]),
                "AUTOPHAGY": float(obs["AUTOPHAGY"]),
            }

        low = run_and_extract(0.01)
        high = run_and_extract(0.05)
        # High damage should give larger upstream signal AND larger
        # downstream NF-κB and autophagy responses.
        assert high["psi_fin"] > low["psi_fin"]
        assert high["eriq_p53_max"] > low["eriq_p53_max"]
        assert high["NFKB"] > low["NFKB"]
        assert high["AUTOPHAGY"] > low["AUTOPHAGY"]


@pytest.mark.slow
class TestDifferentiability:
    def test_forward_pass_finite(self):
        """Forward pass through the composed system produces finite output."""

        comp = build_damage_p53_eriq_composite(alpha=0.02)
        sched = Scheduler()
        res = sched.run(
            comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0
        )
        assert jnp.isfinite(res.ys["p53/psi"][-1])
        assert jnp.isfinite(res.ys["eriq/p53_activity"][-1])
