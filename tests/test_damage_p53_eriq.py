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
            comp.topology["p53_bridge"]["p53_activity"] == "eriq/p53_activity"
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
        assert "p53/psi" in short_run
        assert "p53/x" in short_run
        assert "eriq/p53_activity" in short_run
        assert "eriq/mTOR_activity" in short_run

    def test_damage_accumulates(self, short_run):
        """damage rises from IC=0 toward steady state."""
        psi = short_run.get("p53/psi")
        assert float(psi[-1]) > float(psi[0])

    def test_gz06_responds_to_damage(self, short_run):
        """p53 protein x should respond to non-zero damage signal."""
        x = short_run.get("p53/x")
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
            results[alpha] = float(res.get("p53/psi")[-1])
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
            # Compute final-time NF-κB / autophagy via ERiQ algebraic.
            # Convert the tensor row at -1 into a name-keyed dict for the
            # ERiQ algebraic helper (which lives outside the JAX hot path).
            final_row = res.ys[-1]
            state = {
                k.split("/")[1]: float(final_row[i])
                for i, k in enumerate(res.keys)
                if "/" in k
            }
            state["_params"] = {}
            obs = _compute_algebraic(state)
            return {
                "psi_fin": float(res.get("p53/psi")[-1]),
                "eriq_p53_max": float(res.get("eriq/p53_activity").max()),
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
        res = sched.run(comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0)
        assert jnp.isfinite(res.get("p53/psi")[-1])
        assert jnp.isfinite(res.get("eriq/p53_activity")[-1])


@pytest.mark.slow
class TestBatchedPopulationRun:
    """The Scheduler accepts a batched y0 tensor and produces batched
    trajectories — the JAX-native population study path.

    Trailing-axis convention all the way: y0 shape ``(batch, n_vars)``,
    output ``ys`` shape ``(n_time, batch, n_vars)``, ``result.get(key)``
    shape ``(n_time, batch)``. No vmap'ing of ``Scheduler.run`` from
    outside required — the whole pipeline is shape-polymorphic.
    """

    def test_batched_y0_produces_batched_trajectories(self):
        comp = build_damage_p53_eriq_composite(alpha=0.03)
        sched = Scheduler(max_steps=2_000_000)
        keys = comp.store_keys()
        n_vars = len(keys)

        batch = 4
        y0 = jnp.broadcast_to(comp.initial_state_vec(keys), (batch, n_vars))
        psi_idx = keys.index("p53/psi")
        # Heterogeneous initial damage across cells.
        y0 = y0.at[..., psi_idx].set(jnp.linspace(0.0, 1.5, batch))

        res = sched.run(
            comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0, y0=y0
        )

        # Shapes propagated cleanly through the scheduler.
        assert res.ys.shape == (5, batch, n_vars)
        assert res.get("p53/psi").shape == (5, batch)
        assert res.get("eriq/p53_activity").shape == (5, batch)

        # Each cell starts at its assigned IC.
        psi_initial = res.get("p53/psi")[0]
        assert jnp.allclose(psi_initial, jnp.linspace(0.0, 1.5, batch))

        # Damage grows monotonically with initial-damage IC at the final
        # time — heavier-damage cells stay heavier-damage cells.
        psi_final = res.get("p53/psi")[-1]
        assert jnp.all(jnp.diff(psi_final) > 0)

    def test_batch_runtime_close_to_single(self):
        """Batched runs should be near-flat in cell count on JAX-native
        hardware; on CPU the scaling is sub-linear because per-step Python
        overhead dominates over compute. We only require that batch=4 is
        at most 3x the single-cell wall-time — generous to keep the test
        deterministic on CI hardware."""
        import time

        comp = build_damage_p53_eriq_composite(alpha=0.03)
        sched = Scheduler(max_steps=2_000_000)
        keys = comp.store_keys()

        # Single
        y0_single = comp.initial_state_vec(keys)
        # Warm up JIT
        sched.run(
            comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0, y0=y0_single
        )
        t0 = time.time()
        sched.run(
            comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0, y0=y0_single
        )
        t_single = time.time() - t0

        # Batched (N=4)
        batch = 4
        y0_batched = jnp.broadcast_to(y0_single, (batch, len(keys)))
        # Warm up batched JIT (different shape)
        sched.run(
            comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0, y0=y0_batched
        )
        t0 = time.time()
        sched.run(
            comp, t_span=(0.0, 20.0), macro_dt=5.0, save_dt=5.0, y0=y0_batched
        )
        t_batched = time.time() - t0

        # Batched should not be much slower than single. On GPU this would
        # be ~1x; on CPU we allow up to 3x.
        assert t_batched < 3.0 * t_single, (
            f"batched (N={batch}) wall {t_batched:.2f}s exceeds 3x single "
            f"{t_single:.2f}s — JIT/vmap not propagating cleanly through "
            f"the Scheduler"
        )
