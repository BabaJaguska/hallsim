"""Tests for gene_reporters — single-gene reporter validation.

Covers:
- Reporter table integrity (all expected fields, valid signs, distinct
  observables and genes)
- derive_observables produces the expected keys and types
- compute_concordance correctness on synthetic deltas
- log2_fold_change helper
"""

from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
import pytest

from hallsim.gene_reporters import (
    CANONICAL_REPORTERS,
    MULTI_HALLMARK_REPORTERS,
    GeneExpressionDataset,
    GeneReporter,
    compute_concordance,
    cycle_average,
    derive_multi_hallmark_summaries,
    derive_observable_summaries,
    derive_observables,
    last_value,
    log2_fold_change,
)


# ═══════════════════════════════════════════════════════════════════════════
# Reporter table integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestReporterTable:

    def test_at_least_five_reporters(self):
        assert len(CANONICAL_REPORTERS) >= 5

    def test_all_signs_are_plus_or_minus_one(self):
        for r in CANONICAL_REPORTERS:
            assert r.sign in (
                +1,
                -1,
            ), f"{r.observable}: sign {r.sign} must be ±1"

    def test_all_reporters_have_references(self):
        for r in CANONICAL_REPORTERS:
            assert r.reference, f"{r.observable} missing literature ref"
            assert r.description, f"{r.observable} missing description"

    def test_observables_unique(self):
        obs = [r.observable for r in CANONICAL_REPORTERS]
        assert len(obs) == len(
            set(obs)
        ), "duplicate observable in reporter table"

    def test_genes_unique(self):
        genes = [r.gene_symbol for r in CANONICAL_REPORTERS]
        assert len(genes) == len(
            set(genes)
        ), "duplicate gene symbol in reporter table"


# ═══════════════════════════════════════════════════════════════════════════
# derive_observables — ERiQ state → named observable dict
# ═══════════════════════════════════════════════════════════════════════════


def _stub_eriq_state():
    """Homeostatic IC for ERiQ — all values finite at this state."""
    return {
        "eriq/mito_function": jnp.asarray(3.6239),
        "eriq/glycolysis": jnp.asarray(2.4010),
        "eriq/mito_damage": jnp.asarray(0.0724),
        "eriq/mTOR_activity": jnp.asarray(-0.1936),
        "eriq/p53_activity": jnp.asarray(0.8734),
        "eriq/ROS_activity": jnp.asarray(0.0794),
        "eriq/ROS_integrator_c": jnp.asarray(-0.7944),
    }


class TestDeriveObservables:

    def test_keys_cover_all_reporter_observables(self):
        obs = derive_observables(_stub_eriq_state())
        for r in CANONICAL_REPORTERS:
            assert r.observable in obs, (
                f"derive_observables missing key '{r.observable}' "
                f"required by reporter {r.gene_symbol}"
            )

    def test_values_are_finite(self):
        obs = derive_observables(_stub_eriq_state())
        for name, v in obs.items():
            assert jnp.isfinite(v), f"{name} non-finite at homeostatic IC"


# ═══════════════════════════════════════════════════════════════════════════
# compute_concordance
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeConcordance:

    def test_all_match(self):
        """Every Δ_sim agrees in sign with Δ_data → sign_agreement = 1.0."""
        delta_obs = {
            "p53_activity": +1.0,
            "mito_damage": +1.0,
            "mito_function": -1.0,
            "mTOR_activity_algebraic": +1.0,
            "NFKB_algebraic": +1.0,
            "ROS_algebraic": +1.0,
        }
        delta_data = pd.Series(
            {
                "CDKN1A": +0.5,
                "DDB2": +0.5,
                "CYCS": -0.5,
                "EIF4EBP1": +0.5,
                "NFKBIA": +0.5,
                "HMOX1": +0.5,
            }
        )
        result = compute_concordance(
            delta_observables=delta_obs,
            delta_gene_expression=delta_data,
            condition_name="all_aligned",
        )
        assert result.sign_agreement == 1.0
        assert result.n_compared == 6
        assert result.spearman_r == pytest.approx(1.0)

    def test_all_mismatch(self):
        # Use varying values so Spearman is well-defined (a constant
        # vector makes Spearman undefined → our fallback is 0.0).
        # The (sim, data) pairs in reporter-iteration order should be
        # rank-anticorrelated.
        # Reporter order: p53_activity/CDKN1A, mito_damage/DDB2,
        # ROS_alg/HMOX1, NFKB_alg/NFKBIA, mito_function/CYCS,
        # mTOR_alg/EIF4EBP1.
        delta_obs = {
            "p53_activity": +0.1,
            "mito_damage": +0.5,
            "ROS_algebraic": +1.0,
            "NFKB_algebraic": +1.5,
            "mito_function": +2.0,
            "mTOR_activity_algebraic": +3.0,
        }
        delta_data = pd.Series(
            {
                "CDKN1A": -0.1,
                "DDB2": -0.5,
                "HMOX1": -1.0,
                "NFKBIA": -1.5,
                "CYCS": -2.0,
                "EIF4EBP1": -3.0,
            }
        )
        result = compute_concordance(
            delta_observables=delta_obs,
            delta_gene_expression=delta_data,
            condition_name="all_mismatched",
        )
        assert result.sign_agreement == 0.0
        # Perfectly anticorrelated (largest sim → most-negative data).
        assert result.spearman_r == pytest.approx(-1.0)

    def test_missing_gene_skipped(self):
        """Genes not in the expression series are dropped from comparison."""
        delta_obs = {"p53_activity": +1.0}
        delta_data = pd.Series({"SOMETHING_ELSE": +0.5})
        result = compute_concordance(
            delta_observables=delta_obs,
            delta_gene_expression=delta_data,
            condition_name="empty",
        )
        assert result.n_compared == 0
        assert result.sign_agreement == 0.0  # no rows → defaults

    def test_missing_observable_skipped(self):
        """Observables not in the sim deltas are dropped."""
        delta_obs = {}  # nothing
        delta_data = pd.Series({"CDKN1A": +0.5})
        result = compute_concordance(
            delta_observables=delta_obs,
            delta_gene_expression=delta_data,
            condition_name="empty_sim",
        )
        assert result.n_compared == 0

    def test_inverse_sign_applied(self):
        """A reporter with sign=-1 should treat Δ_sim and -Δ_sim as
        equivalent for matching."""
        inverse_rep = GeneReporter(
            observable="x", gene_symbol="GENE_X", sign=-1
        )
        # Δ_sim positive, Δ_data positive → with sign=-1, Δ_sim_signed is
        # negative ⇒ mismatch
        result = compute_concordance(
            delta_observables={"x": +1.0},
            delta_gene_expression=pd.Series({"GENE_X": +1.0}),
            reporters=[inverse_rep],
        )
        assert result.rows[0].sign_match is False
        # Δ_sim negative, Δ_data positive → with sign=-1, Δ_sim_signed is
        # positive ⇒ match
        result = compute_concordance(
            delta_observables={"x": -1.0},
            delta_gene_expression=pd.Series({"GENE_X": +1.0}),
            reporters=[inverse_rep],
        )
        assert result.rows[0].sign_match is True


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory summaries (last_value / cycle_average) and the
# derive_observable_summaries pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectorySummaries:

    def test_last_value_picks_endpoint(self):
        y = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(last_value(y)) == pytest.approx(5.0)

    def test_cycle_average_quarter(self):
        # Linear ramp 0..99 (100 points). Last 25% are 75..99, mean = 87.
        y = jnp.arange(100, dtype=jnp.float32)
        s = cycle_average(0.25)
        assert float(s(y)) == pytest.approx(87.0)

    def test_cycle_average_full_trajectory(self):
        y = jnp.arange(10, dtype=jnp.float32)
        s = cycle_average(1.0)
        assert float(s(y)) == pytest.approx(4.5)

    def test_cycle_average_phase_insensitive_on_oscillator(self):
        # Two trajectories of pure sinusoid with different phases at
        # endpoint. The cycle_average of the last several cycles should
        # match (both ≈ 0); endpoints differ by ~2.
        t = jnp.linspace(0, 20 * jnp.pi, 2000)
        y1 = jnp.sin(t)
        y2 = jnp.sin(t + 1.3)
        s = cycle_average(0.5)
        # Endpoint readouts differ significantly...
        assert abs(float(y1[-1]) - float(y2[-1])) > 0.5
        # ...but cycle-averaged readouts both ≈ 0.
        assert abs(float(s(y1))) < 0.05
        assert abs(float(s(y2))) < 0.05

    def test_cycle_average_rejects_invalid_fraction(self):
        with pytest.raises(ValueError):
            cycle_average(0.0)
        with pytest.raises(ValueError):
            cycle_average(1.5)

    def test_ddb2_reporter_uses_cycle_average(self):
        ddb2 = next(r for r in CANONICAL_REPORTERS if r.gene_symbol == "DDB2")
        # Sinusoidal oscillator-style trajectory.
        t = jnp.linspace(0, 20 * jnp.pi, 1000)
        # cycle_average should ≈ 5.0 (the mean), not 5 + sin(20π) ≈ 5.0
        # at endpoint by accident — pick a phase that makes endpoint != mean.
        y_offset = 5.0 + jnp.sin(t + 1.7)
        assert abs(float(ddb2.summary(y_offset)) - 5.0) < 0.05
        # last_value would give a phase-dependent value.
        assert abs(float(y_offset[-1]) - 5.0) > 0.1

    def test_derive_observable_summaries_collapses_trajectory(self):
        # Build a trajectory by tiling the homeostatic state along
        # a time axis and adding a sinusoidal wiggle to one channel.
        n_time = 200
        state_traj = {
            k: jnp.broadcast_to(v, (n_time,))
            for k, v in _stub_eriq_state().items()
        }
        out = derive_observable_summaries(state_traj)
        for r in CANONICAL_REPORTERS:
            assert r.observable in out
            assert jnp.isfinite(out[r.observable])


# ═══════════════════════════════════════════════════════════════════════════
# Multi-hallmark reporter table + GeneExpressionDataset
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiHallmarkReporters:

    def test_table_integrity(self):
        # All entries have a store-path observable, a gene symbol, ±1 sign,
        # and a literature anchor.
        for r in MULTI_HALLMARK_REPORTERS:
            assert (
                "/" in r.observable
            ), f"{r.gene_symbol}: observable should be a store path"
            assert r.sign in (+1, -1)
            assert r.gene_symbol
            assert r.reference

    def test_unique_gene_symbols(self):
        genes = [r.gene_symbol for r in MULTI_HALLMARK_REPORTERS]
        assert len(genes) == len(set(genes))

    def test_ddb2_uses_cycle_average(self):
        ddb2 = next(
            r for r in MULTI_HALLMARK_REPORTERS if r.gene_symbol == "DDB2"
        )
        # Sinusoid centered at 5: endpoint phase-dependent, mean ≈ 5.
        t = jnp.linspace(0, 20 * jnp.pi, 1000)
        traj = 5.0 + jnp.sin(t + 1.7)
        assert abs(float(ddb2.summary(traj)) - 5.0) < 0.05

    def test_derive_multi_hallmark_summaries(self):
        n_time = 20
        traj = {
            "dp14/CDKN1A": jnp.linspace(0, 10, n_time),
            "gz06/x": jnp.linspace(0, 1, n_time),
            "dp14/ROS": jnp.linspace(0, 5, n_time),
            "nfkb/IkBa": jnp.linspace(0, 2, n_time),
            "dp14/Mito_mass_new": jnp.linspace(0, 3, n_time),
            "dp14/mTORC1_pS2448": jnp.linspace(0, 4, n_time),
        }
        out = derive_multi_hallmark_summaries(traj)
        for r in MULTI_HALLMARK_REPORTERS:
            assert r.observable in out
            assert jnp.isfinite(out[r.observable])
        # CDKN1A uses last_value → 10.0
        assert float(out["dp14/CDKN1A"]) == pytest.approx(10.0)


class TestGeneExpressionDataset:

    def _toy_df(self):
        return pd.DataFrame(
            {
                "ctrl1": [3.0, 5.0],
                "ctrl2": [3.0, 5.0],
                "ddis1": [4.0, 4.0],
                "ddis2": [4.0, 4.0],
            },
            index=["GENE_A", "GENE_B"],
        )

    def test_delta_uses_named_groups(self):
        ds = GeneExpressionDataset(
            gene_expr=self._toy_df(),
            sample_groups={
                "ctrl": ["ctrl1", "ctrl2"],
                "ddis": ["ddis1", "ddis2"],
            },
        )
        delta = ds.delta("ddis", "ctrl")
        assert delta["GENE_A"] == pytest.approx(1.0)
        assert delta["GENE_B"] == pytest.approx(-1.0)

    def test_delta_unknown_group_raises(self):
        ds = GeneExpressionDataset(
            gene_expr=self._toy_df(),
            sample_groups={"ctrl": ["ctrl1", "ctrl2"]},
        )
        with pytest.raises(KeyError):
            ds.delta("unknown", "ctrl")


# ═══════════════════════════════════════════════════════════════════════════
# log2_fold_change helper
# ═══════════════════════════════════════════════════════════════════════════


class TestLog2FoldChange:

    def test_simple_difference(self):
        """Microarray values are already log2-scaled — the difference of means is
        the log2 fold change."""
        df = pd.DataFrame(
            {
                "s1": [3.0, 5.0],
                "s2": [3.0, 5.0],
                "ctrl1": [2.0, 4.0],
                "ctrl2": [2.0, 4.0],
            },
            index=["GENE_A", "GENE_B"],
        )
        lfc = log2_fold_change(df, ["s1", "s2"], ["ctrl1", "ctrl2"])
        assert lfc["GENE_A"] == pytest.approx(1.0)
        assert lfc["GENE_B"] == pytest.approx(1.0)

    def test_negative_change(self):
        df = pd.DataFrame({"s": [2.0], "ctrl": [4.0]}, index=["GENE_A"])
        lfc = log2_fold_change(df, ["s"], ["ctrl"])
        assert lfc["GENE_A"] == pytest.approx(-2.0)
