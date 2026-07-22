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
    window_mean,
    window_rms,
    zerophase_mean,
)


class TestZerophaseMean:

    def test_constant_preserved(self):
        ts = jnp.linspace(0.0, 20.0, 400)
        y = jnp.full_like(ts, 3.0)
        sm = zerophase_mean(2.0)(ts, y, query_times=ts)
        assert jnp.allclose(sm, 3.0, atol=1e-4)

    def test_ripple_removed_without_lag(self):
        ts = jnp.linspace(0.0, 20.0, 800)
        dc = 2.0 + 0.05 * ts  # gentle DC trend
        y = dc + 0.5 * jnp.sin(2 * jnp.pi * ts / 0.4)  # fast ripple
        qt = jnp.array([5.0, 10.0, 15.0])
        sm = zerophase_mean(2.0)(ts, y, query_times=qt)
        want = 2.0 + 0.05 * qt
        assert jnp.allclose(
            sm, want, atol=0.05
        )  # ripple gone, DC tracked, no lag

    def test_reflection_beats_constant_pad_at_boundary(self):
        # Reflection padding keeps the first smoothed point near y[0] for a
        # signal rising slowly relative to tau, where a constant warm-start
        # would lift it toward the interior.
        ts = jnp.linspace(0.0, 14.0, 280)
        y = 1.0 - jnp.exp(-ts / 10.0)  # slow rise relative to tau=2
        sm = zerophase_mean(2.0)(ts, y, query_times=ts)
        assert abs(float(sm[0]) - float(y[0])) < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Reporter table integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestReporterTable:

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

    def test_empty_intersection_skipped(self):
        """An empty observable/gene intersection yields no comparisons,
        whether the mismatch is on the data side or the sim side."""
        # Drop on the data side: sim observable has no matching gene.
        result_data = compute_concordance(
            delta_observables={"p53_activity": +1.0},
            delta_gene_expression=pd.Series({"SOMETHING_ELSE": +0.5}),
            condition_name="empty",
        )
        assert result_data.n_compared == 0
        assert result_data.sign_agreement == 0.0  # no rows → defaults

        # Drop on the sim side: gene has no matching sim observable.
        result_sim = compute_concordance(
            delta_observables={},
            delta_gene_expression=pd.Series({"CDKN1A": +0.5}),
            condition_name="empty_sim",
        )
        assert result_sim.n_compared == 0

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
        ts = jnp.arange(5, dtype=jnp.float32)
        y = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(last_value(ts, y)) == pytest.approx(5.0)

    def test_cycle_average_quarter(self):
        # Linear ramp 0..99 (100 points). Last 25% are 75..99, mean = 87.
        y = jnp.arange(100, dtype=jnp.float32)
        ts = jnp.arange(100, dtype=jnp.float32)
        s = cycle_average(0.25)
        assert float(s(ts, y)) == pytest.approx(87.0)

    def test_cycle_average_rejects_invalid_fraction(self):
        with pytest.raises(ValueError):
            cycle_average(0.0)
        with pytest.raises(ValueError):
            cycle_average(1.5)

    def test_window_mean_is_exact_flat_mean(self):
        # source = 5 + sin(t); its exact integral A = 5t - cos(t) + 1
        # (so A(0)=0). window_mean over the last save interval recovers the
        # flat mean of the source, phase-insensitively. With save points on
        # multiples of 4π the boundary cos terms cancel → exactly 5.0.
        ts = jnp.linspace(0.0, 20 * jnp.pi, 6)
        A = 5.0 * ts - jnp.cos(ts) + 1.0  # cumulative ∫(5+sin)
        s = window_mean()
        assert float(s(ts, A)) == pytest.approx(5.0, abs=1e-4)

    def test_window_mean_rejects_nonpositive_window(self):
        # The window is a fixed duration in the trajectory's time unit.
        with pytest.raises(ValueError):
            window_mean(0.0)
        with pytest.raises(ValueError):
            window_mean(-1.0)

    def test_summaries_read_at_query_times(self):
        # Trajectory-native contract: passing query_times returns one value
        # per time (not just the endpoint), grid-independently.
        ts = jnp.linspace(0.0, 14.0, 15)
        # point observable y = 2t: value at t=7 is 14, at t=14 is 28.
        y = 2.0 * ts
        got = last_value(ts, y, jnp.asarray([7.0, 14.0]))
        assert got.shape == (2,)
        assert float(got[0]) == pytest.approx(14.0)
        assert float(got[1]) == pytest.approx(28.0)
        # integrated observable A2 = 25t (∫x² for constant x=5) → RMS 5.0 at
        # every query time.
        A2 = 25.0 * ts
        rms = window_rms()(ts, A2, jnp.asarray([7.0, 14.0]))
        assert rms.shape == (2,)
        assert float(rms[0]) == pytest.approx(5.0, abs=1e-4)
        assert float(rms[1]) == pytest.approx(5.0, abs=1e-4)

    def test_derive_observable_summaries_collapses_trajectory(self):
        # Build a trajectory by tiling the homeostatic state along
        # a time axis.
        n_time = 200
        ts = jnp.linspace(0.0, 1.0, n_time)
        state_traj = {
            k: jnp.broadcast_to(v, (n_time,))
            for k, v in _stub_eriq_state().items()
        }
        out = derive_observable_summaries(ts, state_traj)
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

    def test_derive_multi_hallmark_summaries(self):
        n_time = 20
        ts = jnp.linspace(0.0, 25.0, n_time)
        traj = {
            "dp14/CDKN1A": jnp.linspace(0, 10, n_time),
            "gz06/x_integral": jnp.linspace(0, 1, n_time),
            "gz06/y_integral": jnp.linspace(0, 1, n_time),
            "dp14/FoxO3a": jnp.linspace(0, 5, n_time),
            "nfkb/IkBat_integral": jnp.linspace(0, 2, n_time),
            "dp14/Mito_mass_new": jnp.linspace(0, 3, n_time),
            "dp14/mTORC1_pS2448": jnp.linspace(0, 4, n_time),
        }
        out = derive_multi_hallmark_summaries(ts, traj)
        for r in MULTI_HALLMARK_REPORTERS:
            assert r.observable in out
            assert jnp.isfinite(out[r.observable])
        # CDKN1A low-passes the level (zerophase_mean), so a monotone ramp's
        # smoothed endpoint sits below the raw endpoint (10.0).
        assert float(out["dp14/CDKN1A"]) < 10.0


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
        the log2 fold change, in either direction."""
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

        # Opposite sign: control higher than sample → negative fold change.
        df_neg = pd.DataFrame({"s": [2.0], "ctrl": [4.0]}, index=["GENE_A"])
        lfc_neg = log2_fold_change(df_neg, ["s"], ["ctrl"])
        assert lfc_neg["GENE_A"] == pytest.approx(-2.0)
