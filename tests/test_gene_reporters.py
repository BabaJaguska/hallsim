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
    GeneReporter,
    compute_concordance,
    derive_observables,
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
            assert r.sign in (+1, -1), (
                f"{r.observable}: sign {r.sign} must be ±1"
            )

    def test_all_reporters_have_references(self):
        for r in CANONICAL_REPORTERS:
            assert r.reference, f"{r.observable} missing literature ref"
            assert r.description, f"{r.observable} missing description"

    def test_observables_unique(self):
        obs = [r.observable for r in CANONICAL_REPORTERS]
        assert len(obs) == len(set(obs)), "duplicate observable in reporter table"

    def test_genes_unique(self):
        genes = [r.gene_symbol for r in CANONICAL_REPORTERS]
        assert len(genes) == len(set(genes)), (
            "duplicate gene symbol in reporter table"
        )


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
        df = pd.DataFrame(
            {"s": [2.0], "ctrl": [4.0]}, index=["GENE_A"]
        )
        lfc = log2_fold_change(df, ["s"], ["ctrl"])
        assert lfc["GENE_A"] == pytest.approx(-2.0)
