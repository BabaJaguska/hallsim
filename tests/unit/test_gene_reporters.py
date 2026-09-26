"""Tests for gene_reporters — single-gene reporter validation.

Covers:
- Reporter table integrity (all expected fields, valid signs, distinct
  observables and genes)
- derive_observables produces the expected keys and types
- compute_concordance correctness on synthetic deltas
- log2_fold_change helper
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

from hallsim.gene_reporters import (
    CANONICAL_REPORTERS,
    MULTI_HALLMARK_REPORTERS,
    PROTEOSTASIS_REPORTERS,
    GeneExpressionDataset,
    Readout,
    compute_concordance,
    cycle_average,
    last_value,
    log2_fold_change,
    summarize_reporters,
    window_mean,
    window_rms,
    zerophase_mean,
)
from demos.models.eriq import derive_observables


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

    def test_boundary_stays_within_signal_range(self):
        # Even-reflection padding keeps the smoothed endpoints bounded within
        # the data range (no extrapolation past it), unlike point-reflection
        # which continues the local trend and can overshoot at a turning point.
        ts = jnp.linspace(0.0, 14.0, 280)
        y = 1.0 - jnp.exp(-ts / 10.0)  # monotone rise
        sm = zerophase_mean(2.0)(ts, y, query_times=ts)
        assert float(y.min()) - 1e-6 <= float(sm[0]) <= float(y.max()) + 1e-6
        assert float(y.min()) - 1e-6 <= float(sm[-1]) <= float(y.max()) + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Reporter table integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestReporterTable:

    def test_all_signs_are_plus_or_minus_one(self):
        for r in CANONICAL_REPORTERS:
            assert r.sign in (
                +1,
                -1,
            ), f"{r.path}: sign {r.sign} must be ±1"

    def test_all_reporters_have_references(self):
        for r in CANONICAL_REPORTERS:
            assert r.reference, f"{r.path} missing literature ref"
            assert r.description, f"{r.path} missing description"

    def test_observables_unique(self):
        obs = [r.path for r in CANONICAL_REPORTERS]
        assert len(obs) == len(
            set(obs)
        ), "duplicate observable in reporter table"

    def test_genes_unique(self):
        genes = [r.key for r in CANONICAL_REPORTERS]
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
            assert r.path in obs, (
                f"derive_observables missing key '{r.path}' "
                f"required by reporter {r.key}"
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
        inverse_rep = Readout(path="x", key="GENE_X", sign=-1)
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
        out = summarize_reporters(
            ts, state_traj, CANONICAL_REPORTERS, derive=derive_observables
        )
        for r in CANONICAL_REPORTERS:
            assert r.path in out
            assert jnp.isfinite(out[r.path])


# ═══════════════════════════════════════════════════════════════════════════
# Multi-hallmark reporter table + GeneExpressionDataset
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.demo
class TestMultiHallmarkReporters:

    def test_table_integrity(self):
        # All entries have a store-path observable, a gene symbol, ±1 sign,
        # and a literature anchor.
        for r in MULTI_HALLMARK_REPORTERS:
            assert "/" in r.path, f"{r.key}: observable should be a store path"
            assert r.sign in (+1, -1)
            assert r.key
            assert r.reference

    def test_unique_gene_symbols(self):
        genes = [r.key for r in MULTI_HALLMARK_REPORTERS]
        assert len(genes) == len(set(genes))

    def test_derive_multi_hallmark_summaries(self):
        # A monotone ramp on each reporter's own observable path — derive must
        # produce one finite summary per reporter.
        n_time = 20
        ts = jnp.linspace(0.0, 25.0, n_time)
        traj = {
            r.path: jnp.linspace(0, 10, n_time)
            for r in MULTI_HALLMARK_REPORTERS
        }
        out = summarize_reporters(ts, traj, MULTI_HALLMARK_REPORTERS)
        for r in MULTI_HALLMARK_REPORTERS:
            assert r.path in out
            assert jnp.isfinite(out[r.path])
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


class TestPublishedReporterTable:
    """The reporter set is published in three places outside the code. All
    three are checked against ``MULTI_HALLMARK_REPORTERS`` rather than
    trusted: each had drifted from it, and half the documented set named
    store paths and genes the composite does not have."""

    _TABLE_ROW = re.compile(
        r"^\|\s*`([A-Z0-9]+)`[^|]*\|\s*`([A-Za-z0-9_/]+)`\s*\|", re.MULTILINE
    )
    _ARROW = re.compile(r"([A-Z0-9]+)\s*→\s*``([A-Za-z0-9_/]+)``")

    def _live(self):
        return {
            (r.key, r.path)
            for r in MULTI_HALLMARK_REPORTERS + PROTEOSTASIS_REPORTERS
        }

    def _marked_block(self, relative_path: str) -> str:
        """The region a document marks as this table, so prose elsewhere in
        the file is free to mention any gene it likes."""
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        start = text.index("<!-- reporters:start")
        end = text.index("<!-- reporters:end", start)
        return text[start:end]

    def test_calibration_doc_matches_code(self):
        block = self._marked_block("docs/calibration.md")
        assert set(self._TABLE_ROW.findall(block)) == self._live()

    def test_model_docstring_matches_code(self):
        from demos.models import multi_hallmark

        assert set(self._ARROW.findall(multi_hallmark.__doc__)) == self._live()


class TestFetchGeoSeries:
    """The GEO fetch lands the two files ``load_gene_expression`` reads and
    records their checksums beside them."""

    SOFT = (
        "^SERIES = GSE1\n^PLATFORM = GPL1\n!platform_table_begin\n"
        "ID\tgene_assignment\nP1\tNM_1 // GENE1 // a gene\n"
        "!platform_table_end\n^SAMPLE = GSM1\n"
    )
    MATRIX = (
        '!Series_platform_id\t"GPL1"\n!series_matrix_table_begin\n'
        '"ID_REF"\t"GSM1"\nP1\t3.0\n!series_matrix_table_end\n'
    )

    def _fetch(self, monkeypatch, tmp_path):
        import gzip
        import io

        from hallsim import gene_reporters as gr

        bodies = dict(
            zip(
                gr.geo_series_urls("GSE1"),
                (
                    gzip.compress(self.MATRIX.encode()),
                    gzip.compress(self.SOFT.encode()),
                ),
            )
        )
        monkeypatch.setattr(
            gr.urllib.request,
            "urlopen",
            lambda url, timeout: io.BytesIO(bodies[url]),
        )
        matrix, platform = tmp_path / "m.txt", tmp_path / "p.txt"
        gr.fetch_geo_series("GSE1", matrix, platform)
        return gr, matrix, platform

    def test_platform_table_and_matrix_land_and_load(
        self, monkeypatch, tmp_path
    ):
        gr, matrix, platform = self._fetch(monkeypatch, tmp_path)
        assert matrix.read_text() == self.MATRIX
        assert platform.read_text() == (
            "ID\tgene_assignment\nP1\tNM_1 // GENE1 // a gene\n"
        )
        expr = gr.load_gene_expression(matrix, platform)
        assert expr.loc["GENE1", "GSM1"] == 3.0

    def test_checksums_are_recorded_and_checked_on_load(
        self, monkeypatch, tmp_path
    ):
        gr, matrix, platform = self._fetch(monkeypatch, tmp_path)
        sums = (tmp_path / "SHA256SUMS").read_text()
        assert "  m.txt" in sums and "  p.txt" in sums
        gr.load_gene_expression(matrix, platform)
        matrix.write_text(self.MATRIX.replace("3.0", "4.0"))
        with pytest.raises(ValueError, match="SHA-256"):
            gr.load_gene_expression(matrix, platform)

    def test_urls_follow_geo_layout(self):
        from hallsim.gene_reporters import geo_series_urls

        matrix, soft = geo_series_urls("GSE248823")
        assert matrix.endswith(
            "/GSE248nnn/GSE248823/matrix/GSE248823_series_matrix.txt.gz"
        )
        assert soft.endswith(
            "/GSE248nnn/GSE248823/soft/GSE248823_family.soft.gz"
        )


class TestProbeGeneMap:
    """The platform annotation is found by content, whatever the column is
    called, and bare accessions are resolved."""

    IDS = [f"p{i}" for i in range(8)]
    SYMBOLS = [
        "TP53",
        "BRCA1",
        "MDM2",
        "CDKN1A",
        "GLB1",
        "DDB2",
        "BNIP3",
        "HSPA1A",
    ]
    REFSEQ = [f"NM_{i:06d}" for i in range(8)]

    def test_symbol_column_wins_over_flags_sequences_and_accessions(self):
        from hallsim.gene_reporters import choose_annotation, probe_gene_map

        frame = pd.DataFrame(
            {
                "ID": ["A_23_P%d" % i for i in range(8)],
                "CONTROL_TYPE": ["FALSE"] * 8,
                "REFSEQ": self.REFSEQ,
                "GENE_SYMBOL": self.SYMBOLS,
                "SEQUENCE": ["ACGT" * 12] * 8,
                "ENSEMBL_ID": [f"ENST{i:011d}" for i in range(8)],
            }
        )
        assert choose_annotation(frame) == ("GENE_SYMBOL", "symbol")
        assert probe_gene_map(frame) == dict(zip(frame["ID"], self.SYMBOLS))

    def test_affymetrix_assignment_and_listed_symbols(self):
        from hallsim.gene_reporters import probe_gene_map

        assignment = [
            f"{r} // {s} // a gene // 1p36 // {i} /// XR_1 // {s}-AS1 // x"
            for i, (r, s) in enumerate(zip(self.REFSEQ, self.SYMBOLS))
        ]
        uniprot = [
            "P04637",
            "P38398",
            "Q00987",
            "P38936",
            "P16278",
            "Q92466",
            "Q12983",
            "P0DMV8",
        ]
        frame = pd.DataFrame(
            {
                "ID": self.IDS,
                "gene_assignment": assignment,
                # UniProt accessions read as symbols by shape; they must not win.
                "swissprot": [
                    f"{r} // {u}" for r, u in zip(self.REFSEQ, uniprot)
                ],
            }
        )
        assert probe_gene_map(frame) == dict(zip(self.IDS, self.SYMBOLS))
        listed = pd.DataFrame(
            {
                "ID": self.IDS,
                "Gene Symbol": [f"{s} /// MIR1" for s in self.SYMBOLS],
            }
        )
        assert probe_gene_map(listed) == dict(zip(self.IDS, self.SYMBOLS))

    def test_accessions_are_resolved_when_no_field_is_a_symbol(self):
        from hallsim.gene_reporters import choose_annotation, probe_gene_map

        clariom = pd.DataFrame(
            {
                "ID": self.IDS,
                "SPOT_ID": ["Coding"] * 8,
                "SPOT_ID.1": [
                    f"{r} // RefSeq // Homo sapiens some gene ({s}), mRNA. "
                    f"// chr1 // 100 /// ENST{i:011d} // ENSEMBL // a gene"
                    for i, (r, s) in enumerate(zip(self.REFSEQ, self.SYMBOLS))
                ],
            }
        )
        clariom.loc[len(clariom)] = ["p_unannotated", "Coding", "---"]
        assert choose_annotation(clariom) == ("SPOT_ID.1", "accession")
        asked = {}

        def resolve(accessions, *, taxid):
            asked["taxid"] = taxid
            assert accessions == self.REFSEQ
            return dict(zip(self.REFSEQ, self.SYMBOLS))

        assert probe_gene_map(clariom, taxid=9606, resolve=resolve) == dict(
            zip(self.IDS, self.SYMBOLS)
        )
        assert asked["taxid"] == 9606
        hugene = pd.DataFrame(
            {
                "ID": [str(16657436 + i) for i in range(8)],
                "GB_ACC": self.REFSEQ,
            }
        )
        assert choose_annotation(hugene) == ("GB_ACC", "accession")

    def test_no_gene_annotation_raises(self):
        from hallsim.gene_reporters import choose_annotation

        frame = pd.DataFrame(
            {"ID": self.IDS, "start": [str(i) for i in range(8)]}
        )
        with pytest.raises(ValueError, match="names genes"):
            choose_annotation(frame)

    def test_mygene_batches_are_cached(self, monkeypatch, tmp_path):
        import io
        import json

        from hallsim import gene_reporters as gr

        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.data)
            body = [
                {"query": "NM_000001", "symbol": "TP53"},
                {"query": "NM_000002", "notfound": True},
            ]
            return io.BytesIO(json.dumps(body).encode())

        monkeypatch.setattr(gr.urllib.request, "urlopen", fake_urlopen)
        first = gr.symbols_for_accessions(
            ["NM_000002", "NM_000001"], taxid=9606, cache_dir=tmp_path
        )
        again = gr.symbols_for_accessions(
            ["NM_000001", "NM_000002"], taxid=9606, cache_dir=tmp_path
        )
        assert first == again == {"NM_000001": "TP53"}
        assert len(calls) == 1 and b"species=9606" in calls[0]


# ── RNA-seq counts: the other half of GEO's expression deposits ────


def test_identifier_kind_reads_the_index_not_a_filename():
    from hallsim import gene_reporters as gr

    assert gr.identifier_kind(["TP53", "MDM2", "CDKN1A"]) == "symbol"
    assert gr.identifier_kind(["7157", "4193", "1026"]) == "entrez"
    assert gr.identifier_kind(["ENSG00000141510.12", "ENSG00000135679"]) == (
        "ensembl"
    )
    assert gr.identifier_kind(["NM_000546", "NM_002392"]) == "accession"
    assert gr.identifier_kind([]) == "accession"


def test_counts_become_log_cpm_so_depth_cancels():
    import numpy as np
    import pandas as pd

    from hallsim.gene_reporters import counts_to_log_cpm

    # The same composition sequenced twice as deep must give the same CPM.
    shallow = pd.DataFrame({"a": [10.0, 30.0, 60.0]})
    deep = pd.DataFrame({"a": [100.0, 300.0, 600.0]})
    assert np.allclose(
        counts_to_log_cpm(shallow)["a"], counts_to_log_cpm(deep)["a"]
    )
    # A gene seen in nobody is 0, not minus infinity.
    assert counts_to_log_cpm(pd.DataFrame({"a": [0.0, 1.0]}))["a"][0] == 0.0


def test_a_counts_table_becomes_a_gene_by_sample_frame():
    import pandas as pd

    from hallsim.gene_reporters import GeneExpressionDataset, read_counts_table

    frame = pd.DataFrame(
        {
            "ctrl_1": [100, 50, 0, 10],
            "ctrl_2": [110, 45, 2, 12],
            "drug_1": [200, 25, 0, 11],
            # A non-numeric annotation column rides along in real tables.
            "gene_name": ["a", "b", "c", "d"],
        },
        index=["TP53", "MDM2", "IL6", "TP53"],
    )
    out = read_counts_table(frame)
    assert list(out.columns) == ["ctrl_1", "ctrl_2", "drug_1"]
    # The duplicated symbol is summed before normalising, not averaged.
    assert list(out.index) == ["IL6", "MDM2", "TP53"]
    ds = GeneExpressionDataset.from_counts(
        frame, sample_position_groups={"ctrl": [0, 1], "drug": [2]}
    )
    assert ds.sample_groups == {
        "ctrl": ["ctrl_1", "ctrl_2"],
        "drug": ["drug_1"],
    }
    assert "TP53" in ds.gene_expr.index


def test_a_counts_table_with_no_samples_is_refused():
    import pandas as pd
    import pytest

    from hallsim.gene_reporters import read_counts_table

    with pytest.raises(ValueError, match="no numeric sample columns"):
        read_counts_table(pd.DataFrame({"name": ["a"]}, index=["TP53"]))


class TestSeriesMatrixScale:
    """A series matrix carries the submitter's values as deposited: RMA is
    log2, MAS5 and GCOS are linear signal, and the table does not say
    which. The loader puts both on log2, and the dataset refuses a linear
    table handed to it directly."""

    def test_linear_signal_is_logged_and_log2_is_left_alone(self, caplog):
        import logging

        import numpy as np
        import pandas as pd

        from hallsim import gene_reporters as gr

        linear = pd.DataFrame(
            {"GSM1": [3000.0, 0.2], "GSM2": [1500.0, 12.0]},
            index=["P1", "P2"],
        )
        with caplog.at_level(logging.WARNING, logger=gr.log.name):
            out = gr.as_log2(linear, source="GSE1", note="MAS 5.0")
        assert "linear" in caplog.text and "MAS 5.0" in caplog.text
        assert out.loc["P1", "GSM1"] == pytest.approx(np.log2(3000.0))
        assert out.loc["P2", "GSM1"] == 0.0  # floored at 1 before the log
        logged = pd.DataFrame({"GSM1": [11.5, 2.0]}, index=["P1", "P2"])
        assert gr.as_log2(logged).equals(logged.astype(float))

    def test_a_linear_table_is_refused_as_log_values(self):
        import pandas as pd

        from hallsim import gene_reporters as gr

        ds = gr.GeneExpressionDataset(
            gene_expr=pd.DataFrame(
                {"a": [3000.0], "b": [1500.0]}, index=["GENE1"]
            ),
            sample_groups={"ctrl": ["a"], "trt": ["b"]},
        )
        with pytest.raises(ValueError, match="linear signal"):
            ds.delta("trt", "ctrl")

    def test_the_stated_method_is_checked_against_the_range(self, caplog):
        import logging

        import pandas as pd

        from hallsim import gene_reporters as gr

        assert gr.stated_scale("RMA, log2 transformed") == "log2"
        assert gr.stated_scale("MAS 5.0 signal, GCOS") == "linear"
        assert gr.stated_scale("normalized") == ""
        # Values within the log2 range but a header claiming linear signal:
        # the range decides, the disagreement is logged.
        small = pd.DataFrame({"GSM1": [11.5, 2.0]}, index=["P1", "P2"])
        with caplog.at_level(logging.WARNING, logger=gr.log.name):
            gr.as_log2(small, source="GSE9", note="MAS 5.0")
        assert "GSE9" in caplog.text and "disagree" in caplog.text


class TestConcordanceScope:
    """A contrast the model is silent on is outside its scope, not a row of
    mismatches; an undefined rank is NaN, not 0; and a time course is
    compared by rank over time, where a readout that trails its driver
    shows as a negative rank without a lag and a positive one with it."""

    def test_no_predicted_change_is_out_of_scope(self):
        import math

        import pandas as pd

        from hallsim.gene_reporters import Readout, compute_concordance

        reps = [Readout(path="erk", key=g, sign=+1) for g in ("FOS", "EGR1")]
        out = compute_concordance(
            delta_observables={"erk": 0.0},
            delta_gene_expression=pd.Series({"FOS": 0.4, "EGR1": -0.2}),
            reporters=reps,
        )
        assert out.predicted_change is False and out.n_compared == 2
        assert math.isnan(out.sign_agreement) and math.isnan(out.spearman_r)
        scored = compute_concordance(
            delta_observables={"erk": 0.3},
            delta_gene_expression=pd.Series({"FOS": 0.4, "EGR1": -0.2}),
            reporters=reps,
        )
        assert scored.predicted_change is True
        # One observable over several genes is a constant input: undefined.
        assert math.isnan(scored.spearman_r)

    def test_time_course_rank_sees_the_lag(self):
        import numpy as np
        import pandas as pd

        from hallsim.gene_reporters import (
            peak_concordance,
            time_course_concordance,
        )

        t = np.array([5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0])
        early = pd.Series(np.exp(-(t - 5.0) / 20.0), index=t)
        late = pd.DataFrame(
            {tt: [np.exp(-((tt - 45.0) ** 2) / 800.0)] for tt in t},
            index=["FOS"],
        )
        now = time_course_concordance(early, late)
        later = time_course_concordance(early, late, lag=40.0)
        assert now.n_times == 7 and now.per_gene["FOS"] < 0
        assert later.per_gene["FOS"] > now.per_gene["FOS"]
        peaks_sim = pd.Series({"a": 0.2, "b": 0.6, "c": 1.0})
        peaks_data = pd.Series({"c": 3.0, "a": 1.0, "b": 2.0, "d": 9.0})
        assert peak_concordance(peaks_sim, peaks_data) == 1.0

    def test_a_platform_that_maps_no_probe_raises(self, tmp_path):
        from hallsim import gene_reporters as gr

        matrix = tmp_path / "m.txt"
        matrix.write_text(
            '!Series_platform_id\t"GPL1"\n!series_matrix_table_begin\n'
            '"ID_REF"\t"GSM1"\nP1\t3.0\n!series_matrix_table_end\n'
        )
        platform = tmp_path / "p.txt"
        platform.write_text("ID\tgene_assignment\nQ1\tNM_1 // GENE1 // g\n")
        with pytest.raises(ValueError, match="none of 1 probes"):
            gr.load_gene_expression(matrix, platform)
