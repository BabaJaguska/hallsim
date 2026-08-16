"""Run-output and report helpers, now that they ship in ``src/``.

These moved out of the flagship demo, so they went from "whatever that one
script needed" to framework surface: ``make_run_dir`` removes a directory, and
``format_table`` decides which arms are labelled FIT from a parameter rather
than a hardcoded arm name.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from hallsim.calibration_report import (  # noqa: E402
    format_table,
    rows_by_gene,
)
from hallsim.gene_reporters import (  # noqa: E402
    ConcordanceResult,
    GeneReporter,
    ReporterRow,
    last_value,
)
from hallsim.io import make_run_dir  # noqa: E402


def _result(gene, delta_sim, delta_data):
    reporter = GeneReporter(
        observable="pool/x",
        gene_symbol=gene,
        sign=1,
        summary=last_value,
    )
    row = ReporterRow(
        reporter=reporter,
        delta_sim=delta_sim,
        delta_data=delta_data,
        sign_match=(delta_sim > 0) == (delta_data > 0),
    )
    return ConcordanceResult(
        condition_name="arm",
        rows=[row],
        sign_agreement=1.0,
        spearman_r=0.5,
        n_compared=1,
        mean_abs_error=abs(delta_sim - delta_data),
    )


class TestMakeRunDir:
    def test_creates_stamped_dir_and_latest_symlink(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("hallsim.io._ROOT", tmp_path)
        run = make_run_dir("demo_x", stamp="2026-01-01_00-00-00")
        assert run.is_dir()
        latest = tmp_path / "outputs" / "demo_x" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == run.resolve()

    def test_second_run_repoints_latest_and_keeps_the_first(
        self, monkeypatch, tmp_path
    ):
        """A run must never overwrite the previous one — that is the whole
        point of stamping."""
        monkeypatch.setattr("hallsim.io._ROOT", tmp_path)
        first = make_run_dir("demo_x", stamp="run_a")
        (first / "figure.png").write_text("first output")

        second = make_run_dir("demo_x", stamp="run_b")
        assert (first / "figure.png").read_text() == "first output"
        assert (
            tmp_path / "outputs" / "demo_x" / "latest"
        ).resolve() == second.resolve()

    def test_replaces_a_real_latest_directory(self, monkeypatch, tmp_path):
        """``latest`` left behind as a real directory is replaced, and only it
        — sibling runs are untouched."""
        monkeypatch.setattr("hallsim.io._ROOT", tmp_path)
        base = tmp_path / "outputs" / "demo_x"
        keep = make_run_dir("demo_x", stamp="keep_me")
        (keep / "keep.txt").write_text("keep")
        (base / "latest").unlink()
        (base / "latest").mkdir()
        (base / "latest" / "stale.txt").write_text("stale")

        run = make_run_dir("demo_x", stamp="fresh")
        assert (base / "latest").is_symlink()
        assert (base / "latest").resolve() == run.resolve()
        assert (keep / "keep.txt").read_text() == "keep"


class TestFormatTable:
    def test_labels_fit_and_held_out_arms_from_the_argument(self):
        pre = {
            "A_vs_ctrl": {7.0: _result("CDKN1A", 0.5, 0.4)},
            "B_vs_ctrl": {7.0: _result("CDKN1A", 0.2, 0.4)},
        }
        post = {
            "A_vs_ctrl": {7.0: _result("CDKN1A", 0.45, 0.4)},
            "B_vs_ctrl": {7.0: _result("CDKN1A", 0.3, 0.4)},
        }
        table = format_table(pre, post, fit_arms={"A_vs_ctrl"})
        assert "[FIT ] A_vs_ctrl" in table
        assert "[HELD-OUT] B_vs_ctrl" in table

    def test_no_fit_arms_means_everything_is_held_out(self):
        pre = {"A_vs_ctrl": {7.0: _result("CDKN1A", 0.5, 0.4)}}
        table = format_table(pre, pre)
        assert "[HELD-OUT] A_vs_ctrl" in table
        assert "[FIT ]" not in table

    def test_rows_by_gene_keys_on_symbol(self):
        result = _result("MDM2", 1.0, 0.5)
        assert set(rows_by_gene(result)) == {"MDM2"}
