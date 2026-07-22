"""Tests for the pre-flight subsystem screening (hallsim.diagnostics)."""

import pytest

from hallsim.diagnostics import (
    DEAD_SINK,
    SUITABLE,
    ScreenReport,
    coupling_source_verdict,
    recommend_coupling_source,
    screen_process,
    screen_sensitivity,
)
from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS
from hallsim.models.multi_hallmark import (
    GZ06_PSI_DEFAULT,
    GZ06_PSI_NAME,
    GZ06_SBML_PATH,
    build_multi_hallmark_composite,
)
from hallsim.sbml_import import process_from_sbml

# A vendored (committed) model with both a dead-sink species (`s195`) and
# produced-and-consumed species (`s305`) — exercises the coupling-source
# diagnostic locally, no BioModels download.
WNT_SBML_PATH = (
    GZ06_SBML_PATH.parent.parent / "sivakumar2011" / "wnt_BIOMD0000000397.xml"
)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # base flags: exploding / vanishing / tolerance-sensitive gate ok
        ({}, True),
        ({"exploding": True}, False),
        ({"vanishing": True}, False),
        ({"tol_sensitive": True, "tol_delta": 1e2}, False),
        # negative-domain flag (out-of-domain state is not ok)
        ({"negative": True}, False),
        ({"negative": False}, True),
        # 'tunes' half of constituents-first: None/True leave ok, False gates
        ({"tunes": None}, True),
        ({"tunes": True}, True),
        ({"tunes": False}, False),
    ],
)
def test_screenreport_ok_logic(kwargs, expected):
    kwargs.setdefault("tol_delta", 1e-4)
    tol_delta = kwargs.pop("tol_delta")
    report = ScreenReport(
        "m",
        kwargs.pop("exploding", False),
        kwargs.pop("vanishing", False),
        kwargs.pop("tol_sensitive", False),
        1.0,
        tol_delta,
        **kwargs,
    )
    assert report.ok is expected


def test_check_tunability_opt_out_skips_gradient():
    """``check_tunability=False`` skips the jvp probe — ``tunes`` stays None."""
    gz = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_DEFAULT},
    )
    report = screen_process(gz, t_end=100.0, check_tunability=False)
    assert report.tunes is None


def test_dead_sink_rejected_as_coupling_source():
    """A produced-but-never-consumed, read-by-nothing species (the importer
    freezes it) must be rejected: coupling from it feeds a frozen constant /
    diverges if unfrozen."""
    p = process_from_sbml(str(WNT_SBML_PATH), name="wnt")
    v = coupling_source_verdict(p, "s195")
    assert v.verdict == DEAD_SINK
    assert v.frozen and v.produced and not v.consumed
    assert not v.ok


def test_produced_and_consumed_is_the_suitable_source():
    """A produced-and-consumed species is bounded and actively turned over,
    so it is a usable coupling source."""
    p = process_from_sbml(str(WNT_SBML_PATH), name="wnt")
    v = coupling_source_verdict(p, "s305")
    assert v.verdict == SUITABLE
    assert v.produced and v.consumed and v.ok


def test_recommendation_flags_dead_sink_and_clock_mismatch():
    """Focused on a dead-sink + a suitable state, the recommendation must
    pick the suitable one, flag the dead sink as unusable, and warn when the
    native clock is far finer than the composite's."""
    p = process_from_sbml(str(WNT_SBML_PATH), name="wnt")
    canon = p.native_time_seconds * 1000.0  # force a >100x clock ratio
    rec = recommend_coupling_source(
        p,
        target_states=("s195", "s305"),
        canonical_time_seconds=canon,
    )
    assert rec.suitable == ("s305",)
    assert any("s195" in n and "unusable" in n for n in rec.notes)
    assert any("finer than the composite clock" in n for n in rec.notes)


def test_gz06_flagged_tolerance_sensitive():
    """The Geva-Zatorsky p53 oscillator diverges at loose tolerance — the
    screen must catch that it is solver-dependent (the canonical trap)."""
    gz = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_DEFAULT},
    )
    report = screen_process(gz, t_end=100.0)  # native hours
    assert report.tolerance_sensitive, report
    assert report.tol_rel_diff > 1.0  # loose vs tight wildly disagree
    assert not report.ok


@pytest.mark.slow
def test_screen_sensitivity_flags_saturated_reporter():
    """In the default dp14 regime the etoposide dose saturates ψ→1, so DDB2
    (√⟨p53²⟩) is dead to genomic-instability severity while CDKN1A is not. The
    in-regime guardrail must flag DDB2 FLAT and CDKN1A live, with finite
    gradients through the three stiff SBML models (reverse-mode)."""
    comp = build_multi_hallmark_composite(validate=False)
    reports = screen_sensitivity(
        comp,
        MULTI_HALLMARK_REPORTERS,
        ["Genomic Instability"],
        baseline={
            "Genomic Instability": 1.0,
            "Deregulated Nutrient Sensing": 0.5,
        },
        t_end=14.0,
        macro_dt=3.5,
    )
    by = {r.reporter: r for r in reports}
    assert all(r.finite for r in reports), reports
    assert not by["DDB2"].live, by["DDB2"]  # saturated → flat → caught
    assert by["CDKN1A"].live, by["CDKN1A"]  # damage→p21 stays live
