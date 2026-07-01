"""Tests for the pre-flight subsystem screening (hallsim.diagnostics)."""

import pytest

from hallsim.diagnostics import (
    DEAD_SINK,
    SUITABLE,
    ScreenReport,
    coupling_source_verdict,
    recommend_coupling_source,
    screen_process,
)
from hallsim.models.multi_hallmark import (
    GZ06_PSI_DEFAULT,
    GZ06_PSI_NAME,
    GZ06_SBML_PATH,
)
from hallsim.sbml_import import process_from_sbml


def test_screenreport_ok_logic():
    clean = ScreenReport("m", False, False, False, 1.0, 1e-4)
    assert (
        clean.ok
    )  # dataclass positional vs kw — rebuild explicitly to flip one flag
    assert ScreenReport("m", True, False, False, 1.0, 1e-4).ok is False
    assert ScreenReport("m", False, True, False, 1.0, 1e-4).ok is False
    assert ScreenReport("m", False, False, True, 1.0, 1e2).ok is False


def test_tunes_feeds_ok_logic():
    """The 'tunes' half of constituents-first gates ``ok``: a non-tunable
    model is not ok; ``None`` (unprobed) and ``True`` leave it."""
    assert ScreenReport("m", False, False, False, 1.0, 1e-4, tunes=None).ok
    assert ScreenReport("m", False, False, False, 1.0, 1e-4, tunes=True).ok
    assert (
        ScreenReport("m", False, False, False, 1.0, 1e-4, tunes=False).ok
        is False
    )


def test_check_tunability_opt_out_skips_gradient():
    """``check_tunability=False`` skips the jvp probe — ``tunes`` stays None."""
    gz = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_DEFAULT},
    )
    report = screen_process(gz, t_end=100.0, check_tunability=False)
    assert report.tunes is None


@pytest.mark.slow
def test_konrath_tunes_only_on_implicit_solver():
    """Konrath 2023 runs on the explicit solver but its forward sensitivities
    are stiff: the screen must find it tunable only via the implicit solver."""
    kon = process_from_sbml("MODEL2307130001", name="konrath")
    report = screen_process(kon, t_end=600.0)
    assert report.tunes is True
    assert "implicit solver" in report.detail


def test_konrath_pIKK_rejected_as_dead_sink():
    """Konrath's terminal activated-IKK output pIKK is produced but consumed
    by nothing and read by no rate law — the coupling guard must reject it
    (wiring it would feed a frozen constant / diverge if unfrozen)."""
    kon = process_from_sbml("MODEL2307130001", name="konrath")
    v = coupling_source_verdict(kon, "pIKK")
    assert v.verdict == DEAD_SINK
    assert v.frozen and v.produced and not v.consumed
    assert not v.ok


def test_konrath_spIKKg_n_is_the_suitable_source():
    """The live activated-IKK state spIKKg_n is produced and consumed —
    bounded and actively turned over — so it is the usable coupling source."""
    kon = process_from_sbml("MODEL2307130001", name="konrath")
    v = coupling_source_verdict(kon, "spIKKg_n")
    assert v.verdict == SUITABLE
    assert v.produced and v.consumed and v.ok


def test_konrath_recommendation_flags_dead_sink_and_clock_mismatch():
    """The recommendation focused on the activated-IKK outputs must pick
    spIKKg_n over pIKK, and warn that Konrath's second-scale native forcing
    clock is unresolvable on the day-scale composite axis."""
    kon = process_from_sbml("MODEL2307130001", name="konrath")
    rec = recommend_coupling_source(
        kon,
        target_states=("pIKK", "spIKKg_n"),
        canonical_time_seconds=86400.0,
    )
    assert rec.suitable == ("spIKKg_n",)
    assert any("pIKK" in n and "unusable" in n for n in rec.notes)
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
