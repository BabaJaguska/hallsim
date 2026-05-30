"""Tests for the pre-flight subsystem screening (hallsim.diagnostics)."""

from hallsim.diagnostics import ScreenReport, screen_process
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
