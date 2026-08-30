"""`intake.published_fit_chi2` — does a deposit reproduce its paper's fit?

Scored against an analytic first-order decay, so the expected chi2 is known
in closed form rather than recorded from a previous run.
"""

import numpy as np
import pytest

from hallsim.composite import Composite
from hallsim.intake import FitReport, published_fit_chi2
from hallsim.process import Port, PortRole, Process


class Decay(Process):
    """dx/dt = -rate * x, so x(t) = x0 * exp(-rate * t)."""

    rate: float = 0.5

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


def _comp(rate=0.5):
    return Composite(
        processes={"d": Decay(rate=rate)},
        topology={"d": {"x": "cell/x"}},
        validate=False,
        semantic_validation=False,
    )


TS = np.array([0.5, 1.0, 2.0, 4.0])
SD = np.full(4, 0.01)


def _exact(rate=0.5, ts=TS):
    return np.exp(-rate * ts)


def test_perfect_model_scores_near_zero():
    obs = {"cell/x": (TS, _exact(), SD)}
    r = published_fit_chi2(_comp(), obs, save_dt=0.005)
    assert r.n_points == 4
    assert r.chi2 < 1.0  # interpolation error only, against sd=0.01


def test_chi2_grows_with_a_wrong_rate():
    obs = {"cell/x": (TS, _exact(rate=0.5), SD)}
    good = published_fit_chi2(_comp(0.5), obs, save_dt=0.005).chi2
    bad = published_fit_chi2(_comp(0.7), obs, save_dt=0.005).chi2
    assert bad > 100 * max(good, 1e-9)


def test_reproduces_is_a_comparison_not_a_fit_quality():
    """A model can fit badly and still *reproduce* the paper's stated chi2 —
    that is the whole point of the check."""
    obs = {"cell/x": (TS, _exact(rate=0.7), SD)}
    r = published_fit_chi2(_comp(0.5), obs, save_dt=0.005)
    assert r.chi2 > 100  # a bad fit
    same = published_fit_chi2(_comp(0.5), obs, save_dt=0.005, reported=r.chi2)
    assert same.reproduces is True  # and it reproduces exactly
    off = published_fit_chi2(
        _comp(0.5), obs, save_dt=0.005, reported=r.chi2 * 1.5
    )
    assert off.reproduces is False
    assert off.relative_error == pytest.approx(1 / 3, rel=1e-3)


def test_no_reported_value_leaves_the_verdict_undefined():
    obs = {"cell/x": (TS, _exact(), SD)}
    r = published_fit_chi2(_comp(), obs, save_dt=0.005)
    assert r.reproduces is None and r.relative_error is None


def test_per_path_sums_to_the_total():
    obs = {"cell/x": (TS, _exact(rate=0.6), SD)}
    r = published_fit_chi2(_comp(0.5), obs, save_dt=0.005)
    assert sum(r.per_path.values()) == pytest.approx(r.chi2)
    assert set(r.per_path) == {"cell/x"}


def test_unknown_path_is_refused_by_name():
    obs = {"cell/nope": (TS, _exact(), SD)}
    with pytest.raises(KeyError, match="nope"):
        published_fit_chi2(_comp(), obs)


def test_non_positive_sd_is_refused():
    obs = {"cell/x": (TS, _exact(), np.array([0.01, 0.0, 0.01, 0.01]))}
    with pytest.raises(ValueError, match="sd"):
        published_fit_chi2(_comp(), obs, save_dt=0.005)


def test_empty_observations_refused():
    with pytest.raises(ValueError, match="at least one"):
        published_fit_chi2(_comp(), {})


def test_reduced_chi2_divides_by_points():
    r = FitReport(chi2=8.0, n_points=4)
    assert r.reduced_chi2 == 2.0


@pytest.mark.slow
def test_dallepezze_reproduces_its_published_fit():
    """The check that licenses every finding about this deposit: it still
    produces the fit its paper claims, so later defects are the paper's."""
    xlrd = pytest.importorskip("xlrd")
    del xlrd
    from demos.dp14_published_fit import (
        REPORTED_CHI2,
        load_observations,
        observable_map,
        verify,
    )

    # The paper's own count: 14 observables, 127 points.
    assert len(observable_map()) == 14
    data = load_observations()
    assert len(data) == 14
    assert sum(len(t) for t, _, _ in data.values()) == 127

    report = verify()
    assert report.n_points == 127
    assert report.reproduces is True
    assert report.relative_error < 0.01
    assert report.chi2 == pytest.approx(REPORTED_CHI2, rel=0.01)
