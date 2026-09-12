"""The residual scale the verdicts rest on, and the set they recommend."""

import numpy as np
import pytest

from hallsim.identifiability import (
    choose_fit_set,
    report_from_jacobian,
)


def _jacobian():
    """Four candidates: one strong, one a near-copy of it, one weak but
    independent, one that moves nothing."""
    rng = np.random.default_rng(0)
    strong = rng.normal(size=14)
    copy = strong + 1e-6 * rng.normal(size=14)
    weak = 0.02 * rng.normal(size=14)
    return (
        np.stack([strong, copy, weak, np.zeros(14)], axis=1),
        ["strong", "copy", "weak", "dead"],
    )


def test_uncertainty_scales_with_the_residual_scale():
    """σ is the size of a miss; halving it halves every uncertainty. A
    report that omits it has asserted σ = 1."""
    jac, names = _jacobian()
    loose = report_from_jacobian(jac, names, sigma=1.0)
    tight = report_from_jacobian(jac, names, sigma=0.5)
    assert tight.sigma == 0.5
    for n in ("strong", "weak"):
        assert tight.std_decades[n] == pytest.approx(
            loose.std_decades[n] / 2.0, rel=1e-9
        )
    assert "σ = 0.5" in str(tight)


class _Problem:
    """The slice of CalibrationProblem the screen reads."""

    def __init__(self, jac, names):
        self._jac, self._names = jac, names
        self.fit_arms = ["arm"]
        self.data = {"arm": {7.0: {"G": 0.0}}}
        self.reporters = [type("R", (), {"gene_symbol": "G"})()]
        self.param_refs = {n: None for n in names}

    def initial_params(self):
        return {n: 1.0 for n in self._names}

    def model_lfc(self, params, arm, times):
        return np.zeros((1, len(times)))


def test_the_screen_returns_a_set_not_a_verdict_per_parameter(monkeypatch):
    """A near-duplicate is dropped naming what it duplicates, a parameter
    that moves nothing is dropped, and the kept set carries the uncertainty
    each parameter has *within that set*."""
    jac, names = _jacobian()
    problem = _Problem(jac, names)
    monkeypatch.setattr(
        "hallsim.identifiability.sensitivity_jacobian",
        lambda p, params=None: (jac, names),
    )
    choice = choose_fit_set(problem, sigma=0.4)

    # exactly one of the pair survives, and the other names it
    pair = {"strong", "copy"}
    kept, dropped = pair & set(choice.keep), pair & set(choice.drop)
    assert len(kept) == 1 and len(dropped) == 1
    assert kept.pop() in choice.drop[dropped.pop()]
    assert choice.drop["dead"] == "moves no reporter"
    for n in choice.keep:
        assert np.isfinite(choice.std_decades[n])
    assert "drop: " in str(choice)


def test_a_looser_scale_admits_fewer_parameters(monkeypatch):
    """The pool is the same; only the claimed precision of the data
    changes. Calling the data sloppier drops the weak parameter."""
    jac, names = _jacobian()
    monkeypatch.setattr(
        "hallsim.identifiability.sensitivity_jacobian",
        lambda p, params=None: (jac, names),
    )
    problem = _Problem(jac, names)
    tight = choose_fit_set(problem, sigma=0.2)
    loose = choose_fit_set(problem, sigma=20.0)
    assert len(tight.keep) >= len(loose.keep)
    assert "weak" in loose.drop
