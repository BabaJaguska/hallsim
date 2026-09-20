"""The residual scale the verdicts rest on, and the set they recommend."""

import equinox as eqx
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
        self.readouts = [type("R", (), {"key": "G"})()]
        self.fittables = {n: None for n in names}

    def initial_params(self):
        return {n: 1.0 for n in self._names}

    def predicted(self, params, arm, times):
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


def test_the_kept_set_satisfies_its_own_tolerance(monkeypatch):
    """Two near-collinear candidates each look fine on admission and widen
    each other afterwards; the set that comes back keeps only one."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=14)
    near = base + 0.35 * rng.normal(size=14)  # correlated below 0.95
    strong = 4.0 * rng.normal(size=14)
    jac = np.stack([strong, base, near], axis=1)
    names = ["strong", "base", "near"]
    monkeypatch.setattr(
        "hallsim.identifiability.sensitivity_jacobian",
        lambda p, params=None: (jac, names),
    )
    choice = choose_fit_set(_Problem(jac, names), sigma=2.5, std_tol=1.0)
    for n in choice.keep:
        assert choice.std_decades[n] <= 1.0
    assert not ({"base", "near"} <= set(choice.keep))


def test_screen_fittable_pools_the_composite_s_own_surface():
    """The screen's pool is what the processes declare fittable, and the
    problem is rebuilt over it without the caller naming anything."""
    import pandas as pd

    from hallsim.calibration import CalibrationProblem, Condition, FitParam
    from hallsim.composite import Composite
    from hallsim.gene_reporters import Readout
    from hallsim.identifiability import screen_fittable
    from hallsim.process import Port, PortRole, Process, calibratable

    class Decay(Process):
        rate: float = calibratable(0.5, description="decay rate")
        scale: float = 1.0  # not declared fittable
        timescale: float = eqx.field(static=True, default=1.0)

        def ports_schema(self):
            return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

        def derivative(self, t, state):
            return {"x": -self.rate * state["x"]}

    comp = Composite(
        processes={"d": Decay()},
        topology={"d": {"x": "pool/x"}},
        validate=False,
        semantic_validation=False,
    )
    problem = CalibrationProblem(
        composite=comp,
        readouts=[Readout(path="pool/x", key="GX")],
        conditions={"a": Condition("a", {}), "b": Condition("b", {})},
        data={"b_vs_a": pd.Series({"GX": -0.5})},
        arms={"b_vs_a": "a"},
        params={"r": FitParam("d", "rate")},
        fit_arms=["b_vs_a"],
        t_end=2.0,
        n_save=3,
    )
    choice = screen_fittable(problem, sigma=0.3)
    assert set(choice.keep) | set(choice.drop) == {"d.rate"}


def test_a_column_the_data_barely_see_is_dropped_not_certified(monkeypatch):
    """A near-zero sensitivity column makes the normal matrix singular; the
    pseudo-inverse would report zero variance for it. It must be dropped."""
    rng = np.random.default_rng(2)
    strong = rng.normal(size=14)
    faint = 1e-9 * rng.normal(size=14)
    jac = np.stack([strong, faint], axis=1)
    names = ["strong", "faint"]
    monkeypatch.setattr(
        "hallsim.identifiability.sensitivity_jacobian",
        lambda p, params=None: (jac, names),
    )
    choice = choose_fit_set(_Problem(jac, names), sigma=0.4, struct_tol=1e-12)
    assert "faint" in choice.drop
    assert "strong" in choice.keep


def test_must_keep_is_admitted_first_and_never_removed(monkeypatch):
    """A forced member stays even when a stronger candidate duplicates it;
    the duplicate is what gets dropped."""
    jac, names = _jacobian()
    monkeypatch.setattr(
        "hallsim.identifiability.sensitivity_jacobian",
        lambda p, params=None: (jac, names),
    )
    choice = choose_fit_set(
        _Problem(jac, names), sigma=0.4, must_keep=("copy",)
    )
    assert "copy" in choice.keep
    assert "strong" in choice.drop and "copy" in choice.drop["strong"]
