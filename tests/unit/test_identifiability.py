"""Fisher-information identifiability analysis (hallsim.identifiability)."""

import logging

import numpy as np

from hallsim.identifiability import (
    identifiability_report,
    log_summary,
    report_from_jacobian,
)


class TestVerdictsFromJacobian:
    """The linear-algebra core, on synthetic Jacobians (no model solve)."""

    def test_zero_column_is_structural(self):
        # Second parameter moves no residual → structurally non-identifiable.
        jac = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        rep = report_from_jacobian(jac, ["a", "b"])
        assert rep.verdict["b"] == "structural"
        assert rep.verdict["a"] != "structural"
        assert "b" in rep.recommended_freeze
        assert not np.isfinite(rep.std_decades["b"])

    def test_collinear_columns_are_confounded(self):
        # Two identical columns → correlation ±1, one of the pair frozen.
        base = np.array([1.0, 2.0, 3.0, 4.0])
        jac = np.stack([base, base], axis=1)
        rep = report_from_jacobian(jac, ["a", "b"])
        assert rep.confounded, "expected a confounded pair"
        a, b, c = rep.confounded[0]
        assert {a, b} == {"a", "b"}
        assert abs(c) >= 0.95
        assert rep.verdict["a"] == "practical"
        assert len(rep.recommended_freeze) == 1

    def test_orthogonal_well_scaled_columns_are_identifiable(self):
        jac = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        rep = report_from_jacobian(jac, ["a", "b"])
        assert rep.verdict["a"] == "identifiable"
        assert rep.verdict["b"] == "identifiable"
        assert rep.recommended_freeze == []
        assert not rep.confounded

    def test_eigenvalues_ascending_and_correlation_symmetric(self):
        rng = np.random.default_rng(0)
        jac = rng.normal(size=(8, 3))
        rep = report_from_jacobian(jac, ["a", "b", "c"])
        assert np.all(np.diff(rep.eigenvalues) >= -1e-9)
        assert np.allclose(rep.correlation, rep.correlation.T)
        assert np.allclose(np.diag(rep.correlation), 1.0)


class TestLogSummary:
    def test_warns_on_structural(self, caplog):
        jac = np.array([[1.0, 0.0], [2.0, 0.0]])
        rep = report_from_jacobian(jac, ["a", "b"])
        logger = logging.getLogger("hallsim.test_ident")
        with caplog.at_level(logging.INFO, logger="hallsim.test_ident"):
            log_summary(rep, logger)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert any("move no reporter" in r.message for r in caplog.records)

    def test_no_warning_when_all_identifiable(self, caplog):
        jac = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        rep = report_from_jacobian(jac, ["a", "b"])
        logger = logging.getLogger("hallsim.test_ident2")
        with caplog.at_level(logging.INFO, logger="hallsim.test_ident2"):
            log_summary(rep, logger)
        assert not any(
            r.levelno == logging.WARNING for r in caplog.records
        )


def _toy_problem():
    """One-process decay composite with a single tunable rate and two
    reporters reading the same pool — a minimal end-to-end CalibrationProblem."""
    import pandas as pd

    from hallsim.calibration import (
        CalibrationProblem,
        Condition,
        ParameterRef,
    )
    from hallsim.composite import Composite
    from hallsim.gene_reporters import GeneReporter
    from hallsim.process import Port, PortRole, Process

    class Decay(Process):
        rate: float = 0.1

        def ports_schema(self):
            return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

        def derivative(self, t, state):
            return {"x": -self.rate * state["x"]}

    comp = Composite(
        processes={"decay": Decay()},
        topology={"decay": {"x": "pool/x"}},
        validate=False,
        semantic_validation=False,
    )
    return CalibrationProblem(
        composite=comp,
        reporters=[
            GeneReporter(observable="pool/x", gene_symbol="GX", sign=+1),
            GeneReporter(observable="pool/x", gene_symbol="GY", sign=-1),
        ],
        conditions={"ctrl": Condition("ctrl", {}), "hi": Condition("hi", {})},
        data={"hi_vs_ctrl": pd.Series({"GX": -0.5, "GY": +0.5})},
        arm_pairs={"hi_vs_ctrl": ("hi", "ctrl")},
        params={
            "rate": ParameterRef(
                process_name="decay", field="rate", init=0.2, clamp=(1e-3, 5.0)
            )
        },
        fit_arms=["hi_vs_ctrl"],
        t_end=5.0,
        macro_dt=1.0,
        n_save=3,
    )


class TestEndToEnd:
    def test_report_runs_on_composite(self):
        rep = identifiability_report(_toy_problem())
        assert "rate" in rep.names
        assert rep.verdict["rate"] != "structural"  # rate does move the pool
        assert np.isfinite(rep.rel_sensitivity["rate"])

    def test_fit_attaches_identifiability_by_default(self):
        history = _toy_problem().fit(
            steps=2, learning_rate=0.05, verbose=False
        )
        assert history.identifiability is not None
        assert "rate" in history.identifiability.names

    def test_opt_out_leaves_it_none(self):
        history = _toy_problem().fit(
            steps=2, learning_rate=0.05, verbose=False, identifiability=False
        )
        assert history.identifiability is None
