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
        assert not any(r.levelno == logging.WARNING for r in caplog.records)


def _toy_problem():
    """One-process decay composite with a single tunable rate and two
    reporters reading the same pool — a minimal end-to-end CalibrationProblem.
    """
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
                process_name="decay", field="rate", clamp=(1e-3, 5.0)
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

    def test_reverse_mode_fit_still_reports(self):
        # A reverse-mode fit leaves the problem on a custom_vjp adjoint that
        # forward-mode jacfwd cannot differentiate; the report must still run.
        history = _toy_problem().fit(
            steps=2, mode="reverse", learning_rate=0.05, verbose=False
        )
        assert history.identifiability is not None
        assert "rate" in history.identifiability.names


# ── structural redundancy, from the symbolic forms alone ─────────────────

REDUNDANT_MODEL = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="redundant">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="2" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.3" constant="true"/>
      <parameter id="k2" value="0.2" constant="true"/>
      <parameter id="k3" value="0.5" constant="true"/>
      <parameter id="k4" value="1.0" constant="true"/>
      <parameter id="k5" value="0.7" constant="true"/>
      <parameter id="k6" value="0.4" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="r1" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k1</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="r2" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k2</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="r3" reversible="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="C" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k3</ci><ci>B</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="r4" reversible="false">
        <listOfReactants><speciesReference species="C" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k5</ci><ci>k6</ci><ci>C</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


class TestStructuralRedundancy:
    def _composite(self, tmp_path):
        from hallsim.composite import single_process_composite
        from hallsim.sbml_import import process_from_sbml

        path = tmp_path / "redundant.xml"
        path.write_text(REDUNDANT_MODEL)
        return single_process_composite(process_from_sbml(str(path), name="m"))

    def test_same_law_form_on_the_same_reaction_is_one_parameter(
        self, tmp_path
    ):
        """k1·A and k2·A with the same stoichiometry: only k1 + k2 enters
        the dynamics. k5·k6 on one law: only the product does. k4 is read
        by nothing. k3 stands alone."""
        from hallsim.identifiability import structural_redundancy

        comp = self._composite(tmp_path)
        report = structural_redundancy(comp)
        groups = {g.parameters: g.ratios for g in report.groups}
        assert groups == {
            ("m.parameters.k1", "m.parameters.k2"): ("1", "1"),
            ("m.parameters.k5", "m.parameters.k6"): (
                "1",
                "m.parameters.k5/m.parameters.k6",
            ),
        }
        assert report.unassessed == ()
        assert "m.parameters.k3" in report.assessed
        # read by no form, so not assessed unless asked for; then inert
        assert "m.parameters.k4" not in report.assessed
        asked = structural_redundancy(comp, params=["m.parameters.k4"])
        assert asked.inert == ("m.parameters.k4",)
        assert "redundant: m.parameters.k1, m.parameters.k2" in str(report)

    def test_restricting_to_fitted_parameters(self, tmp_path):
        from hallsim.calibration import ParameterRef
        from hallsim.identifiability import structural_redundancy

        report = structural_redundancy(
            self._composite(tmp_path),
            params=[
                ParameterRef("m", "parameters.k1"),
                ParameterRef("m", "parameters.k3"),
                "m.parameters.k6",
            ],
        )
        assert report.groups == ()
        assert report.assessed == (
            "m.parameters.k1",
            "m.parameters.k3",
            "m.parameters.k6",
        )

    def test_two_clamps_on_one_target_are_redundant_and_the_fit_warns(
        self, caplog
    ):
        """Two hand-written edges with the same form on the same path: their
        rates enter only as a sum, seen from the edges' declared forms and
        said at problem construction, before any data."""
        import pandas as pd

        from hallsim.calibration import (
            CalibrationProblem,
            Condition,
            ParameterRef,
        )
        from hallsim.composite import Composite
        from hallsim.gene_reporters import GeneReporter
        from hallsim.identifiability import structural_redundancy
        from hallsim.models.clamp_edge import ClampEdge

        comp = Composite(
            processes={
                "h1": ClampEdge(k_clamp=1.0, target_default=0.2),
                "h2": ClampEdge(k_clamp=0.5),
            },
            topology={
                "h1": {"target": "p/x", "setpoint": "hold/sp"},
                "h2": {"target": "p/x", "setpoint": "hold/sp"},
            },
            initial={"p/x": 0.2, "hold/sp": 1.0},
            validate=False,
            semantic_validation=False,
        )
        report = structural_redundancy(comp)
        assert [g.parameters for g in report.groups] == [
            ("h1.k_clamp", "h2.k_clamp")
        ]
        with caplog.at_level(logging.WARNING, logger="hallsim.calibration"):
            CalibrationProblem(
                composite=comp,
                reporters=[
                    GeneReporter(
                        observable="p/x", gene_symbol="GENE_X", sign=1
                    )
                ],
                conditions={
                    "ctrl": Condition("ctrl", {}),
                    "high": Condition("high", {}),
                },
                data={"high_vs_ctrl": pd.Series({"GENE_X": -0.5})},
                arm_pairs={"high_vs_ctrl": ("high", "ctrl")},
                params={
                    "k1": ParameterRef(process_name="h1", field="k_clamp"),
                    "k2": ParameterRef(process_name="h2", field="k_clamp"),
                },
                fit_arms=["high_vs_ctrl"],
                t_end=5.0,
                macro_dt=1.0,
                n_save=3,
            )
        assert "structural redundancy" in caplog.text
        assert "'k1', 'k2'" in caplog.text

    def test_dallepezze_k33_and_k34_are_one_coordinate(self):
        """The case from the referee pass: two biogenesis reactions with the
        identical law k·Mito_mass_turnover·mTORC1_pS2448 and the same
        stoichiometry, one of them named for AMPK it never reads."""
        from hallsim.composite import single_process_composite
        from hallsim.identifiability import structural_redundancy
        from hallsim.sbml_import import process_from_sbml

        proc = process_from_sbml(
            "demos/models/sbml/dallepezze2014/"
            "dallepezze2014_BIOMD0000000582.xml",
            name="dp14",
        )
        report = structural_redundancy(single_process_composite(proc))
        pair = (
            "dp14.parameters.mito_biogenesis_by_AMPK_pT172",
            "dp14.parameters.mito_biogenesis_by_mTORC1_pS2448",
        )
        assert pair in [g.parameters for g in report.groups]
        group = next(g for g in report.groups if g.parameters == pair)
        assert group.ratios == ("1", "1")
