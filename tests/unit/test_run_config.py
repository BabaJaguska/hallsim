"""A run folder says what produced it: the problem, as JSON, and the fit."""

import json

import pandas as pd

from hallsim.calibration import CalibrationProblem, Condition, FitParam
from hallsim.composite import Composite
from hallsim.gene_reporters import Readout
from hallsim.process import Port, PortRole, Process, calibratable
import equinox as eqx


class Decay(Process):
    rate: float = calibratable(0.5, description="decay rate")
    timescale: float = eqx.field(static=True, default=1.0)

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


def _problem():
    comp = Composite(
        processes={"d": Decay()},
        topology={"d": {"x": "pool/x"}},
        validate=False,
        semantic_validation=False,
    )
    return CalibrationProblem(
        composite=comp,
        readouts=[Readout(path="pool/x", key="GX")],
        conditions={"a": Condition("a", {}), "b": Condition("b", {})},
        data={"b_vs_a": pd.Series({"GX": -0.5})},
        arms={"b_vs_a": "a"},
        params={"r": FitParam("d", "rate", prior=0.5, prior_sigma=0.5)},
        fit_arms=["b_vs_a"],
        t_end=2.0,
        n_save=3,
        notes={"dataset": "toy"},
    )


def test_describe_is_json_and_names_what_matters():
    desc = _problem().describe()
    text = json.dumps(desc)  # must not raise
    assert desc["composite"]["processes"] == {"d": "Decay"}
    assert desc["params"]["r"]["prior"] == 0.5
    assert desc["params"]["r"]["initial"] == 0.5
    assert desc["data"]["b_vs_a"] == {"GX": -0.5}
    assert desc["loss"]["prior_weight"] == 1.0
    assert desc["notes"] == {"dataset": "toy"}
    assert "jax" in desc["versions"]
    assert "GX" in text


def test_the_fit_records_its_settings():
    history = _problem().fit(steps=2, identifiability=False, verbose=False)
    assert history.settings["steps"] == 2
    assert history.settings["method"] in ("adam", "lbfgs")
    json.dumps(history.settings)
