"""A fittable reference is checked against its field when the problem is
wired, not inside a derivative."""

import equinox as eqx
import pandas as pd
import pytest

from hallsim.calibration import CalibrationProblem, Condition, ParameterRef
from hallsim.composite import Composite
from hallsim.gene_reporters import GeneReporter
from hallsim.process import Port, PortRole, Process


class Gate(Process):
    """One scalar rate, one tuple threshold, one static label."""

    rate: float = 0.1
    K: tuple = (0.5,)
    label: str = eqx.field(static=True, default="gate")

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"] / (self.K[0] + state["x"])}


def _problem(field):
    comp = Composite(
        processes={"g": Gate()},
        topology={"g": {"x": "pool/x"}},
        validate=False,
        semantic_validation=False,
    )
    return CalibrationProblem(
        composite=comp,
        reporters=[GeneReporter(observable="pool/x", gene_symbol="GX")],
        conditions={"a": Condition("a", {}), "b": Condition("b", {})},
        data={"b_vs_a": pd.Series({"GX": 0.0})},
        arm_pairs={"b_vs_a": ("a", "b")},
        params={"p": ParameterRef(process_name="g", field=field)},
        fit_arms=["b_vs_a"],
    )


def test_a_scalar_field_is_accepted():
    _problem("rate")


def test_a_tuple_valued_field_is_refused_by_name():
    with pytest.raises(ValueError, match=r"g\.K.*not a scalar"):
        _problem("K")


def test_a_static_field_is_refused():
    with pytest.raises(ValueError, match=r"g\.label.*static"):
        _problem("label")


def test_a_missing_field_is_refused():
    with pytest.raises(ValueError, match=r"g\.nope.*does not have"):
        _problem("nope")
