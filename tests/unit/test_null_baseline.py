"""Every concordance score carries the floor it has to beat."""

import pandas as pd
import pytest

from hallsim.gene_reporters import Readout, compute_concordance


def test_the_no_change_predictor_is_scored_beside_the_model():
    reporters = [
        Readout(path="a", key="A", sign=+1),
        Readout(path="b", key="B", sign=+1),
    ]
    result = compute_concordance(
        delta_observables={"a": 0.9, "b": -0.1},
        delta_gene_expression=pd.Series({"A": 1.0, "B": -0.5}),
        reporters=reporters,
    )
    assert result.null_abs_error == pytest.approx(0.75)
    assert result.mean_abs_error == pytest.approx(0.25)
