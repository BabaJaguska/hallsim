"""A PEtab problem's measurements as contrasts (hallsim.petab_data).

Offline: a PEtab problem is three tables, so the tests build them directly
rather than fetching a benchmark problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hallsim.petab_data import PetabDataset
from hallsim.search.attached import condition_names


class _Problem:
    """What PetabDataset reads off a ``petab.v1.Problem``."""

    def __init__(self, measurement_df, observable_df=None, condition_df=None):
        self.measurement_df = measurement_df
        self.observable_df = observable_df
        self.condition_df = condition_df


def _measurements(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "observableId",
            "simulationConditionId",
            "time",
            "measurement",
        ],
    )


def _observables(scales):
    return pd.DataFrame(
        {
            "observableFormula": {k: f"{k}_expr" for k in scales},
            "observableTransformation": scales,
        }
    )


def _conditions(pairs):
    frame = pd.DataFrame(pairs, columns=["conditionId", "conditionName"])
    return frame.set_index("conditionId")


# --- the contrast ----------------------------------------------------------


def test_a_linear_observable_is_read_on_a_log2_scale():
    problem = _Problem(
        _measurements(
            [
                ("obs", "c1", 0.0, 2.0),
                ("obs", "c1", 1.0, 8.0),
            ]
        ),
        _observables({"obs": "lin"}),
    )
    data = PetabDataset.from_problem(problem)
    course = data.course("c1")
    # 8 against 2 is two doublings.
    assert data.delta(course[1.0], course[0.0])["obs"] == pytest.approx(2.0)


def test_a_declared_log_scale_is_converted_not_logged_again():
    """A measurement PEtab declares on a log scale is already logged; taking
    log2 of it again would report the log of a log."""
    linear = _Problem(
        _measurements([("obs", "c1", 0.0, 2.0), ("obs", "c1", 1.0, 8.0)]),
        _observables({"obs": "lin"}),
    )
    natural = _Problem(
        _measurements(
            [
                ("obs", "c1", 0.0, float(np.log(2.0))),
                ("obs", "c1", 1.0, float(np.log(8.0))),
            ]
        ),
        _observables({"obs": "log"}),
    )
    base_ten = _Problem(
        _measurements(
            [
                ("obs", "c1", 0.0, float(np.log10(2.0))),
                ("obs", "c1", 1.0, float(np.log10(8.0))),
            ]
        ),
        _observables({"obs": "log10"}),
    )
    deltas = []
    for problem in (linear, natural, base_ten):
        data = PetabDataset.from_problem(problem)
        course = data.course("c1")
        deltas.append(float(data.delta(course[1.0], course[0.0])["obs"]))
    assert deltas == pytest.approx([2.0, 2.0, 2.0])


def test_a_non_positive_measurement_is_absent_rather_than_infinite():
    problem = _Problem(
        _measurements(
            [
                ("ok", "c1", 0.0, 1.0),
                ("ok", "c1", 1.0, 2.0),
                ("zero", "c1", 0.0, 0.0),
                ("zero", "c1", 1.0, 4.0),
            ]
        ),
        _observables({"ok": "lin", "zero": "lin"}),
    )
    data = PetabDataset.from_problem(problem)
    course = data.course("c1")
    delta = data.delta(course[1.0], course[0.0])
    assert "ok" in delta
    assert "zero" not in delta
    assert np.isfinite(delta).all()


def test_replicates_become_columns_so_variance_is_available():
    problem = _Problem(
        _measurements(
            [
                ("obs", "c1", 0.0, 1.0),
                ("obs", "c1", 0.0, 1.1),
                ("obs", "c1", 1.0, 4.0),
                ("obs", "c1", 1.0, 4.4),
            ]
        ),
        _observables({"obs": "lin"}),
    )
    data = PetabDataset.from_problem(problem)
    course = data.course("c1")
    assert len(data.sample_groups[course[0.0]]) == 2
    assert float(data.variance(course[1.0], course[0.0])["obs"]) > 0.0


# --- naming ----------------------------------------------------------------


def test_a_condition_resolves_by_id_or_by_name():
    problem = _Problem(
        _measurements([("obs", "c1", 0.0, 1.0), ("obs", "c1", 1.0, 2.0)]),
        _observables({"obs": "lin"}),
        _conditions([("c1", "vehicle")]),
    )
    data = PetabDataset.from_problem(problem)
    assert data.course("c1") == data.course("vehicle")
    assert data.times("c1") == [0.0, 1.0]


def test_an_unknown_condition_names_the_ones_there_are():
    problem = _Problem(
        _measurements([("obs", "c1", 0.0, 1.0)]),
        _observables({"obs": "lin"}),
        _conditions([("c1", "vehicle")]),
    )
    data = PetabDataset.from_problem(problem)
    with pytest.raises(KeyError, match="vehicle"):
        data.course("nope")


def test_conditions_sharing_a_name_stay_separate_arms():
    """PEtab lets several conditions carry one name — Isensee 2018 gives seven
    doses of a compound the same one — and pooling them averages a dose
    response into a single arm."""
    labels = condition_names(
        _conditions(
            [("c1", "dose"), ("c2", "dose"), ("c3", "solo")]
        ).reset_index()
    )
    assert labels["c1"] == "c1"
    assert labels["c2"] == "c2"
    assert labels["c3"] == "solo"

    problem = _Problem(
        _measurements(
            [
                ("obs", "c1", 0.0, 1.0),
                ("obs", "c2", 0.0, 2.0),
            ]
        ),
        _observables({"obs": "lin"}),
        _conditions([("c1", "dose"), ("c2", "dose")]),
    )
    data = PetabDataset.from_problem(problem)
    arms = {g.rpartition(" @ ")[0] for g in data.sample_groups}
    assert arms == {"c1", "c2"}


def test_the_observable_formula_is_carried():
    """The formula is why PEtab needs no proxy: it states the observable over
    the model's own species."""
    problem = _Problem(
        _measurements([("obs", "c1", 0.0, 1.0)]),
        _observables({"obs": "lin"}),
    )
    data = PetabDataset.from_problem(problem)
    assert data.formulas["obs"] == "obs_expr"


def test_a_problem_with_no_measurements_is_refused():
    with pytest.raises(ValueError, match="no measurements"):
        PetabDataset.from_problem(_Problem(_measurements([])))
