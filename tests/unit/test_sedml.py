"""A deposit's authored horizon, and the generated stub that is not one."""

import pytest

from hallsim.intake import DEFAULT_TRIAGE_T_END, triage_horizon
from hallsim.sedml import authored_horizon, horizon_in

SEDML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4" level="1"'
    ' version="4">\n'
    "  <listOfSimulations>\n{courses}  </listOfSimulations>\n"
    "</sedML>\n"
)


def _course(sim_id, end, start="0"):
    return (
        f'    <uniformTimeCourse id="{sim_id}" initialTime="0"'
        f' outputStartTime="{start}" outputEndTime="{end}"'
        ' numberOfSteps="1000"/>\n'
    )


def _doc(*courses):
    return SEDML.format(courses="".join(courses))


def test_an_authored_horizon_is_read():
    assert horizon_in(_doc(_course("sim_14_days", "1209600"))) == 1209600.0


def test_the_repositorys_generated_stub_is_not_a_horizon():
    """BioModels writes this for deposits that shipped no SED-ML, and its 10 is
    the caller's number the horizon is meant to replace."""
    assert horizon_in(_doc(_course("auto_ten_seconds", "10"))) is None


def test_the_longest_authored_course_wins():
    """Several courses are several protocols, and the longest contains them."""
    doc = _doc(
        _course("short", "100"), _course("long", "5000"), _course("mid", "900")
    )
    assert horizon_in(doc) == 5000.0


def test_a_generated_course_beside_an_authored_one_is_ignored():
    doc = _doc(_course("auto_ten_seconds", "10"), _course("real", "864000"))
    assert horizon_in(doc) == 864000.0


def test_a_course_ending_where_it_starts_is_not_a_horizon():
    assert horizon_in(_doc(_course("degenerate", "50", start="50"))) is None


@pytest.mark.parametrize(
    "text",
    ["", "not xml at all", "<sedML><listOfSimulations/></sedML>"],
)
def test_an_unusable_document_states_no_horizon(text):
    assert horizon_in(text) is None


def test_a_malformed_end_time_is_not_a_horizon():
    doc = _doc(_course("sim", "not-a-number"))
    assert horizon_in(doc) is None


def test_a_file_and_a_directory_both_resolve(tmp_path):
    (tmp_path / "a.sedml").write_text(_doc(_course("sim", "777")))
    assert authored_horizon(tmp_path / "a.sedml") == 777.0
    assert authored_horizon(tmp_path) == 777.0


def test_a_directory_takes_the_longest_across_files(tmp_path):
    (tmp_path / "a.sedml").write_text(_doc(_course("one", "10000")))
    (tmp_path / "b.sedml").write_text(_doc(_course("two", "30")))
    assert authored_horizon(tmp_path) == 10000.0


def test_nothing_cached_states_no_horizon(tmp_path):
    assert authored_horizon(tmp_path) is None
    assert authored_horizon("BIOMD0000000000") is None


class TestTriageHorizon:
    """What the screen runs over when the caller names no window."""

    def test_an_authored_horizon_is_preferred_over_the_default(self, tmp_path):
        (tmp_path / "m.sedml").write_text(_doc(_course("sim", "144000")))
        assert triage_horizon(tmp_path) == 144000.0

    def test_a_deposit_stating_nothing_falls_back(self, tmp_path):
        assert triage_horizon(tmp_path) == DEFAULT_TRIAGE_T_END

    def test_the_stub_falls_back_rather_than_asserting_its_ten(self, tmp_path):
        """Both give 10, and they must not be confused: one is the model's
        claim and the other is ours."""
        (tmp_path / "m.sedml").write_text(_doc(_course("auto_x", "10")))
        assert authored_horizon(tmp_path) is None
        assert triage_horizon(tmp_path) == DEFAULT_TRIAGE_T_END
