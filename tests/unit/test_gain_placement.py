"""Placing a gain from two operating points instead of one."""

import pytest

from hallsim.calibration import OperatingRange
from hallsim.diagnostics import operating_range
from hallsim.models.gain_edge import (
    GainEdge,
    place_gain,
    place_gain_from_ranges,
)
from hallsim.process import Port, PortRole, Process


class Decay(Process):
    """dx/dt = -k x from a known start, so its band is [x_end, x_0]."""

    k: float = 1.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=4.0)}

    def derivative(self, t, state):
        return {"x": -self.k * state["x"]}


class TestTwoPointPlacement:
    def test_it_reproduces_both_ends(self):
        line = place_gain_from_ranges((2.0, 10.0), (5.0, 25.0))

        assert line.offset + line.gain * 2.0 == pytest.approx(5.0)
        assert line.offset + line.gain * 10.0 == pytest.approx(25.0)

    def test_it_places_from_a_source_that_rests_at_zero(self):
        line = place_gain_from_ranges((0.0, 8.0), (3.0, 11.0))

        assert line.gain == pytest.approx(1.0)
        assert line.offset == pytest.approx(3.0)
        # The one-point rule cannot express this at all.
        with pytest.raises(ValueError, match="must be positive"):
            place_gain(0.0, 3.0)

    def test_it_agrees_with_place_gain_when_the_line_goes_through_origin(self):
        two = place_gain_from_ranges((0.0, 5.0), (0.0, 20.0))

        assert two.gain == pytest.approx(place_gain(5.0, 20.0))
        assert two.offset == pytest.approx(0.0)

    def test_a_source_that_does_not_move_determines_no_line(self):
        with pytest.raises(ValueError, match="single level"):
            place_gain_from_ranges((3.0, 3.0), (1.0, 9.0))

    def test_it_accepts_operating_ranges(self):
        line = place_gain_from_ranges(
            OperatingRange(lo=2.0, mean=6.0, hi=10.0),
            OperatingRange(lo=5.0, mean=15.0, hi=25.0),
        )

        assert line.offset + line.gain * 10.0 == pytest.approx(25.0)


class TestOperatingRangeFeedsThePlacement:
    def test_a_solo_run_reports_the_band_the_model_occupies(self):
        band = operating_range(Decay(k=1.0), ["Decay/x"], t_end=5.0)["Decay/x"]

        # x(t) = 4 e^-t over [0, 5]: 4 down to 4 e^-5 = 0.02695.
        assert band.hi == pytest.approx(4.0, rel=1e-4)
        assert band.lo == pytest.approx(0.026954, rel=1e-3)

    def test_an_edge_placed_from_two_bands_spans_them(self):
        source = operating_range(Decay(k=1.0), ["Decay/x"], t_end=5.0)[
            "Decay/x"
        ]
        target = OperatingRange(lo=0.5, mean=1.0, hi=2.5)
        line = place_gain_from_ranges(source, target)
        edge = GainEdge(mode="level", gain=line.gain, offset=line.offset)

        assert float(edge.offset + edge.gain * source.hi) == pytest.approx(2.5)
        assert float(edge.offset + edge.gain * source.lo) == pytest.approx(0.5)

    def test_an_unknown_path_says_what_the_model_stores(self):
        with pytest.raises(KeyError, match="not stored paths"):
            operating_range(Decay(), ["nope"], t_end=1.0)
