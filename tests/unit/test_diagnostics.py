"""Tests for the pre-flight subsystem screening (hallsim.diagnostics)."""

import logging

import pytest

import jax.numpy as jnp

from hallsim.composite import single_process_composite
from hallsim.diagnostics import (
    DEAD_SINK,
    SUITABLE,
    ScreenReport,
    coupling_source_verdict,
    recommend_coupling_source,
    rest_timescale,
    screen_process,
    screen_sensitivity,
)
from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS
from hallsim.models.hill_edge import HillActivationEdge
from demos.models.multi_hallmark import (
    GZ06_PSI_PUBLISHED,
    GZ06_PSI_NAME,
    GZ06_SBML_PATH,
    build_multi_hallmark_composite,
)
from demos.models.sbml import sbml_source
from hallsim.process import Port, PortRole, Process
from hallsim.sbml_import import process_from_sbml

# A vendored (committed) model with both a dead-sink species (`s195`) and
# produced-and-consumed species (`s305`) — exercises the coupling-source
# diagnostic locally, no BioModels download.
WNT_SBML_PATH = sbml_source(
    "sivakumar2011", "wnt_BIOMD0000000397.xml", "BIOMD0000000397"
)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # base flags: exploding / vanishing / tolerance-sensitive gate ok
        ({}, True),
        ({"exploding": True}, False),
        ({"vanishing": True}, False),
        ({"tol_sensitive": True, "tol_delta": 1e2}, False),
        # negative-domain flag (out-of-domain state is not ok)
        ({"negative": True}, False),
        ({"negative": False}, True),
        # 'tunes' half of constituents-first: None/True leave ok, False gates
        ({"tunes": None}, True),
        ({"tunes": True}, True),
        ({"tunes": False}, False),
    ],
)
def test_screenreport_ok_logic(kwargs, expected):
    kwargs.setdefault("tol_delta", 1e-4)
    tol_delta = kwargs.pop("tol_delta")
    report = ScreenReport(
        "m",
        kwargs.pop("exploding", False),
        kwargs.pop("vanishing", False),
        kwargs.pop("tol_sensitive", False),
        1.0,
        tol_delta,
        **kwargs,
    )
    assert report.ok is expected


def test_check_tunability_opt_out_skips_gradient():
    """``check_tunability=False`` skips the jvp probe — ``tunes`` stays None."""
    gz = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_PUBLISHED},
    )
    report = screen_process(gz, t_end=100.0, check_tunability=False)
    assert report.tunes is None


def test_bad_scheduler_kwarg_raises_instead_of_flagging_the_model():
    """A caller error must not come back as a verdict about the model.

    ``screen_process`` takes ``**sched_kwargs``; passing an unknown one
    (e.g. ``scheduler_kwargs=``) used to surface as EXPLODING +
    FRAMEWORK-SUSPECT, indistinguishable from a model that won't integrate.
    """
    gz = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_PUBLISHED},
    )
    with pytest.raises(TypeError, match="scheduler_kwargs"):
        screen_process(
            gz, t_end=10.0, scheduler_kwargs={"auto_stiffness": False}
        )


def test_dead_sink_rejected_as_coupling_source():
    """A produced-but-never-consumed, read-by-nothing species (the importer
    freezes it) must be rejected: coupling from it feeds a frozen constant /
    diverges if unfrozen."""
    p = process_from_sbml(str(WNT_SBML_PATH), name="wnt")
    v = coupling_source_verdict(p, "s195")
    assert v.verdict == DEAD_SINK
    assert v.frozen and v.produced and not v.consumed
    assert not v.ok


def test_produced_and_consumed_is_the_suitable_source():
    """A produced-and-consumed species is bounded and actively turned over,
    so it is a usable coupling source."""
    p = process_from_sbml(str(WNT_SBML_PATH), name="wnt")
    v = coupling_source_verdict(p, "s305")
    assert v.verdict == SUITABLE
    assert v.produced and v.consumed and v.ok


def test_recommendation_flags_dead_sink_and_clock_mismatch():
    """Focused on a dead-sink + a suitable state, the recommendation must
    pick the suitable one, flag the dead sink as unusable, and warn when the
    native clock is far finer than the composite's."""
    p = process_from_sbml(str(WNT_SBML_PATH), name="wnt")
    canon = p.native_time_seconds * 1000.0  # force a >100x clock ratio
    rec = recommend_coupling_source(
        p,
        target_states=("s195", "s305"),
        canonical_time_seconds=canon,
    )
    assert rec.suitable == ("s305",)
    assert any("s195" in n and "unusable" in n for n in rec.notes)
    assert any("finer than the composite clock" in n for n in rec.notes)


class _DeadDecay(Process):
    """Decays to zero whatever its input does — genuinely vanishing."""

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=1.0),
            "drive": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"x": -10.0 * state["x"]}


class _AtRest(Process):
    """Starts exactly on its fixed point: dx/dt = k(1 - x), x(0) = 1."""

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": 2.0 * (1.0 - state["x"])}


class _FarFromRest(Process):
    """Same fixed point, released from x(0) = 100 and relaxing far faster than
    the screen can save: tau = 100 / (1e5 * 99) is ~1e-3 of a save step, so the
    saved trajectory never contains the declared initial condition.

    The analytic tau makes the flag checkable against a number rather than a
    threshold.
    """

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=100.0)}

    def derivative(self, t, state):
        return {"x": 1e5 * (1.0 - state["x"])}


class _Kicked(Process):
    """Autonomous at its IC apart from an explicit pulse live at t=0."""

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0)}

    def derivative(self, t, state):
        return {"x": 1e4 * jnp.where(t < 1e-3, 1.0, 0.0)}


def test_rest_state_is_not_flagged():
    r = screen_process(_AtRest(), t_end=5.0, check_tunability=False)
    assert not r.not_at_rest
    assert r.rest_tau == float("inf")


def test_far_from_rest_is_flagged_but_still_ok():
    r = screen_process(_FarFromRest(), t_end=5.0, check_tunability=False)
    assert r.not_at_rest
    assert r.rest_state.endswith("x")
    assert r.rest_tau == pytest.approx(100 / (1e5 * 99), rel=1e-6)
    # Advisory: a stimulus at t=0 legitimately moves the state, so the flag
    # must describe the screen without failing the model.
    assert r.ok
    assert "not a rest state" in r.detail


def test_live_stimulus_is_named_rather_than_called_disequilibrium():
    r = screen_process(_Kicked(), t_end=5.0, check_tunability=False)
    assert r.not_at_rest
    assert "time-dependent term is live at t=0" in r.detail


def test_rest_timescale_names_the_fastest_state():
    """Two states, different rates — the report must name the quicker one."""

    class _TwoRates(Process):
        def ports_schema(self):
            return {
                "slow": Port(role=PortRole.EVOLVED, default=1.0),
                "fast": Port(role=PortRole.EVOLVED, default=1.0),
            }

        def derivative(self, t, state):
            return {"slow": 0.1, "fast": 50.0}

    comp = single_process_composite(_TwoRates())
    tau, state = rest_timescale(comp)
    assert state.endswith("fast")
    assert tau == pytest.approx(1 / 50, rel=1e-6)


def test_driven_edge_is_undriven_not_vanishing():
    """A coupling edge screened solo sits at its port defaults. That is the
    screen having no driver, not the component having no dynamics — it must
    not fail the constituents-first assertion agents are told to write."""
    edge = HillActivationEdge(k_act=1.0, K=(1.0,), n=(2.0,))
    report = screen_process(edge, t_end=100.0)
    assert report.undriven and not report.vanishing, report
    assert report.ok
    assert report.max_abs > 0.0  # flags measured on the driven run
    assert "unfed INPUT" in report.detail


def test_probe_does_not_rescue_a_genuinely_vanishing_process():
    """The probe reclassifies only when the drive actually wakes the model."""
    report = screen_process(_DeadDecay(), t_end=10.0, check_tunability=False)
    assert report.vanishing and not report.undriven, report
    assert not report.ok


def test_solo_screen_does_not_warn_about_its_own_unfed_inputs(caplog):
    """A lone process has unfed INPUTs by construction; warning about them on
    every screen is noise that trains agents to ignore validation."""
    with caplog.at_level(logging.WARNING):
        screen_process(
            HillActivationEdge(k_act=1.0, K=(1.0,), n=(2.0,)),
            t_end=100.0,
            check_tunability=False,
        )
    assert "Unfed input" not in caplog.text


def test_gz06_flagged_tolerance_sensitive():
    """The Geva-Zatorsky p53 oscillator diverges at loose tolerance — the
    screen must catch that it is solver-dependent (the canonical trap)."""
    gz = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_PUBLISHED},
    )
    report = screen_process(gz, t_end=100.0)  # native hours
    assert report.tolerance_sensitive, report
    assert report.tol_rel_diff > 1.0  # loose vs tight wildly disagree
    assert not report.ok


@pytest.mark.demo
@pytest.mark.slow
def test_screen_sensitivity_finite_gradients_and_live_detection():
    """The sensitivity screen must return finite reverse-mode gradients through
    the three stiff SBML models and detect a responsive reporter. CDKN1A
    (damage→p21) is live to genomic-instability severity in any dose regime;
    which reporters saturate is regime-dependent and not asserted here."""
    comp = build_multi_hallmark_composite(validate=False)
    reports = screen_sensitivity(
        comp,
        MULTI_HALLMARK_REPORTERS,
        ["Genomic Instability"],
        baseline={"Genomic Instability": 1.0},
        t_end=14.0,
        macro_dt=3.5,
    )
    by = {r.reporter: r for r in reports}
    assert all(r.finite for r in reports), reports
    assert by["CDKN1A"].live, by["CDKN1A"]  # damage→p21 stays live
