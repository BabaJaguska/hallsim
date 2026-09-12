"""A loop between groups is named at plan time; a forward edge is not."""

import logging

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class Relax(Process):
    """dx/dt = u - x: reads one input, evolves one state."""

    timescale: float = 1.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=0.0),
            "u": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"x": state["u"] - state["x"]}


def _plan(topology, caplog, **kw):
    comp = Composite(
        processes={"a": Relax(), "b": Relax()},
        topology=topology,
        validate=False,
        semantic_validation=False,
    )
    sched = Scheduler(groups={"a": ["a"], "b": ["b"]}, **kw)
    with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
        plan = sched.plan(comp, (0.0, 1.0), macro_dt=0.5, save_dt=0.5)
    warned = any("cycle" in r.getMessage() for r in caplog.records)
    return plan, warned


LOOP = {"a": {"x": "xa", "u": "xb"}, "b": {"x": "xb", "u": "xa"}}
FORWARD = {"a": {"x": "xa", "u": "drive"}, "b": {"x": "xb", "u": "xa"}}


def test_a_loop_run_with_one_sweep_is_warned_about(caplog):
    plan, warned = _plan(LOOP, caplog)
    assert warned
    assert plan.coupling == "interpolated"  # the forward half of the loop


def test_a_forward_edge_is_not_a_loop(caplog):
    plan, warned = _plan(FORWARD, caplog)
    assert not warned
    assert plan.coupling == "interpolated"


def test_a_second_sweep_silences_the_warning(caplog):
    _, warned = _plan(LOOP, caplog, waveform_sweeps=2)
    assert not warned
