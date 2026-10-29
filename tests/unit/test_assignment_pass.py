"""The RHS carries an assignment only when something in the loop reads it."""

import numpy as np
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
import equinox as eqx


class Algebraic(Process):
    """Evolves x and publishes s = 2x, which its own derivative never reads."""

    timescale: float = eqx.field(static=True, default=1.0)

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=1.0),
            "s": Port(role=PortRole.ASSIGNED, default=0.0),
        }

    def assign(self, t, state):
        return {"s": 2.0 * state["x"]}

    def derivative(self, t, state):
        return {"x": -state["x"]}


class Reader(Process):
    """dz/dt = s - z through an INPUT port."""

    timescale: float = eqx.field(static=True, default=1.0)

    def ports_schema(self):
        return {
            "z": Port(role=PortRole.EVOLVED, default=0.5),
            "s": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"z": state["s"] - state["z"]}


class Sweeper(Process):
    """Reads whatever it is given, by iteration."""

    timescale: float = eqx.field(static=True, default=1.0)

    def ports_schema(self):
        return {
            "z": Port(role=PortRole.EVOLVED, default=0.5),
            "s": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"z": sum(v for k, v in state.items() if k != "z")}


def _comp(procs, topo):
    return Composite(
        processes=procs,
        topology=topo,
        validate=False,
        semantic_validation=False,
    )


class Scaled(Algebraic):
    """``s = gain * x``, with the gain a traced value."""

    gain: float = 2.0

    def assign(self, t, state):
        return {"s": self.gain * state["x"]}


def test_the_read_probe_runs_once_per_structure(monkeypatch):
    """Which assignments the derivatives read is a fact about structure,
    found by tracing each process once; a rebuild with new values keeps
    the finding and still computes with the new values."""
    import hallsim.composite as mod

    calls = []
    real = mod._columns_read

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "_columns_read", counted)
    monkeypatch.setattr(mod, "_ASSIGNMENT_PLANS", {})
    comp = _comp(
        {"a": Scaled(), "r": Reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs, keys = comp.build_rhs()
    assert len(calls) == 1 and len(rhs.assign_procs) == 1
    tripled = eqx.tree_at(lambda c: c.processes["a"].gain, comp, 3.0)
    rhs3, _ = tripled.build_rhs()
    assert len(calls) == 1 and len(rhs3.assign_procs) == 1
    y0 = comp.initial_state_vec(keys)
    z = keys.index("z")
    assert float(rhs(0.0, y0)[z]) == pytest.approx(2.0 - 0.5)
    assert float(rhs3(0.0, y0)[z]) == pytest.approx(3.0 - 0.5)
    alone = _comp({"a": Scaled()}, {"a": {"x": "x", "s": "s"}})
    assert alone.build_rhs()[0].assign_procs == ()
    assert len(calls) == 2


def test_an_assigned_value_reaches_its_reader_without_a_state_write():
    """Inside the right-hand side an algebraic value goes straight to the
    ports that read it; writing it into the state and reading it back on
    every stage cost 1.5× on a two-process fit at 2,000 members."""
    import jax

    comp = _comp(
        {"a": Scaled(), "r": Reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs, keys = comp.build_rhs()
    y0 = comp.initial_state_vec(keys)
    prims = {
        e.primitive.name
        for e in jax.make_jaxpr(lambda y: rhs(0.0, y))(y0).jaxpr.eqns
    }
    assert not prims & {"dynamic_update_slice", "scatter", "gather"}, prims
    assert float(rhs(0.0, y0)[keys.index("z")]) == pytest.approx(2.0 - 0.5)
    # the saved trajectory still carries the value
    ys = comp.materialize_assigned(np.array([0.0]), y0[None])
    assert float(ys[0, keys.index("s")]) == pytest.approx(2.0)


def test_an_assignment_nobody_reads_leaves_the_rhs():
    comp = _comp({"a": Algebraic()}, {"a": {"x": "x", "s": "s"}})
    rhs, keys = comp.build_rhs()
    assert rhs.assign_procs == ()
    y0 = comp.initial_state_vec(keys)
    assert float(rhs(0.0, y0)[keys.index("x")]) == pytest.approx(-1.0)
    # the saved trajectory still gets it
    ys = comp.materialize_assigned(np.array([0.0]), y0[None])
    assert float(ys[0, keys.index("s")]) == pytest.approx(2.0)


@pytest.mark.parametrize("reader", [Reader, Sweeper])
def test_an_assignment_a_derivative_reads_stays(reader):
    comp = _comp(
        {"a": Algebraic(), "r": reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs, keys = comp.build_rhs()
    assert len(rhs.assign_procs) == 1
    y0 = comp.initial_state_vec(keys)
    dz = float(rhs(0.0, y0)[keys.index("z")])
    expect = 2.0 - 0.5 if reader is Reader else 2.0
    assert dz == pytest.approx(expect)


def test_a_partial_rhs_keeps_only_what_its_group_reads():
    comp = _comp(
        {"a": Algebraic(), "r": Reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs_a, _ = comp.build_rhs(["a"])
    assert rhs_a.assign_procs == ()


def test_a_group_recomputes_an_assignment_made_in_another_group():
    """An ASSIGNED path is algebraic, so a reader split into its own group
    must recompute it rather than fall through to the persisted slot, which
    no integration writes and which holds the port default."""
    comp = _comp(
        {"a": Algebraic(), "r": Reader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    rhs_r, keys = comp.build_rhs(["r"])
    assert len(rhs_r.assign_procs) == 1, "the writer is in the other group"
    y0 = comp.initial_state_vec(keys)
    # s = 2x = 2 at the default x, so dz/dt = s - z = 2 - 0.5.
    dz = float(rhs_r(0.0, y0)[keys.index("z")])
    assert dz == pytest.approx(1.5)


def test_a_timescale_split_does_not_zero_a_cross_group_assignment():
    """`auto_groups` splits on timescale, so the defect was reachable from
    the default path: the reader saw the port default and integrated zero."""

    class SlowReader(Reader):
        timescale: float = eqx.field(static=True, default=1e4)

    comp = _comp(
        {"a": Algebraic(), "r": SlowReader()},
        {"a": {"x": "x", "s": "s"}, "r": {"z": "z", "s": "s"}},
    )
    groups = comp.auto_groups()
    assert len(groups) > 1, "the timescales must actually split"
    for members in groups.values():
        if "r" not in members:
            continue
        rhs, keys = comp.build_rhs(list(members))
        y0 = comp.initial_state_vec(keys)
        dz = float(rhs(0.0, y0)[keys.index("z")])
        assert dz == pytest.approx(1.5)
