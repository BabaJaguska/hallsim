"""Whether a handle reaches the vector field, measured rather than assumed."""

import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.handles import (
    Handle,
    ParameterMapping,
    handle_effect,
    inert_handles,
    with_handles,
)
from hallsim.process import Port, PortRole, Process


class Decay(Process):
    rate: float = 0.1
    unread: float = 1.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


class Growth(Process):
    rate: float = 0.05

    def ports_schema(self):
        return {"y": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"y": self.rate * state["y"]}


@pytest.fixture
def composite():
    return Composite(
        processes={"a": Decay(), "b": Growth()},
        topology={"a": {"x": "pool/x"}, "b": {"y": "pool/y"}},
        semantic_validation=False,
    )


def _handle(*mappings):
    return Handle(
        name="h", description="", category="test", mappings=list(mappings)
    )


def test_a_handle_that_moves_a_read_rate_is_not_inert(composite):
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            )
        )
    }
    effect = handle_effect(composite, "h", registry, severity=1.0, t_end=10.0)
    assert not effect.inert and not effect.silent
    assert effect.wrote == (("a.rate", 0.1, 0.2),)
    assert [path for path, _ in effect.moved] == ["pool/x"]


def test_a_write_nothing_reads_is_silent_not_merely_inert(composite):
    """The dangerous case: the name resolves, the value changes, the field does
    not, and a severity sweep returns a flat line that looks like biology."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="unread", floor=1.0, slope=1.0
            )
        )
    }
    effect = handle_effect(composite, "h", registry, severity=1.0, t_end=10.0)
    assert effect.wrote == (("a.unread", 1.0, 2.0),)
    assert effect.inert and effect.silent
    assert "INERT" in str(effect) and "read by nothing" in str(effect)


def test_a_handle_naming_no_process_here_is_reported_not_raised(composite):
    """`apply` already raises a good message when no mapping names any process
    in the composite; this check reports it instead of exploding, because
    diagnosing the registry is its whole job."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="ghost", param_name="rate", floor=1.0, slope=1.0
            )
        )
    }
    effect = handle_effect(composite, "h", registry, severity=1.0, t_end=10.0)
    assert effect.inert and not effect.silent
    assert effect.unmapped and "cannot be applied" in effect.unmapped[0]
    assert "ghost" in effect.unmapped[0]
    assert "MAPPING MISSED" in str(effect)


def test_a_mapping_naming_a_parameter_that_is_absent_is_reported(composite):
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="nope", floor=1.0, slope=1.0
            )
        )
    }
    effect = handle_effect(composite, "h", registry, severity=1.0, t_end=10.0)
    assert effect.inert
    assert effect.unmapped and "cannot be applied" in effect.unmapped[0]


def test_a_partial_miss_is_reported_while_the_rest_still_applies(composite):
    """One mapping naming an absent process does not raise, because another
    names a real one — so it is skipped silently unless something looks."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            ),
            ParameterMapping(
                process_name="ghost", param_name="rate", floor=1.0, slope=1.0
            ),
        )
    }
    effect = handle_effect(composite, "h", registry, severity=1.0, t_end=10.0)
    assert not effect.inert
    assert any("no such process" in miss for miss in effect.unmapped)


def test_severity_zero_is_neutral_and_moves_nothing(composite):
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            )
        )
    }
    assert handle_effect(
        composite, "h", registry, severity=0.0, t_end=10.0
    ).inert


def test_a_handle_spanning_two_processes_moves_both(composite):
    """The shape the framework exists for: one named severity across several
    independently-published models."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            ),
            ParameterMapping(
                process_name="b", param_name="rate", floor=1.0, slope=2.0
            ),
        )
    }
    effect = handle_effect(composite, "h", registry, severity=1.0, t_end=10.0)
    assert {path for path, _ in effect.moved} == {"pool/x", "pool/y"}
    assert len(effect.wrote) == 2


class TimedDrive(Process):
    """A driver whose effect starts after ``switch``, so t=0 cannot see it."""

    level: float = 1.0
    switch: float = 5.0

    def ports_schema(self):
        return {"z": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"z": jnp.where(t >= self.switch, self.level, 0.0)}


def test_a_driver_that_steps_later_needs_a_window_that_reaches_it():
    """One of this check's two false-positive modes: a handle on a driver that
    fires after the window shows nothing, and that is the window's fault."""
    comp = Composite(
        processes={"d": TimedDrive()},
        topology={"d": {"z": "pool/z"}},
        semantic_validation=False,
    )
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="d", param_name="level", floor=1.0, slope=1.0
            )
        )
    }
    assert handle_effect(comp, "h", registry, t_end=1.0).inert
    assert not handle_effect(comp, "h", registry, t_end=9.0).inert


class Accumulate(Process):
    """A pool that starts empty, so a rate on it is invisible where it starts."""

    make: float = 1.0
    clear: float = 0.5

    def ports_schema(self):
        return {"agg": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"agg": self.make - self.clear * state["agg"]}


def test_a_rate_on_a_species_that_starts_at_zero_is_found():
    """The check's other false-positive mode, and the one that mattered: the
    demo's proteostasis handle scales a rate on aggregates that begin at 0, so
    it moves no derivative at the initial state *at any time*, while plainly
    changing the answer once they accumulate. Sampling times cannot find this;
    only sampling the states the model reaches can."""
    import jax.numpy as jnp

    comp = Composite(
        processes={"p": Accumulate()},
        topology={"p": {"agg": "pool/agg"}},
        semantic_validation=False,
    )
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="p", param_name="clear", floor=1.0, slope=-1.0
            )
        )
    }
    # At the initial state the two fields are identical, however long you wait.
    keys = comp.store_keys()
    y0 = comp.initial_state_vec(keys)
    base, _ = comp.build_rhs()
    treated, _ = with_handles(comp, {"h": 1.0}, registry=registry).build_rhs()
    for when in (0.0, 1.0, 100.0):
        assert jnp.allclose(base(when, y0, None), treated(when, y0, None))

    # Sampled along the trajectory, it is plainly live.
    effect = handle_effect(comp, "h", registry, t_end=10.0)
    assert not effect.inert
    assert [path for path, _ in effect.moved] == ["pool/agg"]


def test_inert_handles_names_only_the_dead_ones(composite):
    registry = {
        "live": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            )
        ),
        "dead": _handle(
            ParameterMapping(
                process_name="a", param_name="unread", floor=1.0, slope=1.0
            )
        ),
    }
    found = inert_handles(composite, registry, t_end=10.0)
    assert set(found) == {"dead"}
    assert found["dead"].silent


def test_the_trajectory_measure_sees_reach_the_field_cannot(composite):
    """Six of ten round-4 entries reported this: a handle on an upstream rate
    registers only where that constant appears, so the field comparison
    understates a downstream effect that arrives by integration."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="p", param_name="make", floor=1.0, slope=1.0
            )
        )
    }
    comp = Composite(
        processes={"p": Accumulate()},
        topology={"p": {"agg": "pool/agg"}},
        semantic_validation=False,
    )
    effect = handle_effect(comp, "h", registry, t_end=10.0)
    assert effect.diverged, "the trajectory must show the effect"
    assert effect.processes == ("p",)
    # signed and bounded: more production raises the pool
    path, rel = effect.diverged[0]
    assert path == "pool/agg" and rel > 0 and abs(rel) <= 1.0


def test_it_counts_processes_not_just_paths(composite):
    """The two-deposit gate is stated in deposits, which paths do not give."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            ),
            ParameterMapping(
                process_name="b", param_name="rate", floor=1.0, slope=2.0
            ),
        )
    }
    effect = handle_effect(composite, "h", registry, t_end=10.0)
    # Both share the `pool/` namespace, so splitting the path would say one;
    # the topology says two, which is what the gate means.
    assert effect.processes == ("a", "b")
    assert set(effect.reach) == {"pool/x", "pool/y"}


def test_a_write_nothing_reads_stays_inert_under_both_measures(composite):
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="unread", floor=1.0, slope=1.0
            )
        )
    }
    effect = handle_effect(composite, "h", registry, t_end=10.0)
    assert effect.inert and effect.silent
    assert not effect.diverged and not effect.moved


def test_deposits_counts_imported_models_not_coupling_edges(composite):
    """A composite is mostly not deposits — edges, observers and sources are
    processes too, and counting them inflates the two-deposit gate."""
    registry = {
        "h": _handle(
            ParameterMapping(
                process_name="a", param_name="rate", floor=1.0, slope=1.0
            )
        )
    }
    effect = handle_effect(composite, "h", registry, t_end=10.0)
    # Decay/Growth are hand-written Processes, not imported deposits.
    assert effect.processes == ("a",)
    assert effect.deposits == ()
    assert effect.reach == ("pool/x",), "must not reach the other process"


def test_divergence_survives_two_arms_on_different_save_grids(monkeypatch):
    """The regression this check was rewritten to fix: the scheduler may save
    the two arms on different grids once their dynamics differ, and returning
    empty on a shape mismatch reads as a handle that changes nothing."""
    import numpy as np

    from hallsim import handles as H

    class Run:
        def __init__(self, ts, ys, keys):
            self.ts, self.ys, self.keys = ts, ys, keys

    base = Run(
        np.array([0.0, 1.0, 2.0]), np.array([[1.0], [1.0], [1.0]]), ["p/x"]
    )
    # same window, four saves instead of three, and plainly diverging
    treated = Run(
        np.array([0.0, 0.5, 1.0, 2.0]),
        np.array([[1.0], [1.5], [2.0], [3.0]]),
        ["p/x"],
    )
    monkeypatch.setattr(H, "_trajectory_divergence", H._trajectory_divergence)
    monkeypatch.setattr(
        H.Scheduler if hasattr(H, "Scheduler") else H, "__name__", "H"
    )

    import hallsim.scheduler as sched

    monkeypatch.setattr(sched.Scheduler, "run", lambda self, *a, **k: treated)
    out = H._trajectory_divergence(base, object(), 2.0, 8, 0.0)
    assert out, "a diverging pair on mismatched grids must not read as empty"
    path, rel = out[0]
    assert path == "p/x" and rel > 0
