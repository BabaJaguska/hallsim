"""Tests for multi-timescale / hybrid discrete-continuous composites.

Covers:
- ProcessKind enum and kind field
- Timescale declaration
- LATCHED port role
- DISCRETE process with update()
- EVENT process with condition() / handler()
- Topology validation: kind/role checks
- CONTINUOUS-only composites (the simplest case)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.store import validate_topology


# ═══════════════════════════════════════════════════════════════════════════
# Toy processes for testing
# ═══════════════════════════════════════════════════════════════════════════


class ContinuousDecay(Process):
    """Plain continuous process — first-order decay."""

    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


class FastKinetics(Process):
    """Continuous process with explicit timescale."""

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 1.0
    rate: float = 0.1

    def ports_schema(self):
        return {"ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"ros": jnp.asarray(self.rate)}


class SlowDrift(Process):
    """Continuous process with slow timescale."""

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 86400.0 * 30  # ~month
    rate: float = 1e-7

    def ports_schema(self):
        return {
            "methylation": Port(
                role=PortRole.EVOLVED, default=0.0, units="dimensionless"
            ),
            "ros": Port(role=PortRole.INPUT, default=0.0, units="uM"),
        }

    def derivative(self, t, state):
        return {"methylation": self.rate * (1.0 + state["ros"])}


class DivisionCheck(Process):
    """Discrete process: checks if cell should divide."""

    kind: ProcessKind = ProcessKind.DISCRETE
    dt_step: float = 86400.0  # once per day

    def ports_schema(self):
        return {
            "cell_count": Port(
                role=PortRole.LATCHED, default=1.0, units="cells"
            ),
            "damage": Port(
                role=PortRole.INPUT, default=0.0, units="dimensionless"
            ),
        }

    def update(self, t, state):
        can_divide = state["damage"] < 0.8
        return {"cell_count": jnp.where(can_divide, state["cell_count"], 0.0)}


class SenescenceEntry(Process):
    """Event process: triggers senescence on high p53."""

    kind: ProcessKind = ProcessKind.EVENT

    def ports_schema(self):
        return {
            "p53": Port(role=PortRole.INPUT, default=0.0, units="uM"),
            "senescent": Port(
                role=PortRole.LATCHED, default=0.0, units="dimensionless"
            ),
        }

    def condition(self, t, state):
        return state["p53"] > 0.9

    def handler(self, t, state):
        return {"senescent": 1.0 - state["senescent"]}


# ═══════════════════════════════════════════════════════════════════════════
# Bad processes for validation testing
# ═══════════════════════════════════════════════════════════════════════════


class BadContinuousWithLatched(Process):
    """Invalid: continuous process trying to write a LATCHED port."""

    kind: ProcessKind = ProcessKind.CONTINUOUS

    def ports_schema(self):
        return {"x": Port(role=PortRole.LATCHED, default=0.0)}

    def derivative(self, t, state):
        return {"x": jnp.asarray(0.1)}


class BadDiscreteWithEvolved(Process):
    """Invalid: discrete process trying to write an EVOLVED port."""

    kind: ProcessKind = ProcessKind.DISCRETE
    dt_step: float = 100.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

    def update(self, t, state):
        return {"x": jnp.asarray(1.0)}


class BadEventWithExclusive(Process):
    """Invalid: event process trying to write an EXCLUSIVE port."""

    kind: ProcessKind = ProcessKind.EVENT

    def ports_schema(self):
        return {"x": Port(role=PortRole.EXCLUSIVE, default=0.0)}

    def condition(self, t, state):
        return True

    def handler(self, t, state):
        return {"x": jnp.asarray(1.0)}


class BadDiscreteNoDtStep(Process):
    """Invalid: discrete process without dt_step."""

    kind: ProcessKind = ProcessKind.DISCRETE

    def ports_schema(self):
        return {"x": Port(role=PortRole.LATCHED, default=0.0)}

    def update(self, t, state):
        return {"x": jnp.asarray(1.0)}


class LatchedEvolvedConflict(Process):
    """A continuous process that writes EVOLVED to a path that another
    process has as LATCHED — used to test cross-process conflicts."""

    kind: ProcessKind = ProcessKind.CONTINUOUS

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

    def derivative(self, t, state):
        return {"x": jnp.asarray(0.1)}


# ═══════════════════════════════════════════════════════════════════════════
# Tests: ProcessKind and basic properties
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessKind:
    def test_default_kind_is_continuous(self):
        proc = ContinuousDecay()
        assert proc.kind == ProcessKind.CONTINUOUS

    def test_discrete_kind(self):
        proc = DivisionCheck()
        assert proc.kind == ProcessKind.DISCRETE

    def test_event_kind(self):
        proc = SenescenceEntry()
        assert proc.kind == ProcessKind.EVENT

    def test_timescale_default_none(self):
        proc = ContinuousDecay()
        assert proc.timescale is None

    def test_timescale_declared(self):
        proc = FastKinetics()
        assert proc.timescale == 1.0

    def test_dt_step_default_none(self):
        proc = ContinuousDecay()
        assert proc.dt_step is None

    def test_dt_step_declared(self):
        proc = DivisionCheck()
        assert proc.dt_step == 86400.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: LATCHED port role
# ═══════════════════════════════════════════════════════════════════════════


class TestLatchedPort:
    def test_latched_port_exists(self):
        proc = DivisionCheck()
        schema = proc.ports_schema()
        assert schema["cell_count"].role == PortRole.LATCHED

    def test_latched_ports_helper(self):
        proc = DivisionCheck()
        latched = proc.ports_with_role(PortRole.LATCHED)
        assert "cell_count" in latched
        assert "damage" not in latched

    def test_latched_in_output_port_names(self):
        proc = DivisionCheck()
        output = proc.output_port_names()
        assert "cell_count" in output
        assert "damage" not in output


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Process interface methods
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessInterfaces:
    def test_continuous_derivative(self):
        proc = ContinuousDecay()
        result = proc.derivative(0.0, {"x": jnp.array(2.0)})
        assert jnp.isclose(result["x"], -0.2)

    def test_discrete_update(self):
        proc = DivisionCheck()
        result = proc.update(
            0.0,
            {
                "cell_count": jnp.array(1.0),
                "damage": jnp.array(0.5),
            },
        )
        assert "cell_count" in result
        # damage < 0.8 so can_divide = True, delta = cell_count = 1.0
        assert jnp.isclose(result["cell_count"], 1.0)

    def test_discrete_update_high_damage(self):
        proc = DivisionCheck()
        result = proc.update(
            0.0,
            {
                "cell_count": jnp.array(1.0),
                "damage": jnp.array(0.9),
            },
        )
        # damage >= 0.8 so can_divide = False, delta = 0.0
        assert jnp.isclose(result["cell_count"], 0.0)

    def test_event_condition_false(self):
        proc = SenescenceEntry()
        assert not proc.condition(
            0.0,
            {
                "p53": jnp.array(0.5),
                "senescent": jnp.array(0.0),
            },
        )

    def test_event_condition_true(self):
        proc = SenescenceEntry()
        assert proc.condition(
            0.0,
            {
                "p53": jnp.array(0.95),
                "senescent": jnp.array(0.0),
            },
        )

    def test_event_handler(self):
        proc = SenescenceEntry()
        result = proc.handler(
            0.0,
            {
                "p53": jnp.array(0.95),
                "senescent": jnp.array(0.0),
            },
        )
        assert jnp.isclose(result["senescent"], 1.0)

    def test_continuous_raises_on_update(self):
        proc = ContinuousDecay()
        with pytest.raises(NotImplementedError):
            proc.update(0.0, {"x": jnp.array(1.0)})

    def test_continuous_raises_on_condition(self):
        proc = ContinuousDecay()
        with pytest.raises(NotImplementedError):
            proc.condition(0.0, {"x": jnp.array(1.0)})

    def test_discrete_raises_on_derivative(self):
        proc = DivisionCheck()
        with pytest.raises(NotImplementedError):
            proc.derivative(
                0.0, {"cell_count": jnp.array(1.0), "damage": jnp.array(0.0)}
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_metadata_contract(self):
        div_meta = DivisionCheck().metadata()
        assert div_meta["kind"] == "discrete"
        assert div_meta["dt_step"] == 86400.0
        assert FastKinetics().metadata()["timescale"] == 1.0
        assert ContinuousDecay().metadata()["kind"] == "continuous"

    def test_metadata_omits_none_timescale(self):
        proc = ContinuousDecay()
        meta = proc.metadata()
        assert "timescale" not in meta


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Topology validation — kind/role compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestKindRoleValidation:
    def test_continuous_with_latched_is_error(self):
        errors = validate_topology(
            {"bad": BadContinuousWithLatched()},
            {"bad": {"x": "pool/x"}},
        )
        assert any("CONTINUOUS" in e and "LATCHED" in e for e in errors)

    def test_discrete_with_evolved_is_error(self):
        errors = validate_topology(
            {"bad": BadDiscreteWithEvolved()},
            {"bad": {"x": "pool/x"}},
        )
        assert any("DISCRETE" in e and "EVOLVED" in e for e in errors)

    def test_event_with_exclusive_is_error(self):
        errors = validate_topology(
            {"bad": BadEventWithExclusive()},
            {"bad": {"x": "pool/x"}},
        )
        assert any("EVENT" in e and "EXCLUSIVE" in e for e in errors)

    def test_discrete_without_dt_step_is_error(self):
        errors = validate_topology(
            {"bad": BadDiscreteNoDtStep()},
            {"bad": {"x": "pool/x"}},
        )
        assert any("dt_step" in e for e in errors)

    def test_latched_evolved_coexist_on_same_path(self):
        """LATCHED + EVOLVED on the same store path is *allowed*: the
        CONTINUOUS process owns the derivative, the EVENT/DISCRETE
        process applies a one-shot scatter-add delta when fired (e.g.
        a kick at t=10). EXCLUSIVE constrains derivative ownership, not
        state mutation by an event. See ``KickEvent`` for the canonical
        use case and store.py's ``validate_topology`` for the rationale."""
        errors = validate_topology(
            {
                "discrete": DivisionCheck(),
                "continuous": LatchedEvolvedConflict(),
            },
            {
                "discrete": {"cell_count": "pool/x", "damage": "pool/damage"},
                "continuous": {"x": "pool/x"},
            },
        )
        # Should validate without coexistence errors; any errors that
        # remain must be unrelated to the LATCHED ↔ EVOLVED pairing.
        assert not any(
            "LATCHED" in e and "EVOLVED" in e for e in errors
        ), errors

    def test_mixed_valid_composition(self):
        """A valid mix of continuous + discrete + event processes."""
        errors = validate_topology(
            {
                "fast": FastKinetics(),
                "slow": SlowDrift(),
                "div": DivisionCheck(),
                "sen": SenescenceEntry(),
            },
            {
                "fast": {"ros": "cell/ROS"},
                "slow": {"methylation": "cell/methylation", "ros": "cell/ROS"},
                "div": {
                    "cell_count": "pop/count",
                    "damage": "cell/methylation",
                },
                "sen": {"p53": "cell/p53", "senescent": "cell/senescent"},
            },
        )
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Composite with mixed process kinds
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeWithMixedKinds:
    def test_composite_builds_with_mixed_kinds(self):
        """Composite should accept a mix of kinds without raising."""
        composite = Composite(
            processes={
                "fast": FastKinetics(),
                "div": DivisionCheck(),
                "sen": SenescenceEntry(),
            },
            topology={
                "fast": {"ros": "cell/ROS"},
                "div": {"cell_count": "pop/count", "damage": "cell/damage"},
                "sen": {"p53": "cell/p53", "senescent": "cell/senescent"},
            },
        )
        assert "fast" in composite.processes
        assert "div" in composite.processes
        assert "sen" in composite.processes

    def test_composite_initial_state_includes_latched(self):
        """LATCHED ports should appear in the initial state."""
        composite = Composite(
            processes={
                "div": DivisionCheck(),
                "sen": SenescenceEntry(),
            },
            topology={
                "div": {"cell_count": "pop/count", "damage": "cell/damage"},
                "sen": {"p53": "cell/p53", "senescent": "cell/senescent"},
            },
        )
        state = composite.initial_state()
        assert "pop/count" in state
        assert "cell/senescent" in state
        assert jnp.isclose(state["pop/count"], 1.0)
        assert jnp.isclose(state["cell/senescent"], 0.0)

    def test_composite_rejects_invalid_kind_role(self):
        """Composite should reject continuous process with LATCHED port."""
        with pytest.raises(ValueError, match="LATCHED"):
            Composite(
                processes={"bad": BadContinuousWithLatched()},
                topology={"bad": {"x": "pool/x"}},
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: continuous-only composites (the simplest case)
# ═══════════════════════════════════════════════════════════════════════════


class TestContinuousOnly:
    def test_rhs_solves_continuous_only(self):
        """build_rhs() on a CONTINUOUS-only composite."""
        composite = Composite(
            processes={"decay": ContinuousDecay()},
            topology={"decay": {"x": "pool/x"}},
        )
        rhs, keys = composite.build_rhs()
        y_vec = composite.flatten(composite.initial_state(), keys)
        dy = composite.unflatten(rhs(0.0, y_vec), keys)
        assert "pool/x" in dy
        assert jnp.isclose(dy["pool/x"], -0.1)  # -rate * x = -0.1 * 1.0

    def test_evolved_ports_helper(self):
        proc = ContinuousDecay()
        assert "x" in proc.ports_with_role(PortRole.EVOLVED)

    def test_output_port_names_for_continuous(self):
        proc = ContinuousDecay()
        assert proc.output_port_names() == {"x"}


# ═══════════════════════════════════════════════════════════════════════════
# Additional toy processes for Scheduler testing
# ═══════════════════════════════════════════════════════════════════════════


class ConstantProduction(Process):
    """Constant production: dx/dt = +rate."""

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 1.0
    rate: float = 1.0

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"x": jnp.asarray(self.rate)}


class SimpleDecay(Process):
    """First-order decay: dx/dt = -rate * x."""

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 1.0
    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=10.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}


class SlowGrowth(Process):
    """Slow linear growth on a different variable."""

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = 1000.0
    rate: float = 0.001

    def ports_schema(self):
        return {
            "y": Port(
                role=PortRole.EVOLVED, default=0.0, units="dimensionless"
            )
        }

    def derivative(self, t, state):
        return {"y": jnp.asarray(self.rate)}


class ThresholdLatch(Process):
    """Event: sets flag when x exceeds threshold."""

    kind: ProcessKind = ProcessKind.EVENT
    threshold: float = 5.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.INPUT, default=0.0, units="uM"),
            "flag": Port(
                role=PortRole.LATCHED, default=0.0, units="dimensionless"
            ),
        }

    def condition(self, t, state):
        return state["x"] > self.threshold

    def handler(self, t, state):
        return {"flag": 1.0 - state["flag"]}


class PeriodicCounter(Process):
    """Discrete: increments counter every dt_step."""

    kind: ProcessKind = ProcessKind.DISCRETE
    dt_step: float = 10.0

    def ports_schema(self):
        return {
            "count": Port(
                role=PortRole.LATCHED, default=0.0, units="dimensionless"
            ),
        }

    def update(self, t, state):
        return {"count": jnp.asarray(1.0)}


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Composite extensions (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeFiltering:
    def test_continuous_processes(self):
        composite = Composite(
            processes={
                "fast": FastKinetics(),
                "div": DivisionCheck(),
                "sen": SenescenceEntry(),
            },
            topology={
                "fast": {"ros": "cell/ROS"},
                "div": {"cell_count": "pop/count", "damage": "cell/damage"},
                "sen": {"p53": "cell/p53", "senescent": "cell/senescent"},
            },
        )
        cont = composite.continuous_processes()
        assert set(cont.keys()) == {"fast"}

    def test_discrete_processes(self):
        composite = Composite(
            processes={
                "fast": FastKinetics(),
                "div": DivisionCheck(),
            },
            topology={
                "fast": {"ros": "cell/ROS"},
                "div": {"cell_count": "pop/count", "damage": "cell/damage"},
            },
        )
        disc = composite.discrete_processes()
        assert set(disc.keys()) == {"div"}

    def test_event_processes(self):
        composite = Composite(
            processes={
                "fast": FastKinetics(),
                "sen": SenescenceEntry(),
            },
            topology={
                "fast": {"ros": "cell/ROS"},
                "sen": {"p53": "cell/p53", "senescent": "cell/senescent"},
            },
        )
        evts = composite.event_processes()
        assert set(evts.keys()) == {"sen"}


class TestAutoGroups:
    def test_single_timescale_one_group(self):
        composite = Composite(
            processes={
                "a": FastKinetics(timescale=1.0),
                "b": FastKinetics(timescale=10.0),
            },
            topology={"a": {"ros": "cell/ROS_a"}, "b": {"ros": "cell/ROS_b"}},
        )
        groups = composite.auto_groups()
        # Within 100x → same group
        assert len(groups) == 1
        group_members = list(groups.values())[0]
        assert set(group_members) == {"a", "b"}

    def test_two_timescales_two_groups(self):
        composite = Composite(
            processes={
                "fast": FastKinetics(timescale=1.0),
                "slow": SlowDrift(timescale=86400.0 * 30),
            },
            topology={
                "fast": {"ros": "cell/ROS"},
                "slow": {"methylation": "cell/meth", "ros": "cell/ROS"},
            },
        )
        groups = composite.auto_groups()
        assert len(groups) == 2
        # Verify processes are in different groups
        all_members = [set(v) for v in groups.values()]
        assert {"fast"} in all_members
        assert {"slow"} in all_members

    def test_no_timescale_default_group(self):
        composite = Composite(
            processes={"a": ContinuousDecay(), "b": ContinuousDecay()},
            topology={"a": {"x": "pool/a"}, "b": {"x": "pool/b"}},
        )
        groups = composite.auto_groups()
        assert "default" in groups
        assert set(groups["default"]) == {"a", "b"}

    def test_mixed_declared_undeclared(self):
        composite = Composite(
            processes={
                "fast": FastKinetics(timescale=1.0),
                "plain": ContinuousDecay(),  # no timescale
            },
            topology={
                "fast": {"ros": "cell/ROS"},
                "plain": {"x": "pool/x"},
            },
        )
        groups = composite.auto_groups()
        # fast → group_0, plain → default
        all_members = []
        for v in groups.values():
            all_members.extend(v)
        assert "fast" in all_members
        assert "plain" in all_members

    def test_no_continuous_empty_groups(self):
        composite = Composite(
            processes={"div": DivisionCheck()},
            topology={
                "div": {"cell_count": "pop/count", "damage": "cell/damage"}
            },
        )
        groups = composite.auto_groups()
        assert groups == {}


class TestBuildGroupRHS:
    def test_group_rhs_only_includes_named_processes(self):
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "decay": SimpleDecay(rate=0.1),
            },
            topology={
                "prod": {"x": "pool/x"},
                "decay": {"x": "pool/x"},
            },
        )
        # Build RHS for only production
        rhs, keys = composite.build_rhs(["prod"])
        y_vec = composite.flatten({"pool/x": jnp.array(10.0)}, keys)
        dy = composite.unflatten(rhs(0.0, y_vec), keys)
        # Only production contributes, not decay
        assert jnp.isclose(dy["pool/x"], 1.0)

    def test_group_rhs_matches_full_rhs_for_all(self):
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "decay": SimpleDecay(rate=0.1),
            },
            topology={
                "prod": {"x": "pool/x"},
                "decay": {"x": "pool/x"},
            },
        )
        full_rhs, keys = composite.build_rhs()
        group_rhs, _ = composite.build_rhs(["prod", "decay"])
        y_vec = composite.flatten({"pool/x": jnp.array(5.0)}, keys)
        dy_full = composite.unflatten(full_rhs(0.0, y_vec), keys)
        dy_group = composite.unflatten(group_rhs(0.0, y_vec), keys)
        assert jnp.isclose(dy_full["pool/x"], dy_group["pool/x"])


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Scheduler (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════

from hallsim.scheduler import Scheduler, SchedulerResult


class TestSchedulerBasic:
    def test_single_continuous_process(self):
        """Scheduler degrades to one Diffrax solve for a single continuous group."""
        composite = Composite(
            processes={"decay": SimpleDecay(rate=0.1)},
            topology={"decay": {"x": "pool/x"}},
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 10.0), macro_dt=1.0)
        assert result.ts[0] == 0.0
        assert result.ts[-1] == 10.0
        # Exponential decay: x(t) = 10 * exp(-0.1t)
        # At t=10: x ≈ 10 * exp(-1) ≈ 3.68
        final_x = result.get("pool/x")[-1]
        expected = 10.0 * jnp.exp(-0.1 * 10.0)
        assert jnp.allclose(final_x, expected, atol=0.05)

    def test_two_continuous_additive(self):
        """Two continuous processes writing to same path (additive)."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "decay": SimpleDecay(rate=0.1),
            },
            topology={
                "prod": {"x": "pool/x"},
                "decay": {"x": "pool/x"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 50.0), macro_dt=1.0)
        # Steady state: dx/dt = rate - decay*x = 0 → x = rate/decay = 10
        final_x = result.get("pool/x")[-1]
        assert jnp.allclose(final_x, 10.0, atol=0.5)

    def test_result_structure(self):
        composite = Composite(
            processes={"decay": SimpleDecay()},
            topology={"decay": {"x": "pool/x"}},
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 5.0), macro_dt=1.0)
        assert isinstance(result, SchedulerResult)
        assert isinstance(result.ts, jnp.ndarray)
        assert isinstance(result.ys, jnp.ndarray)
        assert isinstance(result.keys, list)
        assert isinstance(result.events, list)
        assert "pool/x" in result
        assert len(result.ts) == len(result.get("pool/x"))


class TestSchedulerInitValidation:
    def test_strang_plus_interpolated_rejected(self):
        """Strang's reverse pass can't consume yet-to-be-produced interpolants."""
        with pytest.raises(ValueError, match="incompatible"):
            Scheduler(splitting="strang", coupling_mode="interpolated")

    def test_strang_plus_frozen_ok(self):
        Scheduler(splitting="strang", coupling_mode="frozen")

    def test_lie_plus_interpolated_ok(self):
        Scheduler(splitting="lie", coupling_mode="interpolated")


class TestInterpolatedCouplingAccuracy:
    """Interpolated coupling feeds a fixed-size sample of the previous
    group's trajectory to the next group, so a downstream integrator sees
    the upstream variable *evolving* across the macro step instead of frozen.
    On a genuine cross-group coupling this is far more accurate than frozen
    and compiles through the scan fast path."""

    def _coupled(self):
        class Decay(Process):
            def ports_schema(self):
                return {"A": Port(role=PortRole.EVOLVED, default=1.0)}

            def derivative(self, t, state):
                return {"A": -state["A"]}

        class Integrate(Process):
            def ports_schema(self):
                return {
                    "A": Port(role=PortRole.INPUT, default=1.0),
                    "B": Port(role=PortRole.EVOLVED, default=0.0),
                }

            def derivative(self, t, state):
                return {"B": state["A"]}

        comp = Composite(
            processes={"decay": Decay(), "integ": Integrate()},
            topology={
                "decay": {"A": "p/A"},
                "integ": {"A": "p/A", "B": "p/B"},
            },
        )
        return comp, {"fast": ["decay"], "slow": ["integ"]}

    def test_interpolated_beats_frozen_on_cross_group_coupling(self):
        # dA/dt=-A, A0=1 -> A=exp(-t); dB/dt=A, B0=0 -> B(T)=1-exp(-T).
        comp, groups = self._coupled()
        T = 4.0
        exact = 1.0 - float(jnp.exp(-jnp.asarray(T)))
        bi = comp.store_keys().index("p/B")

        def run_B(mode, mdt):
            s = Scheduler(coupling_mode=mode, groups=groups)
            return float(s.run(comp, (0.0, T), macro_dt=mdt).ys[-1, bi])

        frozen_err = abs(run_B("frozen", 1.0) - exact)
        interp_err = abs(run_B("interpolated", 1.0) - exact)
        # Frozen (left-rectangle) is badly off at this step; interpolated
        # (trapezoid over the sample) matches the analytic value.
        assert frozen_err > 0.3
        assert interp_err < 1e-2
        assert interp_err < frozen_err / 100

    def test_auto_routes_interpolated_on_forward_edge(self):
        # A forward cross-group edge (fast group writes A, slow group reads
        # it) is exactly the case interpolated helps, so auto must pick it.
        comp, groups = self._coupled()
        s = Scheduler(coupling_mode="auto", groups=groups)
        assert s._effective_coupling(comp, groups, comp.store_keys()) == (
            "interpolated"
        )
        bi = comp.store_keys().index("p/B")
        auto_B = float(s.run(comp, (0.0, 4.0), macro_dt=1.0).ys[-1, bi])
        exact = 1.0 - float(jnp.exp(-jnp.asarray(4.0)))
        assert abs(auto_B - exact) < 1e-2

    def test_auto_routes_frozen_without_forward_edge(self):
        # No cross-group coupling -> interpolated would only add cost, so
        # auto stays frozen.
        comp, groups = self._coupled()
        # Swap the solve order so the reader group runs first: the A->B edge
        # is now backward, which interpolated cannot help.
        rev = {"slow": groups["slow"], "fast": groups["fast"]}
        s = Scheduler(coupling_mode="auto", groups=rev)
        assert s._effective_coupling(comp, rev, comp.store_keys()) == "frozen"

    def test_auto_coupling_traces_under_jit_vmap(self):
        # Auto-detection must use only static structure: population runs wrap
        # run() in jax.jit(jax.vmap(...)) with a traced composite, so any
        # concrete int()/set op on a jnp array would raise ConcretizationError.
        _, groups = self._coupled()

        class DecayR(Process):
            rate: float = 1.0

            def ports_schema(self):
                return {"A": Port(role=PortRole.EVOLVED, default=1.0)}

            def derivative(self, t, state):
                return {"A": -self.rate * state["A"]}

        class Integrate(Process):
            def ports_schema(self):
                return {
                    "A": Port(role=PortRole.INPUT, default=1.0),
                    "B": Port(role=PortRole.EVOLVED, default=0.0),
                }

            def derivative(self, t, state):
                return {"B": state["A"]}

        def run(rate):
            comp = Composite(
                processes={"decay": DecayR(rate=rate), "integ": Integrate()},
                topology={
                    "decay": {"A": "p/A"},
                    "integ": {"A": "p/A", "B": "p/B"},
                },
            )
            return (
                Scheduler(groups=groups)
                .run(comp, (0.0, 4.0), macro_dt=1.0)
                .ys[-1]
            )

        out = jax.jit(jax.vmap(run))(jnp.array([1.0, 1.3, 0.8]))
        assert out.shape == (3, 2)
        assert bool(jnp.isfinite(out).all())


class TestSchedulerDiscrete:
    def test_discrete_counter(self):
        """Discrete process increments counter at fixed intervals."""
        composite = Composite(
            processes={"counter": PeriodicCounter(dt_step=10.0)},
            topology={"counter": {"count": "state/count"}},
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 50.0), macro_dt=5.0)
        # dt_step=10, t_span=50 → fires at t=10,20,30,40,50 → count=5
        final_count = result.get("state/count")[-1]
        assert jnp.isclose(final_count, 5.0)

    def test_discrete_with_continuous(self):
        """Mixed continuous + discrete processes."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "counter": PeriodicCounter(dt_step=10.0),
            },
            topology={
                "prod": {"x": "pool/x"},
                "counter": {"count": "state/count"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 30.0), macro_dt=5.0)
        # x grows linearly: x(30) ≈ 30
        assert jnp.allclose(result.get("pool/x")[-1], 30.0, atol=0.5)
        # counter fires at t=10,20,30 → count=3
        assert jnp.isclose(result.get("state/count")[-1], 3.0)


class TestSchedulerEvent:
    def test_event_fires_on_threshold(self):
        """Event process fires when condition becomes True."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "latch": ThresholdLatch(threshold=5.0),
            },
            topology={
                "prod": {"x": "pool/x"},
                "latch": {"x": "pool/x", "flag": "state/flag"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 10.0), macro_dt=1.0)
        # x grows at rate 1.0, crosses 5.0 around t=5
        # flag should be 1.0 after that
        final_flag = result.get("state/flag")[-1]
        assert jnp.isclose(final_flag, 1.0)
        # Event should be logged
        assert len(result.events) >= 1
        assert result.events[0].process == "latch"

    def test_event_fires_only_once(self):
        """Event fires on False→True transition, not repeatedly."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "latch": ThresholdLatch(threshold=2.0),
            },
            topology={
                "prod": {"x": "pool/x"},
                "latch": {"x": "pool/x", "flag": "state/flag"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 10.0), macro_dt=1.0)
        # Should fire once when x crosses 2.0
        latch_events = [e for e in result.events if e.process == "latch"]
        assert len(latch_events) == 1

    def test_no_event_when_condition_never_met(self):
        """Event doesn't fire if condition is never True."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=0.1),
                "latch": ThresholdLatch(threshold=100.0),
            },
            topology={
                "prod": {"x": "pool/x"},
                "latch": {"x": "pool/x", "flag": "state/flag"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 10.0), macro_dt=1.0)
        assert len(result.events) == 0
        assert jnp.isclose(result.get("state/flag")[-1], 0.0)


class TestSchedulerMultiRate:
    def test_manual_groups(self):
        """Manual group assignment works."""
        composite = Composite(
            processes={
                "fast": ConstantProduction(rate=1.0, timescale=1.0),
                "slow": SlowGrowth(timescale=1000.0),
            },
            topology={
                "fast": {"x": "pool/x"},
                "slow": {"y": "pool/y"},
            },
        )
        scheduler = Scheduler(
            groups={"fast": ["fast"], "slow": ["slow"]},
        )
        result = scheduler.run(composite, t_span=(0.0, 100.0), macro_dt=10.0)
        # x ≈ 100 (linear growth at rate 1.0)
        assert jnp.allclose(result.get("pool/x")[-1], 100.0, atol=1.0)
        # y ≈ 0.1 (linear growth at rate 0.001 for 100s)
        assert jnp.allclose(result.get("pool/y")[-1], 0.1, atol=0.01)

    def test_auto_groups_used_by_default(self):
        """Scheduler uses auto_groups when no manual groups given."""
        composite = Composite(
            processes={
                "fast": ConstantProduction(rate=1.0, timescale=1.0),
                "slow": SlowGrowth(timescale=1e6),
            },
            topology={
                "fast": {"x": "pool/x"},
                "slow": {"y": "pool/y"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 10.0), macro_dt=1.0)
        # Both should evolve correctly
        assert result.get("pool/x")[-1] > 5.0
        assert result.get("pool/y")[-1] > 0.0


class TestSchedulerFullIntegration:
    def test_continuous_discrete_event_together(self):
        """Full integration: continuous + discrete + event in one composite."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "counter": PeriodicCounter(dt_step=10.0),
                "latch": ThresholdLatch(threshold=15.0),
            },
            topology={
                "prod": {"x": "pool/x"},
                "counter": {"count": "state/count"},
                "latch": {"x": "pool/x", "flag": "state/flag"},
            },
        )
        scheduler = Scheduler()
        result = scheduler.run(composite, t_span=(0.0, 30.0), macro_dt=5.0)

        # x grows linearly: x(30) ≈ 30
        assert jnp.allclose(result.get("pool/x")[-1], 30.0, atol=1.0)

        # counter: fires at t=10,20,30 → count=3
        assert jnp.isclose(result.get("state/count")[-1], 3.0)

        # latch: x crosses 15 around t=15, flag should be 1.0
        assert jnp.isclose(result.get("state/flag")[-1], 1.0)

        # Event logged
        assert any(e.process == "latch" for e in result.events)

    def test_save_dt_larger_than_macro_dt(self):
        """save_dt controls output frequency independently of macro_dt."""
        composite = Composite(
            processes={"decay": SimpleDecay()},
            topology={"decay": {"x": "pool/x"}},
        )
        scheduler = Scheduler()
        result = scheduler.run(
            composite,
            t_span=(0.0, 100.0),
            macro_dt=1.0,
            save_dt=10.0,
        )
        # Should have ~11 time points (0, 10, 20, ..., 100)
        assert len(result.ts) <= 12
        assert len(result.ts) >= 10


class TestSchedulerBatchedGuards:
    """Scheduler.run rejects batched y0 against features that rely on
    Python-side branching. The error has to be loud and actionable —
    silent batch-axis collapse on these paths is what the guards prevent.
    """

    def test_batched_y0_with_event_raises(self):
        """Batched y0 + EVENT process → ValueError naming the offender."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "latch": ThresholdLatch(threshold=5.0),
            },
            topology={
                "prod": {"x": "pool/x"},
                "latch": {"x": "pool/x", "flag": "state/flag"},
            },
        )
        keys = composite.store_keys()
        y0 = jnp.broadcast_to(
            composite.initial_state_vec(keys), (4, len(keys))
        )
        with pytest.raises(ValueError, match="EVENT processes"):
            Scheduler().run(composite, t_span=(0.0, 10.0), macro_dt=1.0, y0=y0)

    def test_batched_y0_with_discrete_raises(self):
        """Batched y0 + DISCRETE process → ValueError naming the offender."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "counter": PeriodicCounter(),
            },
            topology={
                "prod": {"x": "pool/x"},
                "counter": {"count": "stats/count"},
            },
        )
        keys = composite.store_keys()
        y0 = jnp.broadcast_to(
            composite.initial_state_vec(keys), (4, len(keys))
        )
        with pytest.raises(ValueError, match="DISCRETE processes"):
            Scheduler().run(composite, t_span=(0.0, 10.0), macro_dt=1.0, y0=y0)

    def test_batched_y0_with_adaptive_dt_raises(self):
        """Batched y0 + adaptive_dt=True → ValueError mentioning adaptive_dt."""
        composite = Composite(
            processes={"prod": ConstantProduction(rate=1.0)},
            topology={"prod": {"x": "pool/x"}},
        )
        keys = composite.store_keys()
        y0 = jnp.broadcast_to(
            composite.initial_state_vec(keys), (4, len(keys))
        )
        with pytest.raises(ValueError, match="adaptive_dt"):
            Scheduler(adaptive_dt=True).run(
                composite, t_span=(0.0, 10.0), macro_dt=1.0, y0=y0
            )

    def test_adaptive_dt_unbatched_runs(self):
        """adaptive_dt=True still works for the unbatched case — guard
        is targeted at batched y0 only."""
        composite = Composite(
            processes={
                "prod": ConstantProduction(rate=1.0),
                "decay": SimpleDecay(rate=0.1),
            },
            topology={
                "prod": {"x": "pool/x"},
                "decay": {"x": "pool/x"},
            },
        )
        result = Scheduler(adaptive_dt=True).run(
            composite, t_span=(0.0, 5.0), macro_dt=0.5, save_dt=0.5
        )
        assert "adaptive_dt" in result.stats
        assert result.ts.shape[0] >= 2
        # State must remain finite throughout — no NaN from a bad dt path.
        assert not jnp.any(jnp.isnan(result.get("pool/x")))


class TestSchedulerIsDue:
    def test_is_due_fires(self):
        assert Scheduler._is_due(0.0, 10.0, 10.0)  # exact multiple
        assert Scheduler._is_due(8.0, 12.0, 10.0)  # crossing a multiple

    def test_not_due_within_interval(self):
        assert not Scheduler._is_due(1.0, 5.0, 10.0)

    def test_zero_dt_step_never_due(self):
        assert not Scheduler._is_due(0.0, 10.0, 0.0)


_OMEGA = 20.0  # rad/time → oscillation period 2π/20 ≈ 0.314


class DampedOscillator(Process):
    """2-state damped oscillator ``dx=ω y − ε x, dy=−ω x − ε y`` — Jacobian
    eigenvalues ``−ε ± iω``, so ``max|Im λ| = ω`` (period ``2π/ω``) and a small
    ``ε`` decay keeps an explicit solver from anti-damping. Fixture for the
    Nyquist save-grid guardrail, which keys off the imaginary spectrum."""

    timescale: float = 1.0
    omega: float = _OMEGA
    damp: float = 1.0

    def ports_schema(self):
        return {
            "x": Port(
                role=PortRole.EVOLVED, default=1.0, units="dimensionless"
            ),
            "y": Port(
                role=PortRole.EVOLVED, default=0.0, units="dimensionless"
            ),
        }

    def derivative(self, t, state):
        x, y = state["x"], state["y"]
        return {
            "x": self.omega * y - self.damp * x,
            "y": -self.omega * x - self.damp * y,
        }


def _osc_composite():
    return Composite(
        processes={"osc": DampedOscillator()},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )


def test_nyquist_guardrail_warns_on_coarse_grid(caplog):
    """save_dt above period/2 (Nyquist) warns once; the message names the
    aliasing and points at RunningIntegral / finer saves."""
    import logging

    comp = _osc_composite()
    s = Scheduler(auto_stiffness=True)
    with caplog.at_level(logging.WARNING, logger="hallsim.scheduler"):
        s.run(
            comp,
            t_span=(0.0, 1.0),
            macro_dt=0.5,
            y0=comp.initial_state_vec(),
            save_dt=0.2,  # > period/2 (≈0.157) → below Nyquist; divides t_span
        )
    assert s._warned_save_res
    assert any("undersamples" in r.message for r in caplog.records)


def test_nyquist_guardrail_silent_on_fine_grid():
    """A Nyquist-satisfying grid (save_dt < period/2) does not warn."""
    comp = _osc_composite()
    s = Scheduler(auto_stiffness=True)
    s.run(
        comp,
        t_span=(0.0, 1.0),
        macro_dt=0.5,
        y0=comp.initial_state_vec(),
        save_dt=0.1,  # < period/2 (≈0.157) → resolves the oscillation
    )
    assert not s._warned_save_res


def test_nyquist_guardrail_opt_out():
    """warn_save_resolution=False suppresses the check even on a coarse grid —
    for runs whose readouts are grid-independent (RunningIntegral-based)."""
    comp = _osc_composite()
    s = Scheduler(auto_stiffness=True)
    s.run(
        comp,
        t_span=(0.0, 1.0),
        macro_dt=0.5,
        y0=comp.initial_state_vec(),
        save_dt=0.2,
        warn_save_resolution=False,
    )
    assert not s._warned_save_res


def test_save_grid_never_overshoots_t1():
    """_save_grid pins both endpoints on [t0, t1] and never places a point past
    t1, for save_step values that do NOT divide the span. linspace fits the
    count to the span."""
    for t0, t1, step in [(0.0, 1.0, 0.06), (0.0, 14.0, 0.5), (2.0, 5.0, 0.7)]:
        ts = Scheduler._save_grid(t0, t1, step)
        assert float(ts[0]) == t0
        assert float(ts[-1]) == pytest.approx(t1)
        assert float(jnp.max(ts)) <= t1 + 1e-9
