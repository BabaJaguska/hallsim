"""Tests for ClampEdge — holding an integrated species at a setpoint against
the model that consumes it — and its rate-placement helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.models.clamp_edge import (
    ClampEdge,
    clamp_species,
    measure_unclamped_flux,
    place_clamp_rate,
)
from hallsim.models.forcing import PulseSource
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler


class LigandUptake(Process):
    """Receptor-mediated uptake of a ligand the model integrates: the case
    ``with_param_input`` (constants) and ``drive_pulse`` (boundary inputs)
    cannot reach."""

    timescale: float | None = 1.0
    v_max: float = 0.5
    K_m: float = 1.0

    def ports_schema(self):
        return {
            "ligand": Port(
                role=PortRole.EVOLVED,
                default=2.0,
                units="uM",
                description="free ligand",
                ontology={"uniprot": "P01137"},
            )
        }

    def derivative(self, t, state):
        L = jnp.maximum(state["ligand"], 0.0)
        return {"ligand": -self.v_max * L / (self.K_m + L)}


def _wired(level=2.0, k_clamp=10.0):
    processes = {"cell": LigandUptake()}
    topology = {"cell": {"ligand": "medium/ligand"}}
    return clamp_species(
        processes,
        topology,
        target="cell",
        species="ligand",
        level=level,
        k_clamp=k_clamp,
    )


class TestClampEdge:
    def test_ports_read_their_own_path(self):
        schema = ClampEdge().ports_schema()
        assert schema["target"].role == PortRole.EVOLVED
        # Negative feedback on the clamped path, not a pure source.
        assert schema["target"].reads_value is True
        assert schema["setpoint"].role == PortRole.INPUT

    def test_units_apply_to_both_ports(self):
        schema = ClampEdge(units="uM").ports_schema()
        assert schema["target"].units == schema["setpoint"].units == "uM"

    def test_restoring_term_is_signed_and_vanishes_at_setpoint(self):
        e = ClampEdge(k_clamp=2.0)
        below = e.derivative(
            0.0, {"target": jnp.array(1.0), "setpoint": jnp.array(3.0)}
        )["target"]
        at = e.derivative(
            0.0, {"target": jnp.array(3.0), "setpoint": jnp.array(3.0)}
        )["target"]
        above = e.derivative(
            0.0, {"target": jnp.array(5.0), "setpoint": jnp.array(3.0)}
        )["target"]
        assert float(below) == pytest.approx(4.0)
        assert float(at) == pytest.approx(0.0)
        assert float(above) == pytest.approx(-4.0)

    def test_differentiable_through_rate_and_setpoint(self):
        gk = jax.grad(
            lambda k: ClampEdge(k_clamp=k).derivative(
                0.0, {"target": jnp.array(1.0), "setpoint": jnp.array(3.0)}
            )["target"]
        )(jnp.array(2.0))
        gs = jax.grad(
            lambda s: ClampEdge(k_clamp=2.0).derivative(
                0.0, {"target": jnp.array(1.0), "setpoint": s}
            )["target"]
        )(jnp.array(3.0))
        assert jnp.isfinite(gk) and float(gk) > 0
        assert jnp.isfinite(gs) and float(gs) > 0

    def test_k_clamp_is_the_calibratable_surface(self):
        assert {p.field for p in ClampEdge().calibratable_params()} == {
            "k_clamp"
        }

    def test_declarative_metadata_folds_in(self):
        m = ClampEdge(hallmark="H", reference="R", description="D").metadata()
        assert (m["hallmark"], m["reference"], m["description"]) == (
            "H",
            "R",
            "D",
        )


class TestSustainedPulseSource:
    def test_no_washout_when_t_end_is_none(self):
        src = PulseSource(amplitude=2.0, t_start=1.0, t_end=None)
        assert float(src.assign(0.5, {})["signal"]) == 0.0
        assert float(src.assign(1.0, {})["signal"]) == 2.0
        assert float(src.assign(1e6, {})["signal"]) == 2.0

    def test_only_the_rising_edge_is_a_discontinuity(self):
        assert PulseSource(t_start=1.0, t_end=None).discontinuity_times() == (
            1.0,
        )
        assert PulseSource(t_start=1.0, t_end=3.0).discontinuity_times() == (
            1.0,
            3.0,
        )

    def test_windowed_pulse_unchanged(self):
        src = PulseSource(amplitude=2.0, t_start=1.0, t_end=3.0)
        assert float(src.assign(2.0, {})["signal"]) == 2.0
        assert float(src.assign(3.0, {})["signal"]) == 0.0


class TestPlaceClampRate:
    def test_rate_meets_the_stated_residual(self):
        s = place_clamp_rate(0.25, 2.0, rel_error=0.01)
        assert s.ok
        # residual flux/k must be rel_error * level
        assert 0.25 / s.k_clamp == pytest.approx(0.01 * 2.0)
        assert s.tau == pytest.approx(1.0 / s.k_clamp)

    def test_sign_of_flux_does_not_matter(self):
        assert (
            place_clamp_rate(-0.25, 2.0).k_clamp
            == place_clamp_rate(0.25, 2.0).k_clamp
        )

    def test_zero_flux_needs_no_clamp(self):
        s = place_clamp_rate(0.0, 2.0)
        assert s.ok and s.k_clamp == 0.0 and "already holds" in s.note

    def test_non_positive_setpoint_flagged(self):
        assert not place_clamp_rate(0.25, 0.0).ok

    def test_stiff_separation_flagged_against_the_model_timescale(self):
        tight = place_clamp_rate(0.25, 2.0, rel_error=1e-4, tau_model=1.0)
        loose = place_clamp_rate(0.25, 2.0, rel_error=1e-1, tau_model=1.0)
        assert not tight.ok and "stiffens" in tight.note
        assert loose.ok


class TestMeasureUnclampedFlux:
    def test_matches_the_models_own_rate_law(self):
        comp = Composite(
            processes={"cell": LigandUptake()},
            topology={"cell": {"ligand": "medium/ligand"}},
        )
        v = measure_unclamped_flux(comp, "medium/ligand", 2.0)
        assert float(v) == pytest.approx(-0.5 * 2.0 / (1.0 + 2.0))

    def test_excludes_a_clamp_already_wired(self):
        procs, topo, _ = _wired(level=2.0, k_clamp=10.0)
        comp = Composite(processes=procs, topology=topo)
        v = measure_unclamped_flux(comp, "medium/ligand", 5.0)
        # Without exclusion the clamp's -10*(5-2) = -30 would dominate.
        assert float(v) == pytest.approx(-0.5 * 5.0 / (1.0 + 5.0))


class TestClampSpecies:
    def test_wires_setpoint_source_and_edge_onto_the_species_path(self):
        procs, topo, name = _wired()
        assert name == "ligand_clamp"
        assert isinstance(procs[name], ClampEdge)
        assert topo[name]["target"] == "medium/ligand"
        assert (
            topo[name]["setpoint"] == topo["ligand_clamp_setpoint"]["signal"]
        )
        assert procs["ligand_clamp_setpoint"].t_end is None

    def test_clamp_inherits_the_species_units_and_ontology(self):
        procs, _, name = _wired()
        schema = procs[name].ports_schema()
        assert schema["target"].units == "uM"
        assert schema["target"].ontology == {"uniprot": "P01137"}
        assert procs["ligand_clamp_setpoint"].ports_schema()[
            "signal"
        ].units == ("uM")

    def test_clamp_cogroups_with_the_target(self):
        procs, topo, name = _wired()
        comp = Composite(processes=procs, topology=topo)
        groups = comp.auto_groups()
        assert len(groups) == 1
        assert set(next(iter(groups.values()))) == set(
            comp.continuous_processes()
        )

    def test_inherited_ontology_leaves_the_semantic_checker_quiet(self):
        from hallsim.validation import CompositeValidator

        procs, topo, _ = _wired()
        report = CompositeValidator().validate(procs, topo)
        assert not [r for r in report.results if r.category == "semantics"]

    def test_the_clamp_loop_is_reported_as_a_feedback_cycle(self):
        # A clamp is negative feedback by construction, so the graph analyser
        # names it. Pinned because it is the one message a correct clamp
        # always emits — and why strict validation rejects one.
        from hallsim.validation import CompositeValidator

        procs, topo, _ = _wired()
        report = CompositeValidator().validate(procs, topo)
        assert [r.category for r in report.warnings] == ["graph"]
        assert "Feedback loop" in report.warnings[0].message


class TestClampedRun:
    """The point of the primitive: the species holds instead of draining."""

    def _run(self, procs, topo):
        comp = Composite(processes=procs, topology=topo)
        res = Scheduler().run(comp, t_span=(0.0, 20.0), macro_dt=1.0)
        return res.get("medium/ligand")

    def test_unclamped_species_drains(self):
        y = self._run(
            {"cell": LigandUptake()}, {"cell": {"ligand": "medium/ligand"}}
        )
        assert float(y[0]) == pytest.approx(2.0)
        assert float(y[-1]) < 0.1

    def test_clamped_species_holds_within_the_predicted_residual(self):
        level = 2.0
        flux = 0.5 * level / (1.0 + level)
        s = place_clamp_rate(flux, level, rel_error=0.01, tau_model=1.0)
        assert s.ok
        procs, topo, _ = _wired(level=level, k_clamp=s.k_clamp)
        y = self._run(procs, topo)
        residual = level - float(y[-1])
        assert 0.0 < residual <= 0.01 * level * 1.05

    def test_a_looser_clamp_leaves_a_larger_residual(self):
        level = 2.0
        tight = self._run(*_wired(level=level, k_clamp=100.0)[:2])
        loose = self._run(*_wired(level=level, k_clamp=1.0)[:2])
        assert (level - float(loose[-1])) > (level - float(tight[-1]))

    def test_setpoint_step_is_tracked(self):
        procs = {"cell": LigandUptake()}
        topo = {"cell": {"ligand": "medium/ligand"}}
        clamp_species(
            procs,
            topo,
            target="cell",
            species="ligand",
            level=5.0,
            k_clamp=20.0,
            t_start=10.0,
        )
        comp = Composite(processes=procs, topology=topo)
        res = Scheduler().run(comp, t_span=(0.0, 20.0), macro_dt=1.0)
        y = res.get("medium/ligand")
        ts = res.ts
        before = float(y[jnp.argmin(jnp.abs(ts - 9.0))])
        assert before < 0.5  # setpoint is 0 until t_start: drains, then held
        assert float(y[-1]) == pytest.approx(5.0, abs=0.05)
