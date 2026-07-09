"""Tests for the semantic validation layer.

Covers all four subsystems:
- UnitChecker: dimension conflicts, scale mismatches, unspecified units
- SemanticChecker: ontology conflicts, partial annotation, bare ports
- GraphAnalyzer: feedback loops, fan-in, coupling density, unfed inputs
- CouplingAuditor: description overlap detection
- CompositeValidator: orchestration + strict mode
- Composite integration: semantic_validation parameter
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.validation import (
    ComposabilityReport,
    CompositeValidator,
    CouplingAuditor,
    GraphAnalyzer,
    SemanticChecker,
    Severity,
    UnitChecker,
    ValidationReport,
    analyze_composability,
)


# ═══════════════════════════════════════════════════════════════════════════
# Toy processes with rich port metadata
# ═══════════════════════════════════════════════════════════════════════════


class ROSProducerMicromolar(Process):
    """Produces ROS in micromolar."""

    rate: float = 0.1

    def ports_schema(self):
        return {
            "ros": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="uM",
                description="ROS production from mitochondrial electron transport chain",
                ontology={"chebi": "CHEBI:26523"},
            ),
        }

    def derivative(self, t, state):
        return {"ros": jnp.asarray(self.rate)}


class ROSProducerNanomolar(Process):
    """Produces ROS in nanomolar — same dimension, different scale."""

    rate: float = 100.0

    def ports_schema(self):
        return {
            "ros": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="nM",
                description="ROS production from mitochondrial complex III leakage",
                ontology={"chebi": "CHEBI:26523"},
            ),
        }

    def derivative(self, t, state):
        return {"ros": jnp.asarray(self.rate)}


class ROSProducerKilograms(Process):
    """Produces 'ROS' but in kilograms — incompatible dimension with concentration."""

    rate: float = 1.0

    def ports_schema(self):
        return {
            "ros": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="kg",
                description="Mass of ROS produced",
            ),
        }

    def derivative(self, t, state):
        return {"ros": jnp.asarray(self.rate)}


class SuperoxideProducer(Process):
    """Produces superoxide specifically (different ChEBI from generic ROS)."""

    rate: float = 0.05

    def ports_schema(self):
        return {
            "ros": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="uM",
                description="Superoxide anion production",
                ontology={
                    "chebi": "CHEBI:18421"
                },  # superoxide, not generic ROS
            ),
        }

    def derivative(self, t, state):
        return {"ros": jnp.asarray(self.rate)}


class BareProducer(Process):
    """Produces something with no units or ontology."""

    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

    def derivative(self, t, state):
        return {"x": jnp.asarray(self.rate)}


class BareProducer2(Process):
    """Another bare producer."""

    rate: float = 0.2

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

    def derivative(self, t, state):
        return {"x": jnp.asarray(self.rate)}


class ReaderProcess(Process):
    """Reads from an input port, writes to an output."""

    coupling: float = 1.0

    def ports_schema(self):
        return {
            "input_val": Port(role=PortRole.INPUT, default=0.0),
            "output_val": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def derivative(self, t, state):
        return {"output_val": self.coupling * state["input_val"]}


class FeedbackA(Process):
    """Part of a feedback loop: reads y, writes x."""

    k: float = 0.1

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
            "y": Port(role=PortRole.INPUT, default=0.0, units="uM"),
        }

    def derivative(self, t, state):
        return {"x": self.k * state["y"]}


class FeedbackB(Process):
    """Part of a feedback loop: reads x, writes y."""

    k: float = 0.2

    def ports_schema(self):
        return {
            "y": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
            "x": Port(role=PortRole.INPUT, default=0.0, units="uM"),
        }

    def derivative(self, t, state):
        return {"y": -self.k * state["x"]}


# ═══════════════════════════════════════════════════════════════════════════
# Unit Checker
# ═══════════════════════════════════════════════════════════════════════════


class TestUnitChecker:

    def test_compatible_same_units(self):
        """Two processes with identical units → no findings."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerMicromolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        results = UnitChecker().check(procs, topo)
        unit_results = [r for r in results if r.category == "units"]
        assert len(unit_results) == 0

    def test_compatible_different_scale(self):
        """uM vs nM → INFO (compatible; auto-reconciled to canonical unit)."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerNanomolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        results = UnitChecker().check(procs, topo)
        infos = [r for r in results if r.level == Severity.INFO]
        assert len(infos) == 1
        assert (
            "reconciled" in infos[0].message.lower()
            or "factor" in infos[0].message.lower()
        )

    def test_incompatible_dimensions(self):
        """uM vs cell → ERROR (incompatible dimensions)."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        results = UnitChecker().check(procs, topo)
        errors = [r for r in results if r.level == Severity.ERROR]
        assert len(errors) == 1
        assert "incompatible" in errors[0].message.lower()

    def test_unspecified_units_warning(self):
        """One port has units, another doesn't → WARNING."""
        procs = {"a": ROSProducerMicromolar(), "b": BareProducer()}
        topo = {"a": {"ros": "pool/x"}, "b": {"x": "pool/x"}}
        results = UnitChecker().check(procs, topo)
        warnings = [r for r in results if r.level == Severity.WARNING]
        assert any("unspecified" in w.message.lower() for w in warnings)

    def test_single_writer_no_check(self):
        """Only one writer → no unit checks needed."""
        procs = {"a": ROSProducerMicromolar()}
        topo = {"a": {"ros": "pool/ros"}}
        results = UnitChecker().check(procs, topo)
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Checker
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticChecker:

    def test_matching_ontology(self):
        """Same chebi ID → no error."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerNanomolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        results = SemanticChecker().check(procs, topo)
        errors = [r for r in results if r.level == Severity.ERROR]
        assert len(errors) == 0

    def test_conflicting_ontology(self):
        """Different chebi IDs → ERROR."""
        procs = {"a": ROSProducerMicromolar(), "b": SuperoxideProducer()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        results = SemanticChecker().check(procs, topo)
        errors = [r for r in results if r.level == Severity.ERROR]
        assert len(errors) == 1
        assert "CHEBI:26523" in errors[0].message
        assert "CHEBI:18421" in errors[0].message

    def test_partial_annotation(self):
        """One annotated, one bare → WARNING."""
        procs = {"a": ROSProducerMicromolar(), "b": BareProducer()}
        topo = {"a": {"ros": "pool/x"}, "b": {"x": "pool/x"}}
        results = SemanticChecker().check(procs, topo)
        warnings = [r for r in results if r.level == Severity.WARNING]
        assert any("partial" in w.message.lower() for w in warnings)

    def test_both_bare(self):
        """Neither annotated → INFO (unverifiable)."""
        procs = {"a": BareProducer(), "b": BareProducer2()}
        topo = {"a": {"x": "pool/x"}, "b": {"x": "pool/x"}}
        results = SemanticChecker().check(procs, topo)
        infos = [r for r in results if r.level == Severity.INFO]
        assert any("no ontology" in i.message.lower() for i in infos)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Analyzer
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphAnalyzer:

    def test_feedback_loop_detected(self):
        """A→B→A cycle → WARNING."""
        procs = {"a": FeedbackA(), "b": FeedbackB()}
        topo = {
            "a": {"x": "pool/x", "y": "pool/y"},
            "b": {"y": "pool/y", "x": "pool/x"},
        }
        results = GraphAnalyzer().analyze(procs, topo)
        loop_warnings = [
            r
            for r in results
            if "feedback" in r.message.lower() or "loop" in r.message.lower()
        ]
        assert len(loop_warnings) >= 1

    def test_high_fan_in(self):
        """4 EVOLVED writers at same path → WARNING."""

        class Writer(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {"x": Port(role=PortRole.EVOLVED, default=0.0)}

            def derivative(self, t, state):
                return {"x": jnp.asarray(self.rate)}

        procs = {f"w{i}": Writer(rate=0.1 * i) for i in range(4)}
        topo = {f"w{i}": {"x": "shared/x"} for i in range(4)}
        results = GraphAnalyzer().analyze(procs, topo)
        fan_in = [r for r in results if "fan-in" in r.message.lower()]
        assert len(fan_in) == 1
        assert "4" in fan_in[0].message

    def test_unfed_input(self):
        """INPUT port with no writer → WARNING."""
        procs = {"reader": ReaderProcess()}
        topo = {"reader": {"input_val": "nowhere", "output_val": "out"}}
        results = GraphAnalyzer().analyze(procs, topo)
        unfed = [r for r in results if "unfed" in r.message.lower()]
        assert len(unfed) == 1
        assert "nowhere" in unfed[0].message

    def test_no_cycle_in_linear_chain(self):
        """A→B with no feedback → no cycle warning."""
        procs = {"prod": BareProducer(), "reader": ReaderProcess()}
        topo = {
            "prod": {"x": "pool/x"},
            "reader": {"input_val": "pool/x", "output_val": "pool/y"},
        }
        results = GraphAnalyzer().analyze(procs, topo)
        loops = [
            r
            for r in results
            if "loop" in r.message.lower() or "feedback" in r.message.lower()
        ]
        assert len(loops) == 0

    def test_graph_export(self):
        """to_dict returns serializable graph."""
        procs = {"a": BareProducer(), "b": BareProducer2()}
        topo = {"a": {"x": "pool/x"}, "b": {"x": "pool/x"}}
        data = GraphAnalyzer().to_dict(procs, topo)
        assert "nodes" in data or "links" in data


# ═══════════════════════════════════════════════════════════════════════════
# Coupling Auditor
# ═══════════════════════════════════════════════════════════════════════════


class TestCouplingAuditor:

    def test_description_overlap(self):
        """Two processes with overlapping descriptions → WARNING."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerNanomolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        results = CouplingAuditor().check(procs, topo)
        warnings = [r for r in results if r.level == Severity.WARNING]
        assert len(warnings) >= 1
        assert "duplication" in warnings[0].message.lower()

    def test_no_overlap_different_descriptions(self):
        """Processes with unrelated descriptions → no warning."""

        class ATPProducer(Process):
            rate: float = 0.1

            def ports_schema(self):
                return {
                    "x": Port(
                        role=PortRole.EVOLVED,
                        default=0.0,
                        description="ATP synthesis via oxidative phosphorylation",
                    )
                }

            def derivative(self, t, state):
                return {"x": jnp.asarray(self.rate)}

        class GlucoseConsumer(Process):
            rate: float = 0.05

            def ports_schema(self):
                return {
                    "x": Port(
                        role=PortRole.EVOLVED,
                        default=0.0,
                        description="Hexokinase catalyzed glucose phosphorylation",
                    )
                }

            def derivative(self, t, state):
                return {"x": jnp.asarray(-self.rate)}

        procs = {"atp": ATPProducer(), "glc": GlucoseConsumer()}
        topo = {"atp": {"x": "pool/energy"}, "glc": {"x": "pool/energy"}}
        results = CouplingAuditor().check(procs, topo)
        warnings = [r for r in results if r.level == Severity.WARNING]
        assert len(warnings) == 0


# ═══════════════════════════════════════════════════════════════════════════
# CompositeValidator (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeValidator:

    def test_full_pipeline_catches_all(self):
        """Dimension conflict + semantic conflict detected together."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        report = CompositeValidator().validate(procs, topo)
        assert not report.is_valid
        assert any(r.category == "units" for r in report.errors)

    def test_valid_composition(self):
        """Compatible processes pass validation."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerMicromolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        report = CompositeValidator().validate(procs, topo)
        assert report.is_valid

    def test_strict_mode_promotes_warnings(self):
        """strict=True turns warnings into errors."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerNanomolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        report = CompositeValidator(strict=True).validate(procs, topo)
        assert not report.is_valid  # warnings promoted to errors

    def test_selective_checks(self):
        """Can disable specific subsystems."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        report = CompositeValidator(check_units=False).validate(procs, topo)
        unit_errors = [r for r in report.errors if r.category == "units"]
        assert len(unit_errors) == 0

    def test_report_summary(self):
        report = ValidationReport()
        assert "0 error" in report.summary()

    def test_report_str(self):
        report = ValidationReport()
        assert isinstance(str(report), str)

    def test_interaction_graph_in_report(self):
        procs = {"a": BareProducer(), "b": BareProducer2()}
        topo = {"a": {"x": "pool/x"}, "b": {"x": "pool/x"}}
        report = CompositeValidator().validate(procs, topo)
        assert report.interaction_graph is not None


# ═══════════════════════════════════════════════════════════════════════════
# Composite integration
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeIntegration:

    def test_semantic_validation_on_by_default(self):
        """Default: semantic validation runs and surfaces errors."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        with pytest.raises(ValueError, match="Semantic validation failed"):
            Composite(procs, topo)

    def test_semantic_validation_can_be_disabled(self):
        """Pass semantic_validation=False to skip the layer entirely."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        composite = Composite(procs, topo, semantic_validation=False)
        assert composite is not None

    def test_semantic_validation_true_raises_on_error(self):
        """semantic_validation=True raises on unit conflict."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        with pytest.raises(ValueError, match="Semantic validation failed"):
            Composite(procs, topo, semantic_validation=True)

    def test_semantic_validation_dict_config(self):
        """Can pass config dict to CompositeValidator."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerKilograms()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        # Disable unit checking → should not raise
        composite = Composite(
            procs, topo, semantic_validation={"check_units": False}
        )
        assert composite is not None

    def test_semantic_validation_warns_on_warnings(self, caplog):
        """Warnings are emitted via logging, not raised as errors."""
        procs = {"a": ROSProducerMicromolar(), "b": ROSProducerNanomolar()}
        topo = {"a": {"ros": "pool/ros"}, "b": {"ros": "pool/ros"}}
        with caplog.at_level(logging.WARNING, logger="hallsim.composite"):
            Composite(procs, topo, semantic_validation=True)
        assert any(
            "Semantic validation" in r.getMessage() for r in caplog.records
        )


# ═══════════════════════════════════════════════════════════════════════════
# Composite-of-composites: merging sub-composites via the constructor
# ═══════════════════════════════════════════════════════════════════════════


class TestSubCompositeFlattening:
    """Composite.__init__ accepts other Composites as values inside its
    processes dict and flattens them with namespace prefixes."""

    def _inner_uM(self) -> Composite:
        return Composite(
            processes={"prod": ROSProducerMicromolar()},
            topology={"prod": {"ros": "pool/ros"}},
            semantic_validation=False,
        )

    def _inner_nM(self) -> Composite:
        return Composite(
            processes={"prod": ROSProducerNanomolar()},
            topology={"prod": {"ros": "pool/ros"}},
            semantic_validation=False,
        )

    def test_two_disjoint_subcomposites_merge_cleanly(self):
        merged = Composite(
            processes={"a": self._inner_uM(), "b": self._inner_nM()},
            semantic_validation=False,
        )
        assert set(merged.processes) == {"a.prod", "b.prod"}
        assert merged.topology["a.prod"] == {"ros": "a/pool/ros"}
        assert merged.topology["b.prod"] == {"ros": "b/pool/ros"}
        assert set(merged.store_paths()) == {"a/pool/ros", "b/pool/ros"}

    def test_rewire_merges_overlapping_paths(self):
        merged = Composite(
            processes={"a": self._inner_uM(), "b": self._inner_nM()},
            rewire={"b/pool/ros": "a/pool/ros"},
            semantic_validation=False,
        )
        # Both inner processes now write to the same canonical path.
        assert merged.topology["a.prod"]["ros"] == "a/pool/ros"
        assert merged.topology["b.prod"]["ros"] == "a/pool/ros"
        assert merged.store_paths() == {"a/pool/ros"}

    def test_subcomposite_with_incompatible_units_raises_on_merge(self):
        """Inner composites validate fine individually; merging onto one
        canonical path surfaces the unit conflict at merge time."""
        kg_inner = Composite(
            processes={"prod": ROSProducerKilograms()},
            topology={"prod": {"ros": "pool/ros"}},
            semantic_validation=False,
        )
        with pytest.raises(ValueError, match="Semantic validation failed"):
            Composite(
                processes={"a": self._inner_uM(), "b": kg_inner},
                rewire={"b/pool/ros": "a/pool/ros"},
            )

    def test_path_already_prefixed_is_not_double_prefixed(self):
        """Idempotence: a sub-composite that already prefixes its paths
        with its outer key is not prefixed twice."""
        pre_namespaced = Composite(
            processes={"prod": ROSProducerMicromolar()},
            topology={"prod": {"ros": "a/pool/ros"}},
            semantic_validation=False,
        )
        merged = Composite(
            processes={"a": pre_namespaced},
            semantic_validation=False,
        )
        assert merged.topology["a.prod"]["ros"] == "a/pool/ros"

    def test_mixed_process_and_composite_values(self):
        """A raw Process can sit alongside a Composite; the Process uses
        the caller-supplied topology arg."""
        merged = Composite(
            processes={
                "a": self._inner_uM(),
                "extra": ROSProducerMicromolar(),
            },
            topology={"extra": {"ros": "a/pool/ros"}},
            semantic_validation=False,
        )
        assert set(merged.processes) == {"a.prod", "extra"}
        assert merged.topology["extra"]["ros"] == "a/pool/ros"

    def test_unknown_value_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="must be a Process or Composite"):
            Composite(processes={"bogus": 42})  # type: ignore[dict-item]

    def test_process_name_collision_raises(self):
        """Two raw Processes can't share the same outer name."""
        with pytest.raises(ValueError, match="name collision"):
            Composite(
                processes={
                    "x": ROSProducerMicromolar(),
                    # Same outer key reused — dict literal can't, but
                    # we simulate by using a sub-composite that produces
                    # the same merged name as a Process.
                },
                topology={"x": {"ros": "pool/ros"}},
            ) and Composite(
                processes={
                    "x": Composite(
                        processes={"x": ROSProducerMicromolar()},
                        topology={"x": {"ros": "pool/ros"}},
                        semantic_validation=False,
                    ),
                    "x.x": ROSProducerMicromolar(),
                },
                topology={"x.x": {"ros": "pool/ros"}},
                semantic_validation=False,
            )


# ═══════════════════════════════════════════════════════════════════════════
# analyze_composability — cross-composite overlap discovery
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalyzeComposability:
    """Surfacing candidate overlaps across composites before merging."""

    def _ros_with_ontology(self) -> Composite:
        return Composite(
            processes={"prod": ROSProducerMicromolar()},  # has chebi ontology
            topology={"prod": {"ros": "pool/ros"}},
            semantic_validation=False,
        )

    def _ros_named_only(self) -> Composite:
        return Composite(
            processes={"prod": BareProducer()},  # no ontology on port
            topology={"prod": {"x": "pool/ros"}},
            semantic_validation=False,
        )

    def test_no_composites_returns_empty_report(self):
        report = analyze_composability()
        assert isinstance(report, ComposabilityReport)
        assert report.matches == ()
        assert report.suggested_rewire == {}

    def test_single_composite_returns_empty_report(self):
        report = analyze_composability(only=self._ros_with_ontology())
        assert report.matches == ()
        assert report.suggested_rewire == {}

    def test_ontology_match_is_high_confidence(self):
        report = analyze_composability(
            eriq=self._ros_with_ontology(),
            dp14=self._ros_with_ontology(),
        )
        assert len(report.matches) == 1
        match = report.matches[0]
        assert match.method == "ontology_id"
        assert match.confidence >= 0.9
        assert match.path_a == "eriq/pool/ros"
        assert match.path_b == "dp14/pool/ros"

    def test_exact_name_match_when_no_ontology(self):
        report = analyze_composability(
            eriq=self._ros_named_only(),
            dp14=self._ros_named_only(),
        )
        assert len(report.matches) == 1
        assert report.matches[0].method == "exact_name"
        assert 0.7 <= report.matches[0].confidence < 0.95

    def test_suggested_rewire_picks_first_kwarg_as_canonical(self):
        report = analyze_composability(
            eriq=self._ros_with_ontology(),
            dp14=self._ros_with_ontology(),
        )
        # eriq comes first → canonical
        assert report.suggested_rewire == {"dp14/pool/ros": "eriq/pool/ros"}

    def test_suggested_rewire_directly_usable_in_composite_merge(self):
        """End-to-end: analyzer suggestion → Composite rewire kwarg."""
        eriq = self._ros_with_ontology()
        dp14 = self._ros_with_ontology()
        report = analyze_composability(eriq=eriq, dp14=dp14)
        merged = Composite(
            processes={"eriq": eriq, "dp14": dp14},
            rewire=report.suggested_rewire,
            semantic_validation=False,
        )
        # Both writers now collapsed onto the canonical path
        assert merged.topology["eriq.prod"]["ros"] == "eriq/pool/ros"
        assert merged.topology["dp14.prod"]["ros"] == "eriq/pool/ros"
        assert merged.store_paths() == {"eriq/pool/ros"}
