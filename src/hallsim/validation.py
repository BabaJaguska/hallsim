"""Semantic validation layer for composable simulations.

Catches unit mismatches, ontology conflicts, feedback loops, coupling
density issues, and potential reaction duplication at composition time —
before the simulation runs.

Four subsystems, orchestrated by :class:`CompositeValidator`:

1. **UnitChecker** — pint-based dimensional analysis across shared store paths.
2. **SemanticChecker** — ontology ID comparison (ChEBI, GO, SBO) for species disambiguation.
3. **GraphAnalyzer** — feedback cycle detection, fan-in analysis, coupling density.
4. **CouplingAuditor** — heuristic duplicate-reaction detection via description overlap.

Usage::

    validator = CompositeValidator()
    report = validator.validate(processes, topology)
    print(report)
    assert report.is_valid

Or integrated into Composite::

    composite = Composite(processes, topology, semantic_validation=True)
"""

from __future__ import annotations

import enum
import logging
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pint

from hallsim.process import Port, PortRole, Process
from hallsim.store import validate_topology

log = logging.getLogger(__name__)

# Shared pint registry — created once, reused everywhere.
_UREG = pint.UnitRegistry()


# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

class Severity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationResult:
    """A single finding from the validation layer."""

    level: Severity
    category: str  # "structure", "units", "semantics", "graph", "coupling"
    message: str

    def __str__(self) -> str:
        return f"[{self.level.value.upper():7s}] ({self.category}) {self.message}"


@dataclass
class ValidationReport:
    """Aggregated output from all validation subsystems."""

    results: list[ValidationResult] = field(default_factory=list)
    interaction_graph: dict | None = None

    @property
    def errors(self) -> list[ValidationResult]:
        return [r for r in self.results if r.level == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if r.level == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        n_info = sum(1 for r in self.results if r.level == Severity.INFO)
        return (
            f"Validation: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), {n_info} info"
        )

    def __str__(self) -> str:
        lines = [self.summary(), ""]
        # Errors first, then warnings, then info
        for sev in (Severity.ERROR, Severity.WARNING, Severity.INFO):
            for r in self.results:
                if r.level == sev:
                    lines.append(str(r))
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build store-path → port mapping
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _PortEntry:
    proc_name: str
    port_name: str
    port: Port


def _store_port_map(
    processes: dict[str, Process],
    topology: dict[str, dict[str, str]],
) -> dict[str, list[_PortEntry]]:
    """Group ports by their resolved store path."""
    result: dict[str, list[_PortEntry]] = {}
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            store_path = topo.get(port_name, port_name)
            result.setdefault(store_path, []).append(
                _PortEntry(proc_name, port_name, port)
            )
    return result


def _writers(entries: list[_PortEntry]) -> list[_PortEntry]:
    """Entries whose role produces derivatives or deltas."""
    return [e for e in entries if e.port.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE, PortRole.LATCHED)]


# ═══════════════════════════════════════════════════════════════════════════
# Subsystem 1: Unit Checker
# ═══════════════════════════════════════════════════════════════════════════

class UnitChecker:
    """Validates unit compatibility across ports wired to the same store path.

    Uses pint for dimensional analysis.

    - ERROR: incompatible dimensionalities (concentration vs count).
    - WARNING: compatible but different units (uM vs nM).
    - WARNING: one or more ports have unspecified units.
    """

    def check(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        spm = _store_port_map(processes, topology)

        for store_path, entries in spm.items():
            writers = _writers(entries)
            if len(writers) < 2:
                continue

            # Parse units
            parsed: list[tuple[_PortEntry, Any]] = []
            unspecified: list[_PortEntry] = []

            for entry in writers:
                unit_str = entry.port.units.strip()
                if not unit_str:
                    unspecified.append(entry)
                    continue
                try:
                    parsed.append((entry, _UREG.parse_expression(unit_str)))
                except (pint.UndefinedUnitError, pint.errors.UndefinedUnitError):
                    results.append(ValidationResult(
                        Severity.WARNING, "units",
                        f"Cannot parse unit {unit_str!r} on "
                        f"{entry.proc_name}.{entry.port_name} -> {store_path!r}",
                    ))

            # Warn about unspecified units on shared paths
            if unspecified and parsed:
                names = ", ".join(f"{e.proc_name}.{e.port_name}" for e in unspecified)
                results.append(ValidationResult(
                    Severity.WARNING, "units",
                    f"Unspecified units at {store_path!r} for: {names}. "
                    f"Cannot verify compatibility with other writers.",
                ))

            # Pairwise comparison of parsed units
            for i, (e1, u1) in enumerate(parsed):
                for e2, u2 in parsed[i + 1:]:
                    if not u1.is_compatible_with(u2):
                        results.append(ValidationResult(
                            Severity.ERROR, "units",
                            f"Incompatible units at {store_path!r}: "
                            f"{e1.proc_name}.{e1.port_name}={u1.units} vs "
                            f"{e2.proc_name}.{e2.port_name}={u2.units}",
                        ))
                    elif u1.units != u2.units:
                        try:
                            factor = u1.to(u2).magnitude
                        except Exception:
                            factor = "?"
                        results.append(ValidationResult(
                            Severity.WARNING, "units",
                            f"Unit scale mismatch at {store_path!r}: "
                            f"{e1.proc_name}.{e1.port_name}={u1.units} vs "
                            f"{e2.proc_name}.{e2.port_name}={u2.units} "
                            f"(factor: {factor})",
                        ))

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Subsystem 2: Semantic Checker
# ═══════════════════════════════════════════════════════════════════════════

class SemanticChecker:
    """Validates that ports wired to the same store path refer to the same
    biological entity, using ontology annotations.

    - ERROR: conflicting ontology IDs on shared store path.
    - WARNING: partial annotation (one port has ontology, another doesn't).
    - INFO: no annotations on shared path (unverifiable).
    """

    def check(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        spm = _store_port_map(processes, topology)

        for store_path, entries in spm.items():
            writers = _writers(entries)
            if len(writers) < 2:
                continue

            onts = [(e, e.port.ontology or {}) for e in writers]

            for i, (e1, ont1) in enumerate(onts):
                for e2, ont2 in onts[i + 1:]:
                    shared_ns = set(ont1.keys()) & set(ont2.keys())

                    # Conflict: same namespace, different value
                    for ns in shared_ns:
                        if ont1[ns] != ont2[ns]:
                            results.append(ValidationResult(
                                Severity.ERROR, "semantics",
                                f"Semantic conflict at {store_path!r}: "
                                f"{e1.proc_name}.{e1.port_name} has {ns}={ont1[ns]}, "
                                f"{e2.proc_name}.{e2.port_name} has {ns}={ont2[ns]}. "
                                f"These may represent different species.",
                            ))

                    # Partial annotation
                    if not shared_ns and (ont1 or ont2):
                        annotated = e1 if ont1 else e2
                        bare = e2 if ont1 else e1
                        results.append(ValidationResult(
                            Severity.WARNING, "semantics",
                            f"Partial annotation at {store_path!r}: "
                            f"{annotated.proc_name}.{annotated.port_name} has "
                            f"ontology {annotated.port.ontology}, but "
                            f"{bare.proc_name}.{bare.port_name} has none.",
                        ))

                    # Both bare
                    if not ont1 and not ont2:
                        results.append(ValidationResult(
                            Severity.INFO, "semantics",
                            f"No ontology annotations at {store_path!r}: "
                            f"cannot verify semantic compatibility between "
                            f"{e1.proc_name}.{e1.port_name} and "
                            f"{e2.proc_name}.{e2.port_name}.",
                        ))

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Subsystem 3: Interaction Graph Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class GraphAnalyzer:
    """Builds and analyzes the process interaction graph.

    Nodes = processes.  An edge A → B exists when A writes (EVOLVED/EXCLUSIVE)
    to a store path that B reads (INPUT or EVOLVED).

    Analyses:
    - Feedback cycle detection.
    - Fan-in: how many EVOLVED writers per store path.
    - Coupling density (edges / possible edges).
    - Unfed INPUT ports (no writer at that store path).
    """

    def build_graph(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        for proc_name in processes:
            G.add_node(proc_name)

        # Who writes / reads each store path?
        path_writers: dict[str, list[str]] = {}
        path_readers: dict[str, list[str]] = {}

        for proc_name, proc in processes.items():
            topo = topology.get(proc_name, {})
            for port_name, port in proc.ports_schema().items():
                sp = topo.get(port_name, port_name)
                if port.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE, PortRole.LATCHED):
                    path_writers.setdefault(sp, []).append(proc_name)
                if port.role == PortRole.INPUT:
                    path_readers.setdefault(sp, []).append(proc_name)
                # EVOLVED ports are both: the process reads the current value
                # (via the port view) and writes a derivative.
                if port.role == PortRole.EVOLVED:
                    path_readers.setdefault(sp, []).append(proc_name)

        # Edges: writer → reader (skip self-loops)
        all_paths = set(path_writers.keys()) | set(path_readers.keys())
        for sp in all_paths:
            for w in path_writers.get(sp, []):
                for r in path_readers.get(sp, []):
                    if w != r:
                        G.add_edge(w, r, store_path=sp)

        return G

    def analyze(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        G = self.build_graph(processes, topology)

        # 1. Feedback cycles
        for cycle in nx.simple_cycles(G):
            path_str = " -> ".join(cycle + [cycle[0]])
            results.append(ValidationResult(
                Severity.WARNING, "graph",
                f"Feedback loop: {path_str}. "
                f"Verify this is intentional and numerically stable.",
            ))

        # 2. Fan-in analysis
        spm = _store_port_map(processes, topology)
        for store_path, entries in spm.items():
            evolved = [e for e in entries if e.port.role == PortRole.EVOLVED]
            if len(evolved) >= 3:
                names = ", ".join(f"{e.proc_name}.{e.port_name}" for e in evolved)
                results.append(ValidationResult(
                    Severity.WARNING, "graph",
                    f"High fan-in at {store_path!r}: {len(evolved)} EVOLVED "
                    f"writers ({names}). Review for potential double-counting.",
                ))

        # 3. Coupling density
        n = len(processes)
        if n > 1:
            max_edges = n * (n - 1)
            actual_edges = G.number_of_edges()
            density = actual_edges / max_edges
            if density > 0.5:
                results.append(ValidationResult(
                    Severity.WARNING, "graph",
                    f"High coupling density: {density:.0%} "
                    f"({actual_edges}/{max_edges} possible edges). "
                    f"Consider decomposing into sub-composites.",
                ))

        # 4. Unfed inputs
        all_written: set[str] = set()
        for proc_name, proc in processes.items():
            topo = topology.get(proc_name, {})
            for port_name, port in proc.ports_schema().items():
                if port.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE, PortRole.LATCHED):
                    all_written.add(topo.get(port_name, port_name))

        for proc_name, proc in processes.items():
            topo = topology.get(proc_name, {})
            for port_name, port in proc.ports_schema().items():
                if port.role == PortRole.INPUT:
                    sp = topo.get(port_name, port_name)
                    if sp not in all_written:
                        results.append(ValidationResult(
                            Severity.WARNING, "graph",
                            f"Unfed input: {proc_name}.{port_name} reads from "
                            f"{sp!r} but no process writes there. "
                            f"Will use default value only.",
                        ))

        return results

    def to_dict(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> dict:
        """Export graph as JSON-serializable dict."""
        G = self.build_graph(processes, topology)
        return nx.node_link_data(G)


# ═══════════════════════════════════════════════════════════════════════════
# Subsystem 4: Coupling Auditor
# ═══════════════════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "by",
    "from", "with", "is", "are", "at", "on", "this", "that",
})


class CouplingAuditor:
    """Detects potential reaction duplication via description overlap.

    Heuristic: if two EVOLVED writers at the same store path have
    descriptions sharing many keywords, they may model the same
    biological mechanism.

    - WARNING: high keyword overlap between descriptions.
    """

    min_shared_words: int = 3

    def check(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        spm = _store_port_map(processes, topology)

        for store_path, entries in spm.items():
            writers = _writers(entries)
            if len(writers) < 2:
                continue

            for i, e1 in enumerate(writers):
                for e2 in writers[i + 1:]:
                    desc1 = (e1.port.description or "").lower()
                    desc2 = (e2.port.description or "").lower()
                    if not desc1 or not desc2:
                        continue

                    words1 = set(desc1.split()) - _STOP_WORDS
                    words2 = set(desc2.split()) - _STOP_WORDS
                    overlap = words1 & words2

                    if len(overlap) >= self.min_shared_words:
                        results.append(ValidationResult(
                            Severity.WARNING, "coupling",
                            f"Potential duplication at {store_path!r}: "
                            f"{e1.proc_name}.{e1.port_name} and "
                            f"{e2.proc_name}.{e2.port_name} share description "
                            f"terms: {overlap}. Verify derivatives are "
                            f"complementary, not duplicating.",
                        ))

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class CompositeValidator:
    """Orchestrates all validation subsystems.

    Parameters
    ----------
    check_units:
        Run pint-based unit analysis.
    check_semantics:
        Run ontology-based semantic analysis.
    check_graph:
        Run interaction graph analysis (cycles, fan-in, density).
    check_coupling:
        Run description-overlap coupling auditor.
    strict:
        If ``True``, promote all warnings to errors.
    """

    def __init__(
        self,
        check_units: bool = True,
        check_semantics: bool = True,
        check_graph: bool = True,
        check_coupling: bool = True,
        strict: bool = False,
    ) -> None:
        self.checkers: list[Any] = []
        if check_units:
            self.checkers.append(UnitChecker())
        if check_semantics:
            self.checkers.append(SemanticChecker())
        self._graph_analyzer: GraphAnalyzer | None = None
        if check_graph:
            self._graph_analyzer = GraphAnalyzer()
            self.checkers.append(self._graph_analyzer)
        if check_coupling:
            self.checkers.append(CouplingAuditor())
        self.strict = strict

    def validate(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
    ) -> ValidationReport:
        """Run all enabled checks and return a :class:`ValidationReport`."""
        # Structural validation first (reuse existing logic)
        structural_errors = validate_topology(processes, topology)
        results = [
            ValidationResult(Severity.ERROR, "structure", e)
            for e in structural_errors
        ]

        # Run each subsystem
        for checker in self.checkers:
            if hasattr(checker, "check"):
                results.extend(checker.check(processes, topology))
            elif hasattr(checker, "analyze"):
                results.extend(checker.analyze(processes, topology))

        # Strict mode: promote warnings to errors
        if self.strict:
            results = [
                ValidationResult(Severity.ERROR, r.category, r.message)
                if r.level == Severity.WARNING else r
                for r in results
            ]

        # Build interaction graph if available
        graph_data = None
        if self._graph_analyzer is not None:
            graph_data = self._graph_analyzer.to_dict(processes, topology)

        return ValidationReport(results=results, interaction_graph=graph_data)
