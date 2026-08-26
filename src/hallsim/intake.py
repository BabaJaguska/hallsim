"""Cheap triage for an imported model — the gate before the review panel.

The panel in ``.claude/agents/`` (bench-scientist, mathematician, physicist)
is the real scientific review, and it costs three deep reviews per model. That
does not scale to a candidate list. This is the pass that runs first and
answers a narrower question: is this model *worth* reviewing, or does it fail
on something a machine can see?

Everything here is mechanical — SBML metadata plus one solve. It cannot tell a
fitted parameter from a measured one, or catch a citation that does not say
what it is cited for; that is exactly what the panel is for. It can reject a
model that will not parse, has no declared clock, sits far from its own rest
state, or blows up. Those never reach a reviewer.

    verdict = triage_sbml(157)
    if verdict.escalate:
        ...  # hand to the panel
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax.numpy as jnp

log = logging.getLogger(__name__)

#: ‖f(y₀)‖ relative to ‖y₀‖ above which the declared IC is not a rest state.
REST_RESIDUAL_FLAG = 1.0
#: Fraction of species carrying an ontology ID below which composition is blind.
ANNOTATION_FLAG = 0.5


@dataclass(frozen=True)
class TriageVerdict:
    """Mechanical screen of one model. ``escalate`` gates the review panel."""

    name: str
    status: str  # "pass" | "flag" | "reject"
    blockers: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    n_species: int = 0
    n_parameters: int = 0
    time_unit_declared: bool = False
    annotation_coverage: float = 0.0
    rest_residual: float = float("nan")
    screen: object | None = field(default=None, repr=False)

    @property
    def escalate(self) -> bool:
        """True when a reviewer's time is worth spending on this model."""
        return self.status != "reject"

    def __str__(self) -> str:
        parts = [f"[{self.status.upper()}] {self.name}"]
        parts.append(
            f"{self.n_species} species, {self.n_parameters} params, "
            f"annot {self.annotation_coverage:.0%}, "
            f"‖f(y0)‖/‖y0‖ {self.rest_residual:.3g}"
        )
        if self.blockers:
            parts.append("blockers: " + "; ".join(self.blockers))
        if self.flags:
            parts.append("flags: " + "; ".join(self.flags))
        return "\n  ".join(parts)


def rest_residual(process) -> float:
    """``‖f(y₀)‖ / ‖y₀‖`` — how hard the model is moving at its declared IC.

    A published initial condition is often a fitted experimental starting
    point rather than a steady state, so the run is mostly relaxation and
    every downstream contrast measures timing. One number, no trajectory.
    """
    from hallsim.composite import Composite

    name = type(process).__name__
    ports = process.ports_schema()
    composite = Composite(
        {name: process},
        {name: {port: f"{name}/{port}" for port in ports}},
        validate=False,
        semantic_validation=False,
    )
    y0 = composite.initial_state_vec()
    rhs, _ = composite.build_rhs()
    dy = jnp.asarray(rhs(0.0, y0))
    scale = float(jnp.linalg.norm(y0))
    return float(jnp.linalg.norm(dy)) / (scale if scale > 0 else 1.0)


def triage_process(
    process,
    t_end: float,
    *,
    xml_path: str | None = None,
    name: str | None = None,
) -> TriageVerdict:
    """Screen one already-imported process. ``xml_path`` adds SBML metadata."""
    from hallsim.diagnostics import screen_process

    label = name or getattr(process, "name", type(process).__name__)
    blockers: list[str] = []
    flags: list[str] = []

    n_species = len(process.ports_schema())
    n_parameters = len(getattr(process, "parameters", {}) or {})

    time_declared, coverage = False, 0.0
    if xml_path is not None:
        from hallsim.sbml_import import (
            _extract_native_time_seconds,
            _extract_species_ontology,
            _precheck_sbml_supported,
            _preprocess_sbml,
        )

        # Match what import consumes: function definitions are expanded
        # first, so the raw file reports calls the pipeline already handles.
        unsupported = _precheck_sbml_supported(_preprocess_sbml(xml_path))
        if unsupported:
            blockers.append(
                f"unsupported constructs: {'; '.join(unsupported)}"
            )
        _, time_declared = _extract_native_time_seconds(xml_path)
        if not time_declared:
            flags.append(
                "no declared time unit — its clock is a guess, and "
                "reconciling it onto a shared axis may be silently wrong"
            )
        ontology = _extract_species_ontology(xml_path)
        if ontology:
            coverage = sum(bool(v) for v in ontology.values()) / len(ontology)
        if coverage < ANNOTATION_FLAG:
            flags.append(
                f"only {coverage:.0%} of species carry an ontology ID — "
                "semantic composition checks are blind here"
            )

    residual = float("nan")
    try:
        residual = rest_residual(process)
        if residual > REST_RESIDUAL_FLAG:
            flags.append(
                f"IC is not a rest state (‖f(y0)‖/‖y0‖ = {residual:.3g}) — "
                "most of the run is relaxation"
            )
    except Exception as exc:
        flags.append(f"rest residual not computable: {exc}")

    report = None
    try:
        report = screen_process(process, t_end)
        if not report.ok:
            blockers.append(f"numerical screen: {report}")
        elif report.not_at_rest:
            flags.append(f"not at rest: tau={report.rest_tau:.3g}")
    except Exception as exc:
        blockers.append(f"screen raised: {exc}")

    status = "reject" if blockers else ("flag" if flags else "pass")
    return TriageVerdict(
        name=label,
        status=status,
        blockers=tuple(blockers),
        flags=tuple(flags),
        n_species=n_species,
        n_parameters=n_parameters,
        time_unit_declared=time_declared,
        annotation_coverage=coverage,
        rest_residual=residual,
        screen=report,
    )


def triage_sbml(
    model_id, t_end: float = 10.0, name: str = "m"
) -> TriageVerdict:
    """Import a BioModels ID or local SBML path, then triage it.

    An import that raises is itself a reject — that is the cheapest possible
    verdict and the most common one on an uncurated candidate.
    """
    from hallsim.sbml_import import _resolve_source, process_from_sbml

    try:
        xml_path, _ = _resolve_source(model_id, name)
        process = process_from_sbml(model_id, name=name)
    except Exception as exc:
        return TriageVerdict(
            name=str(model_id),
            status="reject",
            blockers=(f"import failed: {type(exc).__name__}: {exc}",),
        )
    return triage_process(
        process, t_end, xml_path=str(xml_path), name=str(model_id)
    )


def triage_batch(model_ids, t_end: float = 10.0) -> list[TriageVerdict]:
    """Triage a candidate list, keeping order. Never raises on one bad model."""
    verdicts = [triage_sbml(m, t_end=t_end) for m in model_ids]
    kept = sum(v.escalate for v in verdicts)
    log.info(
        "triage: %d/%d escalate to the panel (%d rejected)",
        kept,
        len(verdicts),
        len(verdicts) - kept,
    )
    return verdicts
