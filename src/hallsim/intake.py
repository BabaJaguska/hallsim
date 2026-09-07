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

from hallsim.process import PortRole

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


@dataclass
class CouplingResponse:
    """How a readout answers a parameter driven across its plausible range.

    A module can pass every numerical check and still be useless as a
    component: if the readout is already near its ceiling at the low end of
    the range you would drive it over, the module cannot discriminate the
    thing you wanted it to report. Dwivedi 2014's IL-6 arm sits at 93.8% of
    its ceiling at the published disease point and gains 5.6 points over the
    next three decades — measured after a reviewer spent half an hour on it,
    which is what this exists to avoid.
    """

    param: str
    readout: str
    multipliers: tuple[float, ...]
    values: tuple[float, ...]
    #: Fraction of the achievable span already reached at the *top of the
    #: plausible range*. Near 1.0 means the range is spent before it starts.
    span_used: float
    monotone: bool

    #: Above this, most of the achievable response is spent before the
    #: coupling edge reaches the top of its plausible drive, so the module
    #: cannot discriminate across the range it would actually see.
    SATURATION = 0.8

    @property
    def saturated(self) -> bool:
        return self.span_used >= self.SATURATION

    def __str__(self) -> str:
        return (
            f"{self.readout} vs {self.param}: "
            + ", ".join(
                f"{m:g}x={v:.4g}"
                for m, v in zip(self.multipliers, self.values)
            )
            + f" | span used by the plausible range {self.span_used:.1%}"
            + ("  SATURATED" if self.saturated else "")
            + ("" if self.monotone else "  NON-MONOTONE")
        )


def emitted_species(process, pattern: str) -> tuple[str, ...]:
    """Species matching ``pattern`` that ``process`` actually *produces*.

    Mentioning a quantity is not emitting it. A module imported to supply an
    output it only ever consumes contributes nothing — the single most common
    way a candidate fails (Ihekwaba 2004, Bekkar 2018, Singh 2006 and
    Dwivedi's cell arm all failed exactly here).
    """
    import re

    from hallsim.diagnostics import coupling_source_verdict

    rx = re.compile(pattern, re.I)
    names = getattr(process, "_species_names", ()) or ()
    labels = dict(getattr(process, "_species_labels", ()) or ())
    if not names:  # a format with no species list (XPP): use the ports
        names = tuple(process.ports_with_role(PortRole.EVOLVED))
    wanted = [
        sid
        for sid in names
        if rx.search(sid) or rx.search(labels.get(sid, ""))
    ]
    if not wanted:
        return ()

    reaction_based = bool(getattr(process, "_stoichiometry", None))
    out = []
    for sid in wanted:
        try:
            produced = (
                coupling_source_verdict(process, sid).produced
                if reaction_based
                else _produced_by_derivative(process, sid)
            )
        except Exception:
            continue
        if produced:
            out.append(labels.get(sid) or sid)
    return tuple(sorted(set(out)))


def _produced_by_derivative(process, name: str) -> bool:
    """Whether ``name`` has a source term, judged from the derivative.

    An XPP model has no reactions at all, and an SBML model may express its
    dynamics as rate rules, so scanning reaction products sees nothing in
    either. The format-agnostic statement is that a species is *produced* when
    something adds to it that does not depend on the species itself — so hold
    it at zero and ask whether its derivative is positive.
    """
    import numpy as np

    from hallsim.composite import single_process_composite

    comp = single_process_composite(process)
    keys = list(comp.store_keys())
    idx = [i for i, k in enumerate(keys) if k.split("/")[-1] == name]
    if not idx:
        return False
    rhs, _ = comp.build_rhs()
    y = jnp.asarray(comp.initial_state_vec()).at[jnp.asarray(idx)].set(0.0)
    dy = np.asarray(rhs(0.0, y))
    return bool(np.any(dy[np.asarray(idx)] > 0))


def coupling_response(
    process,
    param: str,
    readout: str,
    t_end: float,
    multipliers=(1.0, 2.0, 5.0, 10.0, 100.0, 1000.0),
    plausible: float = 10.0,
) -> "CouplingResponse":
    """Drive ``param`` across ``multipliers`` and report where ``readout``
    lands, so a saturated module is caught before a reviewer is spent.

    ``plausible`` is the top of the range the coupling edge would realistically
    drive — measured SASP IL-6 induction spans roughly 3x to 40x, so 10x is a
    middling default. ``span_used`` is the fraction of the full swept span
    already reached there.
    """
    import numpy as np

    from hallsim.composite import single_process_composite
    from hallsim.process import read_param, write_param
    from hallsim.scheduler import Scheduler

    base = float(np.asarray(read_param(process, param)))
    values = []
    for m in multipliers:
        proc = write_param(process, param, base * m)
        comp = single_process_composite(proc)
        idx = comp.store_index()
        key = next(k for k in idx if k.endswith("/" + readout))
        res = Scheduler().run(
            comp,
            (0.0, t_end),
            macro_dt=t_end,
            save_dt=t_end,
            y0=comp.initial_state_vec(),
        )
        values.append(float(np.asarray(res.ys)[-1, idx[key]]))
    v = np.asarray(values)
    span = v.max() - v.min()
    # The sweep is log-spaced, so interpolate in log-multiplier; doing it
    # linearly understates where a plausible drive actually lands.
    at_plausible = float(np.interp(np.log(plausible), np.log(multipliers), v))
    span_used = 1.0 if span == 0 else abs(at_plausible - v[0]) / span
    monotone = bool(np.all(np.diff(v) >= 0) or np.all(np.diff(v) <= 0))
    return CouplingResponse(
        param=param,
        readout=readout,
        multipliers=tuple(float(m) for m in multipliers),
        values=tuple(values),
        span_used=float(span_used),
        monotone=monotone,
    )


def triage_process(
    process,
    t_end: float,
    *,
    xml_path: str | None = None,
    name: str | None = None,
    produces: str | None = None,
    sweep: tuple | None = None,
) -> TriageVerdict:
    """Screen one already-imported process. ``xml_path`` adds SBML metadata.

    ``produces`` is a regex the deposit must *emit* — not mention. A module
    imported to supply an output it only ever consumes contributes nothing,
    and that is the single most common way a candidate fails, so failing it
    **blocks**.

    ``sweep`` is ``(param, readout)`` or ``(param, readout, plausible)``: drive
    the parameter a coupling edge would drive across its plausible range and
    check the readout still moves there. A module already at its ceiling
    cannot discriminate. Flags rather than blocks — a saturated module is
    still usable at a different operating point.
    """
    from hallsim.diagnostics import screen_process

    label = name or getattr(process, "name", type(process).__name__)
    blockers: list[str] = []
    flags: list[str] = []

    n_species = len(process.ports_schema())
    n_parameters = len(getattr(process, "parameters", {}) or {})

    # Trigger defects are properties of the event expressions, so they cost
    # one tree walk and are decided before anything is integrated. They block
    # rather than flag: a model whose output depends on round-off at an event
    # boundary has no reproducible behaviour to screen.
    events = getattr(process, "_events", ())
    if events:
        from hallsim.sbml_events import trigger_pathologies

        blockers.extend(trigger_pathologies(events))

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
        if report.blocking:
            blockers.append(f"numerical screen: {report}")
        else:
            # Everything else the screen found describes a run that produced
            # a usable trajectory, so it qualifies the model rather than
            # disqualifying it. Rejecting on tolerance sensitivity in
            # particular discarded the oscillators for oscillating, and threw
            # away the tolerance the screen had just measured.
            flags.extend(report.advisories)
            if report.not_at_rest:
                flags.append(f"not at rest: tau={report.rest_tau:.3g}")
    except Exception as exc:
        blockers.append(f"screen raised: {exc}")

    if produces is not None:
        emitted = emitted_species(process, produces)
        if emitted:
            flags.append(f"emits {', '.join(emitted)}")
        else:
            blockers.append(
                f"emits nothing matching /{produces}/ — it may consume or "
                f"merely mention the quantity, which composes to nothing"
            )

    if sweep is not None:
        param, readout = sweep[0], sweep[1]
        plausible = sweep[2] if len(sweep) > 2 else 10.0
        try:
            resp = coupling_response(
                process, param, readout, t_end, plausible=plausible
            )
            if resp.saturated:
                flags.append(
                    f"coupling range: {resp.span_used:.0%} of the achievable "
                    f"{readout} response is spent by {plausible:g}x {param} "
                    f"— saturated across the range the edge would drive"
                )
            if not resp.monotone:
                flags.append(
                    f"coupling range: {readout} is non-monotone in " f"{param}"
                )
        except Exception as exc:
            flags.append(f"coupling sweep failed: {type(exc).__name__}: {exc}")

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
    model_id,
    t_end: float = 10.0,
    name: str = "m",
    *,
    produces: str | None = None,
    sweep: tuple | None = None,
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
        process,
        t_end,
        xml_path=str(xml_path),
        name=str(model_id),
        produces=produces,
        sweep=sweep,
    )


@dataclass(frozen=True)
class FitReport:
    """How well a model reproduces the dataset its own paper fitted."""

    chi2: float
    n_points: int
    per_path: dict[str, float] = field(default_factory=dict)
    reported: float | None = None
    rtol: float = 0.01

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.n_points if self.n_points else float("nan")

    @property
    def relative_error(self) -> float | None:
        """|χ² − reported| / reported, or None when nothing was claimed."""
        if self.reported is None or not self.reported:
            return None
        return abs(self.chi2 - self.reported) / abs(self.reported)

    @property
    def reproduces(self) -> bool | None:
        """Whether the deposit reproduces its paper's stated χ² within
        ``rtol``. ``None`` when the paper states none — then the χ² is a
        goodness-of-fit number and not a reproduction check."""
        rel = self.relative_error
        return None if rel is None else rel <= self.rtol

    def __str__(self) -> str:
        head = f"chi2 {self.chi2:.4f} over {self.n_points} points"
        head += f" (reduced {self.reduced_chi2:.4f})"
        if self.reported is not None:
            verdict = "REPRODUCES" if self.reproduces else "DIVERGES"
            head += (
                f" vs reported {self.reported:.4f} "
                f"— {self.relative_error:.2%}, {verdict}"
            )
        worst = sorted(self.per_path.items(), key=lambda kv: -kv[1])[:3]
        tail = ", ".join(f"{k} {v:.3g}" for k, v in worst)
        return f"{head}\n  worst: {tail}" if tail else head


def published_fit_chi2(
    composite,
    observations: dict,
    *,
    scales: dict | None = None,
    reported: float | None = None,
    rtol: float = 0.01,
    save_dt: float | None = None,
    y0=None,
    scheduler=None,
) -> FitReport:
    """``χ² = Σ ((sim − obs) / sd)²`` against the data the model was fitted to.

    ``observations`` maps an observable to ``(times, values, sds)``. The key is
    a store path, or a tuple of store paths that are **summed** — a measured
    quantity is often a total over states the model keeps apart (free + bound,
    new + old). ``scales`` optionally multiplies an observable after summing,
    for the fitted observable-scaling factors a fit deposits alongside its rate
    constants. The run spans ``[0, max(times)]`` through the Scheduler and each
    observable is sampled at its own times by linear interpolation.

    ``reported`` is the paper's own stated χ², which turns the number into a
    **reproduction check**: does this deposit still produce the fit its
    publication claims?

    That question is the one to ask first of any curated model, and asking it
    is what licenses everything after. A deposit that reproduces its paper's χ²
    is the object the paper analysed, so any defect found later belongs to the
    paper rather than to the deposition or to the import. A deposit that does
    not is a broken artefact, and no finding against it means anything.
    """
    import numpy as np

    from hallsim.scheduler import Scheduler

    if not observations:
        raise ValueError("published_fit_chi2 needs at least one observed path")
    scales = scales or {}
    known = set(composite.store_keys())

    def _paths(key):
        return (key,) if isinstance(key, str) else tuple(key)

    missing = sorted(
        {p for key in observations for p in _paths(key) if p not in known}
    )
    if missing:
        raise KeyError(
            f"observed paths absent from the composite: {missing}. "
            "Map the paper's observable names onto store paths first."
        )

    t_end = max(
        float(np.max(np.asarray(t))) for t, _, _ in observations.values()
    )
    if t_end <= 0:
        raise ValueError("observation times must span a positive interval")
    dt = save_dt if save_dt is not None else t_end / 400.0
    y0 = composite.initial_state_vec() if y0 is None else y0
    result = (scheduler or Scheduler()).run(
        composite, t_span=(0.0, t_end), macro_dt=t_end, y0=y0, save_dt=dt
    )
    ts = np.asarray(result.ts)

    per_path: dict[str, float] = {}
    total, n = 0.0, 0
    for key, (obs_t, obs_y, obs_sd) in observations.items():
        obs_t = np.asarray(obs_t, float)
        obs_y = np.asarray(obs_y, float)
        obs_sd = np.asarray(obs_sd, float)
        if np.any(obs_sd <= 0):
            raise ValueError(f"non-positive sd in observations for {key!r}")
        paths = _paths(key)
        traj = sum(np.asarray(result.get(p)) for p in paths)
        sim = float(scales.get(key, 1.0)) * np.interp(obs_t, ts, traj)
        contrib = float(np.sum(((sim - obs_y) / obs_sd) ** 2))
        per_path[" + ".join(paths)] = contrib
        total += contrib
        n += len(obs_t)

    report = FitReport(
        chi2=total,
        n_points=n,
        per_path=per_path,
        reported=reported,
        rtol=rtol,
    )
    if report.reproduces is False:
        log.warning(
            "published_fit_chi2: %s does not reproduce its reported fit "
            "(chi2 %.4f vs %.4f, %.1f%%). A finding against a deposit that "
            "does not reproduce its own paper is a finding about the deposit.",
            getattr(composite, "name", "composite"),
            report.chi2,
            reported,
            100 * report.relative_error,
        )
    return report


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
