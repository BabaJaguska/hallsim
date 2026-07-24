"""Pre-flight stability screening for composed models.

A composite is only as trustworthy as its parts, and the most dangerous
failures are silent: a model that *looks* like it ran but produced
numerical garbage. Before importing a new model, composing it, or
believing any composite output, screen each constituent **on its own**.

The three failure modes to screen for:

- **exploding** — unbounded growth / NaN / Inf. Often a *numerical*
  artifact, not biology: an explicit solver at loose tolerance pumps
  energy into an oscillator until it diverges. The Geva-Zatorsky 2006
  p53 oscillator does exactly this at ``rtol=1e-4`` and is bounded by
  ``rtol=1e-5``.
- **vanishing** — every state collapses to ~0; the subsystem has lost
  its dynamics and contributes nothing.
- **tolerance-sensitive** — the load-bearing check: the trajectory
  changes materially between a loose and a tight solver tolerance. If
  the answer depends on the tolerance, it is solver-dependent and not
  yet a result. A bounded, non-vanishing, tolerance-*insensitive*
  trajectory is one you can trust.

Screening runs each model on its **native clock** (independent of any
composite-level time reconciliation) over a bounded window you choose in
the model's native time unit — a few of its own characteristic times,
*not* the composite's full horizon (integrating a fast oscillator over a
slow horizon is the pathological case the Scheduler grouping exists to
avoid; don't reproduce it in a sanity check). Solver steps are capped so
a pathological model fails fast and is flagged rather than hanging.

Use :func:`screen_process` when bringing a single model in, and
:func:`screen_composite` (pass a per-model window dict) before trusting a
composite. The companion ``demos/subsystem_diagnostics.py`` is the visual
version — it plots each subsystem solo.

**Is it the model or is it us?** When the screen flags an SBML-imported
model as exploding/unintegrable, the screen runs a second, independent
integration with **sbmltoodejax's own** stepper (the generated
``ModelStep``, which bakes the upstream odeint + its generation-time
tolerances) and reports the verdict via ``ScreenReport.framework_suspect``.
A model that HallSim fails but sbmltoodejax integrates bounded-and-finite
is a *framework* problem (solver config, atol scale, timescale grouping),
not a bad model — the decisive check that separates the two failure modes
in one shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from hallsim.composite import Composite, single_process_composite
from hallsim.config import DEFAULT_ATOL, DEFAULT_MAX_STEPS
from hallsim.scheduler import Scheduler


@dataclass
class ScreenReport:
    """Verdict from screening one process solo.

    ``ok`` is True only when none of the three flags fire. The flags are
    advisory — some models legitimately grow or decay — but every flag
    is something to understand before trusting the subsystem.
    """

    name: str
    exploding: bool
    vanishing: bool
    tolerance_sensitive: bool
    max_abs: float
    tol_rel_diff: float
    detail: str = ""
    framework_suspect: bool = False
    tunes: bool | None = None
    negative: bool = False

    @property
    def ok(self) -> bool:
        return (
            not (
                self.exploding
                or self.vanishing
                or self.tolerance_sensitive
                or self.negative
            )
            and self.tunes is not False
        )

    def __str__(self) -> str:
        flags = []
        if self.exploding:
            flags.append("EXPLODING")
        if self.vanishing:
            flags.append("VANISHING")
        if self.tolerance_sensitive:
            flags.append("TOLERANCE-SENSITIVE")
        if self.negative:
            flags.append("NEGATIVE-DOMAIN")
        if self.tunes is False:
            flags.append("NON-TUNABLE")
        if self.framework_suspect:
            flags.append("FRAMEWORK-SUSPECT")
        verdict = "ok" if self.ok else " + ".join(flags)
        return (
            f"{self.name:>14}: {verdict:<38} "
            f"max|y|={self.max_abs:.3g}  tol-rel-diff={self.tol_rel_diff:.3g}"
            + (f"  [{self.detail}]" if self.detail else "")
        )


def _on_native_clock(process):
    """Return the process running on its own native clock (time_scale=1).

    An SBMLProcess reconciled into a composite carries a non-unit
    ``time_scale``; screening should test the model's intrinsic dynamics,
    so normalise it back to native. Non-SBML processes pass through.
    """
    if hasattr(process, "native_time_seconds") and hasattr(
        process, "reconciled_to"
    ):
        return process.reconciled_to(process.native_time_seconds)
    return process


def _solo_run(process, t_end, rtol, atol, n_save, max_steps, sched_kwargs):
    comp = single_process_composite(process)
    res = Scheduler(
        rtol=rtol, atol=atol, max_steps=max_steps, **sched_kwargs
    ).run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=t_end,
        y0=comp.initial_state_vec(),
        save_dt=t_end / n_save,
    )
    return np.asarray(jnp.stack([res.get(k) for k in res.keys], axis=-1))


def _sbmltoodejax_native_finite(process, t_end: float, n_steps: int):
    """Integrate an SBML-imported process with sbmltoodejax's own stepper.

    Rolls the generated ``ModelStep`` (which wraps the upstream odeint at
    its generation-time tolerances) forward over ``[0, t_end]`` on the
    model's native clock — the independent reference for the "is it the
    model or is it us?" check. Returns ``(max_abs, finite)``, or ``None``
    for a non-SBML process (no native reference exists) or if the native
    run itself errors.
    """
    model = getattr(process, "_model", None)
    if model is None or not hasattr(process, "_species_y0"):
        return None
    try:
        y0 = jnp.asarray(process._species_y0)
        c0 = process._c
        dt = t_end / n_steps

        def step(carry, _):
            y, w, c, t = carry
            y, w, c, t = model(y, w, c, t, dt)
            return (y, w, c, t), y

        _, ys = jax.lax.scan(
            step, (y0, process._w0, c0, 0.0), None, length=n_steps
        )
        ys = np.asarray(ys)
    except Exception:
        return None
    finite = bool(np.all(np.isfinite(ys)))
    max_abs = float(np.nanmax(np.abs(ys))) if ys.size else 0.0
    return max_abs, finite


def _tunes(process, t_end: float, n_probe: int = 2, sched_kwargs=None):
    """Forward-mode gradient finiteness — the 'tunes' half of the rule.

    The constituents-first rule requires each model both *runs* and
    *tunes*: a finite forward-mode gradient of a state summary w.r.t. one
    of its parameters. This probes up to ``n_probe`` of the process's
    ``calibratable_params`` with ``jax.jvp`` through ``Scheduler.run``
    (the same ForwardMode path calibration uses). A model whose explicit
    forward sensitivities overflow (stiff) is retried on the implicit
    solver — that is how it would actually be calibrated.

    Returns ``(tunes, needs_implicit)``: ``tunes`` True if some path gives
    finite gradients, False if neither does, ``needs_implicit`` True when
    only the implicit solver works. Returns ``(None, False)`` when the
    process exposes no calibratable parameters.
    """
    from hallsim.calibration import _substitute_param

    sched_kwargs = sched_kwargs or {}
    probes = process.calibratable_params()[:n_probe]
    if not probes:
        return None, False

    def all_finite(sched):
        for cp in probes:

            def loss(val, field=cp.field):
                proc = _substitute_param(process, field, val)
                comp = single_process_composite(proc)
                res = sched.run(
                    comp,
                    t_span=(0.0, t_end),
                    macro_dt=t_end,
                    y0=comp.initial_state_vec(),
                    adjoint=dfx.ForwardMode(),
                )
                return jnp.sum(res.ys[-1])

            try:
                _, tangent = jax.jvp(loss, (cp.default,), (1.0,))
            except Exception:
                return False
            if not bool(jnp.isfinite(tangent)):
                return False
        return True

    if all_finite(Scheduler(**sched_kwargs)):
        return True, False

    # warm_up runs eagerly outside the jvp trace so the stiffness verdict is
    # cached before tracing (analysis can't read tracer Jacobians).
    s_imp = Scheduler(auto_stiffness=True)
    try:
        s_imp.warm_up(
            single_process_composite(process),
            t_span=(0.0, t_end),
            macro_dt=t_end,
        )
    except Exception:
        return False, False
    if all_finite(s_imp):
        return True, True
    return False, False


def screen_process(
    process,
    t_end: float,
    *,
    rtol_loose: float = 1e-3,
    rtol_tight: float = 1e-7,
    tol_rel_threshold: float = 0.05,
    growth_threshold: float = 1e3,
    n_save: int = 400,
    max_steps: int = DEFAULT_MAX_STEPS,
    check_tunability: bool = True,
    **sched_kwargs,
) -> ScreenReport:
    """Screen one process solo over ``[0, t_end]`` native time units.

    ``t_end`` is in the model's **native** time unit — pick a few of its
    own characteristic times (e.g. ~30000 for a second-scale NF-κB
    oscillator, ~100 for an hour-scale p53 oscillator, ~50 for a
    day-scale senescence model). The process is run on its native clock
    regardless of any composite reconciliation.

    Flags ``exploding`` (non-finite, or peak > ``growth_threshold`` ×
    initial scale and still rising), ``vanishing`` (all states end within
    1e-9 of zero), ``tolerance_sensitive`` (loose vs tight tolerance
    differ by more than ``tol_rel_threshold``, peak-normalised), and
    ``negative`` (a state that started non-negative dipped materially below
    zero — a concentration/activity that went out of domain). A solver
    that exceeds ``max_steps`` is reported as exploding/unintegrable.

    When an SBML-imported process is flagged exploding/unintegrable, a
    second integration with sbmltoodejax's own stepper decides whether the
    model is at fault: a bounded, finite native run sets
    ``framework_suspect`` (the failure is HallSim's, not the model's).

    With ``check_tunability`` (default), also verifies the *tunes* half of
    the constituents-first rule — a finite forward-mode gradient through
    ``Scheduler.run`` — setting ``tunes`` (``False`` makes the report not
    ``ok``; a model that tunes only on the implicit solver is noted in
    ``detail``).

    ``sched_kwargs`` are forwarded to the :class:`Scheduler` (e.g.
    ``solver=dfx.Kvaerno5()`` or ``auto_stiffness=True``) so the screen runs
    under the *same* solver configuration production will — required to screen
    a stiff model, which the default explicit solver would flag as exploding.
    """
    proc = _on_native_clock(process)
    name = getattr(proc, "_name", type(proc).__name__)

    try:
        # atol is held at the production default across both runs so the
        # loose-vs-tight comparison isolates rtol sensitivity (and the
        # screen integrates exactly what the Scheduler does in production).
        y_tight = _solo_run(
            proc,
            t_end,
            rtol_tight,
            DEFAULT_ATOL,
            n_save,
            max_steps,
            sched_kwargs,
        )
        y_loose = _solo_run(
            proc,
            t_end,
            rtol_loose,
            DEFAULT_ATOL,
            n_save,
            max_steps,
            sched_kwargs,
        )
    except Exception as exc:  # max_steps / non-finite blow the solve up
        native = _sbmltoodejax_native_finite(proc, t_end, n_save)
        suspect = native is not None and native[1]
        detail = f"solver failed: {type(exc).__name__}"
        if suspect:
            detail += (
                f"; sbmltoodejax integrates it bounded (max|y|={native[0]:.3g})"
                " — framework issue, not the model"
            )
        return ScreenReport(
            name=name,
            exploding=True,
            vanishing=False,
            tolerance_sensitive=True,
            max_abs=float("inf"),
            tol_rel_diff=float("inf"),
            detail=detail,
            framework_suspect=suspect,
        )

    finite = bool(np.all(np.isfinite(y_tight)))
    peak = float(np.nanmax(np.abs(y_tight))) if y_tight.size else 0.0
    init_scale = max(float(np.max(np.abs(y_tight[0]))) if finite else 0.0, 1.0)
    final_peak = (
        float(np.max(np.abs(y_tight[int(0.9 * len(y_tight)) :])))
        if finite
        else float("inf")
    )
    mid_peak = (
        float(np.max(np.abs(y_tight[: int(0.5 * len(y_tight))])))
        if finite
        else 0.0
    )
    exploding = (not finite) or (
        peak > growth_threshold * init_scale and final_peak > 2.0 * mid_peak
    )
    vanishing = finite and bool(np.all(np.abs(y_tight[-1]) < 1e-9))

    # Domain violation: a state that starts non-negative but dips materially
    # below zero during the run — a concentration/activity that went negative.
    # Gating on "started non-negative" exempts legitimately-signed states
    # (e.g. ERiQ's negative-valued regulatory integrators); the scaled
    # threshold ignores sub-percent numerical undershoot near zero.
    negative = False
    if finite and y_tight.size:
        started_nonneg = y_tight[0] >= -1e-9
        most_neg = np.min(y_tight, axis=0)
        neg_thresh = -0.02 * np.maximum(np.max(np.abs(y_tight), axis=0), 1e-12)
        negative = bool(np.any(started_nonneg & (most_neg < neg_thresh)))

    scale = max(peak, 1e-12)
    tol_rel_diff = (
        float(np.nanmax(np.abs(y_loose - y_tight)) / scale)
        if y_loose.shape == y_tight.shape and finite
        else float("inf")
    )
    tolerance_sensitive = tol_rel_diff > tol_rel_threshold

    detail = ""
    if exploding and not finite:
        detail = "non-finite (NaN/Inf)"
    elif exploding:
        detail = f"grew {peak / init_scale:.0f}x initial, still rising"
    elif tolerance_sensitive:
        detail = (
            f"rtol {rtol_loose:.0e} vs {rtol_tight:.0e} disagree "
            f"by {tol_rel_diff * 100:.0f}%"
        )
    elif negative:
        detail = "a non-negative state went materially negative"

    framework_suspect = False
    if exploding:
        native = _sbmltoodejax_native_finite(proc, t_end, n_save)
        if native is not None and native[1]:
            framework_suspect = True
            detail += (
                f"; sbmltoodejax integrates it bounded (max|y|={native[0]:.3g})"
                " — framework issue, not the model"
            )

    tunes = None
    if check_tunability and not exploding:
        tunes, needs_implicit = _tunes(proc, t_end, sched_kwargs=sched_kwargs)
        if tunes is False:
            detail = (detail + "; " if detail else "") + "non-finite gradient"
        elif needs_implicit:
            detail = (detail + "; " if detail else "") + (
                "tunes only under the implicit solver (auto_stiffness=True)"
            )

    return ScreenReport(
        name=name,
        exploding=exploding,
        vanishing=vanishing,
        tolerance_sensitive=tolerance_sensitive,
        max_abs=peak,
        tol_rel_diff=tol_rel_diff,
        detail=detail,
        framework_suspect=framework_suspect,
        tunes=tunes,
        negative=negative,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Coupling-source suitability — "is this model a good addition to a composite?"
# ═══════════════════════════════════════════════════════════════════════════

SUITABLE = "suitable"
DEAD_SINK = "dead_sink"
UNBOUNDED_ACCUMULATOR = "unbounded_accumulator"
DEPLETING_RESERVOIR = "depleting_reservoir"
STATIC = "static"


@dataclass(frozen=True)
class CouplingSource:
    """Verdict for one state of a candidate model used as a coupling source.

    A *coupling source* is a state another model reads (via an INPUT port)
    and adds into its own derivative. Only a **bounded, actively-turned-over**
    state is safe to drive another model with; the other verdicts each name a
    concrete failure mode:

    - ``suitable`` — produced and consumed; reaches a bounded quasi-steady
      level under sustained input. Safe to couple from.
    - ``dead_sink`` — produced, consumed by nothing, and read by no rate law
      (an inert accumulator the importer freezes at its initial value).
      Reads as a constant; unfreezing it makes it diverge. **Never couple.**
    - ``unbounded_accumulator`` — produced, never consumed, but read by the
      dynamics. Grows without bound under sustained input, so an additive
      edge from it diverges. **Never couple.**
    - ``depleting_reservoir`` — consumed, never produced; only declines
      toward zero. Exports a monotone drain, not a signal. Marginal.
    - ``static`` — neither produced nor consumed; carries no dynamics.
    """

    state: str
    verdict: str
    produced: bool
    consumed: bool
    frozen: bool
    reason: str

    @property
    def ok(self) -> bool:
        return self.verdict == SUITABLE


@dataclass
class CouplingRecommendation:
    """Whether a candidate model exposes a usable coupling source.

    ``recommended`` is True when at least one *focused* state (those named
    in ``focus``, else all states) is ``suitable``. ``notes`` carry dynamic
    caveats that structure alone cannot settle — e.g. a native forcing clock
    far finer than the composite's, which makes a built-in input pulse
    unresolvable so the constituent needs a sustained external driver to
    activate under composition.
    """

    name: str
    sources: tuple[CouplingSource, ...]
    focus: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def source(self, state: str) -> CouplingSource:
        for s in self.sources:
            if s.state == state:
                return s
        raise KeyError(f"{self.name!r} has no state {state!r}")

    @property
    def _focused(self) -> tuple[CouplingSource, ...]:
        if not self.focus:
            return self.sources
        return tuple(s for s in self.sources if s.state in self.focus)

    @property
    def suitable(self) -> tuple[str, ...]:
        return tuple(s.state for s in self._focused if s.ok)

    @property
    def recommended(self) -> bool:
        return bool(self.suitable)

    def __str__(self) -> str:
        scope = f" (for {', '.join(self.focus)})" if self.focus else ""
        verdict = "RECOMMENDED" if self.recommended else "NOT RECOMMENDED"
        lines = [
            f"{self.name}: {verdict} as a coupling source{scope}.",
            f"  {'state':<16}{'verdict':<22}why",
        ]
        shown = self._focused if self.focus else self.sources
        for s in sorted(shown, key=lambda x: (x.ok, x.state), reverse=True):
            lines.append(f"  {s.state:<16}{s.verdict:<22}{s.reason}")
        if self.suitable:
            lines.append(f"  → couple from: {', '.join(self.suitable)}")
        for note in self.notes:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


def _reaction_roles(proc):
    """``(produced, consumed)`` boolean arrays per species.

    Read off the stoichiometric matrix sbmltoodejax bakes into the imported
    model (species × reactions): a positive entry produces the species in
    some reaction, a negative one consumes it. This is the same matrix the
    rate law integrates, so producer/consumer structure is exact, not
    re-parsed from the SBML.
    """
    host = getattr(proc._model, "modelstepfunc", proc._model)
    sm = np.asarray(host.ratefunc.stoichiometricMatrix)
    return np.any(sm > 0, axis=1), np.any(sm < 0, axis=1)


def coupling_source_verdict(proc, state: str) -> CouplingSource:
    """Classify one state of an imported model as a coupling source.

    See :class:`CouplingSource` for the verdict meanings. Used both
    standalone (guard a single proposed edge) and by
    :func:`recommend_coupling_source` (survey every state).
    """
    if not hasattr(proc, "_species_names"):
        raise TypeError(
            "coupling-source analysis needs an imported SBML process "
            "(stoichiometry-based); got a generic Process."
        )
    names = list(proc._species_names)
    if state not in names:
        raise KeyError(f"{proc._name!r} has no state {state!r}")
    i = names.index(state)
    produced, consumed = _reaction_roles(proc)
    p, c = bool(produced[i]), bool(consumed[i])
    frozen = i in proc._frozen_indices

    if frozen:
        verdict, reason = DEAD_SINK, (
            "produced, consumed by nothing, read by no rate law — frozen at "
            "its initial value; coupling from it feeds a constant"
        )
    elif p and not c:
        verdict, reason = UNBOUNDED_ACCUMULATOR, (
            "produced but never consumed — grows without bound under "
            "sustained input; an additive edge from it diverges"
        )
    elif c and not p:
        verdict, reason = DEPLETING_RESERVOIR, (
            "consumed but never replenished — only declines toward zero; "
            "exports a monotone drain, not a signal"
        )
    elif p and c:
        verdict, reason = SUITABLE, (
            "produced and consumed — bounded, actively turned over"
        )
    else:
        verdict, reason = STATIC, "neither produced nor consumed — no dynamics"
    return CouplingSource(
        state=state,
        verdict=verdict,
        produced=p,
        consumed=c,
        frozen=frozen,
        reason=reason,
    )


def recommend_coupling_source(
    proc,
    *,
    target_states: tuple[str, ...] = (),
    canonical_time_seconds: float | None = None,
) -> CouplingRecommendation:
    """Recommend whether an imported model is a good coupling addition.

    Surveys every state's coupling-source suitability (structural, from the
    stoichiometry) and, when ``canonical_time_seconds`` is given, adds a
    clock-scale caveat: a model whose native forcing clock is far finer than
    the composite's cannot resolve its own built-in input pulse on the shared
    axis, so it needs a sustained external driver to activate at all.

    Pass ``target_states`` to focus the verdict on the biologically-intended
    output(s) — the states you actually mean to couple from — so
    ``recommended`` answers *for that coupling intent*, not merely whether
    some usable state exists somewhere in the model.
    """
    sources = tuple(
        coupling_source_verdict(proc, s) for s in proc._species_names
    )
    notes: list[str] = []

    bad = [s for s in sources if s.state in target_states and not s.ok]
    if bad:
        notes.append(
            "requested source(s) unusable: "
            + ", ".join(f"{s.state} ({s.verdict})" for s in bad)
        )

    native = getattr(proc, "native_time_seconds", None)
    if canonical_time_seconds is not None and native:
        ratio = canonical_time_seconds / native
        if ratio > 100.0:
            notes.append(
                f"native forcing clock ({native:g}s) is {ratio:.0f}× finer "
                f"than the composite clock ({canonical_time_seconds:g}s); a "
                "built-in input pulse is unresolved on the shared axis — this "
                "constituent needs a sustained external driver to activate"
            )
    return CouplingRecommendation(
        name=proc._name,
        sources=sources,
        focus=tuple(target_states),
        notes=tuple(notes),
    )


def screen_composite(
    composite: Composite,
    t_ends: dict[str, float],
    **kwargs,
) -> list[ScreenReport]:
    """Screen named continuous processes of a composite, each solo.

    ``t_ends`` maps process name → native-time window. Only the named
    processes are screened (each model needs a window matched to its own
    timescale, so there is no single horizon that fits all). Returns one
    :class:`ScreenReport` per entry; ``assert all(r.ok for r in reports)``
    in a test.
    """
    procs = composite.continuous_processes()
    return [
        screen_process(procs[name], t_end, **kwargs)
        for name, t_end in t_ends.items()
        if name in procs
    ]


@dataclass
class SensitivityReport:
    """Whether a reporter responds to a control in a given operating regime.

    A composite can be perfectly composed and still be useless for
    calibrating or validating a reporter *in a particular regime*: if the
    control's effect saturates or dies before it reaches the reporter, the
    reporter is flat there and any fit against it is uninformative (and hides
    bugs). ``live`` is the guardrail — a nonzero in-regime sensitivity means
    there is actually a gradient to trust. ``finite`` catches a broken
    derivative path (the composite is not differentiable here at all).
    """

    reporter: str
    control: str
    value: float
    sensitivity: float
    rel_sensitivity: float
    live: bool
    finite: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.finite and self.live

    def __str__(self) -> str:
        if not self.finite:
            verdict = "NON-FINITE-GRAD"
        elif not self.live:
            verdict = "FLAT (dead in this regime)"
        else:
            verdict = "live"
        return (
            f"{self.reporter:>10} ← {self.control:<24}: {verdict:<26} "
            f"d/dctrl={self.sensitivity:+.3g}  value={self.value:.3g}  "
            f"rel={self.rel_sensitivity:.3g}"
            + (f"  [{self.detail}]" if self.detail else "")
        )


def screen_sensitivity(
    composite: Composite,
    reporters,
    hallmarks,
    *,
    baseline: dict[str, float] | None = None,
    t_end: float = 14.0,
    macro_dt: float | None = None,
    query_time: float | None = None,
    rel_threshold: float = 1e-3,
    auto_stiffness: bool = True,
) -> list["SensitivityReport"]:
    """Flag reporters that are insensitive to a hallmark *in this regime*.

    For each ``hallmark`` it differentiates every reporter's summary with
    respect to that hallmark's severity at ``baseline`` (the operating point —
    sensitivity is regime-dependent, which is the whole point), using the same
    differentiable ``apply_hallmarks`` → ``Scheduler.run`` path the calibrator
    does. A reporter whose relative sensitivity falls below ``rel_threshold``
    is ``FLAT`` — calibrating or validating it in this regime is
    uninformative. A non-finite gradient means the composite is not
    differentiable here. One pre-flight check that would catch a coupling
    driven into saturation or a reporter read outside its dynamic range.

    Reverse-mode (the multi-group scan path is a ``custom_vjp``, so forward
    mode is unavailable); the Scheduler is warmed on the concrete baseline
    first so ``auto_stiffness`` resolves its solvers outside the trace.
    """
    from hallsim.hallmarks import apply_hallmarks

    hallmarks = list(hallmarks)
    base = dict(baseline or {h: 1.0 for h in hallmarks})
    qt = jnp.atleast_1d(
        jnp.asarray(
            query_time if query_time is not None else t_end, dtype=float
        )
    )
    mdt = macro_dt if macro_dt is not None else t_end / 4.0
    sched = Scheduler(auto_stiffness=auto_stiffness)
    base_vec = jnp.asarray([float(base[h]) for h in hallmarks])

    def _build(hm):
        return Composite(
            apply_hallmarks(composite.processes, hm),
            composite.topology,
            validate=False,
            semantic_validation=False,
        )

    def reporter_values(sev_vec):
        hm = dict(base)
        for i, h in enumerate(hallmarks):
            hm[h] = sev_vec[i]
        comp = _build(hm)
        res = sched.run(
            comp,
            t_span=(0.0, t_end),
            macro_dt=mdt,
            y0=comp.initial_state_vec(),
        )
        return jnp.stack(
            [
                jnp.atleast_1d(r.summary(res.ts, res.get(r.observable), qt))[0]
                for r in reporters
            ]
        )

    base_comp = _build(base)
    sched.warm_up(
        base_comp,
        t_span=(0.0, t_end),
        macro_dt=mdt,
        y0=base_comp.initial_state_vec(),
    )
    values = reporter_values(base_vec)
    jac = jax.jacrev(reporter_values)(base_vec)  # (n_reporter, n_hallmark)

    reports = []
    for i, r in enumerate(reporters):
        val = float(values[i])
        for j, h in enumerate(hallmarks):
            sens = float(jac[i, j])
            finite = bool(jnp.isfinite(sens))
            rel = abs(sens) / (abs(val) + 1e-12)
            reports.append(
                SensitivityReport(
                    reporter=getattr(r, "gene_symbol", r.observable),
                    control=h,
                    value=val,
                    sensitivity=sens,
                    rel_sensitivity=rel,
                    live=finite and rel >= rel_threshold,
                    finite=finite,
                )
            )
    return reports
