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
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from hallsim.composite import Composite
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

    @property
    def ok(self) -> bool:
        return not (
            self.exploding or self.vanishing or self.tolerance_sensitive
        )

    def __str__(self) -> str:
        flags = []
        if self.exploding:
            flags.append("EXPLODING")
        if self.vanishing:
            flags.append("VANISHING")
        if self.tolerance_sensitive:
            flags.append("TOLERANCE-SENSITIVE")
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


def _solo_run(process, t_end, rtol, atol, n_save, max_steps):
    comp = Composite(
        processes={process._name: process},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    res = Scheduler(rtol=rtol, atol=atol, max_steps=max_steps).run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=t_end,
        y0=comp.initial_state_vec(),
        save_dt=t_end / n_save,
    )
    return np.asarray(jnp.stack([res.get(k) for k in res.keys], axis=-1))


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
) -> ScreenReport:
    """Screen one process solo over ``[0, t_end]`` native time units.

    ``t_end`` is in the model's **native** time unit — pick a few of its
    own characteristic times (e.g. ~30000 for a second-scale NF-κB
    oscillator, ~100 for an hour-scale p53 oscillator, ~50 for a
    day-scale senescence model). The process is run on its native clock
    regardless of any composite reconciliation.

    Flags ``exploding`` (non-finite, or peak > ``growth_threshold`` ×
    initial scale and still rising), ``vanishing`` (all states end within
    1e-9 of zero), and ``tolerance_sensitive`` (loose vs tight tolerance
    differ by more than ``tol_rel_threshold``, peak-normalised). A solver
    that exceeds ``max_steps`` is reported as exploding/unintegrable.
    """
    proc = _on_native_clock(process)
    name = getattr(proc, "_name", type(proc).__name__)

    try:
        # atol is held at the production default across both runs so the
        # loose-vs-tight comparison isolates rtol sensitivity (and the
        # screen integrates exactly what the Scheduler does in production).
        y_tight = _solo_run(
            proc, t_end, rtol_tight, DEFAULT_ATOL, n_save, max_steps
        )
        y_loose = _solo_run(
            proc, t_end, rtol_loose, DEFAULT_ATOL, n_save, max_steps
        )
    except Exception as exc:  # max_steps / non-finite blow the solve up
        return ScreenReport(
            name=name,
            exploding=True,
            vanishing=False,
            tolerance_sensitive=True,
            max_abs=float("inf"),
            tol_rel_diff=float("inf"),
            detail=f"solver failed: {type(exc).__name__}",
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

    return ScreenReport(
        name=name,
        exploding=exploding,
        vanishing=vanishing,
        tolerance_sensitive=tolerance_sensitive,
        max_abs=peak,
        tol_rel_diff=tol_rel_diff,
        detail=detail,
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
