"""Composable forcing sources for boundary inputs.

A prescribed experimental drive (an irradiation pulse, a drug ramp) is a
Process that emits ``f(t)`` to a store path, wired to an SBML boundary input
via :meth:`hallsim.sbml_import.SBMLProcess.with_input_driver`. So a dose is
composed from the general port-coupling path, not special-cased inside the
importer. :func:`drive_pulse` assembles the common dose-then-washout case.
"""

from __future__ import annotations

import sympy

from hallsim.sbml_math import TIME

import logging

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from hallsim.process import Port, PortRole, Process, calibratable

log = logging.getLogger(__name__)


class PulseSource(Process):
    """Rectangular forcing signal — emits ``dose * amplitude`` on
    ``[t_start, t_end)``, else 0, to its ``signal`` ASSIGNED port. Wire
    ``signal`` to a boundary input's driver port
    (:meth:`hallsim.sbml_import.SBMLProcess.with_input_driver`).
    ``t_end=None`` drops the washout edge, giving a sustained step — the
    setpoint source for a chronic exposure
    (:func:`hallsim.models.clamp_edge.clamp_species`). ``dose`` is the
    magnitude in the input's units, part of the protocol with the window;
    ``amplitude`` is the exposure level as a fraction of it, an input level a
    handle sets (0 = none, 1 = the full dose). A ramp or any other
    piecewise-linear shape is :class:`ProfileSource` over the same port
    mechanism."""

    timescale: float | None = eqx.field(static=True, default=None)
    amplitude: float = calibratable(
        1.0,
        level=True,
        description="exposure level as a fraction of dose; 0 = no exposure.",
    )
    dose: float = eqx.field(static=True, default=1.0)
    t_start: float = eqx.field(static=True, default=0.0)
    t_end: float | None = eqx.field(static=True, default=1.0)
    signal_units: str = eqx.field(static=True, default="dimensionless")
    signal_ontology: dict | None = eqx.field(static=True, default=None)
    handle: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "signal": Port(
                role=PortRole.ASSIGNED,
                default=0.0,
                units=self.signal_units,
                description="rectangular forcing amplitude·1[t_start,t_end)",
                ontology=self.signal_ontology or {},
            )
        }

    def assign(self, t, state):
        on = t >= self.t_start
        if self.t_end is not None:
            on = on & (t < self.t_end)
        return {"signal": self.dose * self.amplitude * jnp.where(on, 1.0, 0.0)}

    def assignment_rules(self):
        on = TIME >= float(self.t_start)
        if self.t_end is not None:
            on = sympy.And(on, TIME < float(self.t_end))
        signal = sympy.Piecewise(
            (float(self.dose) * sympy.Symbol("amplitude"), on), (0, True)
        )
        return (("signal", signal),)

    def discontinuity_times(self):
        if self.t_end is None:
            return (self.t_start,)
        return (self.t_start, self.t_end)


def _attach_driver(proc, input_name):
    """Expose ``input_name`` on ``proc`` as an INPUT port and return
    ``(proc, port)``, whichever part of the driveable surface it lives in: a
    rule-defined boundary input (``with_input_driver``) or a plain constant
    (``with_param_input``) get a ``<name>_in`` port; a species the model
    integrates is read from the store under its own name
    (``with_species_input``). Which one is a fact about the source file,
    not a choice the caller made.
    """
    port = f"{input_name.lower()}_in"
    if input_name in getattr(proc, "_w_names", ()):
        return proc.with_input_driver(input_name, port), port
    if input_name in getattr(proc, "_param_names", ()):
        return proc.with_param_input(input_name, port), port
    if input_name in getattr(proc, "_species_names", ()):
        return proc.with_species_input(input_name), input_name
    raise KeyError(
        f"{input_name!r} is neither a boundary input, a constant nor a "
        f"species on {getattr(proc, '_name', proc)!r}; boundary inputs: "
        f"{sorted(getattr(proc, '_w_names', ()))}; constants: "
        f"{sorted(getattr(proc, '_param_names', ()))}; species: "
        f"{sorted(getattr(proc, '_species_names', ()))}"
    )


def _native_dose(proc, target, input_name, kind):
    """The level ``proc``'s own events set ``input_name`` to, else 1."""
    level = getattr(proc, "native_input_level", None)
    dose = (level(input_name) if level is not None else None) or 1.0
    if dose != 1.0:
        log.info(
            "%s/%s %s: dose %g, the level the model's own events set.",
            target,
            input_name,
            kind,
            dose,
        )
    return dose


def _check_exposure(
    proc,
    target,
    input_name,
    t_start,
    t_end,
    imposed,
    driven_rate,
    warn_factor,
    kind,
):
    """Warn when ``imposed`` exposure over ``[t_start, t_end]`` differs from
    ``proc``'s native one by more than ``warn_factor``×."""
    native = proc.native_input_exposure(input_name, t_start, t_end)
    quantity = "exposure"
    if driven_rate is not None:
        rate_name, native_rate = driven_rate
        params = getattr(proc, "parameters", {}) or {}
        if rate_name not in params:
            raise KeyError(
                f"driven_rate names {rate_name!r}, which is not a parameter "
                f"of {target!r}"
            )
        native *= float(native_rate)
        imposed *= float(params[rate_name])
        quantity = f"dose (exposure x {rate_name})"

    ratio = imposed / max(abs(native), 1e-12)
    if native > 0 and (ratio > warn_factor or ratio < 1.0 / warn_factor):
        log.warning(
            "%s/%s %s [%.4g, %.4g] delivers %s %.4g vs the model's native "
            "%.4g (%.3gx) — rescale the driven rate by ~1/%.3g or match the "
            "native window.",
            target,
            input_name,
            kind,
            t_start,
            t_end,
            quantity,
            imposed,
            native,
            ratio,
            ratio,
        )


def drive_pulse(
    processes,
    topology,
    *,
    target,
    input_name,
    t_start,
    t_end,
    amplitude=1.0,
    dose=None,
    source_name=None,
    signal_ontology=None,
    handle=None,
    driven_rate=None,
    warn_factor=3.0,
):
    """Drive ``target``'s input ``input_name`` with a rectangular pulse on
    ``[t_start, t_end)``, composed from the general port path: adds a
    :class:`PulseSource` to ``processes`` and wires it to the input, a
    boundary input, a constant or a species the model's own events set.
    ``t_end=None`` sustains the drive (no washout). Mutates
    ``processes``/``topology`` in place and returns
    ``(processes, topology, source_name)``.

    ``dose`` is the level at full exposure in the input's units; ``None``
    reads it from the model's own protocol when its events set the input
    (:meth:`SBMLProcess.native_input_level`), else 1. ``amplitude`` is the
    exposure fraction, the level a handle sets: pass 0 for a source that is
    off until a handle turns it on.

    Warns when the dose the pulse would deliver **at full exposure**
    differs from the model's native one by more than ``warn_factor``×. Dose
    is exposure × the rate the input drives, so a window change that is
    compensated by rescaling that rate is not a mismatch. Pass
    ``driven_rate=(param_name, native_value)`` to be scored on the product;
    without it only exposure is compared, which flags a compensated setup as
    if it were off-calibration."""
    src = source_name or f"{input_name.lower()}_pulse"
    path = f"{src}/signal"

    if dose is None:
        dose = _native_dose(processes[target], target, input_name, "pulse")
    processes[src] = PulseSource(
        # co-group with the target so the pulse is evaluated at every solver
        # substep (not frozen across a macro step) — a sub-macro-step window
        # would otherwise be under-sampled by the cross-group coupling.
        timescale=getattr(processes[target], "timescale", None),
        amplitude=amplitude,
        dose=float(dose),
        t_start=float(t_start),
        t_end=None if t_end is None else float(t_end),
        signal_ontology=signal_ontology,
        handle=handle,
    )
    processes[target], port = _attach_driver(processes[target], input_name)
    topology[src] = {"signal": path}
    topology.setdefault(target, {})[port] = path

    if t_end is None:
        # An open-ended drive has no integrated exposure to compare against
        # the native one; the mismatch check below needs a finite window.
        return processes, topology, src

    _check_exposure(
        processes[target],
        target,
        input_name,
        t_start,
        t_end,
        float(dose) * (float(t_end) - float(t_start)),
        driven_rate,
        warn_factor,
        "pulse",
    )
    return processes, topology, src


class ProfileSource(Process):
    """Piecewise-linear forcing signal through ``(times, levels)`` knots,
    scaled by ``dose * amplitude``, to its ``signal`` ASSIGNED port.

    Between two knots the signal is interpolated linearly; before the first
    knot it holds the first level and after the last it holds the last. A
    repeated time is a jump: ``times=(0, 10, 10)``, ``levels=(0, 1, 0)`` is
    a ramp from rest to full over ten time units that stops at once, the
    shape of an incremental exercise test. A rectangle, a ramp-up and hold,
    or a logged trace sampled on a grid are the same source with other knots.

    The knots are protocol, fixed with the composite: the solver lands on
    every one of them, which needs their values before tracing. The level a
    handle or a fit sets is the traced ``amplitude``.
    """

    timescale: float | None = eqx.field(static=True, default=None)
    amplitude: float = calibratable(
        1.0,
        level=True,
        description="exposure level as a fraction of dose; 0 = no exposure.",
    )
    dose: float = eqx.field(static=True, default=1.0)
    times: tuple[float, ...] = eqx.field(static=True, default=(0.0, 1.0))
    levels: tuple[float, ...] = eqx.field(static=True, default=(0.0, 1.0))
    signal_units: str = eqx.field(static=True, default="dimensionless")
    signal_ontology: dict | None = eqx.field(static=True, default=None)
    handle: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if len(self.times) != len(self.levels) or len(self.times) < 1:
            raise ValueError(
                f"times and levels must be the same non-empty length; got "
                f"{len(self.times)} and {len(self.levels)}"
            )
        if any(b < a for a, b in zip(self.times, self.times[1:])):
            raise ValueError(f"times must not decrease: {self.times}")

    def ports_schema(self):
        return {
            "signal": Port(
                role=PortRole.ASSIGNED,
                default=0.0,
                units=self.signal_units,
                description="piecewise-linear forcing through (times, levels)",
                ontology=self.signal_ontology or {},
            )
        }

    def _shape(self, t):
        ts = jnp.asarray(self.times, dtype=float)
        ls = jnp.asarray(self.levels, dtype=float)
        if len(self.times) == 1:
            return ls[0]
        i = jnp.clip(jnp.searchsorted(ts, t, side="right"), 1, len(ts) - 1)
        t0, t1, l0, l1 = ts[i - 1], ts[i], ls[i - 1], ls[i]
        span = t1 - t0
        w = jnp.where(span > 0, (t - t0) / jnp.where(span > 0, span, 1.0), 1.0)
        inside = l0 + jnp.clip(w, 0.0, 1.0) * (l1 - l0)
        return jnp.where(t < ts[0], ls[0], inside)

    def assign(self, t, state):
        return {"signal": self.dose * self.amplitude * self._shape(t)}

    def assignment_rules(self):
        knots = list(zip(map(float, self.times), map(float, self.levels)))
        pieces = [(knots[0][1], TIME < knots[0][0])]
        for (t0, l0), (t1, l1) in zip(knots, knots[1:]):
            if t1 > t0:
                ramp = l0 + (l1 - l0) * (TIME - t0) / (t1 - t0)
                pieces.append((ramp, TIME < t1))
        pieces.append((knots[-1][1], True))
        signal = float(self.dose) * sympy.Symbol("amplitude")
        return (("signal", signal * sympy.Piecewise(*pieces)),)

    def discontinuity_times(self):
        return tuple(sorted(set(map(float, self.times))))


def drive_profile(
    processes,
    topology,
    *,
    target,
    input_name,
    times,
    levels,
    amplitude=1.0,
    dose=None,
    source_name=None,
    signal_ontology=None,
    handle=None,
    driven_rate=None,
    warn_factor=3.0,
):
    """Drive ``target``'s input ``input_name`` with a piecewise-linear
    profile through ``(times, levels)``, composed from the same port path as
    :func:`drive_pulse`. Mutates ``processes``/``topology`` in place and
    returns ``(processes, topology, source_name)``.

    ``dose`` is the level at full exposure in the input's units and the
    levels are fractions of it; ``None`` reads it from the model's own
    protocol as :func:`drive_pulse` does. ``amplitude`` is the exposure
    level a handle or a fit sets. The exposure the profile delivers between
    its first and last knot is checked against the native one, with
    ``driven_rate`` and ``warn_factor`` as in :func:`drive_pulse`.
    """
    src = source_name or f"{input_name.lower()}_profile"
    path = f"{src}/signal"
    if dose is None:
        dose = _native_dose(processes[target], target, input_name, "profile")
    processes[src] = ProfileSource(
        # co-group with the target so the profile is evaluated at every
        # solver substep rather than frozen across a macro step.
        timescale=getattr(processes[target], "timescale", None),
        amplitude=amplitude,
        dose=float(dose),
        times=tuple(float(t) for t in times),
        levels=tuple(float(v) for v in levels),
        signal_ontology=signal_ontology,
        handle=handle,
    )
    processes[target], port = _attach_driver(processes[target], input_name)
    topology[src] = {"signal": path}
    topology.setdefault(target, {})[port] = path

    ts = np.asarray(times, dtype=float)
    _check_exposure(
        processes[target],
        target,
        input_name,
        ts[0],
        ts[-1],
        float(dose) * float(np.trapezoid(np.asarray(levels, float), ts)),
        driven_rate,
        warn_factor,
        "profile",
    )
    return processes, topology, src


class StepSource(Process):
    """Two-level forcing signal — emits ``before`` while ``t < t_step`` and
    ``after`` from ``t_step`` on, to its ``signal`` ASSIGNED port.

    The shape of a drug added partway through, which :class:`PulseSource`
    cannot express because its off-level is 0 rather than a baseline. Both
    levels are calibratable; ``before == after`` is a constant drive.
    """

    timescale: float | None = eqx.field(static=True, default=None)
    after: float = calibratable(
        1.0, description="level from t_step onward (treated level)."
    )
    before: float = calibratable(
        1.0, description="level before t_step (untreated level)."
    )
    t_step: float = eqx.field(static=True, default=0.0)
    # A port default is structure and must stay static; `before` is traced.
    signal_default: float = eqx.field(static=True, default=0.0)
    signal_units: str = eqx.field(static=True, default="dimensionless")
    signal_ontology: dict | None = eqx.field(static=True, default=None)
    handle: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "signal": Port(
                role=PortRole.ASSIGNED,
                default=self.signal_default,
                units=self.signal_units,
                description="two-level forcing: before | after at t_step",
                ontology=self.signal_ontology or {},
            )
        }

    def assign(self, t, state):
        return {"signal": jnp.where(t >= self.t_step, self.after, self.before)}

    def assignment_rules(self):
        signal = sympy.Piecewise(
            (sympy.Symbol("after"), TIME >= float(self.t_step)),
            (sympy.Symbol("before"), True),
        )
        return (("signal", signal),)

    def discontinuity_times(self):
        return (self.t_step,)


def drive_step(
    processes,
    topology,
    *,
    target,
    input_name,
    t_step,
    after=1.0,
    before=1.0,
    source_name=None,
    signal_ontology=None,
    handle=None,
):
    """Drive ``target``'s boundary input ``input_name`` with a two-level step
    at ``t_step``, composed from the same port path as :func:`drive_pulse`:
    adds a :class:`StepSource` and wires it via
    :meth:`SBMLProcess.with_input_driver`. ``before == after`` gives a
    sustained constant drive at the model's own level. Mutates
    ``processes``/``topology`` in place; returns
    ``(processes, topology, source_name)``.

    No exposure check: a step has no finite window to integrate, and holding
    an input at its native level is by construction on-calibration.
    """
    src = source_name or f"{input_name.lower()}_step"
    path = f"{src}/signal"

    processes[src] = StepSource(
        timescale=getattr(processes[target], "timescale", None),
        after=after,
        before=before,
        t_step=float(t_step),
        signal_default=float(before),
        signal_ontology=signal_ontology,
        handle=handle,
    )
    processes[target], port = _attach_driver(processes[target], input_name)
    topology[src] = {"signal": path}
    topology.setdefault(target, {})[port] = path
    return processes, topology, src
