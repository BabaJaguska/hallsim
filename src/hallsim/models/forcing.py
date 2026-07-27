"""Composable forcing sources for boundary inputs.

A prescribed experimental drive (an irradiation pulse, a drug ramp) is a
Process that emits ``f(t)`` to a store path, wired to an SBML boundary input
via :meth:`hallsim.sbml_import.SBMLProcess.with_input_driver`. So a dose is
composed from the general port-coupling path, not special-cased inside the
importer. :func:`drive_pulse` assembles the common dose-then-washout case.
"""

from __future__ import annotations

import logging

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, calibratable

log = logging.getLogger(__name__)


class PulseSource(Process):
    """Rectangular forcing signal — emits ``amplitude`` on ``[t_start, t_end)``,
    else 0, to its ``signal`` ASSIGNED port. Wire ``signal`` to a boundary
    input's driver port (:meth:`hallsim.sbml_import.SBMLProcess.with_input_driver`).
    ``amplitude`` is calibratable (a fittable or severity-driven exposure
    level); the window is structural. Other shapes (ramp, decay) are sibling
    sources over the same port mechanism."""

    timescale: float | None = None
    amplitude: float = calibratable(
        1.0, description="pulse height / exposure level; 0 = no exposure."
    )
    t_start: float = eqx.field(static=True, default=0.0)
    t_end: float = eqx.field(static=True, default=1.0)
    signal_ontology: dict | None = eqx.field(static=True, default=None)
    hallmark: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "signal": Port(
                role=PortRole.ASSIGNED,
                default=0.0,
                units="dimensionless",
                description="rectangular forcing amplitude·1[t_start,t_end)",
                ontology=self.signal_ontology or {},
            )
        }

    def assign(self, t, state):
        gate = jnp.where((t >= self.t_start) & (t < self.t_end), 1.0, 0.0)
        return {"signal": self.amplitude * gate}

    def discontinuity_times(self):
        return (self.t_start, self.t_end)


def drive_pulse(
    processes,
    topology,
    *,
    target,
    input_name,
    t_start,
    t_end,
    amplitude=1.0,
    source_name=None,
    signal_ontology=None,
    hallmark=None,
    warn_factor=3.0,
):
    """Drive ``target``'s boundary input ``input_name`` with a rectangular
    pulse on ``[t_start, t_end)``, composed from the general port path: adds a
    :class:`PulseSource` to ``processes`` and wires it to the input via
    :meth:`SBMLProcess.with_input_driver`. Mutates ``processes``/``topology``
    in place and returns ``(processes, topology, source_name)``.

    Warns when the pulse's integrated exposure differs from the input's native
    SBML drive by more than ``warn_factor``× — the driven rate (e.g. potency)
    is calibrated to the native exposure, so a large mismatch runs the model
    off calibration (a 2-day window on a 5-min pulse ≈ 576×)."""
    src = source_name or f"{input_name.lower()}_pulse"
    port = f"{input_name.lower()}_in"
    path = f"{src}/signal"

    processes[src] = PulseSource(
        # co-group with the target so the pulse is evaluated at every solver
        # substep (not frozen across a macro step) — a sub-macro-step window
        # would otherwise be under-sampled by the cross-group coupling.
        timescale=getattr(processes[target], "timescale", None),
        amplitude=amplitude,
        t_start=float(t_start),
        t_end=float(t_end),
        signal_ontology=signal_ontology,
        hallmark=hallmark,
    )
    processes[target] = processes[target].with_input_driver(input_name, port)
    topology[src] = {"signal": path}
    topology.setdefault(target, {})[port] = path

    native = processes[target].native_input_exposure(
        input_name, t_start, t_end
    )
    imposed = float(amplitude) * (float(t_end) - float(t_start))
    ratio = imposed / max(abs(native), 1e-12)
    if native > 0 and (ratio > warn_factor or ratio < 1.0 / warn_factor):
        log.warning(
            "%s/%s pulse [%.4g, %.4g] delivers exposure %.4g vs the native SBML "
            "drive's %.4g (%.3gx). The rate driven by %r (e.g. potency) is "
            "calibrated to the native exposure — rescale it by ~1/%.3g or match "
            "the native window.",
            target,
            input_name,
            t_start,
            t_end,
            imposed,
            native,
            ratio,
            input_name,
            ratio,
        )
    return processes, topology, src
