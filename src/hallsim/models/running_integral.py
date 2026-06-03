"""RunningIntegral — cumulative time-integral of an observable.

A phase-insensitive readout for oscillating species. Reading an oscillator
at its endpoint is both wrong (phase-dependent) and numerically hostile:
the endpoint's forward/adjoint sensitivity grows with integration time
(accumulating phase drift), blowing calibration gradients up toward NaN.
The cure is to read the *mean over the last window* instead.

``RunningIntegral`` integrates the source alongside the dynamics::

    dA/dt = source          # note: RHS is the source, NOT A — pure
                            # integration, no self-feedback, no exponential

so ``A(t) = ∫₀ᵗ source``. The mean of the source over any trailing window
``[T−W, T]`` is then exactly (fundamental theorem of calculus)::

    ⟨source⟩_[T−W, T] = (A(T) − A(T−W)) / W

Read it with :func:`hallsim.gene_reporters.window_mean`, which differences
the accumulator at two save points and divides by the elapsed time. This is
a *flat* window mean — exact, with no weighting bias (unlike an exponential
moving average) — and its sensitivity stays bounded when the window is a
fixed fraction of the run.

The accumulator must be integrated at the source's own resolution, so set
``timescale`` to the source oscillator's timescale: the Scheduler then
co-solves them in one finely-stepped group and the integral sees the real
oscillation, not a frozen sync-point sample.
"""

from __future__ import annotations

from hallsim.process import Port, PortRole, Process


class RunningIntegral(Process):
    """Cumulative time-integral of an ``INPUT`` observable.

    Wire ``source`` to the oscillating store path; the ``integral`` path
    holds ``∫₀ᵗ source``. Read a trailing-window mean from it via
    :func:`hallsim.gene_reporters.window_mean`. See module docstring.

    Parameters
    ----------
    timescale:
        Must match the source oscillator's timescale so the two share a
        Scheduler group and the integral is taken at the oscillation's
        resolution.
    """

    timescale: float | None = None

    def ports_schema(self):
        return {
            "integral": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="dimensionless",
                description="Cumulative time-integral ∫₀ᵗ source",
            ),
            "source": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="Observable being integrated (read-only)",
            ),
        }

    def derivative(self, t, state):
        # RHS depends only on `source` (a read-only INPUT), never on the
        # accumulator `integral` itself — so this is integration, not an
        # exponential.
        return {"integral": state["source"]}

    def metadata(self):
        base = super().metadata()
        base["description"] = (
            "Cumulative time-integral of an oscillating observable; pair "
            "with window_mean for a phase-insensitive trailing mean with "
            "bounded calibration gradient."
        )
        return base
