"""RunningIntegral — time-integral of an observable, flat or low-pass.

A phase-insensitive readout for oscillating species. Reading an oscillator
at its endpoint is both wrong (phase-dependent) and numerically hostile:
the endpoint's forward/adjoint sensitivity grows with integration time
(accumulating phase drift), blowing calibration gradients up toward NaN.
The cure is to read a *time-average* instead.

Two modes, by ``tau``:

**Flat (``tau=None``, default)** — pure accumulation::

    dA/dt = source          # RHS is the source, NOT A — no self-feedback

so ``A(t) = ∫₀ᵗ source`` and the trailing-window mean over ``[T−W, T]`` is
exactly ``(A(T) − A(T−W)) / W`` (fundamental theorem of calculus). Read it
with :func:`hallsim.gene_reporters.window_mean` / ``window_rms``.

**Leaky (``tau`` set)** — first-order low-pass (exponential moving average)::

    dA/dt = source − A/τ

so ``A(t) = ∫₀ᵗ source·e^{-(t-s)/τ} ds`` and ``A/τ`` is the τ-weighted mean
of the source (the low-pass envelope). Read it with
:func:`hallsim.gene_reporters.leaky_rms`. This models a *stable-product*
reporter that integrates a pulsing drive — the pulses are filtered out and
the readout is smooth, with no window edges or phase to alias, unlike the
flat window. ``τ`` is the integration time constant, in the trajectory's
time unit.

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

    description = (
        "Cumulative time-integral of an oscillating observable; pair with "
        "window_mean for a phase-insensitive trailing mean with bounded "
        "calibration gradient."
    )

    timescale: float | None = None
    # Integrate source**power. Default 2 → ∫x² → √⟨x²⟩: the amplitude-aware
    # readout is the safe default, because a buffered-mean oscillator (e.g.
    # p53, mean analytically damage-blind) is invisible to a plain mean.
    # power=1 → ∫x, for a species whose DC level itself moves with the drive.
    power: float = 2.0
    # None → flat accumulation (∫source**power). Set → leaky low-pass with
    # this time constant: dA/dt = source**power − A/τ, an exponential moving
    # average read via leaky_rms. For a stable-product reporter that filters
    # a pulsing drive into a smooth envelope.
    tau: float | None = None

    def ports_schema(self):
        return {
            "integral": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="dimensionless",
                description="Time-integral of source**power (flat or leaky)",
                reads_value=self.tau is not None,  # leaky term reads A
            ),
            "source": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="Observable being integrated (read-only)",
            ),
        }

    def derivative(self, t, state):
        val = state["source"] ** self.power
        if self.tau is not None:
            val = val - state["integral"] / self.tau
        return {"integral": val}
