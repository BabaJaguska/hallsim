"""RunningIntegral — time-integral of an observable, flat or low-pass.

A phase-insensitive readout for an oscillating species, where an endpoint
read is phase-dependent and its adjoint sensitivity grows with the span.

    tau=None   dA/dt = source**power         A(t) = ∫₀ᵗ source**power
    tau set    dA/dt = source**power − A/τ   τ-weighted moving average

Read the flat form with :func:`hallsim.gene_reporters.window_mean` /
``window_rms`` — its trailing-window mean is exactly ``(A(T) − A(T−W))/W`` —
and the leaky form, whose envelope is ``A/τ``, with
:func:`hallsim.gene_reporters.leaky_rms`.

Set ``timescale`` to the source oscillator's, or the Scheduler groups them
apart and the integral accumulates frozen sync-point samples.
"""

from __future__ import annotations

from hallsim.process import Port, PortRole, Process


class RunningIntegral(Process):
    """Cumulative ``∫ source**power`` on the ``integral`` path; the module
    docstring has the two modes and how to read each."""

    description = (
        "Cumulative time-integral of an oscillating observable; pair with "
        "window_mean for a phase-insensitive trailing mean with bounded "
        "calibration gradient."
    )

    timescale: float | None = None
    # 2 → √⟨x²⟩, amplitude-aware: a buffered-mean oscillator (p53, mean
    # analytically damage-blind) is invisible to a plain mean. 1 → ∫x, for a
    # species whose DC level itself moves with the drive.
    power: float = 2.0
    # Leaky low-pass time constant; None accumulates without decay.
    tau: float | None = None
    # A leaky integral meant to start at rest wants tau * source(0)**power.
    initial: float = 0.0

    def ports_schema(self):
        return {
            "integral": Port(
                role=PortRole.EVOLVED,
                default=self.initial,
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
