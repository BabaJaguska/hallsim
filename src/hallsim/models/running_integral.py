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

import sympy

from hallsim.process import ReactionChannel

from hallsim.process import Port, PortRole, Process
import equinox as eqx


class RunningIntegral(Process):
    """Cumulative ``∫ source**power`` on the ``integral`` path; the module
    docstring has the two modes and how to read each."""

    description = (
        "Cumulative time-integral of an oscillating observable; pair with "
        "window_mean for a phase-insensitive trailing mean with bounded "
        "calibration gradient."
    )

    timescale: float | None = eqx.field(static=True, default=None)
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

    def reaction_channels(self):
        channels = [
            ReactionChannel(
                "accumulate",
                (("integral", 1.0),),
                sympy.Symbol("source") ** sympy.Symbol("power"),
            )
        ]
        if self.tau is not None:
            channels.append(
                ReactionChannel(
                    "decay",
                    (("integral", -1.0),),
                    sympy.Symbol("integral") / sympy.Symbol("tau"),
                )
            )
        return tuple(channels)

    def derivative(self, t, state):
        val = state["source"] ** self.power
        if self.tau is not None:
            val = val - state["integral"] / self.tau
        return {"integral": val}


def transcript_pool(
    composite,
    observable: str,
    *,
    tau: float,
    name: str = "mrna",
    timescale: float | None = None,
):
    """``composite`` with a transcript pool driven by ``observable``: the
    leaky integral ``dm/dt = a − m/τ`` on ``<name>/integral``, which is the
    mRNA of a gene transcribed in proportion to an activity ``a`` and
    decaying with time constant ``τ``, the transcription gain cancelling in
    a log2 fold change. A transcript trails and smooths its driver, so a
    readout of a kinase or factor activity against measured transcripts
    reads this path rather than the activity itself. ``τ`` is a parameter
    to calibrate, held out across deposits.
    """
    from hallsim.composite import Composite

    pool = RunningIntegral(power=1.0, tau=tau, timescale=timescale)
    return Composite(
        processes={**composite.processes, name: pool},
        topology={
            **composite.topology,
            name: {"source": observable, "integral": f"{name}/integral"},
        },
        initial=composite.initial,
        validate=False,
        semantic_validation={"check_semantics": False},
    )
