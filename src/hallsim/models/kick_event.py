"""KickEvent — one-shot perturbation of state variables at a fixed time.

A generic EVENT Process for "at time T, add delta D to store path P".

Replaces the old ``Simulator.run_with_perturbation`` convenience: instead
of a custom run-kick-run-concatenate code path, the kick becomes a
first-class composable Process that the Scheduler dispatches at sync
points — just like any other event in the system.

Why this pattern
----------------
A kick is *not* a derivative contribution; it's a one-shot mutation of
the state. The framework correctly distinguishes these two semantics:

- CONTINUOUS processes own *derivatives* via EVOLVED/EXCLUSIVE ports.
- EVENT processes apply *deltas* via LATCHED ports when their
  ``condition`` becomes true.

A LATCHED-output EVENT can coexist on the same store path as a
CONTINUOUS EVOLVED/EXCLUSIVE writer — the CONTINUOUS process advances
the state via its derivative between sync points, and the EVENT
scatter-adds its delta at the sync point where the trigger fires.

Usage
-----
>>> from hallsim.models.kick_event import KickEvent
>>> from hallsim.scheduler import Scheduler
>>> kick = KickEvent(kick_time=10.0, deltas={"x": 5.0})
>>> # Compose with whatever continuous processes own "x"; topology
>>> # routes kick's "x" port to the same store path.
>>> result = Scheduler().run(comp, t_span=(0.0, 20.0), macro_dt=1.0)
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, ProcessKind


class KickEvent(Process):
    """One-shot additive perturbation at a fixed simulation time.

    Parameters
    ----------
    kick_time:
        Simulation time at which the kick fires. The first sync point
        with ``t >= kick_time`` triggers the False→True transition.
    deltas:
        Mapping from port name to additive delta. Each port name is
        wired in topology to the store path being kicked. Values may
        be Python floats or JAX arrays; arrays must broadcast against
        the store path's value shape (relevant for batched runs).

    Notes
    -----
    The handler returns the same delta on every invocation, but the
    Scheduler's False→True trigger guard fires it exactly once.
    """

    kind: ProcessKind = ProcessKind.EVENT
    kick_time: float = 0.0
    deltas: Mapping[str, float] = None  # type: ignore[assignment]

    def ports_schema(self):
        # All kick targets are LATCHED — that's the role EVENT processes
        # use to apply scatter-add deltas. Other processes' EVOLVED /
        # EXCLUSIVE ports may share these store paths (post topology
        # validator relaxation in store.py).
        return {
            name: Port(
                role=PortRole.LATCHED,
                default=0.0,
                units="dimensionless",
                description=f"kick target: {name}",
            )
            for name in (self.deltas or {})
        }

    def condition(self, t, state):
        return t >= self.kick_time

    def handler(self, t, state):
        # Return the configured deltas. The Scheduler's False→True
        # tracking ensures this fires exactly once at the first sync
        # point where condition becomes true.
        return {
            name: jnp.asarray(value)
            for name, value in (self.deltas or {}).items()
        }

    def metadata(self):
        base = super().metadata()
        base["kick_time"] = self.kick_time
        base["delta_targets"] = list((self.deltas or {}).keys())
        return base
