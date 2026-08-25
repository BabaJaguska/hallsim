"""KickEvent — "at time T, add delta D to store path P", as a Process.

A kick is a one-shot mutation, not a derivative contribution, so it rides the
EVENT/LATCHED path rather than EVOLVED. That lets it share a store path with
the CONTINUOUS process that owns the derivative: the latter advances the state
between sync points, the kick scatter-adds at the one where it fires. There is
no separate "run with perturbation" code path — compose it and run.

>>> kick = KickEvent(kick_time=10.0, deltas={"x": 5.0})
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
        The kick fires at the first sync point with ``t >= kick_time``.
    deltas:
        ``{port_name: delta}``, each port wired in topology to the store path
        being kicked. Arrays must broadcast against that path's shape.
    units:
        ``{port_name: unit}`` so the semantic validator can check the kick
        against whatever CONTINUOUS process owns the path. Missing keys are
        unspecified, which warns if the path declares units elsewhere.
    """

    kind: ProcessKind = ProcessKind.EVENT
    kick_time: float = 0.0
    deltas: Mapping[str, float] = None  # type: ignore[assignment]
    units: Mapping[str, str] = None  # type: ignore[assignment]

    def ports_schema(self):
        units_map = self.units or {}
        return {
            name: Port(
                role=PortRole.LATCHED,
                default=None,  # kicks a path it does not own
                units=units_map.get(name, ""),
                description=f"kick target: {name}",
            )
            for name in (self.deltas or {})
        }

    def condition(self, t, state):
        return t >= self.kick_time

    def handler(self, t, state):
        # Same delta every call; the Scheduler's False→True guard fires it once.
        return {
            name: jnp.asarray(value)
            for name, value in (self.deltas or {}).items()
        }

    def metadata(self):
        base = super().metadata()
        base["kick_time"] = self.kick_time
        base["delta_targets"] = list((self.deltas or {}).keys())
        return base
