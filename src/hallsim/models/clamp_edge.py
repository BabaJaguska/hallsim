"""ClampEdge — hold an integrated species at a setpoint against its own model.

    d(target)/dt += k_clamp · (setpoint − target)

The primitive for a *sustained* condition on a state variable: a ligand held
in the medium while the model consumes it, a metabolite buffered by an
unmodelled pool, chronic exposure where the source model only describes a
bolus. ``with_param_input`` covers constants and ``drive_pulse`` covers
boundary inputs; neither reaches a species the model integrates, which is
what this does.

Additive EVOLVED semantics mean the clamp competes with the model's own flux
rather than overriding it, so the hold is proportional, not absolute: with a
net removal flux ``v`` at the setpoint, the clamped steady state sits at
``setpoint − v/k_clamp``. :func:`measure_unclamped_flux` measures ``v``
through the composite's own RHS and :func:`place_clamp_rate` turns it into
the ``k_clamp`` that meets a stated tolerance — pick the rate, don't guess it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, calibratable
from hallsim.store import as_paths

log = logging.getLogger(__name__)


class ClampEdge(Process):
    """Proportional clamp holding ``target`` at ``setpoint``; see the module
    docstring for the rate law and the residual-offset relation.

    ``target`` is an EVOLVED port that *reads* its own path — the restoring
    term is negative feedback on the clamped species, unlike the pure-source
    :class:`~hallsim.models.hill_edge.HillActivationEdge`. ``setpoint`` is an
    INPUT: wire it to a constant hold
    (:class:`~hallsim.models.forcing.PulseSource` with no washout, which
    :func:`clamp_species` assembles), to a dose schedule, or to another
    model's species to make one composite track the other.

    Declare ``units`` when the clamped path carries them — setpoint, target
    and residual are all in that unit. ``target_ontology`` should name the
    clamped entity so the semantic checker sees an annotated writer.
    """

    timescale: float | None = None

    k_clamp: float = calibratable(
        1.0,
        description="clamp rate (1/time); residual offset is flux/k_clamp. "
        "Place it with hallsim.models.clamp_edge.place_clamp_rate.",
    )

    units: str = eqx.field(static=True, default="dimensionless")
    target_default: float = eqx.field(static=True, default=0.0)
    target_ontology: dict | None = eqx.field(static=True, default=None)
    target_description: str = eqx.field(static=True, default="")
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "target": Port(
                role=PortRole.EVOLVED,
                default=self.target_default,
                units=self.units,
                description=self.target_description,
                ontology=self.target_ontology or {},
            ),
            "setpoint": Port(
                role=PortRole.INPUT,
                default=0.0,
                units=self.units,
                description="level the target is held at",
            ),
        }

    def derivative(self, t, state):
        return {"target": self.k_clamp * (state["setpoint"] - state["target"])}


@dataclass
class ClampRateSuggestion:
    """A ``k_clamp`` placed against a measured flux (see
    :func:`place_clamp_rate`). ``ok`` is False when the rate cannot do the job
    asked of it — read ``note`` for why."""

    k_clamp: float
    tau: float
    rel_offset: float
    ok: bool
    note: str


def place_clamp_rate(
    flux,
    level,
    *,
    rel_error: float = 0.01,
    tau_model: float | None = None,
    max_ratio: float = 100.0,
) -> ClampRateSuggestion:
    """``k_clamp`` holding ``level`` to within ``rel_error`` against a net
    removal ``flux`` (units/time, magnitude — measure it with
    :func:`measure_unclamped_flux`).

    The clamp's steady-state residual is ``flux / k_clamp``, so the rate that
    keeps it under ``rel_error·level`` is ``flux / (rel_error·level)``. Pass
    ``tau_model`` (the clamped model's own timescale) to have the separation
    checked: a clamp more than ``max_ratio`` faster than the dynamics it acts
    on stiffens the solve and lands in a different :meth:`Composite.auto_groups`
    group, which is the same ``max_ratio``.
    """
    v = abs(float(flux))
    L = float(level)
    if L <= 0.0:
        return ClampRateSuggestion(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            "non-positive setpoint; nothing to hold",
        )
    if v == 0.0:
        return ClampRateSuggestion(
            0.0,
            float("inf"),
            0.0,
            True,
            "no flux at the setpoint — the model already holds this level; "
            "a clamp changes nothing",
        )
    k = v / (rel_error * L)
    tau = 1.0 / k
    ok = True
    note = "ok"
    if tau_model is not None and tau_model > 0.0:
        sep = tau_model / tau
        if sep > max_ratio:
            ok = False
            note = (
                f"clamp is {sep:.3g}x faster than the model (tau {tau:.3g} vs "
                f"{tau_model:.3g}): stiffens the solve and auto_groups splits "
                f"it off at {max_ratio:.0f}x. Loosen rel_error, or set the "
                f"edge's timescale to the model's so they co-group"
            )
    return ClampRateSuggestion(k, tau, rel_error, ok, note)


def measure_unclamped_flux(composite, path: str, level, t: float = 0.0):
    """Net ``d(path)/dt`` at ``level``, with every :class:`ClampEdge` in
    ``composite`` excluded — the flux a clamp on ``path`` has to work against.

    Evaluated through the composite's own ``build_rhs`` at its initial state
    with ``path`` set to ``level``, so it measures the assembled model rather
    than a re-derivation of it. Feed the magnitude to
    :func:`place_clamp_rate`.
    """
    names = [
        n
        for n, p in composite.continuous_processes().items()
        if not isinstance(p, ClampEdge)
    ]
    rhs, keys = composite.build_rhs(names)
    idx = keys.index(path)
    y = composite.initial_state_vec(keys).at[idx].set(jnp.asarray(level))
    return rhs(t, y, None)[idx]


def clamp_species(
    processes,
    topology,
    *,
    target,
    species,
    level,
    k_clamp,
    t_start: float = 0.0,
    source_name=None,
    edge_name=None,
    hallmark=None,
):
    """Hold ``target``'s ``species`` port at ``level`` from ``t_start`` on.

    Adds a sustained :class:`~hallsim.models.forcing.PulseSource` (no washout)
    and a :class:`ClampEdge` reading it, wired to the store path ``species``
    already occupies. The clamp inherits that port's units and ontology, so
    the added writer is annotated like the species it holds, and co-groups
    with ``target`` so it is evaluated at every solver substep rather than
    frozen across a macro step. Mutates ``processes``/``topology`` in place and
    returns ``(processes, topology, edge_name)``.

    ``k_clamp`` sets how tightly: measure the flux with
    :func:`measure_unclamped_flux` and place the rate with
    :func:`place_clamp_rate` rather than guessing. For a time-varying setpoint,
    instantiate :class:`ClampEdge` directly and wire ``setpoint`` to the
    driving path.
    """
    from hallsim.models.forcing import PulseSource

    (path,) = as_paths(topology[target][species])
    port = processes[target].ports_schema()[species]
    edge = edge_name or f"{species.lower()}_clamp"
    src = source_name or f"{edge}_setpoint"
    signal_path = f"{src}/signal"
    ts = getattr(processes[target], "timescale", None)

    processes[src] = PulseSource(
        timescale=ts,
        amplitude=level,
        t_start=float(t_start),
        t_end=None,
        signal_units=port.units,
    )
    processes[edge] = ClampEdge(
        timescale=ts,
        k_clamp=k_clamp,
        units=port.units,
        target_default=float(port.default),
        target_ontology=dict(port.ontology or {}),
        target_description=f"{species} held at a setpoint",
        hallmark=hallmark,
    )
    topology[src] = {"signal": signal_path}
    topology[edge] = {"target": path, "setpoint": signal_path}

    log.info(
        "clamping %s (%s) at %s with k_clamp=%.4g (residual %.4g per unit "
        "flux)",
        path,
        species,
        level,
        float(k_clamp),
        1.0 / float(k_clamp) if float(k_clamp) else float("inf"),
    )
    return processes, topology, edge
