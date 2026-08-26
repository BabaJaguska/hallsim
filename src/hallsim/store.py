"""Store utilities — initial state construction and topology validation.

The store is a plain ``dict[str, jnp.ndarray]`` with path-like keys::

    {"cytoplasm/ROS": jnp.array(0.1), "nucleus/p53": jnp.array(0.5)}

It only appears at the API boundary (initial state, simulation results).
Internally everything runs on the flat vector returned by
``Composite.flatten`` / ``unflatten``.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np

from hallsim.process import PortRole, Process, ProcessKind
from hallsim.tracing import is_traced

log = logging.getLogger(__name__)


def build_initial_store(
    processes: dict[str, Process],
    topology: dict[str, dict[str, str]],
    initial: dict[str, float] | None = None,
) -> dict[str, jnp.ndarray]:
    """Create the initial state dict from port defaults.

    For each process, looks up its ports via ``ports_schema()``, maps them
    through the topology to store paths, and collects default values.

    When several ports share a store path, a *writer* port (EVOLVED,
    EXCLUSIVE, LATCHED, ASSIGNED) seeds the value in preference to a reader
    (INPUT) — a reader's default describes what it expects to be handed, not
    what the path holds. Within a tier the lowest process name wins.

    ``initial`` names a path's starting value outright and settles any
    disagreement there. Two same-role ports claiming one path with *different*
    defaults raise instead: which value the path starts at changes the
    trajectory, so it is declared, not inferred.

    The tie-break is by name rather than by dict order on purpose: JAX sorts
    dict keys when it flattens a pytree, so ``processes`` comes back sorted
    from any ``jax.jit`` / ``vmap`` / ``eqx.tree_at`` round-trip of the
    owning Composite. Keyed on insertion order, the same composite would seed
    a different initial value after a round-trip than before it.
    """
    writer_roles = (
        PortRole.EVOLVED,
        PortRole.EXCLUSIVE,
        PortRole.LATCHED,
        PortRole.ASSIGNED,
    )
    # path -> {tier: [(proc_name, default)]}, tier 0 = writer, 1 = reader.
    claims: dict[str, dict[int, list[tuple[str, Any]]]] = {}
    for proc_name in sorted(processes):
        proc = processes[proc_name]
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            store_path = topo.get(port_name, port_name)
            claims.setdefault(store_path, {})
            if port.default is None:
                continue  # writes here, claims nothing about where it starts
            tier = 0 if port.role in writer_roles else 1
            claims[store_path].setdefault(tier, []).append(
                (proc_name, port.default)
            )

    initial = dict(initial or {})
    store: dict[str, jnp.ndarray] = {}
    for store_path, tiers in claims.items():
        if store_path in initial:
            store[store_path] = jnp.asarray(initial.pop(store_path))
            continue
        if not tiers:
            store[store_path] = jnp.asarray(0.0)  # every port abstained
            continue
        winners = tiers[min(tiers)]
        name, default = winners[0]
        rejected = {d for _, d in winners[1:] if not _same_default(d, default)}
        if rejected:
            raise ValueError(
                f"{store_path} is claimed by {len(winners)} ports of the same "
                f"role with differing initial values "
                f"{sorted(rejected | {default}, key=repr)}. Which one the "
                f"path starts at is a modelling decision, not a tie-break: "
                f"declare it with Composite(initial={{{store_path!r}: "
                f"<value>}}), or wire the disagreeing ports to separate paths."
            )
        # No explicit dtype: honor the global JAX default (float64 under
        # jax_enable_x64), so integration state matches the rtol=1e-6 solver
        # tolerance rather than the float32 floor.
        store[store_path] = jnp.asarray(default)
    return store


def _same_default(a: Any, b: Any) -> bool:
    """Whether two port defaults agree. Drives a warning and nothing else.

    Traced defaults count as agreeing: there is no concrete value to compare,
    and this is reached under ``jit`` — ``initial_state_vec`` → ``steady_state``
    → a calibration loss — where raising to report a warning would be absurd.

    The comparison runs in numpy, not ``jnp``. Under ``jit`` a ``jnp`` compare
    of two *concrete* floats still yields a traced bool, so ``bool()`` on it
    raises — guarding the inputs with :func:`is_traced` is not enough.
    """
    if is_traced(a, b):
        return True
    return bool(np.all(np.asarray(a) == np.asarray(b)))


def validate_topology(
    processes: dict[str, Process],
    topology: dict[str, dict[str, str]],
) -> list[str]:
    """Check that the topology is consistent with process port schemas.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    for proc_name, proc in processes.items():
        schema = proc.ports_schema()
        topo = topology.get(proc_name, {})

        # Every port must have a topology entry
        for port_name in schema:
            if port_name not in topo:
                errors.append(
                    f"Process {proc_name!r}: port {port_name!r} has no topology mapping"
                )

        # Every topology entry must correspond to a declared port
        for port_name in topo:
            if port_name not in schema:
                errors.append(
                    f"Process {proc_name!r}: topology maps {port_name!r} "
                    f"but it is not in ports_schema()"
                )

    # Sole-owner conflicts: no two processes may claim the same store path
    # with a sole-owner role (EXCLUSIVE derivative or ASSIGNED algebraic value).
    exclusive_owners: dict[str, str] = {}  # store_path → proc_name
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role in (PortRole.EXCLUSIVE, PortRole.ASSIGNED):
                store_path = topo.get(port_name, port_name)
                if store_path in exclusive_owners:
                    errors.append(
                        f"Sole-owner conflict: store path {store_path!r} claimed "
                        f"by both {exclusive_owners[store_path]!r} and "
                        f"{proc_name!r}"
                    )
                else:
                    exclusive_owners[store_path] = proc_name

    # Check that exclusive store paths are not also evolved by other processes
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role == PortRole.EVOLVED:
                store_path = topo.get(port_name, port_name)
                if (
                    store_path in exclusive_owners
                    and exclusive_owners[store_path] != proc_name
                ):
                    errors.append(
                        f"Exclusive conflict: store path {store_path!r} is EXCLUSIVE "
                        f"in {exclusive_owners[store_path]!r} but EVOLVED in {proc_name!r}"
                    )

    # Check process kind / port role compatibility
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            # Continuous processes must not write to LATCHED ports
            if (
                proc.kind == ProcessKind.CONTINUOUS
                and port.role == PortRole.LATCHED
            ):
                errors.append(
                    f"Process {proc_name!r} is CONTINUOUS but port {port_name!r} "
                    f"is LATCHED. Only DISCRETE/EVENT processes may write LATCHED ports."
                )
            # Discrete/event processes must not write to EVOLVED/EXCLUSIVE ports
            if proc.kind in (ProcessKind.DISCRETE, ProcessKind.EVENT):
                if port.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE):
                    errors.append(
                        f"Process {proc_name!r} is {proc.kind.value.upper()} but port "
                        f"{port_name!r} is {port.role.value.upper()}. "
                        f"DISCRETE/EVENT processes should use LATCHED ports for output."
                    )

    # Check that DISCRETE processes declare dt_step
    for proc_name, proc in processes.items():
        if proc.kind == ProcessKind.DISCRETE and proc.dt_step is None:
            errors.append(
                f"Process {proc_name!r} is DISCRETE but has no dt_step. "
                f"Set dt_step to the interval between update calls (in seconds)."
            )

    # A LATCHED writer (DISCRETE/EVENT) and an EVOLVED/EXCLUSIVE writer
    # (CONTINUOUS) may share a store path: the two are orthogonal. The
    # CONTINUOUS process owns the derivative; the EVENT/DISCRETE process
    # scatter-adds a one-shot delta at sync points without contributing to
    # the derivative. This is the canonical event-"kick" pattern — a one-time
    # perturbation of a continuous state (see
    # ``hallsim.models.kick_event.KickEvent``).
    return errors
