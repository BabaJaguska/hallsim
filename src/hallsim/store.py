"""Store utilities — initial state construction and topology validation.

The store is a plain ``dict[str, jnp.ndarray]`` with path-like keys::

    {"cytoplasm/ROS": jnp.array(0.1), "nucleus/p53": jnp.array(0.5)}

It only appears at the API boundary (initial state, simulation results).
Internally everything runs on the flat vector returned by
``Composite.flatten`` / ``unflatten``.
"""

from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import Any

import jax.numpy as jnp
import numpy as np

from hallsim.process import PortRole, Process, ProcessKind
from hallsim.tracing import is_traced

log = logging.getLogger(__name__)


def as_paths(entry) -> tuple[str, ...]:
    """A topology entry as the store paths it binds.

    A bare string is the one-path case. This is the only place that knows a
    port may bind more than one path; every reader goes through it.
    """
    return (entry,) if isinstance(entry, str) else tuple(entry)


def read_write_paths(schema, topo_p):
    """Store paths a process reads and writes, as ``(reads, writes)``.

    The one definition of "who drives whom", shared by group ordering and
    coupling-mode resolution. A pure source (``reads_value=False``) writes
    without reading, so it drives others and is driven by none.
    """
    reads, writes = set(), set()
    for port, entry in topo_p.items():
        spec = schema.get(port)
        if spec is None:
            continue
        for path in as_paths(entry):
            if spec.role in (
                PortRole.EVOLVED,
                PortRole.EXCLUSIVE,
                PortRole.LATCHED,
                PortRole.ASSIGNED,
            ):
                writes.add(path)
            if spec.role is PortRole.INPUT or (
                spec.role is PortRole.EVOLVED and spec.reads_value
            ):
                reads.add(path)
    return reads, writes


def port_defaults(port, paths, proc_name="", port_name=""):
    """One initial value per path a port binds.

    A block's ``default`` is either one value broadcast over its elements or
    one per element, in element order. The number of bound paths must match
    the number of declared elements — a mismatch means the topology and the
    schema disagree about how wide the port is, which no later stage can
    detect.
    """
    width = port.width
    if width is None:
        if len(paths) != 1:
            raise ValueError(
                f"{proc_name}.{port_name} binds {len(paths)} paths but "
                f"declares no elements; give it elements=(...) to bind a block"
            )
        return (port.default,)
    if len(paths) != width:
        raise ValueError(
            f"{proc_name}.{port_name} declares {width} elements "
            f"{port.elements} but the topology binds {len(paths)} paths"
        )
    default = port.default
    if default is None:
        return (None,) * width
    if np.ndim(default) == 0:
        return (default,) * width
    values = tuple(np.asarray(default).reshape(-1))
    if len(values) != width:
        raise ValueError(
            f"{proc_name}.{port_name} has {len(values)} default values for "
            f"{width} elements"
        )
    return values


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
            paths = as_paths(topo[port_name])
            defaults = port_defaults(port, paths, proc_name, port_name)
            for store_path, value in zip(paths, defaults):
                claims.setdefault(store_path, {})
                if value is None:
                    continue  # writes here, claims nothing about its start
                tier = 0 if port.role in writer_roles else 1
                claims[store_path].setdefault(tier, []).append(
                    (proc_name, value)
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
    if initial:
        unknown = sorted(initial)
        near = get_close_matches(unknown[0], store, n=3)
        raise KeyError(
            f"Composite(initial=...) names paths no port claims: {unknown}"
            + (f"; did you mean {near}?" if near else "")
        )
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
                if port_name not in topo:
                    continue
                for store_path in as_paths(topo[port_name]):
                    if store_path in exclusive_owners:
                        errors.append(
                            f"Sole-owner conflict: store path {store_path!r} "
                            f"claimed by both "
                            f"{exclusive_owners[store_path]!r} and "
                            f"{proc_name!r}"
                        )
                    else:
                        exclusive_owners[store_path] = proc_name

    # Check that exclusive store paths are not also evolved by other processes
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role == PortRole.EVOLVED:
                if port_name not in topo:
                    continue
                for store_path in as_paths(topo[port_name]):
                    owner = exclusive_owners.get(store_path)
                    if owner is not None and owner != proc_name:
                        errors.append(
                            f"Exclusive conflict: store path {store_path!r} "
                            f"is EXCLUSIVE in {owner!r} but EVOLVED in "
                            f"{proc_name!r}"
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
