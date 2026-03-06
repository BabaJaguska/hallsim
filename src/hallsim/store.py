"""Store utilities for the flat-dict state representation.

The store is a plain ``dict[str, jnp.ndarray]`` with path-like keys::

    {"cytoplasm/ROS": jnp.array(0.1), "nucleus/p53": jnp.array(0.5)}

This module provides helpers for building, merging, validating, and
converting stores.  The store is a valid JAX PyTree — it works directly
with ``jax.tree_map``, Diffrax solvers, and ``jax.grad``.
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, ProcessKind


# ---------------------------------------------------------------------------
# Build initial store from processes + topology
# ---------------------------------------------------------------------------

def build_initial_store(
    processes: dict[str, Process],
    topology: dict[str, dict[str, str]],
) -> dict[str, jnp.ndarray]:
    """Create the initial state dict from port defaults.

    For each process, looks up its ports via ``ports_schema()``, maps them
    through the topology to store paths, and collects default values.

    If two ports map to the same store path, the first-seen default wins
    (deterministic because dicts are insertion-ordered in Python 3.7+).

    Parameters
    ----------
    processes:
        ``{process_name: Process}``
    topology:
        ``{process_name: {port_name: store_path}}``

    Returns
    -------
    ``{store_path: jnp.ndarray}`` ready for Diffrax.
    """
    store: dict[str, jnp.ndarray] = {}
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            store_path = topo.get(port_name, port_name)  # identity if no topology
            if store_path not in store:
                store[store_path] = jnp.asarray(port.default, dtype=jnp.float32)
    return store


# ---------------------------------------------------------------------------
# Extract port view / route derivatives back
# ---------------------------------------------------------------------------

def extract_port_view(
    store: dict[str, jnp.ndarray],
    topo: dict[str, str],
) -> dict[str, jnp.ndarray]:
    """Extract values for a single process's ports from the store.

    Parameters
    ----------
    store:
        Full simulation state.
    topo:
        ``{port_name: store_path}`` for this process.

    Returns
    -------
    ``{port_name: value}``
    """
    return {port_name: store[store_path] for port_name, store_path in topo.items()}


def route_derivatives(
    derivs: dict[str, jnp.ndarray],
    topo: dict[str, str],
) -> dict[str, jnp.ndarray]:
    """Map process-local derivative names back to store paths.

    Parameters
    ----------
    derivs:
        ``{port_name: dy/dt}`` from ``process.derivative()``.
    topo:
        ``{port_name: store_path}``

    Returns
    -------
    ``{store_path: dy/dt}``
    """
    return {topo[port_name]: value for port_name, value in derivs.items() if port_name in topo}


# ---------------------------------------------------------------------------
# Zero store (for additive accumulation)
# ---------------------------------------------------------------------------

def zeros_like_store(store: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Return a store with the same keys but all-zero values."""
    return {k: jnp.zeros_like(v) for k, v in store.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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

    # Check exclusive conflicts: no two processes may have EXCLUSIVE ports
    # that map to the same store path.
    exclusive_owners: dict[str, str] = {}  # store_path → proc_name
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role == PortRole.EXCLUSIVE:
                store_path = topo.get(port_name, port_name)
                if store_path in exclusive_owners:
                    errors.append(
                        f"Exclusive conflict: store path {store_path!r} claimed by "
                        f"both {exclusive_owners[store_path]!r} and {proc_name!r}"
                    )
                else:
                    exclusive_owners[store_path] = proc_name

    # Check that exclusive store paths are not also evolved by other processes
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role == PortRole.EVOLVED:
                store_path = topo.get(port_name, port_name)
                if store_path in exclusive_owners and exclusive_owners[store_path] != proc_name:
                    errors.append(
                        f"Exclusive conflict: store path {store_path!r} is EXCLUSIVE "
                        f"in {exclusive_owners[store_path]!r} but EVOLVED in {proc_name!r}"
                    )

    # Check process kind / port role compatibility
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            # Continuous processes must not write to LATCHED ports
            if proc.kind == ProcessKind.CONTINUOUS and port.role == PortRole.LATCHED:
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

    # Check LATCHED port conflicts: no continuous process writes, but also
    # check that LATCHED paths are not mixed with EVOLVED/EXCLUSIVE
    latched_paths: dict[str, str] = {}  # store_path → proc_name
    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role == PortRole.LATCHED:
                store_path = topo.get(port_name, port_name)
                latched_paths[store_path] = proc_name

    for proc_name, proc in processes.items():
        topo = topology.get(proc_name, {})
        for port_name, port in proc.ports_schema().items():
            if port.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE):
                store_path = topo.get(port_name, port_name)
                if store_path in latched_paths:
                    errors.append(
                        f"Store path {store_path!r} is LATCHED by "
                        f"{latched_paths[store_path]!r} but "
                        f"{port.role.value.upper()} by {proc_name!r}. "
                        f"LATCHED paths cannot also be EVOLVED/EXCLUSIVE."
                    )

    return errors
