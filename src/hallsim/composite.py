"""Composite — wires Processes together via a Topology into a single ODE RHS.

A Composite is an Equinox module that bundles:

1. A dict of named Processes (each an Equinox module).
2. A topology mapping each process's port names to store paths.

It builds a combined right-hand-side function ``f(t, y) -> dy/dt`` that:

- Extracts each process's port view from the global store via topology.
- Calls each process's ``derivative()``.
- Routes derivatives back to store paths.
- Sums additive contributions (EVOLVED ports).
- Enforces exclusivity (EXCLUSIVE ports).

The resulting RHS is compatible with Diffrax solvers.

Example
-------
>>> composite = Composite(
...     processes={"decay": Decay(rate=0.1), "growth": Growth(rate=0.05)},
...     topology={
...         "decay": {"x": "pool/x"},
...         "growth": {"x": "pool/x", "nutrient": "env/nutrient"},
...     },
... )
>>> rhs = composite.build_rhs()
>>> y0 = composite.initial_state()
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import PortRole, Process, ProcessKind
from hallsim.store import (
    build_initial_store,
    extract_port_view,
    route_derivatives,
    validate_topology,
    zeros_like_store,
)


class Composite(eqx.Module):
    """A wired bundle of Processes sharing a flat state store.

    Parameters
    ----------
    processes:
        ``{name: Process}`` — each an Equinox module.
    topology:
        ``{name: {port_name: store_path}}`` — maps each process's
        local port names to global store paths.  Two processes writing
        to the same store path contribute additively (EVOLVED) or
        exclusively (EXCLUSIVE).
    """

    processes: dict[str, Process]
    topology: dict[str, dict[str, str]]

    def __init__(
        self,
        processes: dict[str, Process],
        topology: dict[str, dict[str, str]],
        *,
        validate: bool = True,
        semantic_validation: bool | dict = False,
    ) -> None:
        self.processes = processes
        self.topology = topology
        if validate:
            errors = validate_topology(processes, topology)
            if errors:
                raise ValueError(
                    "Topology validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
        if semantic_validation:
            import warnings

            from hallsim.validation import CompositeValidator

            if isinstance(semantic_validation, dict):
                validator = CompositeValidator(**semantic_validation)
            else:
                validator = CompositeValidator()
            report = validator.validate(processes, topology)
            if report.errors:
                raise ValueError(f"Semantic validation failed:\n{report}")
            if report.warnings:
                warnings.warn(
                    f"Semantic validation warnings:\n{report}",
                    UserWarning,
                    stacklevel=2,
                )

    # -----------------------------------------------------------------
    # State flattening: dict ↔ array
    # -----------------------------------------------------------------

    def _store_keys(self) -> list[str]:
        """Sorted list of all store paths — deterministic key order."""
        return sorted(self.store_paths())

    def flatten(
        self, state: dict[str, jnp.ndarray], keys: list[str] | None = None,
    ) -> jnp.ndarray:
        """Convert state dict → 1-D array in sorted key order.

        Parameters
        ----------
        state:
            ``{store_path: scalar_or_array}``
        keys:
            Key order.  If ``None``, uses ``_store_keys()`` (sorted).

        Returns
        -------
        1-D ``jnp.ndarray`` of shape ``(n_vars,)``.
        """
        if keys is None:
            keys = self._store_keys()
        return jnp.array([state[k] for k in keys])

    def unflatten(
        self, vec: jnp.ndarray, keys: list[str] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Convert 1-D array → state dict.

        Parameters
        ----------
        vec:
            1-D array from :meth:`flatten`.
        keys:
            Key order (must match the order used in ``flatten``).
        """
        if keys is None:
            keys = self._store_keys()
        return {k: vec[i] for i, k in enumerate(keys)}

    # -----------------------------------------------------------------
    # Build the combined ODE right-hand side
    # -----------------------------------------------------------------

    def build_rhs(self):
        """Return a JAX-compatible ``f(t, y, args=None) -> dy/dt``.

        The returned function:
        1. For each process: extracts its port view, calls ``derivative()``.
        2. Routes derivatives back to store paths.
        3. Sums EVOLVED contributions; passes through EXCLUSIVE ones.
        4. Returns a dict with the same structure as ``y``.

        The function is safe to JIT-compile and differentiate.
        """
        # Pre-compute static metadata for the RHS closure
        proc_items = list(self.processes.items())
        topo = self.topology

        # Build a set of exclusive store paths for fast lookup
        exclusive_paths: set[str] = set()
        for proc_name, proc in proc_items:
            proc_topo = topo[proc_name]
            for port_name, port in proc.ports_schema().items():
                if port.role == PortRole.EXCLUSIVE:
                    exclusive_paths.add(proc_topo[port_name])

        def rhs(t, y, args=None):
            accum = zeros_like_store(y)

            for proc_name, proc in proc_items:
                proc_topo = topo[proc_name]
                # 1. Extract port view
                view = extract_port_view(y, proc_topo)
                # 2. Compute derivatives
                raw_derivs = proc.derivative(t, view)
                # 3. Route back to store paths
                routed = route_derivatives(raw_derivs, proc_topo)
                # 4. Accumulate
                for store_path, dval in routed.items():
                    accum[store_path] = accum[store_path] + dval

            return accum

        return rhs

    def build_flat_rhs(self):
        """Return a JAX-compatible ``f(t, y_vec, args=None) -> dy_vec``.

        Like :meth:`build_rhs` but operates on a **flat 1-D array** instead
        of a dict.  This compiles orders of magnitude faster under
        ``jax.jit`` / ``jax.grad`` because JAX traces a single array
        value rather than N separate dict entries.

        The key order is ``sorted(store_paths())``, matching
        :meth:`flatten` / :meth:`unflatten`.

        Returns
        -------
        ``(rhs_fn, keys)`` where ``keys`` is the sorted list of store paths
        so callers can ``flatten`` / ``unflatten`` consistently.
        """
        proc_items = list(self.processes.items())
        topo = self.topology
        keys = self._store_keys()
        key_to_idx = {k: i for i, k in enumerate(keys)}
        n = len(keys)

        def rhs(t, y_vec, args=None):
            # Unflatten
            y_dict = {k: y_vec[i] for i, k in enumerate(keys)}
            accum = jnp.zeros(n)

            for proc_name, proc in proc_items:
                proc_topo = topo[proc_name]
                view = extract_port_view(y_dict, proc_topo)
                raw_derivs = proc.derivative(t, view)
                routed = route_derivatives(raw_derivs, proc_topo)
                for store_path, dval in routed.items():
                    accum = accum.at[key_to_idx[store_path]].add(dval)

            return accum

        return rhs, keys

    # -----------------------------------------------------------------
    # Initial state
    # -----------------------------------------------------------------

    def initial_state(self) -> dict[str, jnp.ndarray]:
        """Merge all process port defaults into a single store dict.

        Returns
        -------
        ``{store_path: jnp.ndarray}`` — ready to pass to a Diffrax solver.
        """
        return build_initial_store(self.processes, self.topology)

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def store_paths(self) -> set[str]:
        """All store paths referenced by any process via the topology."""
        paths: set[str] = set()
        for proc_topo in self.topology.values():
            paths.update(proc_topo.values())
        return paths

    def metadata(self) -> dict[str, Any]:
        """Aggregate metadata from all processes."""
        return {name: proc.metadata() for name, proc in self.processes.items()}

    # -----------------------------------------------------------------
    # Process kind filtering
    # -----------------------------------------------------------------

    def continuous_processes(self) -> dict[str, Process]:
        """All CONTINUOUS kind processes."""
        return {
            n: p
            for n, p in self.processes.items()
            if p.kind == ProcessKind.CONTINUOUS
        }

    def discrete_processes(self) -> dict[str, Process]:
        """All DISCRETE kind processes."""
        return {
            n: p
            for n, p in self.processes.items()
            if p.kind == ProcessKind.DISCRETE
        }

    def event_processes(self) -> dict[str, Process]:
        """All EVENT kind processes."""
        return {
            n: p
            for n, p in self.processes.items()
            if p.kind == ProcessKind.EVENT
        }

    # -----------------------------------------------------------------
    # Group-level RHS building
    # -----------------------------------------------------------------

    def build_group_rhs(self, proc_names: list[str]):
        """Build a JAX-compatible RHS for a subset of continuous processes.

        Only the named processes contribute derivatives.  All other store
        paths get zero derivatives.  The returned function has the same
        signature as ``build_rhs()``: ``f(t, y, args=None) -> dy/dt``.
        """
        proc_items = [(n, self.processes[n]) for n in proc_names]
        topo = self.topology

        def rhs(t, y, args=None):
            accum = zeros_like_store(y)
            for proc_name, proc in proc_items:
                proc_topo = topo[proc_name]
                view = extract_port_view(y, proc_topo)
                raw_derivs = proc.derivative(t, view)
                routed = route_derivatives(raw_derivs, proc_topo)
                for store_path, dval in routed.items():
                    accum[store_path] = accum[store_path] + dval
            return accum

        return rhs

    # -----------------------------------------------------------------
    # Timescale auto-grouping
    # -----------------------------------------------------------------

    def auto_groups(self, max_ratio: float = 100.0) -> dict[str, list[str]]:
        """Partition continuous processes into timescale groups.

        Processes within ``max_ratio`` of each other share a group.
        Processes without a declared timescale go into a default group.

        Returns
        -------
        ``{group_name: [proc_name, ...]}``
        """
        continuous = self.continuous_processes()
        if not continuous:
            return {}

        # Separate declared vs undeclared timescales
        with_ts: list[tuple[str, float]] = []
        without_ts: list[str] = []
        for name, proc in continuous.items():
            if proc.timescale is not None:
                with_ts.append((name, proc.timescale))
            else:
                without_ts.append(name)

        if not with_ts:
            # All undeclared → single group
            return {"default": list(continuous.keys())}

        # Sort by timescale and cluster
        with_ts.sort(key=lambda x: x[1])
        groups: dict[str, list[str]] = {}
        group_idx = 0
        current_group: list[str] = [with_ts[0][0]]
        current_min_ts = with_ts[0][1]

        for name, ts in with_ts[1:]:
            if ts / current_min_ts <= max_ratio:
                current_group.append(name)
            else:
                groups[f"group_{group_idx}"] = current_group
                group_idx += 1
                current_group = [name]
                current_min_ts = ts

        groups[f"group_{group_idx}"] = current_group

        # Add undeclared to default group
        if without_ts:
            groups["default"] = without_ts

        return groups
