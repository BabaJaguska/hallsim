"""Composite — wires Processes together via a Topology into a single ODE RHS.

A Composite is an Equinox module that bundles:

1. A dict of named Processes (each an Equinox module).
2. A topology mapping each process's port names to store paths.

It builds a JAX-compatible flat RHS ``f(t, y_vec) -> dy_vec`` that:

- Calls each process's ``derivative()`` with a small per-process port view.
- Scatter-adds each process's contribution into a single accumulator vector.
- Sums additive contributions (EVOLVED ports) implicitly via the scatter.
- Enforces exclusivity (EXCLUSIVE ports) at composition time via topology
  validation.

Example
-------
>>> composite = Composite(
...     processes={"decay": Decay(rate=0.1), "growth": Growth(rate=0.05)},
...     topology={
...         "decay": {"x": "pool/x"},
...         "growth": {"x": "pool/x", "nutrient": "env/nutrient"},
...     },
... )
>>> rhs, keys = composite.build_rhs()
>>> y0_vec = composite.flatten(composite.initial_state(), keys)
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import PortRole, Process, ProcessKind
from hallsim.store import build_initial_store, validate_topology


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
        semantic_validation: bool | dict = True,
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

    def store_keys(self) -> list[str]:
        """Sorted list of all store paths — deterministic key order."""
        return sorted(self.store_paths())

    def flatten(
        self,
        state: dict[str, jnp.ndarray],
        keys: list[str] | None = None,
    ) -> jnp.ndarray:
        """Convert state dict → array in sorted key order.

        Stacks values along the *last* axis so the layout works for
        scalar values (rank-0) and batched values (rank-1+) without
        change. Scalars produce ``(n_vars,)``; ``(batch,)`` values
        produce ``(batch, n_vars)``. This is what makes ``Scheduler.run``
        accept batched y0 dicts natively — no special vmap path needed.

        Parameters
        ----------
        state:
            ``{store_path: scalar_or_array}``. Each value may be a
            JAX scalar or a batched array; all values must share a
            common batch shape.
        keys:
            Key order.  If ``None``, uses ``store_keys()`` (sorted).

        Returns
        -------
        ``jnp.ndarray`` of shape ``(..., n_vars)``.
        """
        if keys is None:
            keys = self.store_keys()
        return jnp.stack([jnp.asarray(state[k]) for k in keys], axis=-1)

    def initial_state_vec(self, keys: list[str] | None = None) -> jnp.ndarray:
        """Initial state as a flat tensor in trailing-axis convention.

        JAX-native counterpart to :meth:`initial_state` (which returns a
        dict). Use this as the default y0 for ``Scheduler.run`` and
        ``Scheduler.run`` — the public API takes a tensor, not a dict.

        Returns
        -------
        ``jnp.ndarray`` of shape ``(n_vars,)``. To override values, use
        ``y0.at[composite.keys.index("path")].set(value)``. To batch over
        a population, ``jnp.broadcast_to(y0, (batch, n_vars))``.
        """
        return self.flatten(self.initial_state(), keys)

    def unflatten(
        self,
        vec: jnp.ndarray,
        keys: list[str] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Convert array (last axis = state vars) → state dict.

        Symmetric inverse of :meth:`flatten`. Indexes the last axis so
        ``(n_vars,)`` produces scalars in the dict and ``(batch, n_vars)``
        produces ``(batch,)`` arrays in the dict.

        Parameters
        ----------
        vec:
            Array from :meth:`flatten` with state vars on the last axis.
        keys:
            Key order (must match the order used in ``flatten``).
        """
        if keys is None:
            keys = self.store_keys()
        return {k: vec[..., i] for i, k in enumerate(keys)}

    # -----------------------------------------------------------------
    # Build the combined ODE right-hand side
    # -----------------------------------------------------------------

    def build_rhs(self, proc_names: list[str] | None = None):
        """Return a JAX-compatible flat ``f(t, y_vec, args=None) -> dy_vec``.

        Operates on a flat 1-D ``jnp.ndarray`` indexed by
        ``sorted(store_paths())``.  Per-process index maps are precomputed
        at composition time; each process contributes via a single batched
        ``.at[idxs].add(vals)`` scatter.

        Parameters
        ----------
        proc_names:
            Subset of processes to include.  ``None`` (default) uses every
            CONTINUOUS process — the whole-system case.  Pass an explicit
            list for operator splitting (the Scheduler does this per
            group); only the named processes contribute, all other store
            entries get zero.  ``keys`` is always the full store layout,
            so a single flat state vector is valid for every group's solve.

        Returns
        -------
        ``(rhs_fn, keys)`` — pair with :meth:`flatten` / :meth:`unflatten`
        at the API boundary if you need dict-shaped state.
        """
        if proc_names is None:
            proc_names = list(self.continuous_processes().keys())

        keys = self.store_keys()
        key_to_idx = {k: i for i, k in enumerate(keys)}

        # Precompute static index maps per process.  The per-process port
        # dict (``view`` below) survives because ``Process.derivative``'s
        # public contract is ``derivative(t, {port_name: value}) -> dict``.
        pre = []
        for proc_name in proc_names:
            proc = self.processes[proc_name]
            proc_topo = self.topology[proc_name]
            read_pairs = tuple(
                (port, key_to_idx[sp]) for port, sp in proc_topo.items()
            )
            schema = proc.ports_schema()
            write_pairs = tuple(
                (port, key_to_idx[proc_topo[port]])
                for port, p in schema.items()
                if p.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE)
            )
            pre.append((proc, read_pairs, write_pairs))

        def rhs(t, y_vec, args=None):
            # Trailing-axis convention: y_vec is (..., n_vars). Scalars
            # for unbatched runs (shape (n,)); (batch, n) for batched
            # population runs through Scheduler.run with a batched y0.
            # accum follows y_vec's shape so the scatter stays aligned.
            accum = jnp.zeros_like(y_vec)
            for proc, read_pairs, write_pairs in pre:
                view = {port: y_vec[..., idx] for port, idx in read_pairs}
                raw = proc.derivative(t, view)
                out = [
                    (idx, raw[port])
                    for port, idx in write_pairs
                    if port in raw
                ]
                if out:
                    idxs = jnp.array([i for i, _ in out])
                    # Stack derivative values along the trailing axis to
                    # match accum's layout. For scalar derivatives this
                    # is shape (n_writes,); for batched (batch,) per-port
                    # derivatives this is (batch, n_writes).
                    vals = jnp.stack([v for _, v in out], axis=-1)
                    accum = accum.at[..., idxs].add(vals)
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
