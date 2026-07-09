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

import logging
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import PortRole, Process, ProcessKind
from hallsim.store import build_initial_store, validate_topology
from hallsim.units import canonical_units, conversion_factor

log = logging.getLogger(__name__)


def _flatten_subcomposites(
    items: dict[str, Process | "Composite"],
    extra_topology: dict[str, dict[str, str]],
) -> tuple[dict[str, Process], dict[str, dict[str, str]]]:
    """Expand sub-Composites into a flat (processes, topology) pair.

    A value that is a :class:`Composite` contributes its internal
    processes under ``<outer_key>.<sub_name>`` and its internal store
    paths under ``<outer_key>/<path>`` (idempotent: paths already
    starting with ``<outer_key>/`` are kept as-is).

    A value that is a :class:`Process` contributes one entry directly
    under ``outer_key``, with its topology taken from
    ``extra_topology[outer_key]`` (the topology arg passed to
    :class:`Composite`).
    """
    flat_processes: dict[str, Process] = {}
    flat_topology: dict[str, dict[str, str]] = {}

    for outer_key, item in items.items():
        if isinstance(item, Composite):
            prefix = f"{outer_key}/"
            for sub_name, sub_proc in item.processes.items():
                merged_name = f"{outer_key}.{sub_name}"
                if merged_name in flat_processes:
                    raise ValueError(
                        f"process name collision while flattening "
                        f"composite {outer_key!r}: {merged_name!r} "
                        f"already exists in the merged composite"
                    )
                flat_processes[merged_name] = sub_proc
                sub_topo = item.topology.get(sub_name, {})
                flat_topology[merged_name] = {
                    port: (path if path.startswith(prefix) else prefix + path)
                    for port, path in sub_topo.items()
                }
        elif isinstance(item, Process):
            if outer_key in flat_processes:
                raise ValueError(
                    f"process name collision: {outer_key!r} already "
                    f"exists in the merged composite"
                )
            flat_processes[outer_key] = item
            # Ports without an explicit topology entry get an
            # auto-prefixed store path ``<outer_key>/<port>``. This
            # matches sub-Composite flattening: the outer key becomes
            # the namespace by default. Caller-provided topology entries
            # win on a per-port basis, so an INPUT port reading from a
            # canonical path elsewhere stays explicit.
            user_topo = extra_topology.get(outer_key, {})
            flat_topology[outer_key] = {
                port: user_topo.get(port, f"{outer_key}/{port}")
                for port in item.ports_schema()
            }
        else:
            raise TypeError(
                f"processes[{outer_key!r}] must be a Process or Composite, "
                f"got {type(item).__name__}"
            )

    return flat_processes, flat_topology


class Composite(eqx.Module):
    """A wired bundle of Processes sharing a flat state store.

    Parameters
    ----------
    processes:
        ``{name: Process | Composite}`` — each an Equinox module. When a
        value is another Composite, it is flattened in place: each
        sub-process is renamed ``<outer_key>.<sub_name>`` and every store
        path the sub-composite references is prefixed with
        ``<outer_key>/`` unless it already carries that prefix. This is
        how independent published composites are merged into one.
    topology:
        ``{name: {port_name: store_path}}`` — maps each top-level
        Process's local port names to global store paths. Sub-composites
        bring their own topology; the caller only writes topology entries
        for raw Process values at the top level.  Two processes writing
        to the same store path contribute additively (EVOLVED) or
        exclusively (EXCLUSIVE).
    rewire:
        Optional ``{old_path: new_path}`` mapping applied after sub-
        composite flattening. Use this to resolve overlapping biology
        across merged composites — e.g.,
        ``rewire={"dp14/mTORC1_pS2448": "eriq/mTOR_activity"}`` declares
        DP14's phospho-mTOR state and ERiQ's mTOR activity refer to the
        same canonical store path. Targets are taken as-is; the rewire
        is a single pass over the flattened topology.
    """

    processes: dict[str, Process]
    topology: dict[str, dict[str, str]]

    def __init__(
        self,
        processes: dict[str, Process | Composite],
        topology: dict[str, dict[str, str]] | None = None,
        *,
        rewire: dict[str, str] | None = None,
        validate: bool = True,
        semantic_validation: bool | dict = True,
    ) -> None:
        flat_processes, flat_topology = _flatten_subcomposites(
            processes, topology or {}
        )
        if rewire:
            flat_topology = {
                proc_name: {
                    port: rewire.get(path, path) for port, path in topo.items()
                }
                for proc_name, topo in flat_topology.items()
            }
        self.processes = flat_processes
        self.topology = flat_topology
        if validate:
            errors = validate_topology(flat_processes, flat_topology)
            if errors:
                raise ValueError(
                    "Topology validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
        if semantic_validation:
            from hallsim.validation import CompositeValidator

            if isinstance(semantic_validation, dict):
                validator = CompositeValidator(**semantic_validation)
            else:
                validator = CompositeValidator()
            report = validator.validate(flat_processes, flat_topology)
            if report.errors:
                raise ValueError(f"Semantic validation failed:\n{report}")
            if report.warnings:
                log.warning("Semantic validation warnings:\n%s", report)

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

        # Canonical unit per store path (whole composite, so it's consistent
        # across operator-splitting groups and with the seeded initial state).
        canon = canonical_units(self.processes, self.topology)

        # Precompute static index maps per process.  The per-process port
        # dict (``view`` below) survives because ``Process.derivative``'s
        # public contract is ``derivative(t, {port_name: value}) -> dict``.
        # Each read carries a factor canonical→port-unit; each write a factor
        # port-unit→canonical — 1.0 unless the port's unit differs from the
        # path's canonical, in which case contributions are reconciled so
        # writers with compatible-but-different units sum correctly.
        pre = []
        for proc_name in proc_names:
            proc = self.processes[proc_name]
            proc_topo = self.topology[proc_name]
            schema = proc.ports_schema()
            read_pairs = tuple(
                (
                    port,
                    key_to_idx[sp],
                    conversion_factor(canon.get(sp, ""), schema[port].units),
                )
                for port, sp in proc_topo.items()
            )
            write_pairs = tuple(
                (
                    port,
                    key_to_idx[proc_topo[port]],
                    conversion_factor(p.units, canon.get(proc_topo[port], "")),
                )
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
                view = {
                    port: y_vec[..., idx] * rf for port, idx, rf in read_pairs
                }
                raw = proc.derivative(t, view)
                out = [
                    (idx, raw[port] * wf)
                    for port, idx, wf in write_pairs
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

    def evolved_indices(
        self,
        proc_names: list[str] | None = None,
        keys: list[str] | None = None,
    ) -> jnp.ndarray:
        """Trailing-axis indices a set of processes writes derivatives to.

        The union of store indices targeted by every ``EVOLVED`` /
        ``EXCLUSIVE`` port of the named processes — i.e. the state
        components that actually evolve when ``build_rhs(proc_names)`` is
        integrated (all other indices keep a zero derivative). Used to
        restrict a group's Jacobian to its own dynamics for stiffness
        analysis and to scope coupling splices.

        Parameters
        ----------
        proc_names:
            Subset of processes. ``None`` (default) uses every CONTINUOUS
            process.
        keys:
            Store layout. ``None`` uses :meth:`store_keys`.

        Returns
        -------
        ``jnp.ndarray`` of sorted int32 indices.
        """
        if proc_names is None:
            proc_names = list(self.continuous_processes().keys())
        if keys is None:
            keys = self.store_keys()
        key_to_idx = {k: i for i, k in enumerate(keys)}
        written: set[int] = set()
        for pname in proc_names:
            proc = self.processes[pname]
            proc_topo = self.topology[pname]
            for port, p in proc.ports_schema().items():
                if p.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE):
                    written.add(key_to_idx[proc_topo[port]])
        return jnp.array(sorted(written), dtype=jnp.int32)

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

    def calibration_targets(
        self,
        *,
        include_hallmark_targets: bool = False,
        registry: dict | None = None,
    ) -> list:
        """Enumerate mechanism parameters across processes, minus hallmark knobs.

        Walks every process, calls each one's ``calibratable_params()``,
        fills in the process namespace, and **subtracts every
        ``(process_name, field)`` pair that's targeted by any
        :class:`hallsim.hallmarks.ParameterMapping` in the active
        hallmark registry**. Hallmarks are knobs — the experimenter /
        modeller sets them via ``Condition.hallmarks[name] = severity``
        — so their target parameters aren't valid Calibrator inputs.
        Everything else in those same processes (upstream regulators,
        downstream rate constants, parallel parameters not touched by
        any hallmark) remains visible and calibratable.

        Parameters
        ----------
        include_hallmark_targets:
            If True, include hallmark-targeted parameters in the
            returned list. Defaults to False.
        registry:
            Hallmark registry to consult. Defaults to
            :data:`hallsim.hallmarks.HALLMARK_REGISTRY`.

        Returns
        -------
        list of :class:`hallsim.calibration.CalibratableParam`, each
        with ``process_name`` filled in. Pass any entry through
        ``ParameterRef(process_name=..., field=..., init=..., clamp=...)``
        to wire it into a :class:`hallsim.calibration.CalibrationProblem`.
        """
        from hallsim.calibration import CalibratableParam
        from hallsim.hallmarks import HALLMARK_REGISTRY

        reg = HALLMARK_REGISTRY if registry is None else registry

        hallmark_targets: set[tuple[str, str]] = set()
        for handle in reg.values():
            for mapping in handle.mappings:
                hallmark_targets.add(
                    (mapping.process_name, mapping.param_name)
                )

        out: list = []
        for proc_name, proc in self.processes.items():
            for item in proc.calibratable_params():
                if (
                    not include_hallmark_targets
                    and (proc_name, item.field) in hallmark_targets
                ):
                    continue
                out.append(
                    CalibratableParam(
                        process_name=proc_name,
                        field=item.field,
                        default=item.default,
                        clamp=item.clamp,
                        description=item.description,
                    )
                )
        return out
