"""Composite — wires Processes together via a Topology into a single ODE RHS.

An Equinox module bundling named Processes with a topology mapping their port
names to store paths. It builds a flat ``f(t, y_vec) -> dy_vec`` that calls
each process's ``derivative()`` with a small per-process port view and
scatter-adds the contributions into one accumulator — so EVOLVED ports sum
implicitly, while EXCLUSIVE ports are enforced at composition time by topology
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
import re
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import PortRole, Process, ProcessKind
from hallsim.store import build_initial_store, validate_topology
from hallsim.units import canonical_units, conversion_factor

log = logging.getLogger(__name__)

_DIGIT_RUN = re.compile(r"(\d+)")


def _natural_key(path: str) -> tuple[Any, ...]:
    """Sort key where digit runs compare numerically, so ``node2`` sorts
    before ``node10``.

    ``re.split`` on a capturing group alternates text/digits from index 0,
    so element types line up positionally and tuples stay comparable.
    """
    return tuple(
        int(part) if part.isdigit() else part
        for part in _DIGIT_RUN.split(path)
    )


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


def _order_assignments(assign_procs: list) -> list:
    """Topologically order algebraic-assignment processes so any that reads a
    path another *assigns* runs after it.

    ``assign_procs`` is a list of ``(proc, read_pairs, assign_pairs)`` where
    each ``*_pairs`` element is ``(port, store_index, factor)``. Raises on an
    algebraic cycle (an ASSIGNED path that depends on itself through others).
    """
    writer_of = {
        idx: i
        for i, (_, _, assigns) in enumerate(assign_procs)
        for _, idx, _ in assigns
    }
    deps = {
        i: {
            writer_of[idx]
            for _, idx, _ in reads
            if idx in writer_of and writer_of[idx] != i
        }
        for i, (_, reads, _) in enumerate(assign_procs)
    }
    order, done = [], set()
    while len(done) < len(assign_procs):
        progress = [
            i
            for i in range(len(assign_procs))
            if i not in done and deps[i] <= done
        ]
        if not progress:
            raise ValueError(
                "Algebraic cycle among ASSIGNED store paths: an assignment "
                "depends on its own output through others."
            )
        for i in progress:
            order.append(assign_procs[i])
            done.add(i)
    return order


def _port_view(y_vec, read_pairs):
    """The ``{port_name: value}`` dict a Process's ``derivative``/``assign``
    consumes, gathered from the flat state and converted to port units."""
    return {port: y_vec[..., idx] * rf for port, idx, rf in read_pairs}


def _apply_assignments(assign_pre, t, y_vec):
    """Inject each ASSIGNED path's algebraic value into ``y_vec`` (returns a
    new array; JAX-functional). Shared by the RHS and materialization."""
    for proc, read_pairs, assign_pairs in assign_pre:
        raw = proc.assign(t, _port_view(y_vec, read_pairs))
        for port, idx, wf in assign_pairs:
            if port in raw:
                y_vec = y_vec.at[..., idx].set(raw[port] * wf)
    return y_vec


class _FlatRHS(eqx.Module):
    """Flat-state RHS as a pytree rather than a closure.

    A closure inside ``ODETerm`` is a static leaf, so a fresh one per
    :meth:`Composite.build_rhs` hashes differently and every solve misses the
    JIT cache. As a Module, repeated builds share a treedef.
    """

    procs: tuple
    assign_procs: tuple
    read_maps: tuple = eqx.field(static=True, default=())
    write_maps: tuple = eqx.field(static=True, default=())
    assign_read_maps: tuple = eqx.field(static=True, default=())
    assign_write_maps: tuple = eqx.field(static=True, default=())

    def __call__(self, t, y_vec, args=None):
        # Assignments run first so the derivative pass reads fresh algebraic
        # values; they are not integrated, so a saved trajectory needs
        # Composite.materialize_assigned to see them.
        y_vec = _apply_assignments(
            zip(
                self.assign_procs,
                self.assign_read_maps,
                self.assign_write_maps,
            ),
            t,
            y_vec,
        )
        accum = jnp.zeros_like(y_vec)
        for proc, read_pairs, write_pairs in zip(
            self.procs, self.read_maps, self.write_maps
        ):
            raw = proc.derivative(t, _port_view(y_vec, read_pairs))
            out = [
                (idx, raw[port] * wf)
                for port, idx, wf in write_pairs
                if port in raw
            ]
            if out:
                idxs = jnp.array([i for i, _ in out])
                vals = jnp.stack([v for _, v in out], axis=-1)
                accum = accum.at[..., idxs].add(vals)
        return accum


class Composite(eqx.Module):
    """A wired bundle of Processes sharing a flat state store.

    Parameters
    ----------
    processes:
        ``{name: Process | Composite}``. A nested Composite is flattened in
        place — sub-processes renamed ``<outer_key>.<sub_name>``, store paths
        prefixed ``<outer_key>/`` — which is how published composites merge.
    topology:
        ``{name: {port_name: store_path}}`` for top-level Processes;
        sub-composites bring their own. Two processes writing one path
        contribute additively (EVOLVED) or exclusively (EXCLUSIVE).
    rewire:
        ``{old_path: new_path}`` applied after flattening, to declare that two
        merged models' states are the same quantity — e.g.
        ``{"dp14/mTORC1_pS2448": "eriq/mTOR_activity"}``.
    """

    processes: dict[str, Process]
    topology: dict[str, dict[str, str]]
    initial: dict[str, float] = eqx.field(static=True, default_factory=dict)

    def __init__(
        self,
        processes: dict[str, Process | Composite],
        topology: dict[str, dict[str, str]] | None = None,
        *,
        rewire: dict[str, str] | None = None,
        initial: dict[str, float] | None = None,
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
        self.initial = dict(initial or {})
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
        """All store paths, in flat-state order. Natural-sorted, so
        ``net/node2`` precedes ``net/node10``."""
        return sorted(self.store_paths(), key=_natural_key)

    def store_index(self) -> dict[str, int]:
        """Store path → its column in the flat state vector. Align any
        externally-built node-indexed array through this, not by re-deriving
        the order."""
        return {k: i for i, k in enumerate(self.store_keys())}

    def flatten(
        self,
        state: dict[str, jnp.ndarray],
        keys: list[str] | None = None,
    ) -> jnp.ndarray:
        """State dict → ``(..., n_vars)`` array in sorted key order.

        Stacking on the *last* axis means scalars give ``(n_vars,)`` and
        ``(batch,)`` values give ``(batch, n_vars)`` with no code change —
        which is why ``Scheduler.run`` takes batched y0 without a vmap path.
        ``keys`` defaults to :meth:`store_keys`.
        """
        if keys is None:
            keys = self.store_keys()
        return jnp.stack([jnp.asarray(state[k]) for k in keys], axis=-1)

    def initial_state_vec(self, keys: list[str] | None = None) -> jnp.ndarray:
        """Initial state as a flat ``(n_vars,)`` tensor — the default y0 for
        ``Scheduler.run``, whose public API takes a tensor, not a dict.

        Override a value with ``y0.at[keys.index("path")].set(v)``; batch a
        population with ``jnp.broadcast_to(y0, (batch, n_vars))``.
        """
        return self.flatten(self.initial_state(), keys)

    def unflatten(
        self,
        vec: jnp.ndarray,
        keys: list[str] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Inverse of :meth:`flatten`: ``(..., n_vars)`` → state dict, so
        ``(n_vars,)`` gives scalars and ``(batch, n_vars)`` gives ``(batch,)``
        arrays. ``keys`` must match the order used in ``flatten``.
        """
        if keys is None:
            keys = self.store_keys()
        return {k: vec[..., i] for i, k in enumerate(keys)}

    # -----------------------------------------------------------------
    # Build the combined ODE right-hand side
    # -----------------------------------------------------------------

    def _assignment_pre(self, proc_names, keys, key_to_idx, canon):
        """Dependency-ordered ``(proc, read_pairs, assign_pairs)`` for the
        ASSIGNED (algebraic) ports of ``proc_names`` — the assignment-rule pass
        shared by :meth:`build_rhs` and :meth:`materialize_assigned`."""
        assign_procs = []
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
            assign_pairs = tuple(
                (
                    port,
                    key_to_idx[proc_topo[port]],
                    conversion_factor(p.units, canon.get(proc_topo[port], "")),
                )
                for port, p in schema.items()
                if p.role == PortRole.ASSIGNED
            )
            if assign_pairs:
                assign_procs.append((proc, read_pairs, assign_pairs))
        # Dependency-order so an assignment reading a path another assigns runs
        # after it (SBML/DAE assignment-rule semantics).
        return _order_assignments(assign_procs)

    _apply_assignments = staticmethod(_apply_assignments)

    def materialize_assigned(self, ts, ys, proc_names=None):
        """Overwrite the ASSIGNED (algebraic) columns of a saved trajectory
        ``ys`` with their true values, recomputed from each saved state.

        Algebraic paths are injected only transiently inside the RHS (they are
        not integrated), so a raw solve buffer would hold their stale *initial*
        value. :meth:`hallsim.scheduler.Scheduler.run` applies this before
        returning, so every SchedulerResult is already self-consistent.
        ``ys`` follows the :class:`hallsim.scheduler.SchedulerResult` layout
        ``(n_time, ..., n_vars)`` (time on axis 0, optional batch dims in the
        middle, vars trailing) and is materialized batch-aware. Returns the same
        shape. No-op when the composite has no ASSIGNED paths."""
        import jax

        if proc_names is None:
            proc_names = list(self.continuous_processes().keys())
        keys = self.store_keys()
        key_to_idx = {k: i for i, k in enumerate(keys)}
        canon = canonical_units(self.processes, self.topology)
        assign_pre = self._assignment_pre(proc_names, keys, key_to_idx, canon)
        if not assign_pre:
            return ys
        # vmap over TIME (axis 0, per SchedulerResult); _apply_assignments
        # indexes vars on the trailing axis so any batch dims ride through.
        return jax.vmap(
            lambda t, y: self._apply_assignments(assign_pre, t, y),
            in_axes=(0, 0),
            out_axes=0,
        )(jnp.asarray(ts), ys)

    def build_rhs(self, proc_names: list[str] | None = None):
        """Flat ``f(t, y_vec, args=None) -> dy_vec`` over a 1-D array indexed
        by :meth:`store_keys`, plus that key list.

        Index maps are precomputed here, so each process contributes via one
        batched ``.at[idxs].add(vals)`` scatter. ``proc_names`` selects a
        subset for operator splitting (the Scheduler does this per group);
        unnamed processes contribute zero, and ``keys`` is always the full
        layout, so one flat state vector serves every group's solve.
        """
        if proc_names is None:
            proc_names = list(self.continuous_processes().keys())

        keys = self.store_keys()
        key_to_idx = {k: i for i, k in enumerate(keys)}

        # Canonical unit per store path, taken over the whole composite so it
        # agrees across splitting groups and with the seeded initial state.
        canon = canonical_units(self.processes, self.topology)

        # Reads carry a canonical→port factor, writes port→canonical; 1.0
        # unless the units differ, so writers with compatible-but-different
        # units still sum correctly.
        pre = []  # derivative contributors: (proc, read_pairs, write_pairs)
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
            if write_pairs:
                pre.append((proc, read_pairs, write_pairs))
        assign_pre = self._assignment_pre(proc_names, keys, key_to_idx, canon)

        return (
            _FlatRHS(
                procs=tuple(p for p, _, _ in pre),
                read_maps=tuple(r for _, r, _ in pre),
                write_maps=tuple(w for _, _, w in pre),
                assign_procs=tuple(p for p, _, _ in assign_pre),
                assign_read_maps=tuple(r for _, r, _ in assign_pre),
                assign_write_maps=tuple(a for _, _, a in assign_pre),
            ),
            keys,
        )

    def evolved_indices(
        self,
        proc_names: list[str] | None = None,
        keys: list[str] | None = None,
    ) -> jnp.ndarray:
        """Sorted int32 trailing-axis indices these processes write derivatives
        to — the union over their EVOLVED / EXCLUSIVE ports, i.e. the states
        that actually evolve under ``build_rhs(proc_names)``.

        Used to restrict a group's Jacobian to its own dynamics and to scope
        coupling splices. Defaults: every CONTINUOUS process, :meth:`store_keys`.
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

    def unfed_input_indices(
        self,
        keys: list[str] | None = None,
    ) -> jnp.ndarray:
        """Trailing-axis indices of INPUT paths that no process writes.

        These components hold their port default for the whole run — the
        state a driven component (coupling edge, clamp, param-input) sees
        when nothing drives it. Screening and validation use this to tell
        "undriven here" apart from "has no dynamics".

        Returns
        -------
        ``jnp.ndarray`` of sorted int32 indices.
        """
        if keys is None:
            keys = self.store_keys()
        key_to_idx = {k: i for i, k in enumerate(keys)}
        read: set[str] = set()
        written: set[str] = set()
        for pname, proc in self.processes.items():
            proc_topo = self.topology[pname]
            for port, p in proc.ports_schema().items():
                path = proc_topo[port]
                (read if p.role == PortRole.INPUT else written).add(path)
        return jnp.array(
            sorted(key_to_idx[p] for p in read - written), dtype=jnp.int32
        )

    # -----------------------------------------------------------------
    # Initial state
    # -----------------------------------------------------------------

    def with_params(self, overrides: dict[str, Any]) -> "Composite":
        """A copy with parameters changed, keyed ``"<process>.<field>"``::

            comp = comp.with_params({"mtor_nfkb.k_act": 0.0})     # ablate
            comp = comp.with_params({"dp14.parameters.kdeg": 2.0})

        The route for an ablation or a sweep outside calibration — a
        bifurcation scan, a robustness probe, a bare :meth:`Scheduler.run`.
        Editing the pytree by hand instead is what
        :meth:`hallsim.calibration.CalibrationProblem.with_overrides` rejects
        for fitted fields, because the next substitution overwrites it and the
        ablation silently does nothing.

        Topology, rewiring and validation settings are untouched; only the
        named fields move.
        """
        import equinox as eqx

        from hallsim.process import write_param

        procs = dict(self.processes)
        for address, value in overrides.items():
            name, _, field = address.partition(".")
            if not field:
                raise ValueError(
                    f"Override key {address!r} must be "
                    f"'<process>.<field>', e.g. 'mtor_nfkb.k_act'."
                )
            if name not in procs:
                raise KeyError(
                    f"No process {name!r} in this composite; "
                    f"available: {sorted(procs)}"
                )
            procs[name] = write_param(procs[name], field, value)
        return eqx.tree_at(lambda c: c.processes, self, procs)

    def initial_state(self) -> dict[str, jnp.ndarray]:
        """All process port defaults merged into one
        ``{store_path: jnp.ndarray}`` store."""
        return build_initial_store(self.processes, self.topology, self.initial)

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

        return self._order_by_coupling(groups)

    def _order_by_coupling(
        self, groups: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Reorder groups so a group runs after the groups that drive it.

        Timescale decides which processes share a solver; the cross-group
        edges decide execution order. A driven group placed first reads its
        driver's previous-step value, and interpolated coupling cannot apply
        to an edge whose source has not run yet. Cycles keep timescale order.
        """
        if len(groups) < 2:
            return groups
        names = list(groups)
        writes, reads = {}, {}
        for gname, procs in groups.items():
            w, r = set(), set()
            for pname in procs:
                topo_p = self.topology.get(pname, {})
                schema = self.processes[pname].ports_schema()
                for port, path in topo_p.items():
                    r.add(path)
                    port_spec = schema.get(port)
                    if port_spec is not None and port_spec.role in (
                        PortRole.EVOLVED,
                        PortRole.EXCLUSIVE,
                        PortRole.ASSIGNED,
                    ):
                        w.add(path)
            writes[gname], reads[gname] = w, r

        drivers = {
            g: {
                other
                for other in names
                if other != g and writes[other] & reads[g]
            }
            for g in names
        }
        ordered, placed = [], set()
        while len(ordered) < len(names):
            ready = [
                g for g in names if g not in placed and drivers[g] <= placed
            ]
            if not ready:  # cycle: keep the remaining timescale order
                ready = [g for g in names if g not in placed]
            for g in ready:
                ordered.append(g)
                placed.add(g)
        return {g: groups[g] for g in ordered}

    def calibration_targets(
        self,
        *,
        include_hallmark_targets: bool = False,
        registry: dict | None = None,
    ) -> list:
        """Every process's ``calibratable_params()`` with the namespace filled
        in, minus the ``(process, field)`` pairs any hallmark mapping targets.

        Hallmarks are experimenter knobs set through
        ``Condition.hallmarks[name] = severity``, so their targets aren't valid
        Calibrator inputs; everything else in those processes stays
        calibratable. Set ``include_hallmark_targets`` to keep them.

        Returns :class:`~hallsim.calibration.CalibratableParam` entries — pass
        one through ``ParameterRef(...)`` to wire it into a
        :class:`~hallsim.calibration.CalibrationProblem`.
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


def single_process_composite(process, name: str | None = None) -> Composite:
    """Wrap one Process as a runnable Composite with identity topology.

    Each port maps to its own name (no cross-model wiring), and validation
    is off — a lone process has no cross-model wiring to check, and its
    INPUT ports are unfed by construction, so the graph analyser's
    unfed-input warning is noise here.  The standard way to run or screen a
    single Process on its own; ``name`` defaults to the process's own
    ``_name`` (imported models) or class name.
    """
    name = name or getattr(process, "_name", None) or type(process).__name__
    return Composite(
        processes={name: process},
        topology={},
        validate=False,
        semantic_validation=False,
    )
