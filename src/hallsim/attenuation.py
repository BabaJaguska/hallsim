"""Where a perturbation dies on its way to a reporter.

A flat reporter has a cause: some node between the control and the readout
stops passing the signal on, a gate the input never reaches or a term a
hundred times weaker than the basal one beside it. :func:`trace_path`
follows the shortest route from the control to the reporter, through the
composite's wiring and, inside a model that declares its reactions, through
the reaction graph itself, runs the composite at two settings of the
control, and reports the relative change at every store path on that route
in order, so the node where the change collapses, and the reactions that
carry that step, are named rather than inferred.

>>> print(trace_path(comp, "Genomic Instability", "dp14/p21",
...                  registry=HALLMARK_REGISTRY, t_end=14.0))
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import networkx as nx

from hallsim.composite import Composite
from hallsim.process import PortRole, read_param, split_param_address
from hallsim.scheduler import Scheduler

_WRITES = (
    PortRole.EVOLVED,
    PortRole.EXCLUSIVE,
    PortRole.ASSIGNED,
    PortRole.LATCHED,
)


@dataclasses.dataclass(frozen=True)
class PathNode:
    """One store path on the route: the process that writes it there, the
    reactions of that process carrying the previous node into it, and its
    time-mean at the two settings of the control."""

    path: str
    process: str
    low: float
    high: float
    rel_change: float
    terms: tuple[str, ...] = ()


@dataclasses.dataclass
class PathTrace:
    """The route from ``control`` to ``reporter`` and how much of the
    perturbation survives at each node. ``gate`` is the node after the last
    one whose relative change clears ``threshold``: from there on nothing
    moves."""

    control: str
    reporter: str
    settings: tuple[float, float]
    nodes: list[PathNode]
    threshold: float

    @property
    def last_alive(self) -> int:
        alive = [
            i
            for i, n in enumerate(self.nodes)
            if n.rel_change >= self.threshold
        ]
        return alive[-1] if alive else -1

    @property
    def gate(self) -> PathNode | None:
        i = self.last_alive + 1
        return self.nodes[i] if i < len(self.nodes) else None

    @property
    def reaches(self) -> bool:
        return bool(self.nodes) and self.gate is None

    def __str__(self) -> str:
        lo, hi = self.settings
        w = max((len(n.path) for n in self.nodes), default=4)
        lines = [
            f"Path trace {self.control} -> {self.reporter} "
            f"(control at {lo:g} vs {hi:g}, time-mean of each node)",
            f"{'node':<{w}}  {'process':<16}{'at low':>12}{'at high':>12}"
            f"{'rel. change':>13}  terms",
            "-" * (w + 62),
        ]
        for n in self.nodes:
            lines.append(
                f"{n.path:<{w}}  {n.process:<16}{n.low:>12.4g}"
                f"{n.high:>12.4g}{n.rel_change:>13.3g}  {', '.join(n.terms)}"
            )
        if not self.nodes:
            lines.append("No route from the control to the reporter.")
            return "\n".join(lines)
        gate = self.gate
        if gate is None:
            lines.append(
                f"The perturbation reaches {self.reporter} "
                f"(rel. change {self.nodes[-1].rel_change:.3g})."
            )
            return "\n".join(lines)
        before = self.nodes[self.last_alive] if self.last_alive >= 0 else None
        where = f"in {gate.process}"
        if before is not None:
            where += f", on the step {before.path} -> {gate.path}"
        if gate.terms:
            where += f" (terms: {', '.join(gate.terms)})"
        lines.append(
            f"The perturbation dies at {gate.path} {where}: rel. change "
            f"{gate.rel_change:.3g} against {self.threshold:g}."
            + (
                f" It was still {before.rel_change:.3g} at {before.path}."
                if before is not None
                else ""
            )
        )
        return "\n".join(lines)


def _paths_of(composite: Composite, name: str) -> dict[str, tuple[str, ...]]:
    """``{port: store paths}`` for one process; a composite keeps a port's
    target as a tuple."""
    out = {}
    for port, p in composite.topology.get(name, {}).items():
        out[port] = (p,) if isinstance(p, str) else tuple(p)
    return out


def _wiring(composite: Composite) -> nx.DiGraph:
    """Store paths and what carries them, as one directed graph.

    A process that declares its symbolic forms (``reaction_channels``,
    ``assignment_rules``, ``rate_rules``) is entered: each reaction is a
    node from the paths its rate law reads to the paths its stoichiometry
    moves, each rule a node from what it reads to what it sets. Any other
    process is one node from every path it reads to every path it writes.
    Every node records the symbols it reads, so a parameter can be found
    where it enters."""
    g = nx.DiGraph()
    for name, proc in composite.processes.items():
        paths_of = _paths_of(composite, name)
        schema = proc.ports_schema()
        declared = False

        def read_paths(symbol):
            """A port's paths, or one element of a block port named
            ``<port>_<i>``."""
            if symbol in paths_of:
                return paths_of[symbol]
            port, _, index = symbol.rpartition("_")
            if port in paths_of and index.isdigit():
                block = paths_of[port]
                if int(index) < len(block):
                    return (block[int(index)],)
            return ()

        def add(node, reads, writes):
            g.add_node(node, reads=frozenset(reads))
            for r in reads:
                for rp in read_paths(r):
                    g.add_edge(("path", rp), node)
            for w in writes:
                for wp in paths_of.get(w, ()):
                    g.add_edge(node, ("path", wp))

        channels = getattr(proc, "reaction_channels", lambda: None)() or ()
        for ch in channels:
            declared = True
            reads = {str(s) for s in getattr(ch.rate_law, "free_symbols", ())}
            writes = [w for w, c in dict(ch.stoichiometry).items() if c]
            add(("rxn", name, str(ch.reaction_id)), reads, writes)
        for kind in ("assignment_rules", "rate_rules"):
            rules = getattr(proc, kind, lambda: None)() or ()
            for target, expr in rules:
                declared = True
                reads = {str(s) for s in getattr(expr, "free_symbols", ())}
                add(("rule", name, str(target)), reads, [str(target)])
        if not declared:
            reads = [p for p in schema if p in paths_of]
            writes = [
                p
                for p, s in schema.items()
                if s.role in _WRITES and p in paths_of
            ]
            add(("proc", name), reads, writes)
    return g


def _starts(g: nx.DiGraph, name: str, param: str | None) -> list:
    """Nodes a control enters ``name`` at: the reactions and rules reading
    ``param`` where the process declares them, else the process node."""
    own = [n for n in g.nodes if n[0] != "path" and n[1] == name]
    if param is not None:
        hits = [n for n in own if param in g.nodes[n].get("reads", ())]
        if hits:
            return hits
    return own


def _route(g: nx.DiGraph, starts: list, reporter: str) -> list[tuple]:
    """The shortest route from any start node to the reporter path, as
    ``[(path, process, terms), ...]`` with the reactions of each step."""
    target = ("path", reporter)
    best = None
    for node in starts:
        if node not in g:
            continue
        try:
            route = nx.shortest_path(g, node, target)
        except nx.NetworkXNoPath:
            continue
        if best is None or len(route) < len(best):
            best = route
    if best is None:
        return []
    out = []
    prev_path = None
    for i in range(1, len(best)):
        kind, *rest = best[i]
        if kind != "path":
            continue
        path = rest[0]
        carrier = best[i - 1]
        process = carrier[1]
        terms = tuple(
            sorted(
                n[2]
                for n in g.predecessors(best[i])
                if n[0] in ("rxn", "rule")
                and n[1] == process
                and (
                    prev_path is None
                    or ("path", prev_path) in g.predecessors(n)
                )
            )
        )
        out.append((path, process, terms))
        prev_path = path
    return out


def trace_path(
    composite: Composite,
    control: str,
    reporter: str,
    *,
    registry: dict | None = None,
    t_end: float,
    macro_dt: float | None = None,
    settings: tuple[float, float] | None = None,
    threshold: float = 1e-3,
    y0=None,
) -> PathTrace:
    """Follow ``control`` to ``reporter`` and report how much of the
    perturbation survives at each store path on the way.

    ``control`` is a handle name in ``registry``, run at severities
    ``settings`` (default 0 and 1), or a parameter address
    ``"<process>.<field>"``, run at its value times ``settings`` (default 1
    and 2). ``reporter`` is a store path. Each node's value is its time-mean
    over ``[0, t_end]`` on a 2000-point grid; ``rel_change`` is the change
    between the two runs relative to the low one. The gate is the node after
    the last one that clears ``threshold``."""
    from hallsim.handles import apply_handles

    g = _wiring(composite)
    if ("path", reporter) not in g:
        raise KeyError(
            f"{reporter!r} is not a store path of this composite; "
            f"paths: {sorted(composite.store_keys())}"
        )
    if registry is not None and control in registry:
        lo, hi = settings or (0.0, 1.0)
        starts = []
        for m in registry[control].mappings:
            if m.process_name in composite.processes:
                starts += _starts(
                    g, m.process_name, m.param_name.split(".")[-1]
                )

        def variant(value):
            return Composite(
                apply_handles(composite.processes, {control: value}, registry),
                composite.topology,
                validate=False,
                semantic_validation=False,
            )

    else:
        proc_name, field = split_param_address(control, composite.processes)
        lo, hi = settings or (1.0, 2.0)
        starts = _starts(g, proc_name, field.split(".")[-1])

        def variant(factor):
            proc = composite.processes[proc_name]
            base = proc.with_param(field, read_param(proc, field) * factor)
            return Composite(
                {**composite.processes, proc_name: base},
                composite.topology,
                validate=False,
                semantic_validation=False,
            )

    route = _route(g, starts, reporter)
    mdt = macro_dt if macro_dt is not None else t_end / 4.0
    sched = Scheduler()
    means = []
    for value in (lo, hi):
        comp = variant(value)
        res = sched.run(
            comp,
            t_span=(0.0, t_end),
            macro_dt=mdt,
            save_dt=t_end / 2000.0,
            y0=y0 if y0 is not None else comp.initial_state_vec(),
        )
        means.append(
            {
                path: float(jnp.mean(jnp.asarray(res.get(path))))
                for path, _, _ in route
            }
        )
    nodes = []
    for path, proc_name, terms in route:
        low, high = means[0][path], means[1][path]
        nodes.append(
            PathNode(
                path=path,
                process=proc_name,
                low=low,
                high=high,
                rel_change=float(abs(high - low) / (abs(low) + 1e-12)),
                terms=terms,
            )
        )
    return PathTrace(
        control=control,
        reporter=reporter,
        settings=(float(lo), float(hi)),
        nodes=nodes,
        threshold=threshold,
    )
