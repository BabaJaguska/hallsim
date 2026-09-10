"""Coupling ablation — the null a composition claim has to beat.

A composite of independently-published models scores better than any one of
them because it carries more mechanism, not necessarily because the wiring
between them does anything. The comparison that separates the two holds the
constituents, the clock, the parameters and the readout fixed and removes only
the *variation* the edges transmit: every coupling edge keeps emitting the
constant it was placed to deliver, and stops following its source.

    result = Scheduler().run(composite, ...)             # the control arm
    levels = trajectory_levels(composite, result)       # what each edge saw
    null = freeze_coupling(composite, levels)           # same models, no signal

Re-score a fitted parameter vector on ``null`` and the difference is what the
coupling buys. Two things this deliberately does not do. It does not cut the
edge to zero — the target would lose a term it was calibrated with, and the
comparison would confound "the edge carries information" with "the edge
carries a level". And it takes no default level: a source's *declared* value
is a published initial condition, which for most deposits is a fitted
experimental starting point rather than a rest point, so freezing there
measures the composite against an operating point it never occupies.
"""

from __future__ import annotations

import logging

import equinox as eqx
import numpy as np

from hallsim.store import as_paths

log = logging.getLogger(__name__)


def coupling_edges(composite) -> dict:
    """``{process_name: process}`` for every process that can be frozen."""
    return {
        name: proc
        for name, proc in composite.processes.items()
        if hasattr(proc, "frozen_at")
    }


def _source_paths(composite, name, proc) -> dict:
    """``{source_port: store_path}`` for one edge, from the topology."""
    wiring = composite.topology.get(name, {})
    ports = getattr(proc, "sources", None) or (getattr(proc, "source", None),)
    missing = [p for p in ports if p not in wiring]
    if missing:
        raise ValueError(
            f"{name} reads {missing} but the topology wires "
            f"{sorted(wiring)}; cannot resolve what to freeze it at."
        )
    return {p: wiring[p] for p in ports}


def _declared_defaults(composite) -> dict:
    """``{store_path: default}`` for every path a process declares one for."""
    out = {}
    for name, proc in composite.processes.items():
        schema = proc.ports_schema()
        for port, entry in composite.topology.get(name, {}).items():
            port_default = schema[port].default if port in schema else None
            if port_default is None:
                continue
            for path in as_paths(entry):
                out.setdefault(path, float(port_default))
    return out


def declared_levels(composite, edges=None) -> dict:
    """``{edge: {source_port: level}}`` at each source's declared default.

    Only meaningful where the source genuinely rests at its declared value.
    Check that before using it: a published initial condition is usually a
    fitted experimental starting point, and freezing an edge at one holds it
    at a level the composite never occupies. :func:`trajectory_levels` is
    what a measured null uses.
    """
    declared = _declared_defaults(composite)
    out = {}
    for name, proc in (edges or coupling_edges(composite)).items():
        levels = {}
        for port, entry in _source_paths(composite, name, proc).items():
            total = 0.0
            for path in as_paths(entry):
                default = declared.get(path)
                if default is None:
                    raise ValueError(
                        f"{name} reads {path}, which no process declares a "
                        "default for, so it has no declared level. Pass "
                        "levels explicitly."
                    )
                total += float(default)
            levels[port] = total
        out[name] = levels
    return out


def trajectory_levels(composite, result, edges=None) -> dict:
    """``{edge: {source_port: level}}`` at each source's mean over ``result``.

    The stronger null: the edge delivers its own time-average, so the
    comparison isolates the *variation* it transmits rather than its level.
    """
    out = {}
    for name, proc in (edges or coupling_edges(composite)).items():
        out[name] = {
            port: float(
                sum(
                    np.mean(np.asarray(result.get(p))) for p in as_paths(entry)
                )
            )
            for port, entry in _source_paths(composite, name, proc).items()
        }
    return out


def freeze_coupling(composite, levels: dict, edges=None):
    """A copy of ``composite`` with every coupling edge held at ``levels``.

    ``levels`` is ``{edge: {source_port: level}}``, from
    :func:`trajectory_levels` or :func:`declared_levels`. Constituents,
    topology, clock and every non-edge parameter are untouched, so the result
    is the same models on the same axis with the coupling carrying no signal.
    """
    found = edges or coupling_edges(composite)
    if not found:
        raise ValueError(
            "no process in this composite exposes frozen_at(), so there is "
            "no coupling to ablate. Coupling edges are the GainEdge / "
            "HillEdge family; a hand-written edge needs the same method."
        )
    absent = sorted(set(found) - set(levels))
    if absent:
        raise ValueError(
            f"no level given for {absent}; freeze_coupling needs one per "
            f"edge, and this composite couples through {sorted(found)}."
        )
    frozen = dict(composite.processes)
    for name, proc in found.items():
        frozen[name] = proc.frozen_at(levels[name])
    log.info("froze %d coupling edges: %s", len(found), sorted(found))
    return eqx.tree_at(lambda c: c.processes, composite, frozen)
