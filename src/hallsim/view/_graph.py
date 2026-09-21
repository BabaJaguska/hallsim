"""The wiring tab: what is connected to what, and where a signal dies."""

from __future__ import annotations

import logging
import threading

import networkx as nx

from hallsim.attenuation import _wiring, trace_path
from hallsim.view._page import Page
from hallsim.view._theme import TOKENS

log = logging.getLogger(__name__)

TABS_LAYOUT = {
    "name": "klay",
    "klay": {
        "direction": "RIGHT",
        "spacing": 28,
        "edgeRouting": "ORTHOGONAL",
        "nodePlacement": "LINEAR_SEGMENTS",
    },
    "animate": False,
    "fit": True,
    "padding": 24,
}


def wiring(composite) -> nx.DiGraph:
    """Store paths and what carries them, as one directed graph."""
    return _wiring(composite)


def owners(g: nx.DiGraph) -> dict[str, str | None]:
    """The process whose reaction, rule or derivative writes each path."""
    out: dict[str, str | None] = {}
    for node in g.nodes:
        if node[0] != "path":
            continue
        writers = sorted(u[1] for u in g.predecessors(node))
        out[node[1]] = writers[0] if writers else None
    return out


def _representative(node, owner, expanded, shown=None):
    """The element standing for ``node``: itself inside an opened process,
    else the process. ``shown`` limits an opened process to those nodes,
    the rest folding back into it."""
    kind = node[0]
    if kind == "path":
        p = node[1]
        proc = owner.get(p)
        if proc is not None and proc not in expanded:
            return f"proc:{proc}"
        if shown is not None and proc is not None and node not in shown:
            return f"proc:{proc}"
        return f"path:{p}"
    proc = node[1]
    if proc not in expanded:
        return f"proc:{proc}"
    if shown is not None and node not in shown:
        return f"proc:{proc}"
    if kind == "proc":
        return f"deriv:{proc}"
    return ":".join(node)


def _plain(v):
    """A JSON-safe rendering of a detail value."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "name") and hasattr(v, "value"):
        return str(v.name).lower()
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _port_info(composite) -> dict[str, dict]:
    """Per path: units, description, writers, readers."""
    info: dict[str, dict] = {}
    for name, proc in composite.processes.items():
        schema = proc.ports_schema()
        topo = composite.topology.get(name, {})
        for port, target in topo.items():
            spec = schema.get(port)
            if spec is None:
                continue
            for p in (target,) if isinstance(target, str) else target:
                d = info.setdefault(
                    p,
                    {
                        "units": "",
                        "description": "",
                        "writers": [],
                        "readers": [],
                    },
                )
                units = getattr(spec, "units", None)
                if units and not d["units"]:
                    d["units"] = str(units)
                desc = getattr(spec, "description", None)
                if desc and not d["description"]:
                    d["description"] = str(desc)
                role = spec.role.name
                side = "readers" if role == "INPUT" else "writers"
                d[side].append(f"{name}.{port}")
                if role == "EVOLVED":
                    d["readers"].append(f"{name}.{port}")
    return info


def _reactions(composite) -> dict[tuple[str, str], dict]:
    out = {}
    for name, proc in composite.processes.items():
        channels = getattr(proc, "reaction_channels", lambda: None)() or ()
        for ch in channels:
            stoich = ", ".join(
                f"{c:+g} {s}" for s, c in dict(ch.stoichiometry).items() if c
            )
            out[(name, str(ch.reaction_id))] = {
                "rate_law": str(ch.rate_law),
                "stoichiometry": stoich,
            }
        for kind in ("assignment_rules", "rate_rules"):
            rules = getattr(proc, kind, lambda: None)() or ()
            for target, expr in rules:
                out[(name, str(target))] = {
                    "rate_law": str(expr),
                    "stoichiometry": kind.replace("_", " ")[:-1],
                }
    return out


def elements(
    composite,
    page: Page,
    expanded=None,
    trace=None,
    reactions=False,
    observables=False,
) -> list[dict]:
    """Cytoscape elements: one node per process, opened ones showing the
    states they write with their reactions folded into the edges between
    states, or drawn as their own nodes when ``reactions`` is set. Edges
    between processes are labelled with the paths they carry. With
    ``trace``, the route's nodes carry their relative change and everything
    else is dimmed; ``expanded=None`` opens the route's processes onto the
    route alone. Frozen sinks are never drawn, and states set only by rules
    and read by nothing (an import's observables) only when ``observables``
    is set; a state on the route always is."""
    g = wiring(composite)
    owner = owners(g)
    info = _port_info(composite)
    rxn = _reactions(composite)
    route = {n.path: n for n in (trace.nodes if trace else ())}
    route_terms = {
        (n.process, t) for n in (trace.nodes if trace else ()) for t in n.terms
    }
    shown = None
    if expanded is None:
        expanded = {n.process for n in trace.nodes} if trace else set()
        if trace:
            shown = {("path", p) for p in route} | {
                n
                for n in g.nodes
                if n[0] != "path"
                and (n[1], n[2] if len(n) > 2 else "") in route_terms
            }
    expanded = set(expanded)
    quiet = set(composite.frozen_paths())
    if not observables:
        for node in g.nodes:
            if node[0] != "path":
                continue
            writers = list(g.predecessors(node))
            readers = list(g.successors(node))
            if (
                writers
                and not readers
                and all(w[0] == "rule" for w in writers)
            ):
                quiet.add(node[1])
    hidden = {("path", q) for q in quiet if q not in route}

    def ident_of(node):
        return node[2] if len(node) > 2 else "derivative"

    def folded(node):
        """A carrier drawn as edges between the states it joins."""
        return (
            node[0] != "path"
            and node[1] in expanded
            and shown is None
            and not reactions
        )

    out = []
    for name, proc in composite.processes.items():
        classes = ["process"]
        if name in expanded:
            classes.append("expanded")
        if trace and name not in {n.process for n in trace.nodes}:
            classes.append("dim")
        n_states = len([p for p, o in owner.items() if o == name])
        out.append(
            {
                "data": {
                    "id": f"proc:{name}",
                    "label": page.titles.get(name, name),
                    "kind": "process",
                    "process": name,
                    "detail": {
                        "process": name,
                        "kind": _plain(getattr(proc, "kind", "")),
                        "deposit": page.deposits.get(name, ""),
                        "states written": n_states,
                        "timescale": _plain(getattr(proc, "timescale", "")),
                    },
                },
                "classes": " ".join(classes),
            }
        )
    for node in g.nodes:
        if folded(node) or node in hidden:
            continue
        rep = _representative(node, owner, expanded, shown)
        if rep.startswith("proc:"):
            continue
        kind = node[0]
        if kind == "path":
            p = node[1]
            d = info.get(p, {})
            on_route = p in route
            data = {
                "id": rep,
                "label": p.split("/", 1)[-1],
                "kind": "path",
                "path": p,
                "detail": {
                    "path": p,
                    "units": d.get("units", ""),
                    "description": d.get("description", ""),
                    "written by": ", ".join(d.get("writers", ())),
                    "read by": ", ".join(d.get("readers", ())),
                },
            }
            if owner.get(p) in expanded:
                data["parent"] = f"proc:{owner[p]}"
            classes = ["path"]
            if on_route:
                n = route[p]
                data["rel"] = min(max(n.rel_change, 0.0), 1.0)
                data["detail"].update(
                    {
                        "at low": f"{n.low:.4g}",
                        "at high": f"{n.high:.4g}",
                        "rel. change": f"{n.rel_change:.3g}",
                        "terms": ", ".join(n.terms),
                    }
                )
                classes.append("route")
            elif trace:
                classes.append("dim")
            out.append({"data": data, "classes": " ".join(classes)})
        else:
            proc, ident = node[1], ident_of(node)
            d = rxn.get((proc, ident), {})
            classes = [kind]
            if (proc, ident) in route_terms:
                classes.append("route")
            elif trace:
                classes.append("dim")
            out.append(
                {
                    "data": {
                        "id": rep,
                        "label": ident,
                        "kind": kind,
                        "parent": f"proc:{proc}",
                        "detail": {
                            "process": proc,
                            "term": ident,
                            "rate law": d.get("rate_law", ""),
                            "moves": d.get("stoichiometry", ""),
                        },
                    },
                    "classes": " ".join(classes),
                }
            )
    parent_of = {
        e["data"]["id"]: e["data"].get("parent")
        for e in out
        if "id" in e["data"]
    }

    def joinable(ru, rv):
        if ru == rv:
            return False
        return not (parent_of.get(ru) == rv or parent_of.get(rv) == ru)

    edges: dict[tuple[str, str], dict] = {}

    def join(ru, rv, path=None, term=None):
        if not joinable(ru, rv):
            return
        e = edges.setdefault((ru, rv), {"paths": set(), "terms": set()})
        if path:
            e["paths"].add(path)
        if term:
            e["terms"].add(term)

    for u, v in g.edges:
        if folded(u) or folded(v) or u in hidden or v in hidden:
            continue
        ru, rv = (
            _representative(u, owner, expanded, shown),
            _representative(v, owner, expanded, shown),
        )
        join(ru, rv, path=u[1] if u[0] == "path" else v[1])
    for node in g.nodes:
        if not folded(node):
            continue
        reads = [
            u
            for u in g.predecessors(node)
            if u[0] == "path" and u not in hidden
        ]
        writes = [
            v for v in g.successors(node) if v[0] == "path" and v not in hidden
        ]
        for u in reads:
            for v in writes:
                join(
                    _representative(u, owner, expanded, shown),
                    _representative(v, owner, expanded, shown),
                    term=ident_of(node),
                )
    ids = {e["data"]["id"] for e in out}
    route_ids = {
        e["data"]["id"] for e in out if "route" in e["classes"].split()
    } | {f"proc:{n.process}" for n in (trace.nodes if trace else ())}
    for (ru, rv), carried in edges.items():
        if ru not in ids or rv not in ids:
            continue
        names = sorted(carried["paths"]) or sorted(carried["terms"])
        what = "paths" if carried["paths"] else "reactions"
        label = ", ".join(names) if len(names) <= 3 else f"{len(names)} {what}"
        classes = []
        if trace:
            classes.append(
                "route" if ru in route_ids and rv in route_ids else "dim"
            )
        if parent_of.get(ru) is not None or parent_of.get(rv) is not None:
            classes.append("inner")
        out.append(
            {
                "data": {
                    "source": ru,
                    "target": rv,
                    "label": label,
                    "kind": "edge",
                    "detail": {
                        "from": ru.split(":", 1)[-1],
                        "to": rv.split(":", 1)[-1],
                        "carries": ", ".join(sorted(carried["paths"])),
                        "via": ", ".join(sorted(carried["terms"])),
                    },
                },
                "classes": " ".join(classes),
            }
        )
    return out


def place(els: list[dict], positions: dict | None, spacing: float = 60.0):
    """Give every node a position: its last one where it had one, a compact
    grid at its parent's last position for a node just revealed inside an
    opened process (the browser lays those out by their edges and moves
    whatever the finished box overlaps), else the centre of its placed
    neighbours. Returns the elements with ``position`` set, for a preset
    layout."""
    positions = dict(positions or {})
    if not positions:
        return els
    nodes = [e for e in els if "id" in e["data"]]
    edges = [e for e in els if "source" in e["data"]]
    by_parent: dict[str, list[dict]] = {}
    for n in nodes:
        nid = n["data"]["id"]
        if nid in positions:
            n["position"] = dict(positions[nid])
        elif n["data"].get("parent"):
            by_parent.setdefault(n["data"]["parent"], []).append(n)
    for parent, children in by_parent.items():
        centre = positions.get(parent)
        if centre is None:
            placed = [
                positions[c["data"]["id"]]
                for c in nodes
                if c["data"].get("parent") == parent
                and c["data"]["id"] in positions
            ]
            if placed:
                centre = {
                    "x": sum(q["x"] for q in placed) / len(placed),
                    "y": sum(q["y"] for q in placed) / len(placed),
                }
        centre = centre or {"x": 0.0, "y": 0.0}
        cols = max(1, int(len(children) ** 0.5 + 0.999))
        rows = -(-len(children) // cols)
        for i, c in enumerate(children):
            c["position"] = {
                "x": centre["x"] + (i % cols - (cols - 1) / 2) * spacing,
                "y": centre["y"] + (i // cols - (rows - 1) / 2) * spacing,
            }
            positions[c["data"]["id"]] = c["position"]
            c["classes"] = (c.get("classes", "") + " placed").strip()
    for n in nodes:
        if "position" in n:
            continue
        nid = n["data"]["id"]
        near = [
            positions[o]
            for e in edges
            for o in (e["data"]["source"], e["data"]["target"])
            if nid in (e["data"]["source"], e["data"]["target"])
            and o != nid
            and o in positions
        ]
        n["position"] = (
            {
                "x": sum(q["x"] for q in near) / len(near),
                "y": sum(q["y"] for q in near) / len(near),
            }
            if near
            else {"x": 0.0, "y": 0.0}
        )
        positions[nid] = n["position"]
    return els


PRESET_LAYOUT = {"name": "preset", "animate": False, "fit": False}


def stylesheet(page: Page) -> list[dict]:
    accent = TOKENS["color-accent"]
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-family": TOKENS["font-mono"],
                "font-size": 9,
                "color": TOKENS["color-text"],
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "#fff",
                "border-width": 1,
                "border-color": TOKENS["color-neutral-400"],
            },
        },
        {
            "selector": ".process",
            "style": {
                "shape": "round-rectangle",
                "width": "label",
                "height": "label",
                "padding": "10px",
                "font-family": TOKENS["font-heading"],
                "font-size": 12,
                "font-weight": 600,
                "text-transform": "uppercase",
                "border-color": TOKENS["color-text"],
                "background-color": "#f2f2f3",
            },
        },
        {
            "selector": ".process.expanded",
            "style": {
                "text-valign": "top",
                "text-halign": "center",
                "padding": "14px",
                "background-color": "#f7f7f8",
                "background-opacity": 0.6,
            },
        },
        {
            "selector": ".path",
            "style": {
                "shape": "ellipse",
                "width": "label",
                "height": "label",
                "padding": "6px",
                "background-color": "#fff",
            },
        },
        {
            "selector": ".rxn, .rule, .proc",
            "style": {
                "shape": "diamond",
                "width": 14,
                "height": 14,
                "font-size": 7,
                "text-valign": "bottom",
                "text-margin-y": 3,
                "color": TOKENS["color-neutral-700"],
                "background-color": TOKENS["color-neutral-400"],
                "border-width": 0,
            },
        },
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "font-family": TOKENS["font-mono"],
                "font-size": 8,
                "color": TOKENS["color-neutral-700"],
                "text-rotation": "autorotate",
                "text-margin-y": -6,
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "arrow-scale": 0.8,
                "line-color": TOKENS["color-neutral-400"],
                "target-arrow-color": TOKENS["color-neutral-400"],
                "width": 1.2,
            },
        },
        {
            "selector": ".route",
            "style": {
                "border-color": accent,
                "border-width": 2.5,
                "line-color": accent,
                "target-arrow-color": accent,
                "width": 2.2,
                "color": TOKENS["color-text"],
            },
        },
        {
            "selector": "node.path.route",
            "style": {
                "background-color": f"mapData(rel, 0, 1, #ffffff, {accent})"
            },
        },
        {
            "selector": ".rxn.route, .rule.route, .proc.route",
            "style": {"background-color": accent, "width": 18, "height": 18},
        },
        {"selector": ".dim", "style": {"opacity": 0.22}},
        {
            "selector": "node.hl",
            "style": {"border-color": accent, "border-width": 2.5},
        },
        {
            "selector": "edge.hl",
            "style": {
                "line-color": accent,
                "target-arrow-color": accent,
                "text-opacity": 1,
                "width": 2.2,
            },
        },
        {
            "selector": "edge.dim, edge.route, edge.inner",
            "style": {"text-opacity": 0},
        },
        {
            "selector": "node.path.route",
            "style": {"font-size": 10, "font-weight": 600, "padding": "9px"},
        },
        {
            "selector": ":selected",
            "style": {
                "border-color": TOKENS["color-pulse"],
                "border-width": 3,
            },
        },
    ]


class TraceRunner:
    """Traces requested by the page, taken on host threads and kept."""

    def __init__(self, page: Page):
        self.page = page
        self._done: dict[tuple, object] = {}
        self._errors: dict[tuple, str] = {}
        self._running: set[tuple] = set()
        self._lock = threading.Lock()

    def request(self, composite, variant, control, reporter):
        key = (variant, control, reporter)
        if key in self._done or key in self._errors:
            return key
        with self._lock:
            if key in self._running:
                return key
            self._running.add(key)

        def run():
            try:
                self._done[key] = trace_path(
                    composite,
                    control,
                    reporter,
                    registry=self.page.registry or None,
                    t_end=self.page.t_end,
                    macro_dt=self.page.macro_dt,
                )
            except Exception as e:
                log.warning("trace %s -> %s failed: %s", control, reporter, e)
                self._errors[key] = f"{type(e).__name__}: {e}"
            finally:
                with self._lock:
                    self._running.discard(key)

        threading.Thread(target=run, daemon=True).start()
        return key

    def result(self, key):
        return self._done.get(tuple(key)) if key else None

    def error(self, key):
        return self._errors.get(tuple(key)) if key else None

    def running(self, key) -> bool:
        return tuple(key) in self._running if key else False


def detail_items(data: dict):
    from dash import html

    if not data:
        return [html.Span("hover a node or an edge", className="k")]
    d = data.get("detail") or {}
    kind = data.get("kind", "")
    out = [html.B(kind or "node")]
    for k, v in d.items():
        if v in ("", None, ()):
            continue
        out += [
            html.Span(f"{k}  ", className="k"),
            html.Span(str(v)),
            html.Br(),
        ]
    return out


def layout(page: Page, bank):
    from dash import dcc, html
    import dash_cytoscape as cyto

    cyto.load_extra_layouts()
    composite = page.composite
    handles = [lv.handle for lv in page.levers] or [
        name
        for name, h in page.registry.items()
        if any(m.process_name in composite.processes for m in h.mappings)
    ]
    keys = list(composite.store_keys())
    side = html.Div(
        className="side",
        children=[
            html.Div(
                className="card",
                children=[
                    html.H3("Trace a signal"),
                    dcc.Dropdown(
                        id="graph-control",
                        options=[{"label": h, "value": h} for h in handles],
                        placeholder="handle",
                        clearable=True,
                    ),
                    dcc.Input(
                        id="graph-param",
                        type="text",
                        placeholder="or process.parameter",
                        debounce=True,
                        style={"marginTop": "6px"},
                    ),
                    dcc.Dropdown(
                        id="graph-reporter",
                        options=[{"label": k, "value": k} for k in keys],
                        placeholder="reporter path",
                        clearable=True,
                        style={"marginTop": "6px"},
                    ),
                    html.Button("trace", id="graph-run", className="btn"),
                    html.Div(id="graph-note", className="note"),
                    dcc.Store(id="graph-trace", data=None),
                    dcc.Store(id="graph-expanded", data=[]),
                    dcc.Store(id="graph-fitted", data=0),
                    dcc.Store(id="graph-positions", data=None),
                    dcc.Store(id="graph-sublayout", data=0),
                    dcc.Store(id="graph-hover", data=False),
                    dcc.Interval(
                        id="graph-poll", interval=1000, n_intervals=0
                    ),
                ],
            ),
            html.Div(
                className="card",
                children=[
                    html.H3("View"),
                    dcc.Checklist(
                        id="graph-show",
                        options=[
                            {"label": " reactions", "value": "reactions"},
                            {"label": " observables", "value": "observables"},
                        ],
                        value=[],
                        className="check",
                    ),
                    html.Button(
                        "reset", id="graph-reset", className="btn ghost"
                    ),
                ],
            ),
            html.Div(
                className="card",
                children=[
                    html.H3("Node"),
                    html.Div(
                        id="graph-detail",
                        className="detail",
                        children=detail_items({}),
                    ),
                ],
            ),
            html.Div(
                "click a box to open it · hover a state to see what it "
                "touches",
                className="note",
            ),
        ],
    )
    main = html.Div(
        className="main",
        children=[
            cyto.Cytoscape(
                id="graph",
                elements=elements(composite, page),
                layout=TABS_LAYOUT,
                stylesheet=stylesheet(page),
                className="graph",
                style={"width": "100%"},
                minZoom=0.2,
                maxZoom=3,
            ),
            html.Pre(id="graph-table", className="trace"),
        ],
    )
    return html.Div(className="wrap", children=[side, main])


def register(app, page: Page, bank):
    from dash import Input, Output, State, no_update

    runner = TraceRunner(page)
    # The graph is laid out while its tab is hidden, at zero size: refit
    # it once the tab shows, and whenever the reset button asks.
    app.clientside_callback(
        """
        function(tab, _clicks, n) {
            if (tab !== "wiring") { return window.dash_clientside.no_update; }
            setTimeout(function() {
                const g = document.querySelector("#graph");
                const cy = g && g._cyreg && g._cyreg.cy;
                if (cy) { cy.resize(); cy.fit(undefined, 24); }
            }, 60);
            return (n || 0) + 1;
        }
        """,
        Output("graph-fitted", "data"),
        [Input("tab", "data"), Input("graph-reset", "n_clicks")],
        State("graph-fitted", "data"),
    )

    # Where every node sits, read before a click's redraw so the map keeps
    # its shape and only what the click reveals is placed.
    app.clientside_callback(
        """
        function(_node, _clicks) {
            const g = document.querySelector("#graph");
            const cy = g && g._cyreg && g._cyreg.cy;
            if (!cy) { return window.dash_clientside.no_update; }
            const out = {};
            cy.nodes().forEach(function(n) {
                if (!n.isParent()) { out[n.id()] = n.position(); }
            });
            return out;
        }
        """,
        Output("graph-positions", "data"),
        [Input("graph", "tapNodeData"), Input("graph-run", "n_clicks")],
        prevent_initial_call=True,
    )

    # States just revealed inside an opened process are laid out by their
    # edges within the room made for them.
    app.clientside_callback(
        """
        function(_elements, n) {
            const g = document.querySelector("#graph");
            const cy = g && g._cyreg && g._cyreg.cy;
            if (!cy) { return window.dash_clientside.no_update; }
            setTimeout(function() {
                const placed = cy.nodes(".placed");
                if (placed.length === 0) {
                    cy.animate({
                        fit: {eles: cy.elements(), padding: 24},
                        duration: 400, easing: "ease-in-out-cubic",
                    });
                    return;
                }
                const parents = {};
                placed.forEach(function(n) { parents[n.data("parent")] = true; });
                Object.keys(parents).forEach(function(pid) {
                    const kids = cy.nodes('[parent = "' + pid + '"]');
                    if (kids.length < 2) { return; }
                    const room = kids.boundingBox();
                    const inner = kids.edgesWith(kids);
                    kids.union(inner).layout({
                        name: "cose-bilkent", fit: false, animate: false,
                        randomize: true, nodeDimensionsIncludeLabels: true,
                        idealEdgeLength: 80, nodeRepulsion: 6000, tile: true,
                        tilingPaddingVertical: 16, tilingPaddingHorizontal: 16,
                    }).run();
                    // Centre the result where the room was, then move only
                    // what the finished box overlaps, by the overlap.
                    const got = kids.boundingBox({includeLabels: true});
                    const cx = (room.x1 + room.x2) / 2, cy0 = (room.y1 + room.y2) / 2;
                    const gx = (got.x1 + got.x2) / 2, gy = (got.y1 + got.y2) / 2;
                    kids.positions(function(n) {
                        const q = n.position();
                        return {x: q.x - gx + cx, y: q.y - gy + cy0};
                    });
                    // Then treat every top-level node and opened box as a
                    // rigid rectangle, this box fixed, and separate any two
                    // that overlap by exactly their overlap until none do.
                    const M = 28;
                    const items = cy.nodes().filter(function(n) {
                        return n.parent().length === 0;
                    });
                    const rectOf = function(ele) {
                        const bb = ele.boundingBox({includeLabels: true});
                        return {ele: ele, x1: bb.x1 - M, y1: bb.y1 - M,
                                x2: bb.x2 + M, y2: bb.y2 + M,
                                fixed: ele.id() === pid};
                    };
                    const moveBy = function(r, dx, dy) {
                        if (r.ele.isParent()) {
                            r.ele.children().positions(function(n) {
                                const q = n.position();
                                return {x: q.x + dx, y: q.y + dy};
                            });
                        } else {
                            const q = r.ele.position();
                            r.ele.position({x: q.x + dx, y: q.y + dy});
                        }
                        r.x1 += dx; r.x2 += dx; r.y1 += dy; r.y2 += dy;
                    };
                    for (let it = 0; it < 60; it++) {
                        const rs = items.map(rectOf);
                        let moved = false;
                        for (let i = 0; i < rs.length; i++) {
                            for (let j = i + 1; j < rs.length; j++) {
                                const a = rs[i], b = rs[j];
                                const ox = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
                                const oy = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
                                if (ox <= 0 || oy <= 0) { continue; }
                                moved = true;
                                const acx = (a.x1 + a.x2) / 2, bcx = (b.x1 + b.x2) / 2;
                                const acy = (a.y1 + a.y2) / 2, bcy = (b.y1 + b.y2) / 2;
                                let dx = 0, dy = 0;
                                if (ox < oy) { dx = (bcx >= acx ? ox : -ox) + 1; }
                                else { dy = (bcy >= acy ? oy : -oy) + 1; }
                                if (a.fixed) { moveBy(b, dx, dy); }
                                else if (b.fixed) { moveBy(a, -dx, -dy); }
                                else { moveBy(b, dx / 2, dy / 2); moveBy(a, -dx / 2, -dy / 2); }
                            }
                        }
                        if (!moved) { break; }
                    }
                });
                placed.removeClass("placed");
                // The map keeps its arrangement, so bring all of it into
                // view rather than leave the eye on the gap it just made.
                cy.animate({
                    fit: {eles: cy.elements(), padding: 24},
                    duration: 400, easing: "ease-in-out-cubic",
                });
            }, 150);
            return (n || 0) + 1;
        }
        """,
        Output("graph-sublayout", "data"),
        Input("graph", "elements"),
        State("graph-sublayout", "data"),
    )
    # Hovering a state fades everything but its neighbourhood.
    app.clientside_callback(
        """
        function(_fitted, installed) {
            const g = document.querySelector("#graph");
            const cy = g && g._cyreg && g._cyreg.cy;
            if (!cy || cy.scratch("hallsimHover")) {
                return window.dash_clientside.no_update;
            }
            cy.scratch("hallsimHover", true);
            cy.on("mouseover", "node", function(e) {
                const n = e.target;
                if (n.isParent()) { return; }
                n.closedNeighborhood().addClass("hl");
            });
            const clear = function() { cy.elements().removeClass("hl"); };
            cy.on("mouseout", "node", clear);
            cy.on("tap", clear);
            g.addEventListener("mouseleave", clear);
            return true;
        }
        """,
        Output("graph-hover", "data"),
        Input("graph-fitted", "data"),
        State("graph-hover", "data"),
    )

    def composite_for(window):
        vm = bank.get(window)
        return vm.base if vm is not None else page.variants[window]

    @app.callback(
        Output("graph-expanded", "data"),
        Input("graph", "tapNodeData"),
        State("graph-expanded", "data"),
        prevent_initial_call=True,
    )
    def toggle(node, expanded):
        if not node or node.get("kind") != "process":
            return no_update
        expanded = list(expanded or [])
        name = node["process"]
        if name in expanded:
            expanded.remove(name)
        else:
            expanded.append(name)
        return expanded

    @app.callback(
        Output("graph-detail", "children"),
        Input("graph", "mouseoverNodeData"),
        Input("graph", "mouseoverEdgeData"),
        Input("graph", "tapNodeData"),
    )
    def detail(hovered, edge, tapped):
        from dash import ctx

        latest = {
            "mouseoverNodeData": hovered,
            "mouseoverEdgeData": edge,
            "tapNodeData": tapped,
        }.get(
            (ctx.triggered_id and ctx.triggered[0]["prop_id"].split(".")[-1])
            or "",
            None,
        )
        return detail_items(latest or hovered or tapped or {})

    @app.callback(
        [
            Output("graph-expanded", "data", allow_duplicate=True),
            Output("graph-trace", "data", allow_duplicate=True),
            Output("graph-positions", "data", allow_duplicate=True),
        ],
        Input("graph-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset(_n):
        """Back to the opening view: every process closed, no trace, laid
        out afresh."""
        return [], None, None

    @app.callback(
        [
            Output("graph-trace", "data"),
            Output("graph-expanded", "data", allow_duplicate=True),
            Output("graph-note", "children", allow_duplicate=True),
        ],
        Input("graph-run", "n_clicks"),
        [
            State("graph-control", "value"),
            State("graph-param", "value"),
            State("graph-reporter", "value"),
            State("window", "data"),
        ],
        prevent_initial_call=True,
    )
    def start(_n, control, param, reporter, window):
        control = (param or "").strip() or control
        if not control or not reporter:
            return None, no_update, "pick a control and a reporter"
        key = runner.request(composite_for(window), window, control, reporter)
        return list(key), [], "tracing…"

    @app.callback(
        [
            Output("graph", "elements"),
            Output("graph", "layout"),
            Output("graph-table", "children"),
            Output("graph-note", "children"),
            Output("graph-poll", "disabled"),
        ],
        [
            Input("graph-expanded", "data"),
            Input("graph-trace", "data"),
            Input("graph-poll", "n_intervals"),
            Input("window", "data"),
            Input("graph-show", "value"),
        ],
        State("graph-positions", "data"),
    )
    def redraw(expanded, key, _n, window, show, positions):
        from dash import ctx

        fresh = ctx.triggered_id in ("window", None) or not positions
        composite = composite_for(window)
        trace = runner.result(key)
        error = runner.error(key)
        if key and trace is None and error is None:
            return no_update, no_update, no_update, "tracing…", False
        show = set(show or ())
        els = elements(
            composite,
            page,
            expanded or None,
            trace,
            reactions="reactions" in show,
            observables="observables" in show,
        )
        note = error or ""
        table = str(trace) if trace is not None else ""
        layout = dict(TABS_LAYOUT) if fresh else dict(PRESET_LAYOUT)
        if not fresh:
            els = place(els, positions)
        return els, layout, table, note, True
