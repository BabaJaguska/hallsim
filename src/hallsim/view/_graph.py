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


def elements(composite, page: Page, expanded=None, trace=None) -> list[dict]:
    """Cytoscape elements: one node per process, expanded ones opened into
    their reactions and the paths they write, edges labelled with the paths
    they carry. With ``trace``, the route's nodes carry their relative
    change and everything else is dimmed; ``expanded=None`` opens the
    route's processes onto the route alone."""
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
            proc = node[1]
            ident = node[2] if len(node) > 2 else "derivative"
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
    edges: dict[tuple[str, str], set[str]] = {}
    parent_of = {
        e["data"]["id"]: e["data"].get("parent")
        for e in out
        if "id" in e["data"]
    }
    for u, v in g.edges:
        ru, rv = (
            _representative(u, owner, expanded, shown),
            _representative(v, owner, expanded, shown),
        )
        if ru == rv or parent_of.get(ru) == rv or parent_of.get(rv) == ru:
            continue
        label = u[1] if u[0] == "path" else (v[1] if v[0] == "path" else "")
        edges.setdefault((ru, rv), set())
        if label:
            edges[(ru, rv)].add(label)
    ids = {e["data"]["id"] for e in out}
    route_ids = {
        e["data"]["id"] for e in out if "route" in e["classes"].split()
    } | {f"proc:{n.process}" for n in (trace.nodes if trace else ())}
    for (ru, rv), labels in edges.items():
        if ru not in ids or rv not in ids:
            continue
        names = sorted(labels)
        label = ", ".join(names) if len(names) <= 3 else f"{len(names)} paths"
        classes = []
        if trace:
            classes.append(
                "route" if ru in route_ids and rv in route_ids else "dim"
            )
        out.append(
            {
                "data": {"source": ru, "target": rv, "label": label},
                "classes": " ".join(classes),
            }
        )
    return out


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
        {"selector": "edge.dim, edge.route", "style": {"text-opacity": 0}},
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
        return [html.Span("hover a node", className="k")]
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
                    dcc.Interval(
                        id="graph-poll", interval=1000, n_intervals=0
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
                className="about",
                children=[
                    html.P(
                        "Boxes are processes; click one to open it into its "
                        "reactions and the states it writes. Edges carry "
                        "store paths."
                    ),
                    html.P(
                        "A trace runs the composite at two settings of the "
                        "control and colours the route by how much of the "
                        "relative change each state keeps."
                    ),
                ],
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
    # it once the tab shows.
    app.clientside_callback(
        """
        function(tab, n) {
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
        Input("tab", "data"),
        State("graph-fitted", "data"),
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
        Input("graph", "tapNodeData"),
    )
    def detail(hovered, tapped):
        return detail_items(hovered or tapped or {})

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
        ],
    )
    def redraw(expanded, key, _n, window):
        composite = composite_for(window)
        trace = runner.result(key)
        error = runner.error(key)
        if key and trace is None and error is None:
            return no_update, no_update, no_update, "tracing…", False
        els = elements(composite, page, expanded or None, trace)
        note = error or ("" if trace is None else "")
        table = str(trace) if trace is not None else ""
        return els, dict(TABS_LAYOUT), table, note, True
