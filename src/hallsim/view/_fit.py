"""The fit tab: a saved calibration run's history, parameters and
concordance, read from the folder :func:`hallsim.calibration_report.save_outputs`
wrote."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hallsim.view._theme import TOKENS, font


def find_runs(paths) -> list[Path]:
    """Those of ``paths`` that are run folders holding a ``summary.json``."""
    return [Path(p) for p in paths if (Path(p) / "summary.json").exists()]


def load_run(folder) -> dict:
    folder = Path(folder)
    with open(folder / "summary.json") as f:
        summary = json.load(f)
    config = {}
    if (folder / "config.json").exists():
        with open(folder / "config.json") as f:
            config = json.load(f)
    return {
        "name": folder.name,
        "folder": str(folder),
        "summary": summary,
        "config": config,
    }


def _axes(fig, **kw):
    axis = dict(
        showline=True,
        linecolor=TOKENS["color-neutral-400"],
        linewidth=1,
        ticks="",
        showgrid=False,
        zeroline=False,
        tickfont=font("font-mono", 10, "color-neutral-700"),
        title_font=font("font-mono", 10, "color-neutral-700"),
        title_standoff=4,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=52, r=10, t=30, b=40),
        font=font("font-body", 11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=font("font-mono", 10), orientation="h", y=1.12),
        **kw,
    )
    return fig


def loss_figure(summary: dict):
    import plotly.graph_objects as go

    losses = summary.get("loss_history", [])
    epochs = np.arange(1, len(losses) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            name="loss",
            line=dict(color=TOKENS["color-text"], width=1.6),
        )
    )
    val = summary.get("val_loss_history") or []
    if val:
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(val) + 1),
                y=val,
                name="held-out",
                line=dict(color=TOKENS["color-accent"], width=1.6),
            )
        )
    fig.update_yaxes(type="log", title_text="loss")
    fig.update_xaxes(title_text="epoch")
    return _axes(fig, height=260)


def gradient_figure(summary: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gn = summary.get("grad_norm_history") or []
    lrs = summary.get("lr_history") or []
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if gn:
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(gn) + 1),
                y=gn,
                name="|grad|",
                line=dict(color=TOKENS["color-red"], width=1.4),
            )
        )
    if lrs:
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(lrs) + 1),
                y=lrs,
                name="learning rate",
                line=dict(color=TOKENS["color-blue"], width=1.4),
            ),
            secondary_y=True,
        )
    fig.update_yaxes(type="log", title_text="|grad|", secondary_y=False)
    fig.update_yaxes(type="log", title_text="lr", secondary_y=True)
    fig.update_xaxes(title_text="epoch")
    return _axes(fig, height=260)


def parameter_figure(summary: dict):
    """Each parameter as its log-position within its clamp range, or its own
    travelled range dashed when unclamped."""
    import plotly.graph_objects as go

    history = summary.get("param_history") or []
    specs = dict(summary.get("params", {}))
    specs.update(summary.get("handle_coeffs", {}))
    fig = go.Figure()
    names = [k for k in (history[0] if history else {}) if k in specs]
    for name in names:
        vals = np.asarray(
            [ph.get(name, np.nan) for ph in history], dtype=float
        )
        if not np.isfinite(vals).any():
            continue
        clamp = specs[name].get("clamp")
        lo, hi = clamp if clamp else (np.nanmin(vals), np.nanmax(vals))
        if lo > 0 and hi > lo:
            y = (np.log(vals) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            y = np.full_like(vals, 0.5)
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(vals) + 1),
                y=y,
                name=name,
                line=dict(width=1.4, dash="solid" if clamp else "dash"),
            )
        )
    fig.update_yaxes(range=[-0.02, 1.02], title_text="log-position in range")
    fig.update_xaxes(title_text="epoch")
    return _axes(fig, height=300)


def _table(header, rows, classes=()):
    from dash import html

    def cell(v, cls=""):
        return html.Td(v, className=cls)

    return html.Table(
        className="fit",
        children=[
            html.Thead(html.Tr([html.Th(h) for h in header])),
            html.Tbody(
                [
                    html.Tr(
                        [cell(v, c) for v, c in row],
                        className=cls,
                    )
                    for row, cls in zip(rows, classes or [""] * len(rows))
                ]
            ),
        ],
    )


def parameter_table(summary: dict):
    specs = dict(summary.get("params", {}))
    specs.update(summary.get("handle_coeffs", {}))
    init = summary.get("init_params", {})
    final = summary.get("fitted_params", {})
    rows = []
    for name in final:
        if name not in specs:
            continue
        spec = specs[name]
        clamp = spec.get("clamp")
        rows.append(
            [
                (name, ""),
                (f"{init.get(name, float('nan')):.4g}", "num"),
                (f"{final[name]:.4g}", "num"),
                (
                    (
                        f"{final[name] / init[name]:.3g}×"
                        if init.get(name)
                        else ""
                    ),
                    "num",
                ),
                (f"[{clamp[0]:g}, {clamp[1]:g}]" if clamp else "", ""),
                (spec.get("description", ""), ""),
            ]
        )
    return _table(
        ["parameter", "start", "fitted", "moved", "clamp", "what it is"],
        rows,
    )


def concordance_tables(summary: dict):
    """One table per arm: measured against the model before and after the
    fit, per readout and timepoint, with the arm's sign agreement and rank
    correlation."""
    from dash import html

    pre = summary.get("concordance_pre") or {}
    post = summary.get("concordance_post") or {}
    fit = set(summary.get("fit_arms") or [])
    out = []
    for arm in pre:
        held = arm not in fit
        out.append(
            html.H3(
                [
                    arm,
                    html.Span(
                        "  held-out" if held else "  fit", className="dep"
                    ),
                ],
                className="fit-arm",
            )
        )
        for t in sorted(pre[arm], key=float):
            a, b = pre[arm][t], post.get(arm, {}).get(t, pre[arm][t])
            rows, classes = [], []
            post_rows = {r["gene"]: r for r in b["rows"]}
            for r in a["rows"]:
                q = post_rows.get(r["gene"], r)
                e0 = abs(r["delta_sim_signed"] - r["delta_data"])
                e1 = abs(q["delta_sim_signed"] - q["delta_data"])
                rows.append(
                    [
                        (r["gene"], ""),
                        (f"{r['delta_data']:+.3f}", "num"),
                        (f"{r['delta_sim_signed']:+.3f}", "num"),
                        (f"{q['delta_sim_signed']:+.3f}", "num"),
                        (f"{e0:.3f} → {e1:.3f}", "num"),
                        (
                            "✓" if q["sign_match"] else "✗",
                            "" if q["sign_match"] else "miss",
                        ),
                    ]
                )
                classes.append("held" if held else "")
            n = max(len(a["rows"]), 1)
            mae0 = (
                sum(
                    abs(r["delta_sim_signed"] - r["delta_data"])
                    for r in a["rows"]
                )
                / n
            )
            mae1 = sum(
                abs(r["delta_sim_signed"] - r["delta_data"]) for r in b["rows"]
            ) / max(len(b["rows"]), 1)
            rows.append(
                [
                    ("mean |err|", ""),
                    ("", ""),
                    ("", ""),
                    ("", ""),
                    (f"{mae0:.3f} → {mae1:.3f}", "num"),
                    (
                        f"sign {a['sign_agreement'] * 100:.0f}→{b['sign_agreement'] * 100:.0f}%  "
                        f"ρ {a['spearman_r']:+.2f}→{b['spearman_r']:+.2f}",
                        "",
                    ),
                ]
            )
            classes.append("")
            out.append(
                html.Div(
                    [
                        html.Div(f"t = {float(t):g}", className="note"),
                        _table(
                            [
                                "readout",
                                "measured",
                                "model before",
                                "model after",
                                "|err| before → after",
                                "sign",
                            ],
                            rows,
                            classes,
                        ),
                    ]
                )
            )
    return out


def run_view(run: dict):
    from dash import dcc, html

    s = run["summary"]
    facts = [
        ("run", run["name"]),
        ("best loss", f"{s.get('best_loss', float('nan')):.4g}"),
        (
            "epochs",
            str(s.get("stopped_epoch", len(s.get("loss_history", [])))),
        ),
        ("wall time", f"{s.get('wall_time_s', 0):.0f} s"),
        ("fit arms", ", ".join(s.get("fit_arms", []))),
        ("held out", ", ".join(s.get("held_out_arms", [])) or "none"),
        ("processes", ", ".join(s.get("processes", []))),
    ]
    settings = (run.get("config") or {}).get("fit") or {}
    return html.Div(
        [
            html.Div(
                className="detail",
                children=[
                    x
                    for k, v in facts
                    for x in (
                        html.Span(f"{k}  ", className="k"),
                        html.Span(v),
                        html.Br(),
                    )
                ],
            ),
            html.Div(
                className="fit-grid",
                children=[
                    html.Div(
                        [
                            html.H3("loss"),
                            dcc.Graph(
                                figure=loss_figure(s),
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.H3("gradient and learning rate"),
                            dcc.Graph(
                                figure=gradient_figure(s),
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                ],
            ),
            html.H3("parameters"),
            dcc.Graph(
                figure=parameter_figure(s), config={"displayModeBar": False}
            ),
            parameter_table(s),
            html.H3("concordance"),
            *concordance_tables(s),
            html.Details(
                [
                    html.Summary("fit settings and problem"),
                    html.Pre(
                        json.dumps(
                            {
                                "fit": settings,
                                "problem": (run.get("config") or {}).get(
                                    "problem", {}
                                ),
                            },
                            indent=1,
                        )[:20000],
                        className="trace",
                    ),
                ]
            ),
        ]
    )


def layout(runs: list[Path], selected: str | None = None):
    """The tab: a run picker over ``runs``, ``selected`` open."""
    from dash import dcc, html

    options = [{"label": p.name, "value": str(p)} for p in runs]
    side = html.Div(
        className="side",
        children=[
            html.Div(
                className="card",
                children=[
                    html.H3("Run"),
                    dcc.Dropdown(
                        id="fit-run",
                        options=options,
                        value=selected,
                        placeholder="pick a run",
                        clearable=True,
                    ),
                    dcc.Input(
                        id="fit-path",
                        type="text",
                        placeholder="or a folder path",
                        debounce=True,
                        style={"marginTop": "6px"},
                    ),
                    html.Div(id="fit-note", className="note"),
                ],
            ),
            html.Div(
                className="about",
                children=[
                    html.P(
                        "What a calibration run wrote: its loss and "
                        "gradient history, where each parameter went "
                        "within its range, and the readouts measured "
                        "against the model before and after the fit."
                    )
                ],
            ),
        ],
    )
    main = html.Div(className="main", children=[html.Div(id="fit-body")])
    return html.Div(className="wrap", children=[side, main])


def register(app):
    from dash import Input, Output, html

    @app.callback(
        [Output("fit-body", "children"), Output("fit-note", "children")],
        [Input("fit-run", "value"), Input("fit-path", "value")],
    )
    def show(selected, typed):
        folder = (typed or "").strip() or selected
        if not folder:
            return html.Div("no run loaded", className="note"), ""
        if not (Path(folder) / "summary.json").exists():
            return (
                html.Div("no run loaded", className="note"),
                f"no summary.json in {folder}",
            )
        return run_view(load_run(folder)), ""
