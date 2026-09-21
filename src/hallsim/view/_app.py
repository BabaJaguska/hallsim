"""The page: a tab strip over the levers and wiring views, and a fit view
when a run is named."""

from __future__ import annotations

import logging

from hallsim.view import _fit, _graph, _levers
from hallsim.view._model import ModelBank
from hallsim.view._page import Page
from hallsim.view._theme import CSS, GOOGLE_FONTS, root_css

log = logging.getLogger(__name__)


def build_app(page: Page, bank: ModelBank | None = None, *, runs=()):
    """The Dash app for ``page``: the levers and wiring tabs, and a fit tab
    when ``runs`` names calibration run folders."""
    try:
        from dash import ALL, Dash, Input, Output, ctx, dcc, html
    except ImportError as e:  # pragma: no cover - install hint
        raise SystemExit(
            'The page needs Dash: pip install "hallsim[app]"'
        ) from e

    bank = bank or ModelBank(page)
    app = Dash(
        __name__,
        title=page.title,
        update_title=None,
        external_stylesheets=[GOOGLE_FONTS],
        assets_folder=page.assets_folder or "assets",
    )
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{root_css()}{CSS}</style></head>"
    )
    found = _fit.find_runs(runs)
    tab_views = {
        "levers": _levers.layout(page, bank, app),
        "wiring": _graph.layout(page, bank),
    }
    if found:
        tab_views["fit"] = _fit.layout(found, selected=str(found[0]))
    tabs = tuple(tab_views)
    app.layout = html.Div(
        [
            html.Div(
                className="top",
                children=[
                    html.Div(
                        className="brand",
                        children=[
                            html.H1(page.title, id="brand", title="levers")
                        ]
                        + (
                            [html.Div(page.tagline, className="tag")]
                            if page.tagline
                            else []
                        ),
                    ),
                    html.Div(
                        className="seg-group tabs",
                        children=[
                            html.Button(
                                [html.Span(name, className="seg-top")],
                                id={"tab": name},
                                className="seg selected" if i == 0 else "seg",
                            )
                            for i, name in enumerate(tabs)
                        ],
                    ),
                    dcc.Store(id="tab", data=tabs[0]),
                ],
            ),
            *[
                html.Div(
                    view,
                    id=f"tab-{name}",
                    style={} if i == 0 else {"display": "none"},
                )
                for i, (name, view) in enumerate(tab_views.items())
            ],
        ]
    )

    @app.callback(
        Output("tab", "data"),
        [Input({"tab": ALL}, "n_clicks"), Input("brand", "n_clicks")],
        prevent_initial_call=True,
    )
    def choose_tab(_clicks, _brand):
        if ctx.triggered_id == "brand":
            return tabs[0]
        return ctx.triggered_id["tab"]

    @app.callback(
        [Output(f"tab-{name}", "style") for name in tabs]
        + [Output({"tab": ALL}, "className")],
        Input("tab", "data"),
    )
    def show_tab(selected):
        styles = [
            {} if name == selected else {"display": "none"} for name in tabs
        ]
        classes = [
            "seg selected" if name == selected else "seg" for name in tabs
        ]
        return (*styles, classes)

    _levers.register(app, page, bank)
    _graph.register(app, page, bank)
    if found:
        _fit.register(app)
    return app


def serve(
    page: Page,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    runs=(),
):
    """Compile the page's first variant, then serve it."""
    log.info(
        "compiling the first variant; the page is served as soon as that "
        "is done. Other variants and any population sample fill in behind."
    )
    bank = ModelBank(page)
    app = build_app(page, bank, runs=runs)
    log.info("serving on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
