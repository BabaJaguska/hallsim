"""The page: a tab strip over the levers, wiring and fit views."""

from __future__ import annotations

import logging
from pathlib import Path

from hallsim.view import _fit, _graph, _levers
from hallsim.view._model import ModelBank
from hallsim.view._page import Page
from hallsim.view._theme import CSS, GOOGLE_FONTS, root_css

log = logging.getLogger(__name__)

TABS = (("levers", "levers"), ("wiring", "wiring"), ("fit", "fit"))


def build_app(
    page: Page, bank: ModelBank | None = None, *, runs=(), runs_dir=None
):
    """The Dash app for ``page``. ``runs`` are calibration run folders for
    the fit tab, beside every one found under ``runs_dir``."""
    try:
        from dash import ALL, Dash, Input, Output, ctx, dcc, html
    except ImportError as e:  # pragma: no cover - install hint
        raise SystemExit(
            'The page needs Dash: pip install "hallsim[app]"'
        ) from e

    bank = bank or ModelBank(page)
    assets = page.assets_folder or str(Path(__file__).parent / "assets")
    app = Dash(
        __name__,
        title=page.title,
        update_title=None,
        external_stylesheets=[GOOGLE_FONTS],
        assets_folder=assets,
    )
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{root_css()}{CSS}</style></head>"
    )
    found = _fit.find_runs(runs_dir, runs)
    tab_views = {
        "levers": _levers.layout(page, bank, app),
        "wiring": _graph.layout(page, bank),
        "fit": _fit.layout(found),
    }
    app.layout = html.Div(
        [
            html.Div(
                className="top",
                children=[
                    html.Div(
                        className="brand",
                        children=[html.H1(page.title)]
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
                                [html.Span(label, className="seg-top")],
                                id={"tab": name},
                                className="seg selected" if i == 0 else "seg",
                            )
                            for i, (name, label) in enumerate(TABS)
                        ],
                    ),
                    dcc.Store(id="tab", data=TABS[0][0]),
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
        Input({"tab": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def choose_tab(_clicks):
        return ctx.triggered_id["tab"]

    @app.callback(
        [Output(f"tab-{name}", "style") for name, _ in TABS]
        + [Output({"tab": ALL}, "className")],
        Input("tab", "data"),
    )
    def show_tab(selected):
        styles = [
            {} if name == selected else {"display": "none"} for name, _ in TABS
        ]
        classes = [
            "seg selected" if name == selected else "seg" for name, _ in TABS
        ]
        return (*styles, classes)

    _levers.register(app, page, bank)
    _graph.register(app, page, bank)
    _fit.register(app)
    return app


def serve(
    page: Page,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    runs=(),
    runs_dir=None,
):
    """Compile the page's first variant, then serve it."""
    log.info(
        "compiling the first variant; the page is served as soon as that "
        "is done. Other variants and any population sample fill in behind."
    )
    bank = ModelBank(page)
    app = build_app(page, bank, runs=runs, runs_dir=runs_dir)
    log.info("serving on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
