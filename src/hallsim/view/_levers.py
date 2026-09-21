"""The levers tab: one slider per handle, one trajectory row per process."""

from __future__ import annotations

import numpy as np

from hallsim.view._model import ModelBank, ViewModel, series
from hallsim.view._page import Page
from hallsim.view._theme import (
    CURVE_WIDTH,
    HOVER,
    REFERENCE_LINE,
    TOKENS,
    color,
    font,
    rgba,
)


def shades_of(page: Page, variant: str | None):
    """The page's shades, or the variant's own when they differ by
    variant."""
    if isinstance(page.shades, dict):
        return page.shades.get(variant, ())
    return page.shades


def spans(page: Page, severities, variant=None):
    """``(on, x0, x1, color, opacity)`` per shade at this setting."""
    sev = page.severities(severities)
    return tuple(
        (
            sh.on(sev),
            sh.start,
            page.t_end if sh.end is None else sh.end,
            color(sh.color),
            sh.opacity,
        )
        for sh in shades_of(page, variant)
    )


def legend_items(page: Page, severities, variant=None):
    from dash import html

    def entry(sample, name):
        return html.Span([sample, name])

    sev = page.severities(severities)
    items = [
        entry(html.Span(className="ln"), "current setting"),
        entry(html.Span(className="ln dotted"), page.reference_label),
    ]
    items += [
        entry(
            html.Span(
                className="sw", style={"background": rgba(col, op * 2.5)}
            ),
            sh.name(sev),
        )
        for sh, (on, _, _, col, op) in zip(
            shades_of(page, variant), spans(page, severities, variant)
        )
        if on
    ]
    return items


def _grid(n: int, ncols) -> tuple[int, int]:
    cols = n if not ncols else max(1, min(n, int(ncols)))
    return -(-n // cols), cols


def _cell(j: int, cols: int) -> tuple[int, int]:
    return j // cols + 1, j % cols + 1


def row_figure(
    page: Page, name: str, severities, variant=None, height=210, ncols=None
):
    """An empty figure for one process's panels, the shaded periods in
    every panel, the shared axes."""
    from plotly.subplots import make_subplots

    row = page.panels[name]
    n = len(row)
    rows, cols = _grid(n, ncols)
    total_height = height * rows + 64 * (rows - 1)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[pn.label for pn in row],
        horizontal_spacing=0.045,
        vertical_spacing=(64 / total_height) if rows > 1 else 0.1,
    )
    for on, x0, x1, col, opacity in spans(page, severities, variant):
        if not on:
            continue
        for j in range(n):
            r, c = _cell(j, cols)
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=col,
                opacity=opacity,
                line_width=0,
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
    axis = dict(
        showline=True,
        linecolor=TOKENS["color-neutral-400"],
        linewidth=1,
        ticks="",
        showgrid=False,
        zeroline=False,
        tickfont=font("font-mono", 10, "color-neutral-700"),
    )
    title = dict(
        title_font=font("font-mono", 10, "color-neutral-700"),
        title_standoff=4,
    )
    xaxis = dict(range=[0.0, page.t_end], title_text="t", **title, **axis)
    if page.ticks:
        xaxis["tickvals"] = list(page.ticks)
    fig.update_xaxes(**xaxis)
    fig.update_yaxes(rangemode="tozero", **axis)
    if page.units.get(name):
        for r in range(1, rows + 1):
            fig.update_yaxes(
                title_text=page.units[name], **title, row=r, col=1
            )
    fig.update_layout(
        template="plotly_white",
        height=total_height,
        margin=dict(l=52, r=10, t=30, b=40),
        font=font("font-body", 11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        hoverdistance=-1,
        hoverlabel=dict(
            bgcolor=TOKENS["color-bg"],
            bordercolor=TOKENS["color-neutral-400"],
            font=font("font-mono", 11),
        ),
        showlegend=False,
    )
    fig.update_annotations(
        font=font("font-heading", 11.5, weight=500),
        selector=dict(xref="paper"),
    )
    return fig


def figure_for(
    page: Page, name: str, vm: ViewModel, ys, severities, ncols=None
):
    """Deterministic row: the reference dotted, this setting solid."""
    import plotly.graph_objects as go

    col = color(page.colors[name])
    fig = row_figure(page, name, severities, vm.variant, ncols=ncols)
    _, grid_cols = _grid(len(page.panels[name]), ncols)
    shown = vm.ts >= 0.0
    t = vm.ts[shown]
    for j, cols in enumerate(vm.columns[name]):
        r, c = _cell(j, grid_cols)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=series(vm.reference, shown, cols),
                line=REFERENCE_LINE,
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=series(ys, shown, cols),
                line=dict(color=col, width=CURVE_WIDTH),
                hovertemplate=HOVER,
            ),
            row=r,
            col=c,
        )
    return fig


def spread(cells):
    """``(lo, hi)`` edges of the band across the cell axis: the mean ± one
    sample standard deviation, the lower edge held at zero."""
    mean = cells.mean(axis=1)
    sd = cells.std(axis=1, ddof=1) if cells.shape[1] > 1 else 0.0
    return np.maximum(mean - sd, 0.0), mean + sd


def _band(fig, t, cells, fill, row, col):
    import plotly.graph_objects as go

    lo, hi = spread(cells)
    for edge, mode in ((hi, None), (lo, "tonexty")):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=edge,
                mode="lines",
                line=dict(width=0),
                fill=mode,
                fillcolor=fill,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def population_figure(
    page: Page, name: str, vm: ViewModel, ys_pop, severities, ncols=None
):
    """Reaction-level row: the spread across cells as a band, the population
    mean bold, the reference population in grey. ``ys_pop=None`` draws the
    reference population alone."""
    import plotly.graph_objects as go

    col = color(page.colors[name])
    fig = row_figure(
        page, name, severities, vm.variant, height=240, ncols=ncols
    )
    _, grid_cols = _grid(len(page.panels[name]), ncols)
    shown = vm.ts >= 0.0
    t = vm.ts[shown]
    control = vm.reference_population
    for j, cols in enumerate(vm.columns[name]):
        r, c = _cell(j, grid_cols)
        control_cells = None
        if control is not None:
            control_cells = series(control, shown, cols)
            _band(
                fig,
                t,
                control_cells,
                rgba(TOKENS["color-neutral-700"], 0.18),
                r,
                c,
            )
        if ys_pop is None:
            if control_cells is not None:
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=control_cells.mean(axis=1),
                        line=REFERENCE_LINE,
                        hoverinfo="skip",
                    ),
                    row=r,
                    col=c,
                )
            continue
        cells = series(ys_pop, shown, cols)
        _band(fig, t, cells, rgba(col, 0.22), r, c)
        if control is not None:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=control_cells.mean(axis=1),
                    line=REFERENCE_LINE,
                    hoverinfo="skip",
                ),
                row=r,
                col=c,
            )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=cells.mean(axis=1),
                line=dict(color=col, width=CURVE_WIDTH),
                hovertemplate=HOVER,
            ),
            row=r,
            col=c,
        )
    return fig


def mark_pending(fig, text: str):
    fig.update_traces(opacity=0.35)
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=font("font-heading", 15, weight=500),
        bgcolor="rgba(250,250,250,0.85)",
        borderpad=6,
    )
    return fig


def badge(text: str, pending: bool = False):
    from dash import html

    if pending:
        return html.Span(
            [html.Span(className="spin"), text], className="pop pending"
        )
    return html.Span(text, className="pop")


def population_badge(vm: ViewModel | None, sev):
    if vm is None:
        return badge("compiling…", pending=True)
    if not vm.n_cells:
        return badge("no population")
    if vm.population_cached(sev) is None:
        return badge("sampling…", pending=True)
    return badge("")


def chip_classes(page: Page, sev) -> list[str]:
    key = ViewModel.key(sev)
    return [
        "chip selected" if ViewModel.key(p) == key else "chip"
        for p in page.presets.values()
    ]


def render(
    page: Page,
    vm: ViewModel | None,
    fallback: ViewModel,
    *severities,
    ncols=None,
):
    """Everything a lever pull shows, in the callback's output order: one
    figure per process row, one value per lever, the population badge, the
    legend, the preset chips' classes. ``vm=None`` is a variant still
    compiling: ``fallback``'s rows, faded and marked."""
    sev = np.asarray(severities, dtype=float)
    model = vm or fallback
    _, ys = model.solve(sev)
    figures = []
    for name in page.panels:
        sample = model.population_cached(sev) if model.n_cells else None
        if name in page.population and model.n_cells:
            fig = population_figure(
                page,
                name,
                model,
                None if sample is None else sample[0],
                sev,
                ncols=ncols,
            )
            if sample is None:
                mark_pending(fig, "sampling…")
        else:
            fig = figure_for(page, name, model, ys, sev, ncols=ncols)
        if vm is None:
            mark_pending(fig, "compiling…")
        figures.append(fig)
    values = [
        f"{s:+.2f}" if lever.lo < 0 else f"{s:.2f}"
        for lever, s in zip(page.levers, sev)
    ]
    return (
        *figures,
        *values,
        population_badge(vm, sev),
        legend_items(page, sev, model.variant),
        chip_classes(page, sev),
    )


def render_population(page: Page, vm: ViewModel, *severities, ncols=None):
    """The reaction-level rows for one setting, or ``None`` while its sample
    is being taken: one figure per population process, then the badge."""
    sev = np.asarray(severities, dtype=float)
    sample = vm.population_async(sev)
    if sample is None:
        return None
    figures = [
        population_figure(page, m, vm, sample[0], sev, ncols=ncols)
        for m in page.population
    ]
    return (*figures, population_badge(vm, sev))


def layout(page: Page, bank: ModelBank, app):
    """The tab: levers, presets and the variant switch beside the rows."""
    from dash import dcc, html

    def lever_card(i, lever):
        handle = page.registry[lever.handle]
        marks = {
            v: {"label": ""}
            for v in np.round(np.linspace(lever.lo, lever.hi, 5), 2)
        }
        return html.Div(
            className="card",
            children=[
                html.H3(
                    [lever.name, html.Span(id=f"value-{i}", className="val")],
                    title=f"{lever.handle}: {handle.description}",
                ),
                dcc.Slider(
                    id=f"lever-{i}",
                    min=lever.lo,
                    max=lever.hi,
                    step=0.05,
                    value=page.presets[page.reference][i],
                    marks=marks,
                    tooltip={"placement": "bottom"},
                    updatemode="mouseup",
                ),
            ],
        )

    def segment(name):
        top, bottom = page.variant_label(name)
        return html.Button(
            [
                html.Span(top, className="seg-top"),
                html.Span(bottom, className="seg-sub"),
            ],
            id={"window": name},
            className="seg",
        )

    def model_row(name):
        heading = [page.titles[name]]
        if page.deposits.get(name):
            heading.append(
                html.Span(f"  {page.deposits[name]}", className="dep")
            )
        if name in page.population:
            heading.append(html.Span(id="population-status"))
        graph = dcc.Loading(
            children=dcc.Graph(
                id=f"panel-{name}", config={"displayModeBar": False}
            ),
            type="circle",
            color=color(page.colors[name]),
            delay_show=400,
        )
        return html.Div(className="row", children=[html.H2(heading), graph])

    variant_card = []
    if len(page.variants) > 1:
        variant_card = [
            html.H3(page.variant_title),
            html.Div(
                [segment(name) for name in page.variants],
                className="seg-group",
            ),
            html.Div(id="window-note", className="seg-note"),
        ]
    else:
        variant_card = [
            html.Div(id="window-note", style={"display": "none"}),
        ]
    if not page.population:
        variant_card.append(
            html.Span(id="population-status", style={"display": "none"})
        )
    side = [
        html.Div(
            [
                html.Button(name, id={"preset": name}, className="chip")
                for name in page.presets
            ],
            className="chips",
        ),
        *[lever_card(i, lever) for i, lever in enumerate(page.levers)],
        html.Div(
            className="card",
            children=[
                *variant_card,
                dcc.Store(id="window", data=bank.first),
                dcc.Store(id="drawn", data=None),
                dcc.Store(id="viewport", data=None),
                dcc.Interval(id="viewport-poll", interval=1000, n_intervals=0),
                dcc.Interval(id="window-poll", interval=2000, n_intervals=0),
                dcc.Interval(
                    id="population-poll", interval=1000, n_intervals=0
                ),
            ],
        ),
    ]
    if page.about:
        side.append(
            html.Div(
                className="about", children=[html.P(p) for p in page.about]
            )
        )
    if page.logos:
        side.append(
            html.Div(
                className="logos",
                children=[
                    html.Img(src=app.get_asset_url(f), alt=alt)
                    for f, alt in page.logos
                ],
            )
        )
    return html.Div(
        className="wrap",
        children=[
            html.Div(className="side", children=side),
            html.Div(
                className="main",
                children=[
                    html.Div(id="legend", className="legend"),
                    *[model_row(name) for name in page.panels],
                ],
            ),
        ],
    )


def register(app, page: Page, bank: ModelBank):
    from dash import ALL, Input, Output, State, ctx, no_update

    first = bank.get(bank.first)
    n_levers = len(page.levers)
    lever_inputs = [Input(f"lever-{i}", "value") for i in range(n_levers)]
    window_input = Input("window", "data")
    app.clientside_callback(
        """
        function(n, current) {
            const cols = window.innerWidth > 600 ? 5 : 2;
            return cols === current ? window.dash_clientside.no_update : cols;
        }
        """,
        Output("viewport", "data"),
        Input("viewport-poll", "n_intervals"),
        State("viewport", "data"),
    )
    viewport_input = Input("viewport", "data")

    if len(page.variants) > 1:

        @app.callback(
            Output("window", "data"),
            Input({"window": ALL}, "n_clicks"),
            prevent_initial_call=True,
        )
        def choose_window(_clicks):
            return ctx.triggered_id["window"]

        @app.callback(
            [
                Output({"window": ALL}, "className"),
                Output({"window": ALL}, "disabled"),
                Output("window-note", "children"),
                Output("window-poll", "disabled"),
            ],
            [Input("window-poll", "n_intervals"), window_input],
        )
        def poll_windows(_n, selected):
            classes, disabled = [], []
            for name in page.variants:
                ready = bank.ready(name)
                cls = "seg" + (" selected" if name == selected else "")
                cls += "" if ready else " compiling"
                classes.append(cls)
                disabled.append(not ready)
            pending = sum(1 for d in disabled if d)
            note = (
                f"{pending} of {len(page.variants)} compiling"
                if pending
                else ""
            )
            return classes, disabled, note, not any(disabled)

    if n_levers:

        @app.callback(
            [Output(f"lever-{i}", "value") for i in range(n_levers)],
            Input({"preset": ALL}, "n_clicks"),
            prevent_initial_call=True,
        )
        def apply_preset(_clicks):
            return list(page.presets[ctx.triggered_id["preset"]])

    @app.callback(
        [Output(f"panel-{m}", "figure") for m in page.panels]
        + [Output(f"value-{i}", "children") for i in range(n_levers)]
        + [
            Output("population-status", "children"),
            Output("legend", "children"),
            Output({"preset": ALL}, "className"),
        ],
        [window_input, viewport_input, *lever_inputs],
    )
    def on_pull(window, ncols, *severities):
        return render(page, bank.get(window), first, *severities, ncols=ncols)

    if not first.n_cells:
        return

    pop = page.population
    population_outputs = [
        Output(f"panel-{m}", "figure", allow_duplicate=True) for m in pop
    ] + [
        Output("population-status", "children", allow_duplicate=True),
        Output("drawn", "data", allow_duplicate=True),
        Output("population-poll", "disabled", allow_duplicate=True),
    ]
    idle = (no_update,) * (len(pop) + 3)
    lever_states = [State(f"lever-{i}", "value") for i in range(n_levers)]

    def asleep(vm) -> bool:
        return vm is not None and vm.reference_population is not None

    def drawn_key(window, severities, ncols):
        vm = bank.get(window)
        return [
            window,
            vm is not None and vm.reference_population is not None,
            ncols,
            *(round(float(s), 6) for s in severities),
        ]

    def population_rows(window, severities, ncols):
        vm = bank.get(window)
        if vm is None:
            return (*(no_update,) * (len(pop) + 2), False)
        rows = render_population(page, vm, *severities, ncols=ncols)
        if rows is None:
            return (*(no_update,) * (len(pop) + 1), None, False)
        return (*rows, drawn_key(window, severities, ncols), asleep(vm))

    @app.callback(
        population_outputs,
        [window_input, viewport_input, *lever_inputs],
        prevent_initial_call=True,
    )
    def on_pull_population(window, ncols, *severities):
        return population_rows(window, severities, ncols)

    @app.callback(
        population_outputs,
        Input("population-poll", "n_intervals"),
        [
            State("window", "data"),
            State("drawn", "data"),
            State("viewport", "data"),
            *lever_states,
        ],
        prevent_initial_call=True,
    )
    def poll_population(_n, window, drawn, ncols, *severities):
        vm = bank.get(window)
        if drawn == drawn_key(window, severities, ncols):
            return (*idle[:-1], asleep(vm))
        if vm is None or vm.population_cached(severities) is None:
            return idle
        return population_rows(window, severities, ncols)
