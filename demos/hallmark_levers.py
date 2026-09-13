"""Hallmark levers — pull a hallmark severity and watch three published
models re-solve.

``simulate hallmark-levers`` serves a page with one slider per hallmark of
aging wired into the multi-hallmark composite (DallePezze 2014, Geva-Zatorsky
2006, Proctor 2007). Moving a slider applies the severity through
:func:`hallsim.hallmarks.with_hallmarks` — the same call the calibration arms
are built with — re-solves the whole composite through the Scheduler and
redraws every panel against control (all levers at 0).

Genomic Instability is an etoposide pulse: the exposure window is shaded
in every row and named in the page legend, as is the rapamycin period,
and a switch offers longer windows than the published two days. Each
window is its own compiled composite; the first is built before the page
is served and the others in the background.

Proctor 2007 is a stochastic model and is drawn as one: its row runs the
member at reaction level as a population of cells (one Gillespie path each,
on the composite's clock, driven by DallePezze's ROS and phospho-mTORC1),
as a spread band with the population mean and the mean field the
calibration used over it. A sample is seconds of serial event loop, so the
presets are sampled at startup, every sample is kept, and a new setting
shows its mean field, faded and marked, until the sample lands.
``--cells 0`` serves the page without a population.

Needs the ``app`` extra: ``pip install "hallsim[app]"``.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demos.models.multi_hallmark import (
    DDIS_ETOPOSIDE_DOSE_WINDOW,
    MULTI_HALLMARK_GRID,
    RAPA_INTERVENTION_DAY,
    build_multi_hallmark_composite,
)
from demos.multi_hallmark_calibrate import (
    PREROLL_DAYS,
    RAPA_INTENSITY,
    _registry_with_intensity,
)
from demos.multi_hallmark_figures import MODEL_BLOCKS, REACTION_LEVEL
from hallsim.composite import Composite
from hallsim.hallmarks import with_hallmarks
from hallsim.scheduler import Scheduler

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lever:
    """One slider: a registry hallmark and the severity range it is defined
    on (``[0, 1]`` for a hallmark with no opposite, ``[-1, 1]`` otherwise;
    −1 on Deregulated Nutrient Sensing is rapamycin)."""

    hallmark: str
    lo: float
    hi: float
    label: str


LEVERS = (
    Lever("Genomic Instability", 0.0, 1.0, "Genomic instability"),
    Lever(
        "Deregulated Nutrient Sensing",
        -1.0,
        1.0,
        "Deregulated nutrient sensing",
    ),
    Lever("Loss of Proteostasis", 0.0, 1.0, "Loss of proteostasis"),
)

# The calibration arms, as severity vectors in LEVERS order.
PRESETS = {
    "control": (0.0, 0.0, 0.0),
    "etoposide": (1.0, 0.0, 0.0),
    "+rapamycin": (1.0, -1.0, 0.0),
}
# The arm the page opens on.
DEFAULT_PRESET = "etoposide"

# Etoposide exposure windows on offer, in days from the start; the first is
# the published protocol the composite was calibrated on. Rapamycin still
# enters at its own day whatever the window.
DOSE_WINDOWS = {
    "days 0–2": DDIS_ETOPOSIDE_DOSE_WINDOW,
    "days 0–7": (0.0, 7.0),
    "whole run": (0.0, MULTI_HALLMARK_GRID.t_end),
}
# Two-line face of each segment of the exposure switch.
WINDOW_LABELS = {
    "days 0–2": ("0–2", "days"),
    "days 0–7": ("0–7", "days"),
    "whole run": ("whole", "run"),
}

# What each publication's row shows: (store path or paths summed, label).
# Membership is checked against the composite at build, so a renamed
# species fails loudly; a species the import froze as an inert sink is
# unfrozen for the page, since a panel is a reader.
PANELS = {
    "dp14": (
        ("dp14/DNA_damage", "DNA damage"),
        ("dp14/mTORC1_pS2448", "phospho-mTORC1"),
        ("dp14/ROS", "ROS"),
        ("dp14/CDKN1A", "p21 (CDKN1A)"),
        ("dp14/SA_beta_gal", "SA-β-gal"),
    ),
    "gz06": (("gz06/x", "p53"), ("gz06/y", "Mdm2")),
    "p07": (
        ("p07/MisP", "misfolded protein"),
        # A free aggregate is sequestered or binds the proteasome within
        # minutes; what accumulates is the sum.
        (("p07/AggP", "p07/AggP_Proteasome", "p07/SeqAggP"), "aggregates"),
        ("p07/Ub", "free ubiquitin"),
    ),
}


def panel_paths(spec) -> tuple[str, ...]:
    return (spec,) if isinstance(spec, str) else tuple(spec)


MODEL_TITLES = {
    "dp14": "DallePezze 2014",
    "gz06": "Geva-Zatorsky 2006",
    "p07": "Proctor 2007",
}

POPULATION_MODELS = tuple(m for m in PANELS if m in REACTION_LEVEL)

# What the deposits declare, shown only where it is a unit. DallePezze:
# substance and volume dimensionless (its `_obs` rules carry the fitted
# scale to blot intensity). Geva-Zatorsky: `item`, a curation placeholder
# on levels the paper normalised to 0–1. Proctor: `item` on amount-only
# species, molecule counts. Time is the composite's clock, days.
Y_UNITS = {"dp14": None, "gz06": None, "p07": "molecules"}

# Dash serves this folder at /assets/. Each mark is looked up by stem in
# the formats below; one whose file is missing is simply not shown.
ASSETS = Path(__file__).resolve().parent / "assets"
LOGOS = (
    ("buck-logo", "Buck Institute for Research on Aging"),
    ("furmanlab", "Furman Lab"),
)
LOGO_FORMATS = (".svg", ".png", ".webp")


def logo_files() -> list[tuple[str, str]]:
    """``(file name, alt text)`` for every mark present in the assets."""
    found = []
    for stem, alt in LOGOS:
        for ext in LOGO_FORMATS:
            if (ASSETS / f"{stem}{ext}").exists():
                found.append((f"{stem}{ext}", alt))
                break
    return found


# A sample is one Gillespie chain per cell and the Scheduler runs a
# stochastic batch one host thread per member, so the sample costs a chain
# rather than the sum of them: 4 cells 11.9 s, 8 cells 12.2 s, 16 cells
# 12.1 s, 32 cells 12.8 s, against 10.1 s for a single chain, on 80 cores
# (docs/benchmarks.md section 6). Fewer cores than cells runs them in
# waves, so there the sample costs ceil(cells / cores) chains. Sixteen is
# what four cost on the vectorised lane, measured through the page: a move
# is 10.5 s at both 4 and 16 cells, 12.3 s at 32.
DEFAULT_CELLS = 16


def _unfreeze_panel_species(composite):
    """The composite with every panel species integrated: the import holds
    a species nothing reads at its initial value, and a panel reads it."""
    procs = dict(composite.processes)
    for name, proc in procs.items():
        frozen = getattr(proc, "_frozen_indices", ())
        if not frozen:
            continue
        held = {proc._species_names[i] for i in frozen}
        wanted = {
            p.split("/", 1)[1]
            for row in PANELS.values()
            for spec, _ in row
            for p in panel_paths(spec)
            if p.startswith(f"{name}/")
        }
        if held & wanted:
            procs[name] = proc.with_unfrozen(*sorted(held & wanted))
    if all(procs[n] is composite.processes[n] for n in procs):
        return composite
    return Composite(
        processes=procs,
        topology=composite.topology,
        validate=False,
        semantic_validation=False,
    )


def _series(ys, shown, cols):
    """One panel's series: the summed paths, over the shown times, any
    cell axis kept."""
    return ys[shown][..., cols].sum(axis=-1)


class LeverModel:
    """The composite behind the page for one exposure window, compiled once.

    ``solve(severities)`` returns the mean-field trajectory; ``population(
    severities)`` runs the reaction-level members as ``n_cells`` cells on the
    same grid, common random numbers across settings so a lever's effect is
    not confounded with the noise draw. The presets are sampled at
    construction and every sample is kept, so a setting is paid for once.
    ``n_cells=0`` skips the population and the reaction-level rows draw the
    mean field.
    """

    def __init__(
        self,
        rapa_intensity: float = RAPA_INTENSITY,
        n_cells: int = DEFAULT_CELLS,
        seed: int = 0,
        dose_window=DDIS_ETOPOSIDE_DOSE_WINDOW,
    ):
        self.dose_window = tuple(float(t) for t in dose_window)
        built = build_multi_hallmark_composite(dose_window=dose_window)
        self.base = _unfreeze_panel_species(built)
        self.registry = _registry_with_intensity(rapa_intensity)
        self.keys = list(self.base.store_keys())
        index = {k: i for i, k in enumerate(self.keys)}
        wanted = [
            p
            for row in PANELS.values()
            for spec, _ in row
            for p in panel_paths(spec)
        ]
        missing = [p for p in wanted if p not in index]
        if missing:
            raise KeyError(f"panel paths not in the composite: {missing}")
        self.columns = {
            model: [
                np.array([index[p] for p in panel_paths(spec)])
                for spec, _ in row
            ]
            for model, row in PANELS.items()
        }
        self.n_cells = int(n_cells)
        self.seed = int(seed)
        self._scheduler = Scheduler()
        self._solve = jax.jit(self._trajectory)
        t0 = time.perf_counter()
        self.ts, self.control = self.solve(np.zeros(len(LEVERS)))
        self.compile_seconds = time.perf_counter() - t0
        self._populations: dict[tuple, tuple] = {}
        self.population_compile_seconds = 0.0
        if self.n_cells:
            procs = dict(self.base.processes)
            for name in REACTION_LEVEL:
                procs[name] = procs[name].as_stochastic()
            self.stochastic_base = Composite(
                processes=procs,
                topology=self.base.topology,
                validate=False,
                semantic_validation=False,
            )
            # Eager: the Scheduler maps a stochastic batch over host
            # threads, and a jit around this would trace that lane away.
            self._sample = self._population
            t0 = time.perf_counter()
            # Serial: each sample already runs one host thread per cell, so
            # a pool over presets on top of that oversubscribes the box.
            for preset in PRESETS.values():
                self.population(preset)
            self.population_compile_seconds = time.perf_counter() - t0
        log.info(
            "window %s ready: %d states, mean field compiled in %.1f s, "
            "%d-cell population presets in %.1f s",
            self.dose_window,
            len(self.keys),
            self.compile_seconds,
            self.n_cells,
            self.population_compile_seconds,
        )

    @property
    def control_population(self) -> np.ndarray:
        return self._populations[PRESETS["control"]][0]

    def _severities(self, severities) -> dict[str, float]:
        return {
            lever.hallmark: severities[i] for i, lever in enumerate(LEVERS)
        }

    def _run_kwargs(self) -> dict:
        return dict(
            t_span=(-PREROLL_DAYS, MULTI_HALLMARK_GRID.t_end),
            macro_dt=MULTI_HALLMARK_GRID.macro_dt,
            save_dt=MULTI_HALLMARK_GRID.save_dt,
        )

    def _trajectory(self, severities):
        comp = with_hallmarks(
            self.base, self._severities(severities), registry=self.registry
        )
        res = self._scheduler.run(
            comp, y0=comp.initial_state_vec(), **self._run_kwargs()
        )
        return res.ts, res.ys

    def _population(self, severities, key):
        comp = with_hallmarks(
            self.stochastic_base,
            self._severities(severities),
            registry=self.registry,
        )
        y0 = comp.initial_state_vec()
        res = self._scheduler.run(
            comp,
            y0=jnp.tile(y0[None], (self.n_cells, 1)),
            key=key,
            **self._run_kwargs(),
        )
        events = jnp.stack(
            [res.stats[name]["num_events"] for name in REACTION_LEVEL]
        )
        return res.ys, events

    @lru_cache(maxsize=32)
    def _solve_cached(self, severities: tuple):
        ts, ys = self._solve(jnp.asarray(severities, dtype=jnp.float64))
        return np.asarray(ts), np.asarray(ys)

    def solve(self, severities):
        """``(ts, ys)`` as numpy, ``ys`` being ``(n_time, n_vars)``. Memoised
        on the severity vector, so the population request reuses the lever
        request's solve."""
        return self._solve_cached(self._key(severities))

    @staticmethod
    def _key(severities) -> tuple:
        return tuple(round(float(s), 6) for s in severities)

    def population_cached(self, severities):
        """The sample already taken at ``severities``, or ``None``."""
        return self._populations.get(self._key(severities))

    def population(self, severities):
        """``(ys, events_per_cell)``: ``ys`` is ``(n_time, n_cells, n_vars)``
        with the reaction-level members' species sampled per cell. Sampled
        once per setting and kept."""
        key = self._key(severities)
        if key not in self._populations:
            ys, events = self._sample(
                jnp.asarray(key, dtype=jnp.float64),
                jax.random.PRNGKey(self.seed),
            )
            self._populations[key] = (
                np.asarray(ys),
                float(np.asarray(events).mean()),
            )
        return self._populations[key]

    def moved(self, lever: Lever, severity: float) -> list[dict]:
        """What the lever set, for the mappings that target this composite:
        ``{target, control, now, description}`` per parameter."""
        handle = self.registry[lever.hallmark]
        procs = self.base.processes
        present = [m for m in handle.mappings if m.process_name in procs]
        at_control = handle.summary(0.0, procs)
        at_now = handle.summary(float(severity), procs)
        return [
            {
                "target": f"{m.process_name} · {m.param_name.split('.')[-1]}",
                "control": float(
                    at_control[f"{m.process_name}.{m.param_name}"]
                ),
                "now": float(at_now[f"{m.process_name}.{m.param_name}"]),
                "description": m.description,
            }
            for m in present
        ]


class LeverBank:
    """One :class:`LeverModel` per exposure window. The first window is
    built here, the rest on a background thread, so the page is up after
    one compile; a window still compiling reports ``None``."""

    def __init__(self, windows=DOSE_WINDOWS, **model_kwargs):
        self.windows = dict(windows)
        self._models: dict[str, LeverModel] = {}
        self._kwargs = model_kwargs
        first = next(iter(self.windows))
        self._models[first] = LeverModel(
            dose_window=self.windows[first], **model_kwargs
        )
        self.first = first
        self._thread = threading.Thread(target=self._build_rest, daemon=True)
        self._thread.start()

    def _build_rest(self):
        rest = [
            (name, window)
            for name, window in self.windows.items()
            if name not in self._models
        ]
        if not rest:
            return
        # Separate composites with separate compiles and separate solves,
        # so the remaining windows cost the slowest of them rather than
        # their sum. Each is published the moment it lands.
        with ThreadPoolExecutor(max_workers=len(rest)) as pool:
            pending = {
                pool.submit(
                    LeverModel, dose_window=window, **self._kwargs
                ): name
                for name, window in rest
            }
            for future in as_completed(pending):
                self._models[pending[future]] = future.result()

    def get(self, name: str) -> LeverModel | None:
        return self._models.get(name)

    def ready(self, name: str) -> bool:
        return name in self._models

    def wait(self):
        """Block until every window is built (tests, not the page)."""
        self._thread.join()


# ── page ─────────────────────────────────────────────────────────────────

# Design tokens: the one source for the stylesheet (assets/hallsim.css reads
# them as CSS custom properties, emitted into :root by the page) and for the
# Plotly figures.
TOKENS = {
    "font-heading": '"Barlow Condensed", "Arial Narrow", sans-serif',
    "font-body": 'Barlow, "Helvetica Neue", Arial, sans-serif',
    "font-mono": '"IBM Plex Mono", ui-monospace, Menlo, monospace',
    "color-text": "#1d1f20",
    "color-bg": "#fafafa",
    "color-neutral-400": "#b7b7ba",
    "color-neutral-700": "#5d5d60",
    "color-divider": "rgba(29,31,32,0.16)",
    "color-accent": "#5980a6",
    "color-accent-900": "#1d2d3d",
    "color-pulse": "#de8f05",
    "color-blue": "#123d63",
}
GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700"
    "&family=Barlow:ital,wght@0,400;0,500;1,400"
    "&family=IBM+Plex+Mono:wght@400;500&display=swap"
)
# Curves: DallePezze in the text ink, Geva-Zatorsky in the accent, Proctor
# in the data blue.
MODEL_COLORS = {
    "dp14": TOKENS["color-text"],
    "gz06": TOKENS["color-accent"],
    "p07": TOKENS["color-blue"],
}
CONTROL_COLOR = TOKENS["color-neutral-700"]
PULSE_COLOR = TOKENS["color-pulse"]
RAPA_COLOR = TOKENS["color-accent"]
CURVE_WIDTH = 1.8
# Control is true dots, the mean field long dashes, so the two read apart
# even over a noisy population mean.
CONTROL_LINE = dict(
    color=TOKENS["color-neutral-700"], width=1.4, dash="1px,3px"
)
MEAN_FIELD_DASH = "9px,5px"
TIME_RANGE = (0.0, MULTI_HALLMARK_GRID.t_end)
# Washout day and the two sampled days of the calibration data.
TIME_TICKS = (0, RAPA_INTERVENTION_DAY, 7, MULTI_HALLMARK_GRID.t_end)


def spread(cells):
    """``(lo, hi)`` edges of the band across the cell axis of ``cells``
    (``(n_time, n_cells)``): the mean ± one sample standard deviation,
    the lower edge held at zero since these are counts."""
    mean = cells.mean(axis=1)
    sd = cells.std(axis=1, ddof=1) if cells.shape[1] > 1 else 0.0
    return np.maximum(mean - sd, 0.0), mean + sd


HOVER = "day %{x:.1f}: %{y:.3g}<extra></extra>"


def root_css() -> str:
    """The tokens as CSS custom properties; the stylesheet in ``assets/``
    reads nothing else."""
    return ":root{" + ";".join(f"--{k}:{v}" for k, v in TOKENS.items()) + "}"


def _font(role: str, size: float, color: str = "color-text", **kw) -> dict:
    return dict(family=TOKENS[role], size=size, color=TOKENS[color], **kw)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _spans(lm: LeverModel, severities):
    """The shaded periods at this setting: ``(on, x0, x1, color, opacity)``
    for the etoposide window and the rapamycin period — a step held from
    its day to the end of the run, not a pulse."""
    gi, dns = severities[0], severities[1]
    return (
        (gi > 0, *lm.dose_window, PULSE_COLOR, 0.14),
        (dns != 0, RAPA_INTERVENTION_DAY, TIME_RANGE[1], RAPA_COLOR, 0.09),
    )


def legend_items(lm: LeverModel, severities):
    """The line styles, then a swatch and name for each shaded period that
    is on."""
    from dash import html

    def entry(sample, name):
        return html.Span([sample, name])

    items = [
        entry(html.Span(className="ln"), "current setting"),
        entry(html.Span(className="ln dotted"), "control"),
    ]
    if lm.n_cells:
        items.append(entry(html.Span(className="ln dashed"), "mean field"))
    names = ("etoposide", "rapamycin" if severities[1] < 0 else "mTORC1 drive")
    items += [
        entry(
            html.Span(
                className="sw", style={"background": _rgba(color, op * 2.5)}
            ),
            name,
        )
        for (on, _, _, color, op), name in zip(_spans(lm, severities), names)
        if on
    ]
    return items


def _row_figure(
    model: str, lm: LeverModel, titles: list[str], severities, height=210
):
    """An empty one-row figure for ``model``'s panels: titles, the
    etoposide window and the rapamycin period, each labelled in the first
    panel, the shared axes."""
    from plotly.subplots import make_subplots

    n = len(PANELS[model])
    fig = make_subplots(
        rows=1, cols=n, subplot_titles=titles, horizontal_spacing=0.045
    )
    # The traces come later, so the spans must not skip empty subplots.
    # Names live in the page legend, not in the panels.
    for on, x0, x1, color, opacity in _spans(lm, severities):
        if not on:
            continue
        for col in range(1, n + 1):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=color,
                opacity=opacity,
                line_width=0,
                row=1,
                col=col,
                exclude_empty_subplots=False,
            )
    axis = dict(
        showline=True,
        linecolor=TOKENS["color-neutral-400"],
        linewidth=1,
        ticks="",
        showgrid=False,
        zeroline=False,
        tickfont=_font("font-mono", 10, "color-neutral-700"),
    )
    title = dict(
        title_font=_font("font-mono", 10, "color-neutral-700"),
        title_standoff=4,
    )
    fig.update_xaxes(
        range=list(TIME_RANGE),
        tickvals=TIME_TICKS,
        title_text="days",
        **title,
        **axis,
    )
    fig.update_yaxes(rangemode="tozero", **axis)
    if Y_UNITS[model]:
        fig.update_yaxes(title_text=Y_UNITS[model], **title, row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=52, r=10, t=30, b=40),
        font=_font("font-body", 11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # One quiet readout per panel: only the main line answers a hover,
        # nearest point from anywhere in the panel, no label on the axis.
        hovermode="closest",
        hoverdistance=-1,
        hoverlabel=dict(
            bgcolor=TOKENS["color-bg"],
            bordercolor=TOKENS["color-neutral-400"],
            font=_font("font-mono", 11),
        ),
        showlegend=False,
    )
    fig.update_annotations(
        font=_font("font-heading", 11.5, weight=500),
        selector=dict(xref="paper"),
    )
    return fig


def _figure_for(model: str, lm: LeverModel, ys, severities):
    """Mean-field row: control dotted, this setting solid."""
    import plotly.graph_objects as go

    row = PANELS[model]
    color = MODEL_COLORS[model]
    fig = _row_figure(model, lm, [label for _, label in row], severities)
    shown = lm.ts >= TIME_RANGE[0]
    t = lm.ts[shown]
    for j, cols in enumerate(lm.columns[model]):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=_series(lm.control, shown, cols),
                line=CONTROL_LINE,
                hoverinfo="skip",
            ),
            row=1,
            col=j + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=_series(ys, shown, cols),
                line=dict(color=color, width=CURVE_WIDTH),
                hovertemplate=HOVER,
            ),
            row=1,
            col=j + 1,
        )
    return fig


def _band(fig, t, cells, color, name, row, col):
    """A filled band across ``cells`` (``(n_time, n_cells)``), edges from
    :func:`spread`: two traces, the upper edge then the lower one filled up
    to it."""
    import plotly.graph_objects as go

    lo, hi = spread(cells)
    for edge, fill in ((hi, None), (lo, "tonexty")):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=edge,
                mode="lines",
                line=dict(width=0),
                fill=fill,
                fillcolor=color,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _population_figure(model: str, lm: LeverModel, ys_pop, ys_mf, severities):
    """Reaction-level row: the spread across cells as a band, the population
    mean bold, the mean field dashed, the control population in grey."""
    import plotly.graph_objects as go

    row = PANELS[model]
    color = MODEL_COLORS[model]
    fig = _row_figure(
        model, lm, [label for _, label in row], severities, height=240
    )
    shown = lm.ts >= TIME_RANGE[0]
    t = lm.ts[shown]
    for j, cols in enumerate(lm.columns[model]):
        control_cells = _series(lm.control_population, shown, cols)
        cells = _series(ys_pop, shown, cols)
        _band(
            fig,
            t,
            control_cells,
            _rgba(CONTROL_COLOR, 0.18),
            "control",
            1,
            j + 1,
        )
        _band(fig, t, cells, _rgba(color, 0.22), "this setting", 1, j + 1)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=control_cells.mean(axis=1),
                line=CONTROL_LINE,
                hoverinfo="skip",
            ),
            row=1,
            col=j + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=_series(ys_mf, shown, cols),
                line=dict(color=color, width=1.5, dash=MEAN_FIELD_DASH),
                hoverinfo="skip",
            ),
            row=1,
            col=j + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=cells.mean(axis=1),
                line=dict(color=color, width=CURVE_WIDTH),
                hovertemplate=HOVER,
            ),
            row=1,
            col=j + 1,
        )
    return fig


def _mark_pending(fig, text: str):
    """Fade a row and stamp it: what it should show is on its way."""
    fig.update_traces(opacity=0.35)
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=_font("font-heading", 15, weight=500),
        bgcolor="rgba(250,250,250,0.85)",
        borderpad=6,
    )
    return fig


def _badge(text: str, pending: bool = False):
    """The population row's header badge; pending ones carry a spinner."""
    from dash import html

    if pending:
        return html.Span(
            [html.Span(className="spin"), text], className="pop pending"
        )
    return html.Span(text, className="pop")


def population_badge(lm: LeverModel | None, sev):
    if lm is None:
        return _badge("compiling…", pending=True)
    if not lm.n_cells:
        return _badge("mean field")
    if lm.population_cached(sev) is None:
        return _badge("sampling…", pending=True)
    return _badge("")


def render(lm: LeverModel | None, fallback: LeverModel, *severities):
    """Everything the lever request shows for one severity vector, in the
    order the callback's outputs are declared: one figure per publication,
    one value per lever, the population badge, the legend for the shaded
    periods, the scenario chips' classes. ``lm`` is the chosen window's model, or
    ``None`` while it compiles, in which case ``fallback``'s rows are drawn
    faded and marked."""
    sev = np.asarray(severities, dtype=float)
    model = lm or fallback
    _, ys = model.solve(sev)
    figures = []
    for name in PANELS:
        sample = model.population_cached(sev) if model.n_cells else None
        if name in POPULATION_MODELS and sample is not None:
            fig = _population_figure(name, model, sample[0], ys, sev)
        else:
            fig = _figure_for(name, model, ys, sev)
            if name in POPULATION_MODELS and model.n_cells:
                _mark_pending(fig, "sampling…")
        if lm is None:
            _mark_pending(fig, "compiling…")
        figures.append(fig)
    values = [
        f"{s:+.2f}" if lever.lo < 0 else f"{s:.2f}"
        for lever, s in zip(LEVERS, sev)
    ]
    return (
        *figures,
        *values,
        population_badge(lm, sev),
        legend_items(model, sev),
        chip_classes(sev),
    )


def chip_classes(sev) -> list[str]:
    """One class per scenario chip, the one matching this setting selected."""
    key = LeverModel._key(sev)
    return [
        "chip selected" if LeverModel._key(p) == key else "chip"
        for p in PRESETS.values()
    ]


def render_population(lm: LeverModel, *severities):
    """The reaction-level rows for one severity vector, sampled if they
    have not been: one figure per population publication, then the badge."""
    sev = np.asarray(severities, dtype=float)
    _, ys_mf = lm.solve(sev)
    ys_pop, _ = lm.population(sev)
    figures = [
        _population_figure(m, lm, ys_pop, ys_mf, sev)
        for m in POPULATION_MODELS
    ]
    return (*figures, population_badge(lm, sev))


def build_app(bank: LeverBank):
    try:
        from dash import ALL, Dash, Input, Output, ctx, dcc, html, no_update
    except ImportError as e:  # pragma: no cover - install hint
        raise SystemExit(
            'The lever page needs Dash: pip install "hallsim[app]"'
        ) from e

    first = bank.get(bank.first)
    # The stylesheet is assets/hallsim.css, served by Dash with the marks;
    # the page only emits the tokens it is written against.
    app = Dash(__name__, title="hallsim", external_stylesheets=[GOOGLE_FONTS])
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{root_css()}</style></head>"
    )

    def lever_card(i: int, lever: Lever):
        handle = first.registry[lever.hallmark]
        marks = {
            v: {"label": ""}
            for v in np.round(np.linspace(lever.lo, lever.hi, 5), 2)
        }
        return html.Div(
            className="card",
            children=[
                html.H3(
                    [
                        lever.label,
                        html.Span(id=f"value-{i}", className="val"),
                    ],
                    title=f"{lever.hallmark}: {handle.description}",
                ),
                dcc.Slider(
                    id=f"lever-{i}",
                    min=lever.lo,
                    max=lever.hi,
                    step=0.05,
                    value=PRESETS[DEFAULT_PRESET][i],
                    marks=marks,
                    tooltip={"placement": "bottom"},
                    updatemode="mouseup",
                ),
            ],
        )

    def logos():
        return [
            html.Img(src=app.get_asset_url(name), alt=alt)
            for name, alt in logo_files()
        ]

    def segment_state(selected: str):
        """Class and disabled flag per exposure segment, and the compile
        count for the line beneath them."""
        classes, disabled = [], []
        for name in bank.windows:
            ready = bank.ready(name)
            cls = "seg"
            if name == selected:
                cls += " selected"
            if not ready:
                cls += " compiling"
            classes.append(cls)
            disabled.append(not ready)
        pending = sum(1 for name in bank.windows if not bank.ready(name))
        note = f"{pending} of {len(bank.windows)} compiling" if pending else ""
        return classes, disabled, note

    def segment(name: str):
        top, bottom = WINDOW_LABELS[name]
        return html.Button(
            [
                html.Span(top, className="seg-top"),
                html.Span(bottom, className="seg-sub"),
            ],
            id={"window": name},
            className="seg",
        )

    def model_row(model: str):
        heading = [
            MODEL_TITLES[model],
            html.Span(f"  {MODEL_BLOCKS[model]['deposit']}", className="dep"),
        ]
        if model in POPULATION_MODELS:
            heading.append(html.Span(id="population-status"))
        graph = dcc.Graph(
            id=f"panel-{model}", config={"displayModeBar": False}
        )
        return html.Div(className="row", children=[html.H2(heading), graph])

    app.layout = html.Div(
        className="wrap",
        children=[
            html.Div(
                className="side",
                children=[
                    html.H1("hallsim"),
                    html.Div("hallmarks of aging simulator", className="tag"),
                    html.Div(
                        [
                            html.Button(
                                name, id={"preset": name}, className="chip"
                            )
                            for name in PRESETS
                        ]
                    ),
                    *[lever_card(i, lever) for i, lever in enumerate(LEVERS)],
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Etoposide exposure"),
                            html.Div(
                                [segment(name) for name in bank.windows],
                                className="seg-group",
                            ),
                            html.Div(id="window-note", className="seg-note"),
                            dcc.Store(id="window", data=bank.first),
                            dcc.Interval(
                                id="window-poll", interval=2000, n_intervals=0
                            ),
                        ],
                    ),
                    html.Div(className="logos", children=logos()),
                ],
            ),
            html.Div(
                className="main",
                children=[
                    html.Div(id="legend", className="legend"),
                    *[model_row(model) for model in PANELS],
                ],
            ),
        ],
    )

    lever_inputs = [Input(f"lever-{i}", "value") for i in range(len(LEVERS))]
    window_input = Input("window", "data")

    @app.callback(
        Output("window", "data"),
        Input({"window": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def choose_window(_clicks):
        return ctx.triggered_id["window"]

    @app.callback(
        [Output(f"lever-{i}", "value") for i in range(len(LEVERS))],
        Input({"preset": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def apply_preset(_clicks):
        name = ctx.triggered_id["preset"]
        return list(PRESETS[name])

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
        classes, disabled, note = segment_state(selected)
        return classes, disabled, note, not any(disabled)

    @app.callback(
        [Output(f"panel-{m}", "figure") for m in PANELS]
        + [Output(f"value-{i}", "children") for i in range(len(LEVERS))]
        + [
            Output("population-status", "children"),
            Output("legend", "children"),
            Output({"preset": ALL}, "className"),
        ],
        [window_input, *lever_inputs],
    )
    def on_pull(window, *severities):
        return render(bank.get(window), first, *severities)

    if first.n_cells:
        # Fires alongside the lever request and replaces the marked
        # mean-field row with the sample once it exists; instant for a
        # setting already sampled. A window still compiling changes nothing.
        @app.callback(
            [
                Output(f"panel-{m}", "figure", allow_duplicate=True)
                for m in POPULATION_MODELS
            ]
            + [Output("population-status", "children", allow_duplicate=True)],
            [window_input, *lever_inputs],
            prevent_initial_call=True,
        )
        def on_pull_population(window, *severities):
            lm = bank.get(window)
            if lm is None:
                return (no_update,) * (len(POPULATION_MODELS) + 1)
            return render_population(lm, *severities)

    return app


def main(
    port: int = 8050,
    host: str = "127.0.0.1",
    debug: bool = False,
    cells: int = DEFAULT_CELLS,
    seed: int = 0,
):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info(
        "building the composite and compiling the solves (about a minute; "
        "the other exposure windows follow in the background)…"
    )
    bank = LeverBank(n_cells=cells, seed=seed)
    app = build_app(bank)
    log.info("serving on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
