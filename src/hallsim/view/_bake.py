"""A page baked to a static site: every lever setting on a grid, solved
once and written as JSON beside a page that reads them back.

:func:`bake` is the opt-in alternative to :func:`hallsim.view.serve` for
a page that will live on a static host. Nothing runs after the bake, so
the sliders snap to the grid; the lever rows are what it writes, and the
wiring tab stays with the server.
"""

from __future__ import annotations

import html
import itertools
import json
import logging
import re
import shutil
import time
from pathlib import Path

import numpy as np

from hallsim.view._levers import row_figure, shades_of, spread
from hallsim.view._model import ModelBank, ViewModel, series
from hallsim.view._page import Page
from hallsim.view._theme import (
    CSS,
    CURVE_WIDTH,
    GOOGLE_FONTS,
    HOVER,
    REFERENCE_LINE,
    TOKENS,
    color,
    rgba,
    root_css,
)

log = logging.getLogger(__name__)

#: Panel columns a row is laid out in, wide screens first; the served page
#: picks between the same two by viewport width, and so does the baked one.
COLUMN_BUCKETS = (5, 2)
#: Significant digits a series is written at.
SIGNIFICANT = 5
_HERE = Path(__file__).parent


def clean(value) -> float:
    """A setting value as the page keys it: six decimals, no minus zero."""
    return round(float(value), 6) + 0.0


def setting_key(severities) -> str:
    """The file stem of one setting; ``bake.js`` computes the same."""
    return "_".join(f"{clean(v):.2f}" for v in severities)


def slug(name: str) -> str:
    """A variant name as a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "variant"


def grid(page: Page, step: float) -> list[tuple[float, ...]]:
    """Every setting on the lever grid at ``step``, a preset added when it
    falls between."""
    axes = []
    for lever in page.levers:
        n = int(round((lever.hi - lever.lo) / step))
        axes.append([clean(lever.lo + i * step) for i in range(n + 1)])
    settings = [tuple(s) for s in itertools.product(*axes)]
    seen = set(settings)
    for preset in page.presets.values():
        p = tuple(clean(v) for v in preset)
        if p not in seen:
            settings.append(p)
            seen.add(p)
    return settings


def compact(values) -> list:
    """A series at ``SIGNIFICANT`` digits of its largest value; a value
    that is not finite is written as null."""
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    top = float(np.max(np.abs(a[finite]))) if finite.any() else 0.0
    if top > 0.0:
        digits = SIGNIFICANT - 1 - int(np.floor(np.log10(top)))
        a = np.round(a, max(digits, 0))
    return [float(v) if ok else None for v, ok in zip(a, finite)]


def _setting_rows(page: Page, vm: ViewModel, ys, sample) -> dict:
    """One setting's panel series: ``series[row][j]`` for a deterministic
    row; ``population[row][j] = {lo, mean, hi}`` across the cells for a
    population row."""
    shown = vm.ts >= 0.0
    out_series: dict[str, list] = {}
    out_pop: dict[str, list] = {}
    for name in page.panels:
        cols = vm.columns[name]
        if name in page.population and sample is not None:
            rows = []
            for c in cols:
                cells = series(sample, shown, c)
                lo, hi = spread(cells)
                rows.append(
                    {
                        "lo": compact(lo),
                        "mean": compact(cells.mean(axis=1)),
                        "hi": compact(hi),
                    }
                )
            out_pop[name] = rows
        else:
            out_series[name] = [compact(series(ys, shown, c)) for c in cols]
    return {"series": out_series, "population": out_pop}


def _layouts(page: Page, variant: str, columns) -> dict:
    """The row figures' layouts per column count, every shaded period in
    under its name so the page keeps the ones a setting switches on."""
    import plotly.io as pio

    zero = tuple(0.0 for _ in page.levers)
    out: dict[str, dict] = {}
    for ncols in columns:
        out[str(ncols)] = {}
        for name in page.panels:
            tall = name in page.population and page.cells
            fig = row_figure(
                page,
                name,
                zero,
                variant,
                height=240 if tall else 210,
                ncols=ncols,
                all_shades=True,
            )
            out[str(ncols)][name] = json.loads(pio.to_json(fig))["layout"]
    return out


def _index(page: Page, step: float, columns) -> dict:
    rows = []
    for name in page.panels:
        col = color(page.colors[name])
        rows.append(
            {
                "name": name,
                "title": page.titles[name],
                "deposit": page.deposits.get(name, ""),
                "color": col,
                "band": rgba(col, 0.22),
                "population": bool(name in page.population and page.cells),
                "panels": [pn.label for pn in page.panels[name]],
            }
        )
    levers = []
    for lever in page.levers:
        handle = page.registry.get(lever.handle)
        levers.append(
            {
                "name": lever.name,
                "handle": lever.handle,
                "lo": lever.lo,
                "hi": lever.hi,
                "signed": lever.lo < 0,
                "description": getattr(handle, "description", "") or "",
            }
        )
    shades = {
        slug(variant): [
            {
                "label": sh.label,
                "negative_label": sh.negative_label,
                "lever": sh.lever,
                "when": sh.when,
                "color": rgba(color(sh.color), sh.opacity * 2.5),
            }
            for sh in shades_of(page, variant)
        ]
        for variant in page.variants
    }
    return {
        "title": page.title,
        "tagline": page.tagline,
        "about": list(page.about),
        "logos": [[f, alt] for f, alt in page.logos],
        "levers": levers,
        "step": step,
        "presets": {
            k: [clean(v) for v in vals] for k, vals in page.presets.items()
        },
        "reference": page.reference,
        "reference_label": page.reference_label,
        "variants": [
            {
                "name": v,
                "slug": slug(v),
                "top": page.variant_label(v)[0],
                "sub": page.variant_label(v)[1],
            }
            for v in page.variants
        ],
        "variant_title": page.variant_title,
        "rows": rows,
        "shades": shades,
        "layouts": {
            slug(v): _layouts(page, v, columns) for v in page.variants
        },
        "style": {
            "reference_line": REFERENCE_LINE,
            "curve_width": CURVE_WIDTH,
            "hover": HOVER,
            "control_band": rgba(TOKENS["color-neutral-700"], 0.18),
        },
        "columns": list(columns),
    }


def _write_page(page: Page, out: Path) -> None:
    import plotly

    plotly_js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    shutil.copy(plotly_js, out / "plotly.min.js")
    shutil.copy(_HERE / "bake.js", out / "bake.js")
    text = (_HERE / "bake.html").read_text()
    text = (
        text.replace("{{TITLE}}", html.escape(page.title))
        .replace("{{FONTS}}", GOOGLE_FONTS)
        .replace("{{CSS}}", root_css() + CSS)
    )
    (out / "index.html").write_text(text)
    if page.logos and page.assets_folder:
        (out / "assets").mkdir(exist_ok=True)
        for f, _ in page.logos:
            src = Path(page.assets_folder) / f
            if src.exists():
                shutil.copy(src, out / "assets" / f)


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")))


def bake(
    page: Page,
    out_dir,
    *,
    step: float = 0.25,
    columns=COLUMN_BUCKETS,
    quiet: bool = False,
) -> Path:
    """Write ``page`` as a static site under ``out_dir``: ``index.html``
    with its script and plotly.js; ``index.json`` (levers, presets, rows,
    shaded periods, the row layouts); ``reference/<variant>.json``; and
    ``data/<variant>/<setting>.json`` for every lever setting on the grid
    at ``step``. A population row is sampled at the page's cell count and
    written as the band's edges and mean. Returns the directory."""
    if step <= 0:
        raise ValueError("step must be positive")
    from tqdm import tqdm

    out = Path(out_dir)
    settings = grid(page, step)
    bank = ModelBank(page)
    bank.wait()
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(exist_ok=True)
    total = len(settings) * len(page.variants)
    log.info(
        "baking %d settings x %d variant(s)", len(settings), len(page.variants)
    )
    started = time.perf_counter()
    done = 0
    bar = tqdm(total=total, disable=quiet, unit="setting")
    for variant in page.variants:
        vm = bank.get(variant)
        folder = out / "data" / slug(variant)
        folder.mkdir(exist_ok=True)
        for sev in settings:
            _, ys = vm.solve(sev)
            sample = vm.population(sev)[0] if vm.n_cells else None
            _dump(
                folder / f"{setting_key(sev)}.json",
                _setting_rows(page, vm, ys, sample),
            )
            vm.discard_population(sev)
            done += 1
            bar.update(1)
            if done == 1:
                per = time.perf_counter() - started
                log.info(
                    "first setting %.1f s; about %.0f min for all %d",
                    per,
                    per * total / 60,
                    total,
                )
        ref = page.presets[page.reference]
        _, ys = vm.solve(ref)
        sample = vm.population(ref)[0] if vm.n_cells else None
        _dump(
            out / "reference" / f"{slug(variant)}.json",
            {
                "t": compact(vm.ts[vm.ts >= 0.0]),
                **_setting_rows(page, vm, ys, sample),
            },
        )
    bar.close()
    _dump(out / "index.json", _index(page, step, columns))
    _write_page(page, out)
    log.info(
        "baked %d settings to %s in %.0f s",
        total,
        out,
        time.perf_counter() - started,
    )
    return out
