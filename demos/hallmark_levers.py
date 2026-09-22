"""Hallmark levers: pull a hallmark severity and watch three published
models re-solve.

``simulate demo hallmark-levers`` serves the multi-hallmark composite (Dalle
Pezze 2014, Geva-Zatorsky 2006, Proctor 2007) through :mod:`hallsim.view`:
one slider per hallmark of aging, every pull applied through
:func:`hallsim.handles.with_handles` and re-solved by the Scheduler, every
panel drawn against the etoposide arm. The etoposide exposure window and
the rapamycin period are shaded; a switch offers longer exposure windows,
each its own compiled composite. Proctor 2007 is drawn as a population of
cells at reaction level with the population mean over it. ``--cells 0``
draws its mean field instead.

Needs the ``app`` extra: ``pip install "hallsim[app]"``.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
from hallsim.view import Lever, Page, Shade, panel

log = logging.getLogger(__name__)

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

PRESETS = {
    "etoposide": (1.0, 0.0, 0.0),
    "etoposide + rapamycin": (1.0, -1.0, 0.0),
}
REFERENCE_PRESET = "etoposide"

DOSE_WINDOWS = {
    "days 0–2": DDIS_ETOPOSIDE_DOSE_WINDOW,
    "days 0–7": (0.0, 7.0),
}
WINDOW_LABELS = {
    "days 0–2": ("0–2", "days"),
    "days 0–7": ("0–7", "days"),
}

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
        (("p07/AggP", "p07/AggP_Proteasome", "p07/SeqAggP"), "aggregates"),
    ),
}
MODEL_TITLES = {
    "dp14": "Dalle Pezze 2014",
    "gz06": "Geva-Zatorsky 2006",
    "p07": "Proctor 2007",
}
MODEL_COLORS = {
    "dp14": "color-text",
    "gz06": "color-accent",
    "p07": "color-blue",
}
# Shown only where the deposit declares a unit: Proctor's species are
# molecule counts; the other two are dimensionless or placeholders.
Y_UNITS = {"dp14": None, "gz06": None, "p07": "molecules"}
POPULATION_MODELS = tuple(m for m in PANELS if m in REACTION_LEVEL)
# Sixteen cells cost what four do on the threaded batch lane.
DEFAULT_CELLS = 16

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


def shades(dose_window) -> tuple[Shade, ...]:
    return (
        Shade(
            "etoposide pulse",
            *dose_window,
            lever="Genomic Instability",
            when="positive",
            color="color-pulse",
            opacity=0.14,
        ),
        Shade(
            "mTORC1 drive",
            RAPA_INTERVENTION_DAY,
            None,
            lever="Deregulated Nutrient Sensing",
            when="nonzero",
            negative_label="rapamycin treatment",
            color="color-accent",
            opacity=0.09,
        ),
    )


def page(
    cells: int = DEFAULT_CELLS,
    seed: int = 0,
    rapa_intensity: float = RAPA_INTENSITY,
    windows: dict | None = None,
) -> Page:
    """The lever page's configuration over the multi-hallmark composite."""
    windows = dict(windows or DOSE_WINDOWS)
    return Page(
        variants={
            name: build_multi_hallmark_composite(dose_window=w)
            for name, w in windows.items()
        },
        registry=_registry_with_intensity(rapa_intensity),
        levers=LEVERS,
        t_end=MULTI_HALLMARK_GRID.t_end,
        macro_dt=MULTI_HALLMARK_GRID.macro_dt,
        save_dt=MULTI_HALLMARK_GRID.save_dt,
        preroll=PREROLL_DAYS,
        presets=PRESETS,
        reference=REFERENCE_PRESET,
        reference_label="etoposide alone",
        panels={
            m: tuple(panel(spec, label) for spec, label in row)
            for m, row in PANELS.items()
        },
        titles=MODEL_TITLES,
        deposits={m: MODEL_BLOCKS[m]["deposit"] for m in PANELS},
        units=Y_UNITS,
        colors=MODEL_COLORS,
        population=POPULATION_MODELS,
        cells=cells,
        seed=seed,
        shades={name: shades(w) for name, w in windows.items()},
        ticks=(0, RAPA_INTERVENTION_DAY, 7, MULTI_HALLMARK_GRID.t_end),
        variant_title="Etoposide exposure",
        variant_labels=WINDOW_LABELS,
        title="hallsim",
        tagline="hallmarks of aging simulator",
        about=(
            "Three published models composed into one system.",
            "DNA damage-induced senescence and its rapamycin rescue follow "
            "the experiment of Tighanimine et al. 2024 (GSE248823): "
            "etoposide for two days, then rapamycin from day 2.",
        ),
        logos=tuple(logo_files()),
        assets_folder=str(ASSETS),
    )


def main(
    port: int = 8050,
    host: str = "127.0.0.1",
    debug: bool = False,
    cells: int = DEFAULT_CELLS,
    seed: int = 0,
    bake: str | None = None,
    step: float = 0.25,
):
    """Serve the page, or with ``bake`` write it as a static site: every
    slider setting on a grid at ``step``, solved once. Returns the site's
    directory when baking."""
    from hallsim.view import bake as bake_page
    from hallsim.view import serve

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    built = page(cells=cells, seed=seed)
    if bake:
        return bake_page(built, bake, step=step)
    serve(built, host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
