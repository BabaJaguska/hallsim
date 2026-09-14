"""The lever page renders what it claims: every deterministic panel carries
the etoposide reference and a current trace, the population row carries the spread band and
the three summary lines, a setting not yet sampled is drawn faded and
marked, a pulled lever moves the parameters it names, and the exposure
window is shaded. Bound to the multi-hallmark composite, so it is a demo
test and a slow one."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dash")
pytest.importorskip("plotly")

from demos.hallmark_levers import (  # noqa: E402
    DOSE_WINDOWS,
    LEVERS,
    PANELS,
    POPULATION_MODELS,
    PRESETS,
    LeverBank,
    LeverModel,
    build_app,
    population_badge,
    render,
    render_population,
)

pytestmark = [pytest.mark.demo, pytest.mark.slow]


@pytest.fixture(scope="module")
def lever_model():
    return LeverModel(n_cells=2)


def _annotation_texts(fig) -> list[str]:
    return [a.text or "" for a in fig.layout.annotations]


def test_etoposide_starts_p53_pulsing(lever_model):
    lm = lever_model
    col = lm.keys.index("gz06/x")
    _, ys = lm.solve(PRESETS["etoposide"])
    _, ys_zero = lm.solve(np.zeros(len(LEVERS)))
    late = lm.ts >= 7.0
    swing_ctrl = np.ptp(ys_zero[late, col])
    swing_ddis = np.ptp(ys[late, col])
    assert swing_ddis > 10 * max(swing_ctrl, 1e-9)


def test_render_draws_every_panel(lever_model):
    """A sampled preset's lever request draws the population row; an
    unsampled setting draws the etoposide population alone, marked, and the
    badge says so. The presets are sampled in the background so the page
    paints without them, which a test has to wait for."""
    lm = lever_model
    lm.wait_for_presets()
    n_models, n_levers = len(PANELS), len(LEVERS)
    out = render(lm, lm, *PRESETS["etoposide + rapamycin"])
    assert len(out) == n_models + n_levers + 3
    figures = out[:n_models]
    values = out[n_models : n_models + n_levers]
    badge, legend, chips = out[-3], out[-2], out[-1]
    assert chips == ["chip", "chip selected"]
    for model, fig in zip(PANELS, figures):
        per_panel = 6 if model in POPULATION_MODELS else 2
        assert len(fig.data) == per_panel * len(PANELS[model])
        for trace in fig.data:
            assert np.isfinite(np.asarray(trace.y, dtype=float)).all()
        # Etoposide and rapamycin are on: both periods shaded in every
        # panel, named only in the legend.
        assert len(fig.layout.shapes) == 2 * len(PANELS[model])
        assert _annotation_texts(fig) == [label for _, label in PANELS[model]]
    assert list(values) == ["1.00", "-1.00", "0.00"]
    assert badge.className == "pop"
    assert badge.children == ""
    names = [item.children[-1] for item in legend]
    assert names[:2] == ["current setting", "etoposide alone"]
    assert names[-2:] == ["etoposide pulse", "rapamycin treatment"]

    out = render(lm, lm, 0.5, 0.0, 0.0)
    for model, fig in zip(PANELS, out[:n_models]):
        # an unsampled population row: the reference band and its mean only
        per_panel = 3 if model in POPULATION_MODELS else 2
        assert len(fig.data) == per_panel * len(PANELS[model])
        assert len(fig.layout.shapes) == len(PANELS[model])
        texts = _annotation_texts(fig)
        assert ("sampling…" in texts) == (model in POPULATION_MODELS)
    badge, legend, chips = out[-3], out[-2], out[-1]
    assert badge.className == "pop pending"
    assert badge.children[-1] == "sampling…"
    names = [item.children[-1] for item in legend]
    assert (
        names[-1] == "etoposide pulse" and "rapamycin treatment" not in names
    )
    assert chips == ["chip", "chip"]

    # A window still compiling: the fallback's rows, all marked.
    out = render(None, lm, 0.0, 0.0, 0.0)
    for fig in out[:n_models]:
        assert "compiling…" in _annotation_texts(fig)
        assert len(fig.layout.shapes) == 0
    assert out[-3].children[-1] == "compiling…"
    names = [item.children[-1] for item in out[-2]]
    assert not {
        "etoposide pulse",
        "rapamycin treatment",
        "mTORC1 drive",
    } & set(names)
    assert out[-1] == ["chip", "chip"]


def test_population_row_carries_the_spread(lever_model):
    lm = lever_model
    # The page starts a sample and draws it when it lands; a test says when.
    lm.wait_for_presets()
    lm.population((0.5, 0.0, 0.0))
    *figures, badge = render_population(lm, 0.5, 0.0, 0.0)
    assert len(figures) == len(POPULATION_MODELS)
    # Per panel: two band edges for the reference, two for this setting,
    # then the reference mean and the population mean.
    for model, fig in zip(POPULATION_MODELS, figures):
        assert len(fig.data) == 6 * len(PANELS[model])
    # The member is coupled: with ROS read from DallePezze it misfolds, so
    # free ubiquitin is drawn down from the deposit's full pool in every cell.
    ub = lm.keys.index("p07/Ub")
    ys, events = lm.population((0.5, 0.0, 0.0))
    assert events > 1e5
    assert (ys[-1, :, ub] < 300).all()
    assert badge.className == "pop"
    # Sampled once: a second request for the same setting is the same
    # object.
    assert lm.population((0.5, 0.0, 0.0)) is lm.population_cached(
        (0.5, 0.0, 0.0)
    )


def test_a_narrow_viewport_wraps_the_panels(lever_model):
    """At three panels across, the five-panel row becomes two rows: same
    traces and titles, a taller figure."""
    lm = lever_model
    wide = render(lm, lm, *PRESETS["etoposide"])[0]
    narrow = render(lm, lm, *PRESETS["etoposide"], ncols=3)[0]
    assert len(narrow.data) == len(wide.data)
    assert _annotation_texts(narrow) == _annotation_texts(wide)
    assert narrow.layout.height > wide.layout.height


def test_moved_reports_the_registry_targets(lever_model):
    lm = lever_model
    gi, dns, lop = LEVERS
    assert [m["target"] for m in lm.moved(gi, 1.0)] == [
        "irradiation_pulse · amplitude"
    ]
    (rapa,) = lm.moved(dns, -1.0)
    assert rapa["target"] == "rapamycin_drive · after"
    assert rapa["now"] < rapa["published"]
    (k69,) = lm.moved(lop, 1.0)
    assert k69["now"] == 0.0


def test_a_new_setting_does_not_hold_the_request_open(lever_model):
    """A sample is seconds of serial event loop. The lever request starts it
    and returns, so the browser is not held for it; the page's poll draws the
    rows when the sample lands.

    Discrimination: sampling inside the request returns the rows on the first
    call, which the second assertion rejects.
    """
    import time

    lm = lever_model
    sev = (0.35, 0.1, 0.0)
    assert lm.population_cached(sev) is None
    assert render_population(lm, *sev) is None

    deadline = time.monotonic() + 600
    while lm.population_cached(sev) is None:
        assert time.monotonic() < deadline, "the sample never landed"
        time.sleep(0.5)
    rows = render_population(lm, *sev)
    assert rows is not None
    *figures, _ = rows
    assert len(figures) == len(POPULATION_MODELS)


def test_the_page_says_it_is_working_while_it_is(tmp_path):
    """Every panel sits under a spinner and the population row carries a
    pending badge, so a row that has not been drawn yet reads as working
    rather than as broken. The poll and the store it compares against have
    to be in the layout or the band never arrives.

    One window: the bank builds the first synchronously and the rest on a
    thread, and this needs neither the rest nor a sample.
    """
    from dash import dcc

    first = next(iter(DOSE_WINDOWS.items()))
    bank = LeverBank(windows=dict([first]), n_cells=2)
    app = build_app(bank)

    def children(node):
        kids = getattr(node, "children", None)
        if kids is None:
            return []
        return list(kids) if isinstance(kids, (list, tuple)) else [kids]

    spinners, ids = {}, set()

    def walk(node):
        if getattr(node, "id", None) is not None:
            ids.add(str(node.id))
        for child in children(node):
            if isinstance(node, dcc.Loading) and isinstance(child, dcc.Graph):
                spinners[child.id] = node
            walk(child)

    walk(app.layout)
    for model in PANELS:
        wrap = spinners.get(f"panel-{model}")
        assert wrap is not None, f"panel-{model} draws with no spinner"
        # Instant callbacks must not flash it: the population poll answers
        # once a second and usually with nothing.
        assert wrap.delay_show > 0
    assert {"population-poll", "drawn", "window-poll"} <= ids

    lm = bank.get(bank.first)
    unsampled = population_badge(lm, (0.35, 0.1, 0.0))
    assert "pending" in unsampled.className
    assert "pending" in population_badge(None, (0.0,) * len(LEVERS)).className
