"""A page derived from any composite: levers from the registry, a panel per
written state, the wiring graph collapsed and opened, a trace overlaid, and a
saved run rendered."""

from __future__ import annotations

import json

import equinox as eqx
import numpy as np
import pytest

from hallsim.composite import Composite
from hallsim.handles import Handle, ParameterMapping
from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.view import ViewModel, page_for
from hallsim.view._graph import elements, owners, wiring


class Fast(Process):
    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = eqx.field(static=True, default=1.0)
    rate: float = 0.1

    def ports_schema(self):
        return {"ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"ros": self.rate - 0.5 * state["ros"]}


class Slow(Process):
    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float = eqx.field(static=True, default=100.0)
    rate: float = 1e-2

    def ports_schema(self):
        return {
            "meth": Port(
                role=PortRole.EVOLVED, default=0.0, units="dimensionless"
            ),
            "ros": Port(role=PortRole.INPUT, default=0.0, units="uM"),
        }

    def derivative(self, t, state):
        return {"meth": self.rate * state["ros"]}


@pytest.fixture(scope="module")
def composite():
    return Composite(
        processes={"fast": Fast(), "slow": Slow()},
        topology={
            "fast": {"ros": "cell/ROS"},
            "slow": {"meth": "cell/meth", "ros": "cell/ROS"},
        },
    )


@pytest.fixture(scope="module")
def registry():
    return {
        "Oxidative load": Handle(
            name="Oxidative load",
            description="more ROS production",
            mappings=[
                ParameterMapping(
                    process_name="fast",
                    param_name="rate",
                    floor=1.0,
                    slope=1.0,
                )
            ],
        ),
        "Elsewhere": Handle(
            name="Elsewhere",
            mappings=[
                ParameterMapping(
                    process_name="other", param_name="k", floor=1.0, slope=1.0
                )
            ],
        ),
    }


@pytest.fixture(scope="module")
def page(composite, registry):
    return page_for(composite, registry, t_end=20.0, macro_dt=1.0)


def test_page_is_derived_from_the_composite(page):
    assert [lv.handle for lv in page.levers] == ["Oxidative load"]
    assert {k: [p.label for p in v] for k, v in page.panels.items()} == {
        "fast": ["ROS"],
        "slow": ["meth"],
    }
    assert page.units == {"fast": "uM", "slow": None}
    assert page.presets == {"control": (0.0,)}
    assert page.reference == "control"


def test_a_lever_moves_the_solve_and_the_rows(page):
    pytest.importorskip("plotly")
    pytest.importorskip("dash")
    from hallsim.view._levers import render

    vm = ViewModel(page)
    _, at_zero = vm.solve((0.0,))
    _, at_one = vm.solve((1.0,))
    ros = vm.keys.index("cell/ROS")
    assert at_one[-1, ros] > 1.5 * at_zero[-1, ros]
    out = render(page, vm, vm, 1.0)
    figures, values = out[:2], out[2:3]
    badge, chips = out[3], out[5]
    assert [len(f.data) for f in figures] == [2, 2]
    assert list(values) == ["1.00"]
    assert badge.className == "pop"
    assert chips == ["chip"]
    (moved,) = vm.moved(page.levers[0], 1.0)
    assert moved["target"] == "fast · rate"
    assert moved["now"] == pytest.approx(2 * moved["published"])


def test_the_wiring_collapses_to_processes_and_opens_into_paths(
    composite, page
):
    g = wiring(composite)
    assert owners(g) == {"cell/ROS": "fast", "cell/meth": "slow"}
    els = elements(composite, page)
    nodes = {e["data"]["id"] for e in els if "id" in e["data"]}
    edges = {
        (e["data"]["source"], e["data"]["target"], e["data"]["label"])
        for e in els
        if "source" in e["data"]
    }
    assert nodes == {"proc:fast", "proc:slow"}
    assert edges == {("proc:fast", "proc:slow", "cell/ROS")}
    els = elements(composite, page, expanded={"fast"})
    by_id = {e["data"]["id"]: e["data"] for e in els if "id" in e["data"]}
    assert by_id["path:cell/ROS"]["parent"] == "proc:fast"
    assert "deriv:fast" not in by_id
    edges = {
        (e["data"]["source"], e["data"]["target"])
        for e in els
        if "source" in e["data"]
    }
    assert edges == {("path:cell/ROS", "proc:slow")}
    els = elements(composite, page, expanded={"fast"}, reactions=True)
    by_id = {e["data"]["id"]: e["data"] for e in els if "id" in e["data"]}
    assert by_id["deriv:fast"]["parent"] == "proc:fast"
    edges = {
        (e["data"]["source"], e["data"]["target"])
        for e in els
        if "source" in e["data"]
    }
    assert ("deriv:fast", "path:cell/ROS") in edges
    assert ("path:cell/ROS", "proc:slow") in edges
    assert ("path:cell/ROS", "deriv:fast") in edges


def test_a_trace_marks_its_route(composite, page, registry):
    from hallsim.attenuation import trace_path

    trace = trace_path(
        composite,
        "Oxidative load",
        "cell/meth",
        registry=registry,
        t_end=20.0,
        macro_dt=1.0,
    )
    assert trace.reaches
    els = elements(composite, page, trace=trace)
    classes = {
        e["data"].get("id")
        or (e["data"]["source"], e["data"]["target"]): e["classes"].split()
        for e in els
    }
    assert "route" in classes["path:cell/ROS"]
    assert "route" in classes["path:cell/meth"]
    assert "dim" not in classes["proc:fast"]
    rel = {
        e["data"]["id"]: e["data"]["rel"] for e in els if "rel" in e["data"]
    }
    assert set(rel) == {"path:cell/ROS", "path:cell/meth"}
    assert all(0.0 <= v <= 1.0 for v in rel.values())


def test_a_page_bakes_to_a_static_site(page, tmp_path):
    from hallsim.view import bake
    from hallsim.view._bake import setting_key

    out = bake(page, tmp_path / "site", step=0.5, quiet=True)
    index = json.loads((out / "index.json").read_text())
    assert [lv["handle"] for lv in index["levers"]] == ["Oxidative load"]
    assert [r["name"] for r in index["rows"]] == ["fast", "slow"]
    assert set(index["layouts"]["composite"]) == {"5", "2"}
    names = sorted(p.name for p in (out / "data" / "composite").glob("*"))
    assert names == ["0.00.json", "0.50.json", "1.00.json"]
    assert setting_key((1.0,)) == "1.00"
    ref = json.loads((out / "reference" / "composite.json").read_text())
    top = json.loads((out / "data" / "composite" / "1.00.json").read_text())
    assert len(top["series"]["fast"][0]) == len(ref["t"])
    # the handle doubles the production rate, so more ROS at the end
    assert top["series"]["fast"][0][-1] > ref["series"]["fast"][0][-1]
    assert top["population"] == {}
    for name in ("index.html", "bake.js", "plotly.min.js"):
        assert (out / name).exists()


def test_a_saved_run_renders(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("dash")
    from hallsim.view import _fit

    run = tmp_path / "run1"
    run.mkdir()
    row = {
        "gene": "G1",
        "path": "cell/ROS",
        "delta_sim_signed": 0.1,
        "delta_data": 0.3,
        "sign_match": True,
    }
    block = {
        "timepoint": 5,
        "sign_agreement": 1.0,
        "spearman_r": 0.5,
        "n_compared": 1,
        "rows": [row],
    }
    summary = {
        "params": {
            "fast.rate": {
                "process_name": "fast",
                "field": "rate",
                "clamp": [0.01, 1.0],
                "description": "ROS production",
            }
        },
        "handle_coeffs": {},
        "init_params": {"fast.rate": 0.1},
        "fitted_params": {"fast.rate": 0.2},
        "loss_history": [1.0, 0.5],
        "val_loss_history": [],
        "grad_norm_history": [1.0, 0.5],
        "lr_history": [0.01, 0.01],
        "lr_scale_history": [1, 1],
        "param_history": [{"fast.rate": 0.1}, {"fast.rate": 0.2}],
        "best_loss": 0.5,
        "stopped_epoch": 2,
        "wall_time_s": 1.0,
        "conditions": {},
        "arms": {},
        "processes": ["fast", "slow"],
        "fit_arms": ["a"],
        "held_out_arms": [],
        "t_end": 20,
        "macro_dt": 1,
        "concordance_pre": {"a": {"5": block}},
        "concordance_post": {"a": {"5": block}},
        "readouts": [],
    }
    (run / "summary.json").write_text(json.dumps(summary))
    assert _fit.find_runs([run, tmp_path / "missing"]) == [run]
    view = _fit.run_view(_fit.load_run(run))
    figures = [c for c in view.children if type(c).__name__ == "Graph"]
    assert len(figures) == 1
    losses = np.asarray(_fit.loss_figure(summary).data[0].y)
    assert losses.tolist() == [1.0, 0.5]
