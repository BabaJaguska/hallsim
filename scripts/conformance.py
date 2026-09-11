"""Compare imported models with libRoadRunner and COPASI, on demand.

A development instrument, not a test and not a dependency: the reference
engines are not part of the project. Install them by hand to run it::

    pip install libroadrunner copasi-basico

    python scripts/conformance.py                 # every vendored deposit
    python scripts/conformance.py path/to/a.xml   # one file
    python scripts/conformance.py --synthetic     # the three synthetic models
    python scripts/conformance.py --composite     # a two-model, three-edge composite

Each model runs in HallSim, libRoadRunner (CVODE) and COPASI (LSODA) over
two windows the reference engine picks from a log grid — the transient after
the first 1% motion and the settling time — and every species is scored with
``hallsim.diagnostics.trajectory_agreement``, cycle statistics for an
oscillator, pointwise otherwise. The two reference engines are scored against
each other as well: HallSim must sit within ``REFERENCE_MARGIN`` of that, and
never below ``LEVEL_TOL`` / ``CYCLE_TOL``. Exit status is the number of
failing cases.
"""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from pathlib import Path

import numpy as np

from demos.models.sbml import sbml_dir
from hallsim.composite import Composite, single_process_composite
from hallsim.diagnostics import trajectory_agreements
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler

POINTS = 1001
RTOL, ATOL = 1e-9, 1e-12
# A model whose largest state is tiny is integrated under absolute tolerance
# alone; the absolute tolerance follows the model's scale.
ATOL_FRACTION = 1e-9
LEVEL_TOL, CYCLE_TOL = 1e-4, 1e-3
# HallSim must sit as close to roadrunner as COPASI does, within this factor.
REFERENCE_MARGIN = 10.0
SETTLE_CAP = 1000.0
# A species is scored against its own peak, floored at this fraction of the
# model's largest state.
FLOOR_FRACTION = 1e-6


def _rr(path, atol=ATOL):
    import roadrunner

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    rr = roadrunner.RoadRunner(path)
    rr.integrator.relative_tolerance = RTOL
    rr.integrator.absolute_tolerance = atol
    ids = rr.model.getFloatingSpeciesIds()
    rr.timeCourseSelections = ["time"] + ids
    return rr, ids


def windows(path):
    """``({"transient": t, "settling": t}, peak)`` from the reference engine."""
    rr, ids = _rr(path)
    grid = np.logspace(-2, np.log10(SETTLE_CAP), 200)
    rows = [np.r_[0.0, rr.model.getFloatingSpeciesAmounts()]]
    for t in grid:
        try:
            rr.simulate(rows[-1][0], t, 2)
        except RuntimeError:
            break  # the reference cannot go further; use what it reached
        rows.append(np.r_[t, rr.model.getFloatingSpeciesAmounts()])
    res = np.asarray(rows)
    t, y = res[:, 0], res[:, 1:]
    scale = np.maximum(np.max(np.abs(y), axis=0), 1e-12)
    motion = (np.abs(y - y[0]) / scale).max(axis=1)
    unsettled = (np.abs(y - y[-1]) / scale).max(axis=1)
    first = t[np.argmax(motion > 0.01)] if (motion > 0.01).any() else 1.0
    settle = (
        t[np.flatnonzero(unsettled > 0.01).max()]
        if (unsettled > 0.01).any()
        else first
    )
    return (
        {"transient": 100.0 * first, "settling": min(settle, SETTLE_CAP)},
        float(scale.max()),
    )


def run_roadrunner(path, t_end, atol=ATOL):
    rr, ids = _rr(path, atol)
    res = np.asarray(rr.simulate(0.0, t_end, POINTS))
    return res[:, 0], dict(zip(ids, res[:, 1:].T))


def run_copasi(path, t_end, atol=ATOL):
    import basico
    from basico import T

    dm = basico.load_model(path)
    basico.set_task_settings(
        T.TIME_COURSE,
        {
            "method": {
                "name": "Deterministic (LSODA)",
                "Relative Tolerance": RTOL,
                "Absolute Tolerance": atol,
            }
        },
        model=dm,
    )
    # Concentration × compartment size is the amount in the model's own
    # substance unit; COPASI's "amount" is a particle count.
    tc = basico.run_time_course(
        duration=t_end,
        intervals=POINTS - 1,
        model=dm,
        use_sbml_id=True,
        use_concentrations=True,
    )
    species = basico.get_species(model=dm).set_index("sbml_id")
    sizes = basico.get_compartments(model=dm)["initial_size"]
    volume = {
        sid: float(sizes[row["compartment"]])
        for sid, row in species.iterrows()
    }
    return tc.index.to_numpy(float), {
        c: tc[c].to_numpy(float) * volume.get(c, 1.0) for c in tc
    }


def run_hallsim(path, t_end, atol=ATOL):
    proc = process_from_sbml(path)
    frozen = [proc._species_names[i] for i in proc._frozen_indices]
    if frozen:
        proc = proc.with_unfrozen(*frozen)
    comp = single_process_composite(proc)
    dt = t_end / (POINTS - 1)
    res = Scheduler(rtol=RTOL, atol=atol).run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=dt if proc._events else t_end,
        save_dt=dt,
        antialias=False,
    )
    prefix = proc._name + "/"
    series = {
        k[len(prefix) :]: np.asarray(res.ys[:, i])
        for i, k in enumerate(res.keys)
        if k.startswith(prefix)
    }
    return np.asarray(res.ts), series


def _on_grid(ts, source_ts, series):
    return {k: np.interp(ts, source_ts, v) for k, v in series.items()}


def _bars(reference_table):
    bars = {"level": LEVEL_TOL, "cycle": CYCLE_TOL}
    for a in reference_table.values():
        bars[a.kind] = max(bars[a.kind], REFERENCE_MARGIN * a.rel_dev)
    return bars


def _verdict(label, table, bars=None):
    bars = bars or {"level": LEVEL_TOL, "cycle": CYCLE_TOL}
    if not table:
        print(f"  {label}: no species in common")
        return False
    name, worst = max(table.items(), key=lambda kv: kv[1].rel_dev)
    failing = [n for n, a in table.items() if a.rel_dev > bars[a.kind]]
    print(
        f"  {label}: {len(table)} species, worst {name} {worst.kind} "
        f"{worst.rel_dev:.2e}"
        + (f"  FAIL ({len(failing)} over)" if failing else "")
    )
    return not failing


def check_file(path, t_end=None, label=None) -> int:
    """Print the verdicts for one SBML file; return the number of failures."""
    label = label or Path(path).name
    if t_end is None:
        wins, peak = windows(path)
    else:
        wins = {"fixed": t_end}
        _, ref = run_roadrunner(path, t_end)
        peak = max(np.max(np.abs(v)) for v in ref.values())
    atol = min(ATOL, ATOL_FRACTION * peak)
    floor = FLOOR_FRACTION * peak
    failures = 0
    for window, t_end in wins.items():
        print(f"{label} [{window} @ {t_end:g}]")
        ts, ref = run_roadrunner(path, t_end, atol)
        ts_c, cop = run_copasi(path, t_end, atol)
        between = trajectory_agreements(
            ts, ref, _on_grid(ts, ts_c, cop), floor=floor
        )
        ok = _verdict("roadrunner vs copasi", between)
        try:
            ts_h, hs = run_hallsim(path, t_end, atol)
        except Exception as exc:  # noqa: BLE001
            print(f"  hallsim: import or solve failed: {exc}")
            failures += 1
            continue
        table = trajectory_agreements(
            ts, ref, _on_grid(ts, ts_h, hs), floor=floor
        )
        ok &= _verdict("hallsim vs roadrunner", table, _bars(between))
        failures += not ok
    return failures


# ── Synthetic models: the semantics the vendored corpus never exercises ──

_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="{id}" substanceUnits="mole" timeUnits="second" volumeUnits="litre" extentUnits="mole">
<listOfUnitDefinitions>
  <unitDefinition id="per_second"><listOfUnits><unit kind="second" exponent="-1" scale="0" multiplier="1"/></listOfUnits></unitDefinition>
</listOfUnitDefinitions>
"""
_TAIL = "</model></sbml>\n"

SYNTHETIC = {
    # Two compartments of different size; a concentration species, an
    # amount-only species, a boundary species; second-order mass action
    # across compartments and a local parameter.
    "compartments": _HEAD.format(id="compartments")
    + """
<listOfCompartments>
  <compartment id="cyt" spatialDimensions="3" size="2.5" constant="true"/>
  <compartment id="nuc" spatialDimensions="3" size="0.4" constant="true"/>
</listOfCompartments>
<listOfSpecies>
  <species id="A" compartment="cyt" initialConcentration="1.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="B" compartment="cyt" initialAmount="0.5" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
  <species id="C" compartment="nuc" initialConcentration="0.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="E" compartment="cyt" initialConcentration="0.3" hasOnlySubstanceUnits="false" boundaryCondition="true" constant="true"/>
</listOfSpecies>
<listOfParameters>
  <parameter id="k1" value="0.7" constant="true"/>
  <parameter id="k2" value="0.2" constant="true"/>
</listOfParameters>
<listOfReactions>
  <reaction id="bind" reversible="false">
    <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="C" stoichiometry="1" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cyt</ci><ci>k1</ci><ci>A</ci><ci>B</ci><ci>E</ci></apply></math></kineticLaw>
  </reaction>
  <reaction id="release" reversible="false">
    <listOfReactants><speciesReference species="C" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="A" stoichiometry="2" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>nuc</ci><ci>kloc</ci><ci>C</ci></apply></math>
      <listOfLocalParameters><localParameter id="kloc" value="0.15"/></listOfLocalParameters>
    </kineticLaw>
  </reaction>
</listOfReactions>
"""
    + _TAIL,
    # A function definition, an assignment rule feeding a rate law, a rate
    # rule on a concentration species, an initial assignment, a piecewise
    # in time, and a parameter that another rule defines.
    "rules": _HEAD.format(id="rules")
    + """
<listOfFunctionDefinitions>
  <functionDefinition id="hill"><math xmlns="http://www.w3.org/1998/Math/MathML"><lambda><bvar><ci>x</ci></bvar><bvar><ci>K</ci></bvar><bvar><ci>n</ci></bvar>
    <apply><divide/><apply><power/><ci>x</ci><ci>n</ci></apply><apply><plus/><apply><power/><ci>K</ci><ci>n</ci></apply><apply><power/><ci>x</ci><ci>n</ci></apply></apply></apply></lambda></math></functionDefinition>
</listOfFunctionDefinitions>
<listOfCompartments><compartment id="cell" spatialDimensions="3" size="1.5" constant="true"/></listOfCompartments>
<listOfSpecies>
  <species id="S" compartment="cell" initialConcentration="2.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="P" compartment="cell" initialConcentration="0.1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="D" compartment="cell" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  <species id="R" compartment="cell" initialConcentration="0.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters>
  <parameter id="vmax" value="1.2" constant="true"/>
  <parameter id="K" value="0.8" constant="true"/>
  <parameter id="kdeg" value="0.3" constant="true"/>
  <parameter id="drive" constant="false"/>
  <parameter id="total" constant="false"/>
</listOfParameters>
<listOfInitialAssignments>
  <initialAssignment symbol="D"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>0.5</cn><ci>S</ci></apply></math></initialAssignment>
</listOfInitialAssignments>
<listOfRules>
  <assignmentRule variable="total"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>S</ci><ci>P</ci></apply></math></assignmentRule>
  <assignmentRule variable="drive"><math xmlns="http://www.w3.org/1998/Math/MathML"><piecewise><piece><apply><times/><cn>2</cn><ci>total</ci></apply><apply><lt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol><cn>3</cn></apply></piece><otherwise><ci>total</ci></otherwise></piecewise></math></assignmentRule>
  <rateRule variable="R"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><apply><times/><cn>0.4</cn><ci>drive</ci></apply><apply><times/><ci>kdeg</ci><ci>R</ci></apply></apply></math></rateRule>
</listOfRules>
<listOfReactions>
  <reaction id="convert" reversible="false">
    <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="P" stoichiometry="1" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cell</ci><ci>vmax</ci><apply><ci>hill</ci><ci>S</ci><ci>K</ci><cn>2</cn></apply><apply><ci>hill</ci><ci>D</ci><cn>0.5</cn><cn>1</cn></apply></apply></math></kineticLaw>
  </reaction>
  <reaction id="decay" reversible="false">
    <listOfReactants><speciesReference species="D" stoichiometry="1" constant="true"/></listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cell</ci><ci>kdeg</ci><ci>D</ci></apply></math></kineticLaw>
  </reaction>
</listOfReactions>
"""
    + _TAIL,
    # One timed event that resets a species and steps a parameter.
    "event": _HEAD.format(id="event")
    + """
<listOfCompartments><compartment id="cell" spatialDimensions="3" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>
  <species id="X" compartment="cell" initialConcentration="1.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters><parameter id="k" value="0.5" constant="false"/></listOfParameters>
<listOfReactions>
  <reaction id="decay" reversible="false">
    <listOfReactants><speciesReference species="X" stoichiometry="1" constant="true"/></listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>cell</ci><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
  </reaction>
</listOfReactions>
<listOfEvents>
  <event id="pulse" useValuesFromTriggerTime="true">
    <trigger initialValue="true" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol><cn>2</cn></apply></math></trigger>
    <listOfEventAssignments>
      <eventAssignment variable="X"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>3</cn></math></eventAssignment>
      <eventAssignment variable="k"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1.5</cn></math></eventAssignment>
    </listOfEventAssignments>
  </event>
</listOfEvents>
"""
    + _TAIL,
}


def build_composite(tmp: Path) -> Composite:
    """Two imported models sharing a species, three edges, one model on a
    2× rescaled clock."""
    from hallsim.models.clamp_edge import ClampEdge
    from hallsim.models.gain_edge import GainEdge
    from hallsim.models.hill_edge import HillEdge

    (tmp / "a.xml").write_text(SYNTHETIC["compartments"])
    (tmp / "b.xml").write_text(SYNTHETIC["rules"])
    a = process_from_sbml(str(tmp / "a.xml"), name="a").reconciled_to(2.0)
    b = process_from_sbml(
        str(tmp / "b.xml"), name="b", native_time_seconds=2.0
    ).reconciled_to(2.0)
    return Composite(
        processes={
            "a": a,
            "b": b,
            "drive": HillEdge(basal=0.0, hi=0.4, K=(0.3,), n=(2.0,)),
            "bridge": GainEdge(offset=0.1, gain=2.0, mode="level"),
            "hold": ClampEdge(k_clamp=1.5),
        },
        topology={
            "drive": {"source": "a/C", "target": "b/P"},
            "bridge": {"source": "b/S", "signal": "bridge/level"},
            "hold": {"target": "a/B", "setpoint": "hold/setpoint"},
        },
        rewire={"b/S": "a/A"},
        initial={"hold/setpoint": 0.4, "a/A": 2.5, "a/B": 0.5},
        semantic_validation=False,
    )


def check_composite(tmp: Path, t_end: float = 6.0) -> int:
    """Export the composite and compare the document's solution, in both
    reference engines, with HallSim's run of the composite in one group."""
    from hallsim.sbml_export import _sid

    comp = build_composite(tmp)
    path = str(tmp / "composite.xml")
    comp.to_sbml(path)
    print(f"composite [@ {t_end:g}]")
    ts, ref = run_roadrunner(path, t_end)
    floor = FLOOR_FRACTION * max(np.max(np.abs(v)) for v in ref.values())
    ts_c, cop = run_copasi(path, t_end)
    between = trajectory_agreements(
        ts, ref, _on_grid(ts, ts_c, cop), floor=floor
    )
    ok = _verdict("roadrunner vs copasi", between)
    dt = t_end / (POINTS - 1)
    res = Scheduler(
        rtol=RTOL, atol=ATOL, groups={"all": list(comp.processes)}
    ).run(
        comp, t_span=(0.0, t_end), macro_dt=t_end, save_dt=dt, antialias=False
    )
    hs = {_sid(k): np.asarray(res.ys[:, i]) for i, k in enumerate(res.keys)}
    table = trajectory_agreements(
        ts, ref, _on_grid(ts, np.asarray(res.ts), hs), floor=floor
    )
    missing = set(ref) - set(table)
    if missing:
        print(f"  hallsim lacks {sorted(missing)}")
    ok &= (
        _verdict("hallsim vs roadrunner", table, _bars(between))
        and not missing
    )
    return int(not ok)


def check_event_composite(tmp: Path, t_end: float = 5.0) -> int:
    """The synthetic event model composed (its event becomes an EVENT
    process), exported, and compared. HallSim fires an event at the first
    macro-step sync point after its trigger, so the grid is chosen to land
    on the trigger time (t = 2 on a 5/1000 grid); off the grid the deviation
    is the step times the post-event rate, 6e-3 at a 6/1000 grid here."""
    from hallsim.sbml_export import _sid

    (tmp / "ev.xml").write_text(SYNTHETIC["event"])
    comp = single_process_composite(
        process_from_sbml(str(tmp / "ev.xml"), name="m")
    )
    path = str(tmp / "events.xml")
    comp.to_sbml(path)
    print(f"event composite [@ {t_end:g}]")
    ts, ref = run_roadrunner(path, t_end)
    floor = FLOOR_FRACTION * max(np.max(np.abs(v)) for v in ref.values())
    ts_c, cop = run_copasi(path, t_end)
    between = trajectory_agreements(
        ts, ref, _on_grid(ts, ts_c, cop), floor=floor
    )
    ok = _verdict("roadrunner vs copasi", between)
    dt = t_end / (POINTS - 1)
    res = Scheduler(rtol=RTOL, atol=ATOL).run(
        comp, t_span=(0.0, t_end), macro_dt=dt, save_dt=dt, antialias=False
    )
    hs = {_sid(k): np.asarray(res.ys[:, i]) for i, k in enumerate(res.keys)}
    table = trajectory_agreements(
        ts, ref, _on_grid(ts, np.asarray(res.ts), hs), floor=floor
    )
    ok &= _verdict("hallsim vs roadrunner", table, _bars(between))
    return int(not ok)


def check_multi_hallmark(tmp: Path, t_end: float = 2.0) -> int:
    """The multi-hallmark demo composite (two deposits on reconciled clocks,
    two Hill edges, a pulse and a step), exported and compared in one group
    over ``t_end`` days: the transcription of the composite, not the
    Scheduler's splitting."""
    from demos.models.multi_hallmark import build_multi_hallmark_composite
    from hallsim.sbml_export import _sid

    comp = build_multi_hallmark_composite()
    path = str(tmp / "multi_hallmark.xml")
    comp.to_sbml(path)
    print(f"multi-hallmark composite [@ {t_end:g} days]")
    ts, ref = run_roadrunner(path, t_end)
    floor = FLOOR_FRACTION * max(np.max(np.abs(v)) for v in ref.values())
    ts_c, cop = run_copasi(path, t_end)
    between = trajectory_agreements(
        ts, ref, _on_grid(ts, ts_c, cop), floor=floor
    )
    ok = _verdict("roadrunner vs copasi", between)
    dt = t_end / (POINTS - 1)
    res = Scheduler(
        rtol=RTOL, atol=ATOL, groups={"all": list(comp.continuous_processes())}
    ).run(comp, t_span=(0.0, t_end), macro_dt=dt, save_dt=dt, antialias=False)
    hs = {_sid(k): np.asarray(res.ys[:, i]) for i, k in enumerate(res.keys)}
    table = trajectory_agreements(
        ts, ref, _on_grid(ts, np.asarray(res.ts), hs), floor=floor
    )
    missing = set(ref) - set(table)
    if missing:
        print(f"  hallsim lacks {sorted(missing)}")
    ok &= (
        _verdict("hallsim vs roadrunner", table, _bars(between))
        and not missing
    )
    return int(not ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "paths", nargs="*", help="SBML files; default: every vendored deposit"
    )
    ap.add_argument(
        "--synthetic", action="store_true", help="the three synthetic models"
    )
    ap.add_argument(
        "--composite",
        action="store_true",
        help="the two-model, three-edge composite",
    )
    ap.add_argument(
        "--multi-hallmark",
        action="store_true",
        help="the multi-hallmark demo composite",
    )
    ap.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="one fixed window instead of the two derived ones",
    )
    args = ap.parse_args(argv)
    failures = 0
    tmp = Path(tempfile.mkdtemp(prefix="hallsim-conformance-"))
    paths = args.paths
    if not (paths or args.synthetic or args.composite or args.multi_hallmark):
        paths = sorted(glob.glob(str(sbml_dir("") / "*" / "*.xml")))
    for path in paths:
        failures += check_file(path, args.t_end)
    if args.synthetic:
        for name, text in sorted(SYNTHETIC.items()):
            p = tmp / f"{name}.xml"
            p.write_text(text)
            failures += check_file(
                str(p), args.t_end or 10.0, label=f"synthetic:{name}"
            )
    if args.composite:
        failures += check_composite(tmp)
        failures += check_event_composite(tmp)
    if args.multi_hallmark:
        failures += check_multi_hallmark(tmp)
    print(f"{failures} failing case(s)")
    return failures


if __name__ == "__main__":
    sys.exit(main())
