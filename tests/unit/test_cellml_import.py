"""CellML import — conversion at the boundary.

A CellML file is components, connected variables, units and equations, with
no species and no reactions. Myokit reads it; the converter writes states as
rate rules, derived quantities as assignment rules and constants as
parameters, and everything downstream sees an ordinary SBML import.
"""

import math

import libsbml
import numpy as np
import pytest

from hallsim.cellml_import import (
    UnsupportedCellMLFeatureError,
    cellml_to_sbml,
    document_from_cellml,
    is_cellml,
)
from hallsim.composite import single_process_composite
from hallsim.io import file_sha256
from hallsim.process import PortRole
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler

# Relaxation towards a target with a one-millisecond source pulse: the
# rate constant lives in its own component and reaches the cell through a
# connection, the target is a constant derived from another constant, and
# the pulse is a piecewise over time.
DECAY_WITH_PULSE = """<?xml version="1.0"?>
<model xmlns="http://www.cellml.org/cellml/1.0#"
       xmlns:cellml="http://www.cellml.org/cellml/1.0#"
       name="decay_with_pulse">
  <units name="millisecond"><unit prefix="milli" units="second"/></units>
  <units name="per_millisecond">
    <unit exponent="-1" prefix="milli" units="second"/>
  </units>
  <units name="millimolar">
    <unit prefix="milli" units="mole"/><unit exponent="-1" units="litre"/>
  </units>
  <units name="millimolar_per_millisecond">
    <unit units="millimolar"/><unit units="per_millisecond"/>
  </units>
  <component name="environment">
    <variable name="time" public_interface="out" units="millisecond"/>
  </component>
  <component name="rates">
    <variable name="k" initial_value="0.5" public_interface="out"
              units="per_millisecond"/>
  </component>
  <component name="cell">
    <variable name="time" public_interface="in" units="millisecond"/>
    <variable name="k" public_interface="in" units="per_millisecond"/>
    <variable name="x" initial_value="2" units="millimolar"/>
    <variable name="x0" initial_value="1" units="millimolar"/>
    <variable name="target" units="millimolar"/>
    <variable name="pulse" units="millimolar_per_millisecond"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>target</ci>
        <apply><times/><cn cellml:units="dimensionless">3</cn><ci>x0</ci>
        </apply>
      </apply>
      <apply><eq/><ci>pulse</ci>
        <piecewise>
          <piece>
            <cn cellml:units="millimolar_per_millisecond">1</cn>
            <apply><and/>
              <apply><geq/><ci>time</ci>
                <cn cellml:units="millisecond">1</cn></apply>
              <apply><lt/><ci>time</ci>
                <cn cellml:units="millisecond">2</cn></apply>
            </apply>
          </piece>
          <otherwise><cn cellml:units="millimolar_per_millisecond">0</cn>
          </otherwise>
        </piecewise>
      </apply>
      <apply><eq/>
        <apply><diff/><bvar><ci>time</ci></bvar><ci>x</ci></apply>
        <apply><plus/>
          <apply><times/><ci>k</ci>
            <apply><minus/><ci>target</ci><ci>x</ci></apply>
          </apply>
          <ci>pulse</ci>
        </apply>
      </apply>
    </math>
  </component>
  <connection>
    <map_components component_1="environment" component_2="cell"/>
    <map_variables variable_1="time" variable_2="time"/>
  </connection>
  <connection>
    <map_components component_1="rates" component_2="cell"/>
    <map_variables variable_1="k" variable_2="k"/>
  </connection>
</model>
"""

MODULAR = """<?xml version="1.0"?>
<model xmlns="http://www.cellml.org/cellml/1.1#"
       xmlns:xlink="http://www.w3.org/1999/xlink" name="modular">
  <import xlink:href="elsewhere.cellml">
    <component component_ref="a" name="a"/>
  </import>
</model>
"""


def _closed_form(t, k=0.5, x_init=2.0, target=3.0):
    """x(t) for dx/dt = k (target - x) + [1 <= t < 2]."""

    def relax(x_from, goal, dt):
        return goal - (goal - x_from) * math.exp(-k * dt)

    if t <= 1.0:
        return relax(x_init, target, t)
    x1 = relax(x_init, target, 1.0)
    if t <= 2.0:
        return relax(x1, target + 1.0 / k, t - 1.0)
    x2 = relax(x1, target + 1.0 / k, 1.0)
    return relax(x2, target, t - 2.0)


@pytest.fixture(scope="module")
def cellml_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("cellml") / "decay_with_pulse.cellml"
    path.write_text(DECAY_WITH_PULSE)
    return path


def test_is_cellml_is_by_extension():
    assert is_cellml("model.cellml") and is_cellml("MODEL.CELLML")
    assert not is_cellml("model.xml") and not is_cellml(42)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        cellml_to_sbml(tmp_path / "absent.cellml")


def test_conversion_is_cached_on_content(cellml_file):
    first = cellml_to_sbml(cellml_file)
    assert cellml_to_sbml(cellml_file) == first, "same bytes, same path"


def test_the_document_is_consistent_sbml(cellml_file):
    doc = document_from_cellml(cellml_file)
    doc.checkConsistency()
    errors = [
        doc.getError(i).getMessage()
        for i in range(doc.getNumErrors())
        if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    ]
    assert errors == []


def test_states_rules_and_constants_land_where_the_compiler_reads_them(
    cellml_file,
):
    model = document_from_cellml(cellml_file).getModel()
    rules = {
        model.getRule(i).getVariable(): model.getRule(i)
        for i in range(model.getNumRules())
    }
    assert rules["cell_x"].isRate()
    assert rules["cell_target"].isAssignment()
    assert rules["cell_pulse"].isAssignment()
    constants = {
        model.getParameter(i).getId()
        for i in range(model.getNumParameters())
        if model.getParameter(i).getConstant()
    }
    assert constants == {"rates_k", "cell_x0"}
    assert model.getParameter("cell_x").getValue() == 2.0
    assert model.getParameter("cell_x").getName() == "cell.x"


def test_a_connected_variable_is_one_quantity(cellml_file):
    model = document_from_cellml(cellml_file).getModel()
    ids = {
        model.getParameter(i).getId() for i in range(model.getNumParameters())
    }
    assert "rates_k" in ids and "cell_k" not in ids
    assert "rates_k" in libsbml.formulaToL3String(
        model.getRule("cell_x").getMath()
    )


def test_units_become_unit_definitions(cellml_file):
    model = document_from_cellml(cellml_file).getModel()
    concentration = model.getUnitDefinition(
        model.getParameter("cell_x").getUnits()
    )
    kinds = {
        concentration.getUnit(i).getKind()
        for i in range(concentration.getNumUnits())
    }
    assert kinds == {libsbml.UNIT_KIND_MOLE, libsbml.UNIT_KIND_METRE}
    clock = model.getUnitDefinition(model.getTimeUnits())
    assert clock.getNumUnits() == 1
    assert clock.getUnit(0).getKind() == libsbml.UNIT_KIND_SECOND
    assert clock.getUnit(0).getMultiplier() == pytest.approx(1e-3)


def test_process_from_sbml_routes_a_cellml_path(cellml_file):
    """The point of the whole module: downstream needs no special case."""
    proc = process_from_sbml(str(cellml_file))
    schema = proc.ports_schema()
    assert schema["cell_x"].role is PortRole.EVOLVED
    assert schema["cell_target"].role is PortRole.ASSIGNED
    assert schema["cell_pulse"].role is PortRole.ASSIGNED
    assert set(proc.parameters) == {"rates_k", "cell_x0"}
    assert proc.native_time_seconds == pytest.approx(1e-3)
    assert proc.native_time_source == "declared"
    assert proc.source_sha256 == file_sha256(cellml_file)


def test_the_trajectory_matches_the_closed_form(cellml_file):
    proc = process_from_sbml(str(cellml_file))
    result = Scheduler().run(
        single_process_composite(proc),
        t_span=(0.0, 3.0),
        macro_dt=0.01,
        save_dt=0.01,
    )
    x = result.get(f"{proc._name}/cell_x")
    for t, got in ((0.5, x[50]), (1.5, x[150]), (3.0, x[-1])):
        assert float(got) == pytest.approx(_closed_form(t), abs=1e-3)


def test_a_modular_model_is_refused_with_the_reason(tmp_path):
    path = tmp_path / "modular.cellml"
    path.write_text(MODULAR)
    with pytest.raises(UnsupportedCellMLFeatureError, match="modular.cellml"):
        cellml_to_sbml(path)


@pytest.mark.network
def test_luo_rudy_from_physiome_matches_the_biomodels_copy():
    """Luo-Rudy 1991 as deposited in CellML against the SBML that BioModels
    converted from it. The two deposits differ only in their initial state
    (BioModels converted an earlier revision), so the copy's initial values
    are imposed on the CellML import and the voltage traces must coincide.
    """
    from hallsim.search.models import download_physiome_models

    cellml = download_physiome_models("2d2ce7737b42a4f72d6bf8b67f6eb5a2")[0]
    own = process_from_sbml(str(cellml), name="cellml")
    copy = process_from_sbml("MODEL0479527919", name="biomodels")
    copy_initial = {
        name: float(port.default)
        for name, port in copy.ports_schema().items()
        if port.role is PortRole.EVOLVED
    }

    def same_quantity(cellml_name):
        variable = cellml_name.rsplit("_", 1)[-1]
        return next(
            k
            for k in copy_initial
            if k == variable or k.startswith(variable + "_")
        )

    initial = {
        f"cellml/{name}": copy_initial[same_quantity(name)]
        for name, port in own.ports_schema().items()
        if port.role is PortRole.EVOLVED
    }

    def voltage(composite, path):
        result = Scheduler().run(
            composite, t_span=(0.0, 400.0), macro_dt=0.1, save_dt=0.1
        )
        return np.asarray(result.get(path))

    ours = voltage(
        single_process_composite(own).with_initial(initial),
        "cellml/membrane_V",
    )
    theirs = voltage(single_process_composite(copy), "biomodels/V_membrane")
    assert np.max(np.abs(ours - theirs)) < 0.05
    assert np.max(ours) == pytest.approx(47.05, abs=0.5)
