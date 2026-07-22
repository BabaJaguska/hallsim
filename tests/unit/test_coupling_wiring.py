"""Tests for the coupling-wiring checker (driver-target semantic validity)."""

from hallsim.coupling_wiring import (
    classify_driver_target,
    classify_topology_edge,
    topology_writer_verdicts,
    validate_couplings,
    CouplingReport,
)
from hallsim.process import Port, PortRole


class _FakeProc:
    """Minimal stand-in carrying just what the checker reads."""

    def __init__(self, meta, drivers=()):
        self._name = "m"
        self._meta = meta
        self._param_drivers = drivers

    def coupling_structure(self):
        return self._meta


# kd2 is the constant baseline of `kd2_0 = kd2*(1+DNAdamage)`; DNAdamage is the
# model's damage channel. Mirrors Zhang 2007.
_META = {
    "param_constant": {"kd2": True, "Dam0": True, "ks2": True, "kdeg": True},
    "param_sbo": {"kd2": -1, "Dam0": -1, "ks2": -1, "kdeg": 17},
    "variables": frozenset({"DNAdamage", "p53", "MDM2", "kd2_0"}),
    "rules": (("kd2_0", frozenset({"kd2", "DNAdamage"})),),
}


def test_bypass_flagged_with_suggestion():
    v = classify_driver_target(_FakeProc(_META), "kd2")
    assert v.status == "review"
    assert v.is_warning
    assert v.suggested_target == "DNAdamage"
    assert "kd2_0" in v.message and "Confirm" in v.message


def test_clean_input_ok():
    # Dam0 is constant but modulates nothing via a variable-bearing rule.
    v = classify_driver_target(_FakeProc(_META), "Dam0")
    assert v.status == "ok"
    assert not v.is_warning


def test_kinetic_sbo_flagged():
    # kdeg carries a rate-constant SBO term and is not rule-modulated.
    v = classify_driver_target(_FakeProc(_META), "kdeg")
    assert v.status == "rate-constant"
    assert v.is_warning


def test_non_sbml_target_is_non_structural():
    class Bare:
        _name = "bare"

    v = classify_driver_target(Bare(), "k_act")
    assert v.status == "non-structural"
    assert not v.is_warning


def test_unknown_param_is_non_structural():
    v = classify_driver_target(_FakeProc(_META), "not_a_param")
    assert v.status == "non-structural"


def test_validate_couplings_collects_driver_edges():
    class _D:
        param_name = "kd2"

    class _Composite:
        processes = {"m": _FakeProc(_META, drivers=(_D(),))}
        topology = {}

    report = validate_couplings(_Composite())
    assert isinstance(report, CouplingReport)
    assert len(report.warnings) == 1
    assert report.warnings[0].param == "kd2"


class _Owner:
    def __init__(self):
        self._name = "owner"

    def coupling_structure(self):
        return {
            "param_constant": {},
            "param_sbo": {},
            "variables": frozenset({"p53", "kd2_0", "held"}),
            "rules": (("kd2_0", frozenset({"kd2", "p53"})),),
            "boundary": frozenset({"held"}),
        }

    def ports_schema(self):
        return {}


class _Writer:
    _name = "writer"

    def ports_schema(self):
        return {"out": Port(role=PortRole.EVOLVED, default=0.0, units="x")}


def test_topology_write_to_rule_target_flagged():
    procs = {"owner": _Owner(), "writer": _Writer()}
    v = classify_topology_edge(procs, "writer", "out", "owner/kd2_0")
    assert v.status == "review" and "assignment rule" in v.message


def test_topology_write_to_boundary_flagged():
    procs = {"owner": _Owner(), "writer": _Writer()}
    v = classify_topology_edge(procs, "writer", "out", "owner/held")
    assert v.status == "review" and "boundary" in v.message


def test_topology_write_to_state_ok():
    procs = {"owner": _Owner(), "writer": _Writer()}
    v = classify_topology_edge(procs, "writer", "out", "owner/p53")
    assert v.status == "ok"


def test_topology_self_write_and_new_path_skipped():
    procs = {"owner": _Owner(), "writer": _Writer()}
    # self-write (writer's own namespace) is skipped by topology_writer_verdicts
    topo = {"writer": {"out": "writer/x"}}
    assert topology_writer_verdicts(procs, topo) == []
    # cross-namespace write to a rule target is flagged
    topo2 = {"writer": {"out": "owner/kd2_0"}}
    v = topology_writer_verdicts(procs, topo2)
    assert len(v) == 1 and v[0].status == "review"


def test_extra_targets_checked():
    # Hallmark severity mappings arrive as extra_targets, not _param_drivers.
    class _Composite:
        processes = {}
        topology = {}

    report = validate_couplings(
        _Composite(), extra_targets=[(_FakeProc(_META), "kd2")]
    )
    assert len(report.warnings) == 1
    assert report.warnings[0].suggested_target == "DNAdamage"
