"""A sum of store paths is one path any reader can use."""

import numpy as np

from hallsim.composite import Composite
from hallsim.models.observer import SumObserver
from hallsim.process import Port, PortRole, Process


class Two(Process):
    timescale: float = 1.0

    def ports_schema(self):
        return {
            "a": Port(role=PortRole.EVOLVED, default=1.0),
            "b": Port(role=PortRole.EVOLVED, default=2.0),
        }

    def derivative(self, t, state):
        return {"a": -state["a"], "b": -state["b"]}


def test_the_total_is_materialised_and_tracks_its_parts():
    comp = Composite(
        processes={
            "m": Two(),
            "sum": SumObserver(elements=("a", "b"), what="a plus b"),
        },
        topology={
            "m": {"a": "m/a", "b": "m/b"},
            "sum": {"parts": ("m/a", "m/b"), "total": "m/total"},
        },
        validate=False,
        semantic_validation=False,
    )
    keys = comp.store_keys()
    y0 = comp.initial_state_vec(keys)
    ys = comp.materialize_assigned(np.array([0.0]), y0[None])
    assert float(ys[0, keys.index("m/total")]) == 3.0
    # nothing in the loop reads the total, so the RHS carries no assignment
    rhs, _ = comp.build_rhs()
    assert rhs.assign_procs == ()


class TwoDecays(Two):
    """``Two`` with the symbolic form the exporter needs."""

    def reaction_channels(self):
        import sympy

        from hallsim.process import ReactionChannel

        return tuple(
            ReactionChannel(f"decay_{p}", ((p, -1.0),), sympy.Symbol(p))
            for p in ("a", "b")
        )


def test_the_total_exports_as_an_assignment_rule_over_its_parts():
    import libsbml

    from hallsim.sbml_export import composite_to_sbml

    comp = Composite(
        processes={
            "m": TwoDecays(),
            "sum": SumObserver(elements=("a", "b"), what="a plus b"),
        },
        topology={
            "m": {"a": "m/a", "b": "m/b"},
            "sum": {"parts": ("m/a", "m/b"), "total": "sum/total"},
        },
        validate=False,
        semantic_validation=False,
    )
    doc = libsbml.readSBMLFromString(composite_to_sbml(comp))
    model = doc.getModel()
    assert doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) == 0
    rules = {
        r.getVariable(): libsbml.formulaToL3String(r.getMath())
        for r in model.getListOfRules()
        if r.isAssignment()
    }
    (formula,) = [f for v, f in rules.items() if "total" in v]
    a = [
        s.getId() for s in model.getListOfSpecies() if s.getId().endswith("a")
    ]
    b = [
        s.getId() for s in model.getListOfSpecies() if s.getId().endswith("b")
    ]
    assert a and b
    assert a[0] in formula and b[0] in formula
