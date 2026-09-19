"""Handles are suggested from annotations; the registry stays named."""

import pytest
import sympy

from hallsim.composite import Composite
from hallsim.handles import (
    Intent,
    IntentTarget,
    suggest_handle,
    suggest_mappings,
    targets_for,
)
from hallsim.process import Port, PortRole, Process, ReactionChannel


class Damage(Process):
    """A -> D at rate k_make, D -> 0 at rate k_fix, both catalysed by E."""

    k_make: float = 0.1
    k_fix: float = 0.2

    def ports_schema(self):
        return {
            "A": Port(role=PortRole.EVOLVED, default=1.0),
            "D": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                ontology={"chebi": "CHEBI:16991"},
            ),
            "E": Port(
                role=PortRole.INPUT, default=1.0, ontology={"go": "GO:1"}
            ),
        }

    def derivative(self, t, state):
        make = self.k_make * state["A"] * state["E"]
        fix = self.k_fix * state["D"] * state["E"]
        return {"A": -make, "D": make - fix}

    def reaction_channels(self):
        A, D, E, k_make, k_fix = sympy.symbols("A D E k_make k_fix")
        return (
            ReactionChannel("make", (("A", -1.0), ("D", 1.0)), k_make * A * E),
            ReactionChannel("fix", (("D", -1.0),), k_fix * D * E),
        )


def _comp():
    return Composite(
        processes={"dmg": Damage()},
        topology={"dmg": {"A": "x/A", "D": "x/D", "E": "x/E"}},
        semantic_validation=False,
    )


def test_roles_select_the_reactions_the_species_plays_in():
    comp = _comp()
    made = targets_for(comp, {"chebi": "CHEBI:16991"}, "production")
    assert [(t.param, t.reaction) for t in made] == [("k_make", "make")]
    fixed = targets_for(comp, {"chebi": "CHEBI:16991"}, "consumption")
    assert [(t.param, t.reaction) for t in fixed] == [("k_fix", "fix")]
    catalysed = targets_for(comp, {"go": "GO:1"}, "modifier")
    assert sorted((t.param, t.reaction) for t in catalysed) == [
        ("k_fix", "fix"),
        ("k_make", "make"),
    ]
    assert targets_for(comp, {"uniprot": "nope"}) == []


def test_suggestions_are_mappings_that_apply():
    comp = _comp()
    (m,) = suggest_mappings(
        comp, {"chebi": "CHEBI:16991"}, "production", slope=2.0
    )
    assert (m.process_name, m.param_name) == ("dmg", "k_make")
    assert "production of D" in m.description and "make" in m.description
    intent = Intent(
        "More damage",
        targets=[
            IntentTarget({"chebi": "CHEBI:16991"}, "production", 1.0, 2.0)
        ],
    )
    handle = suggest_handle(intent, comp)
    moved = handle.apply(comp.processes, 1.0)
    assert float(moved["dmg"].k_make) == pytest.approx(0.3)
    assert float(moved["dmg"].k_fix) == pytest.approx(0.2)
    nothing = suggest_handle(
        Intent("x", targets=[IntentTarget({"go": "GO:9"})]), comp
    )
    assert nothing.mappings == []
