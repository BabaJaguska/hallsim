"""The screens between a search hit and the framework: what a model
produces, what a dataset measures of a composite, how a platform loads."""

import pandas as pd

from hallsim import screens
from hallsim.search.datasets import DatasetCandidate, Measured


class TestSpeciesRoleScreen:
    class _Model:
        def __init__(
            self,
            n_species=0,
            reactions=(),
            names=None,
            packages=(),
            fixed=(),
        ):
            self._n, self._rx = n_species, reactions
            if names is None:
                # No display names declared: id is the label, as in a
                # hand-written or COPASI-exported deposit.
                names = {
                    p: p
                    for r in reactions
                    for p in (*r._p, *r._r, *r._m, *r._law)
                }
            self._names = names
            self._fixed = set(fixed)
            self._packages = set(packages)

        def getPlugin(self, name):
            return object() if name in self._packages else None

        def getNumSpecies(self):
            return len(self._names) or self._n

        def getSpecies(self, i):
            sid = sorted(self._names)[i]
            fixed = sid in self._fixed
            return type(
                "_S",
                (),
                {
                    "getId": lambda s, v=sid: v,
                    "getName": lambda s, v=self._names[sid]: v,
                    "getBoundaryCondition": lambda s, v=fixed: v,
                    "getConstant": lambda s: False,
                },
            )()

        def getNumReactions(self):
            return len(self._rx)

        def getReaction(self, i):
            return self._rx[i]

    class _Reaction:
        def __init__(
            self,
            products=(),
            reactants=(),
            modifiers=(),
            law=(),
            kinetic=True,
        ):
            self._p, self._r, self._m = products, reactants, modifiers
            self._law = law
            self._kinetic = kinetic

        def isSetKineticLaw(self):
            return self._kinetic

        @staticmethod
        def _ref(sid):
            return type("_Ref", (), {"getSpecies": lambda s: sid})()

        def getNumProducts(self):
            return len(self._p)

        def getProduct(self, j):
            return self._ref(self._p[j])

        def getNumReactants(self):
            return len(self._r)

        def getReactant(self, j):
            return self._ref(self._r[j])

        def getNumModifiers(self):
            return len(self._m)

        def getModifier(self, j):
            return self._ref(self._m[j])

        def getKineticLaw(self):
            import libsbml

            formula = " * ".join(("k", *self._r, *self._m, *self._law))
            math = libsbml.parseL3Formula(formula)
            return type("_K", (), {"getMath": lambda s: math})()

    def _patch(self, monkeypatch, model):
        monkeypatch.setattr(screens, "model_files", lambda *a: ["/tmp/x.xml"])
        monkeypatch.setattr(
            screens, "_first_readable_sbml", lambda paths: (model, "")
        )

    def _one(self, monkeypatch, model, pattern, **kw):
        self._patch(monkeypatch, model)
        (row,) = screens.screen_species_roles(["MODEL1"], pattern, **kw)
        return row

    def test_a_reactionless_deposit_is_not_a_no_match(self, monkeypatch):
        """SBML-qual parses to an empty core model. Calling that `no-match` says
        the reactions were read and produced nothing — the opposite conclusion,
        and how three Boolean deposits read as screened negatives."""
        row = self._one(monkeypatch, self._Model(n_species=0), "IL6")
        assert row.status == "no-reactions"
        assert "no reactions" in row.note

    def test_the_package_names_the_formalism(self, monkeypatch):
        """A reactionless deposit carrying the qual package is a logical
        model and one carrying fbc is constraint-based; each is reported as
        what it is, not lumped under an absence."""
        model = self._Model(n_species=3, packages=["qual"])
        row = self._one(monkeypatch, model, "IL6")
        assert row.status == "qualitative"
        assert "SBML-qual" in row.note and row.n_species == 3
        model = self._Model(n_species=3, packages=["fbc"])
        row = self._one(monkeypatch, model, "IL6")
        assert row.status == "constraint-based"
        assert "SBML-fbc" in row.note

    def test_no_match_still_means_read_and_absent(self, monkeypatch):
        model = self._Model(reactions=(self._Reaction(["TNFR"]),))
        row = self._one(monkeypatch, model, r"\bIL6\b")
        assert row.status == "no-match" and row.roles == ()

    def test_a_product_is_produced(self, monkeypatch):
        model = self._Model(reactions=(self._Reaction(["IL6", "junk"]),))
        row = self._one(monkeypatch, model, r"IL6")
        assert row.status == "match" and row.produced == ("IL6",)
        assert row.roles == (("IL6", ("produces",)),)

    def test_a_consumer_is_a_match_not_a_failure(self, monkeypatch):
        """A deposit that binds IL6 to its receptor and never makes it is
        the downstream half of a composite; the producer is the edge."""
        model = self._Model(
            reactions=(self._Reaction(["IL6_IL6R"], ["IL6", "IL6R"]),)
        )
        row = self._one(monkeypatch, model, r"\bIL6\b")
        assert row.status == "match"
        assert row.having("consumes") == ("IL6",)
        assert row.produced == ()

    def test_a_modifier_or_rate_law_symbol_is_read(self, monkeypatch):
        model = self._Model(
            reactions=(
                self._Reaction(["STAT3p"], ["STAT3"], modifiers=["IL6"]),
                self._Reaction(["SOCS3"], law=["IL6R"]),
            )
        )
        row = self._one(monkeypatch, model, r"IL6")
        assert dict(row.roles) == {"IL6": ("reads",), "IL6R": ("reads",)}

    def test_a_boundary_species_is_an_input(self, monkeypatch):
        model = self._Model(
            reactions=(self._Reaction(["pSTAT"], ["IL6"]),), fixed=["IL6"]
        )
        row = self._one(monkeypatch, model, r"IL6")
        assert row.roles == (("IL6", ("consumes", "input")),)

    def test_roles_narrow_what_counts_as_a_match(self, monkeypatch):
        model = self._Model(reactions=(self._Reaction(["x"], ["IL6"]),))
        row = self._one(monkeypatch, model, "IL6", roles=("produces",))
        assert row.status == "no-match"
        # The role is still reported, so a caller can see why.
        assert row.having("consumes") == ("IL6",)

    def test_an_unknown_role_is_refused(self, monkeypatch):
        import pytest

        with pytest.raises(ValueError, match="unknown roles"):
            self._one(monkeypatch, self._Model(), "IL6", roles=("emits",))

    def test_matches_a_display_name_when_the_id_is_a_uuid(self, monkeypatch):
        """CellDesigner exports — a large part of BioModels — put the gene
        symbol in the name and a UUID in the id. Dwivedi2014 produces IL6 in
        three compartments and scored `no-match` on an id-only screen."""
        uuid = "mwf626e95e_543f_41e4_aad4_c6bf60ab345b"
        model = self._Model(
            reactions=(self._Reaction([uuid]),), names={uuid: "IL6"}
        )
        row = self._one(monkeypatch, model, r"\bIL6\b")
        assert row.status == "match"
        assert row.produced == ("IL6",), "report the name, not the UUID"

    def test_a_drawn_map_does_not_produce_anything(self, monkeypatch):
        """A CellDesigner disease map draws reactions with no rate law. Wu2010
        has 254 of them and 0 parameters; counting an arrow as production
        promotes a diagram to a model."""
        model = self._Model(
            reactions=(self._Reaction(["IL6"], kinetic=False),)
        )
        row = self._one(monkeypatch, model, r"\bIL6\b")
        assert row.status == "no-rate-laws"
        assert row.n_reactions == 1

    def test_an_unfetchable_source_is_reported_not_dropped(self, monkeypatch):
        def refuse(model_id, source, timeout):
            raise LookupError(f"no SBML fetcher for source {source!r}")

        monkeypatch.setattr(screens, "model_files", refuse)
        (row,) = screens.screen_species_roles(["12345"], "IL6")
        assert row.status == "unscreenable"
        assert "no SBML fetcher" in row.note


def _annotated_composite():
    import equinox as eqx

    from hallsim.composite import Composite
    from hallsim.process import Port, PortRole, Process, ProcessKind

    class P53(Process):
        kind: ProcessKind = ProcessKind.CONTINUOUS
        timescale: float = eqx.field(static=True, default=1.0)

        def ports_schema(self):
            return {
                "p53": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    ontology={"uniprot": "P04637"},
                ),
                "atp": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    ontology={"chebi": "CHEBI:15422"},
                ),
            }

        def derivative(self, t, state):
            return {"p53": -state["p53"], "atp": -state["atp"]}

    return Composite(
        processes={"m": P53()},
        topology={"m": {"p53": "cell/p53", "atp": "cell/atp"}},
        semantic_validation=False,
    )


def _candidate(measured, **kw):
    base = dict(
        source="x",
        accession="A",
        title="",
        kind="",
        organism="",
        n_samples=0,
        platform="",
        url="",
    )
    base.update(kw)
    return DatasetCandidate(measured=measured, **base)


def test_coverage_joins_measured_quantities_to_store_paths():
    comp = _annotated_composite()
    M = Measured
    cov = screens.coverage(
        _candidate(M("metabolomics", False, ("chebi:15422",))), comp
    )
    assert (cov.paths, cov.via) == (("cell/atp",), "direct")
    cov = screens.coverage(_candidate(M("proteomics", True)), comp)
    assert (cov.paths, cov.via) == (("cell/p53",), "complete")
    cov = screens.coverage(_candidate(M("expression", True)), comp)
    assert (cov.paths, cov.via) == (("cell/p53",), "regulon")  # p53 is a TF
    assert not screens.coverage(_candidate(M("imaging")), comp)
    assert not screens.coverage(
        _candidate(M("metabolomics", False, ("chebi:1",))), comp
    )


def test_loader_route_names_the_column_or_the_failure():
    head = pd.DataFrame(
        {
            "ID": [f"TC0100000{i}.hg.1" for i in range(8)],
            "probeset_id": [f"TC0100000{i}.hg.1" for i in range(8)],
            "gene_assignment": [f"NM_{i} // GENE{i} // x" for i in range(8)],
        }
    )
    assert screens.loader_route(head) == (
        "the loader reads symbols from column 'gene_assignment'"
    )
    assert "cannot map" in screens.loader_route(head[["ID", "probeset_id"]])
    assert "no platform table" in screens.loader_route(pd.DataFrame())
