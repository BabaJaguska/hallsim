"""The screens between a search hit and the framework: what a model
produces, what a dataset measures of a composite, how a platform loads."""

import pandas as pd

from hallsim import screens
from hallsim.search.datasets import DatasetCandidate, Measured


class TestProducedSpeciesScreen:
    class _Model:
        def __init__(self, n_species=0, reactions=(), names=None, packages=()):
            self._n, self._rx = n_species, reactions
            if names is None:
                # No display names declared: id is the label, as in a
                # hand-written or COPASI-exported deposit.
                names = {p: p for r in reactions for p in getattr(r, "_p", ())}
            self._names = names
            self._packages = set(packages)

        def getPlugin(self, name):
            return object() if name in self._packages else None

        def getNumSpecies(self):
            return len(self._names) or self._n

        def getSpecies(self, i):
            sid = sorted(self._names)[i]
            return type(
                "_S",
                (),
                {
                    "getId": lambda s, v=sid: v,
                    "getName": lambda s, v=self._names[sid]: v,
                },
            )()

        def getNumReactions(self):
            return len(self._rx)

        def getReaction(self, i):
            return self._rx[i]

    class _Reaction:
        def __init__(self, products, kinetic=True):
            self._p = products
            self._kinetic = kinetic

        def isSetKineticLaw(self):
            return self._kinetic

        def getNumProducts(self):
            return len(self._p)

        def getProduct(self, j):
            class _P:
                def __init__(self, s):
                    self._s = s

                def getSpecies(self):
                    return self._s

            return _P(self._p[j])

    def _patch(self, monkeypatch, model):
        monkeypatch.setattr(screens, "model_files", lambda *a: ["/tmp/x.xml"])
        monkeypatch.setattr(
            screens, "_first_readable_sbml", lambda paths: (model, "")
        )

    def test_a_reactionless_deposit_is_not_a_no_match(self, monkeypatch):
        """SBML-qual parses to an empty core model. Calling that `no-match` says
        the reactions were read and produced nothing — the opposite conclusion,
        and how three Boolean deposits read as screened negatives."""
        self._patch(monkeypatch, self._Model(n_species=0))
        (row,) = screens.screen_produced_species(["MODEL1"], "IL6")
        assert row.status == "no-reactions"
        assert "no reactions" in row.note

    def test_the_package_names_the_formalism(self, monkeypatch):
        """A reactionless deposit carrying the qual package is a logical
        model and one carrying fbc is constraint-based; each is reported as
        what it is, not lumped under an absence."""
        self._patch(monkeypatch, self._Model(n_species=3, packages=["qual"]))
        (row,) = screens.screen_produced_species(["MODEL1"], "IL6")
        assert row.status == "qualitative"
        assert "SBML-qual" in row.note and row.n_species == 3
        self._patch(monkeypatch, self._Model(n_species=3, packages=["fbc"]))
        (row,) = screens.screen_produced_species(["MODEL1"], "IL6")
        assert row.status == "constraint-based"
        assert "SBML-fbc" in row.note

    def test_no_match_still_means_read_and_absent(self, monkeypatch):
        self._patch(
            monkeypatch,
            self._Model(n_species=2, reactions=(self._Reaction(["TNFR"]),)),
        )
        (row,) = screens.screen_produced_species(["MODEL1"], r"\bIL6\b")
        assert row.status == "no-match"

    def test_produces_reports_the_matching_products(self, monkeypatch):
        self._patch(
            monkeypatch,
            self._Model(
                n_species=3,
                reactions=(self._Reaction(["IL6", "junk"]),),
            ),
        )
        (row,) = screens.screen_produced_species(["MODEL1"], r"IL6")
        assert row.status == "produces" and row.produced == ("IL6",)

    def test_matches_a_display_name_when_the_id_is_a_uuid(self, monkeypatch):
        """CellDesigner exports — a large part of BioModels — put the gene
        symbol in the name and a UUID in the id. Dwivedi2014 produces IL6 in
        three compartments and scored `no-match` on an id-only screen."""
        uuid = "mwf626e95e_543f_41e4_aad4_c6bf60ab345b"
        model = self._Model(
            reactions=(self._Reaction([uuid]),), names={uuid: "IL6"}
        )
        self._patch(monkeypatch, model)
        (row,) = screens.screen_produced_species(["MODEL1"], r"\bIL6\b")
        assert row.status == "produces"
        assert row.produced == ("IL6",), "report the name, not the UUID"

    def test_a_drawn_map_does_not_produce_anything(self, monkeypatch):
        """A CellDesigner disease map draws reactions with no rate law. Wu2010
        has 254 of them and 0 parameters; counting an arrow as production
        promotes a diagram to a model."""
        model = self._Model(
            reactions=(self._Reaction(["IL6"], kinetic=False),)
        )
        self._patch(monkeypatch, model)
        (row,) = screens.screen_produced_species(["MODEL1"], r"\bIL6\b")
        assert row.status == "no-rate-laws"
        assert row.n_reactions == 1

    def test_an_unfetchable_source_is_reported_not_dropped(self, monkeypatch):
        def refuse(model_id, source, timeout):
            raise LookupError(f"no SBML fetcher for source {source!r}")

        monkeypatch.setattr(screens, "model_files", refuse)
        (row,) = screens.screen_produced_species(["12345"], "IL6")
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
