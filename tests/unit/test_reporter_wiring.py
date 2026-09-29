"""Objective reporter-wiring checker — verdicts come from vendored
reference tables (MIRIAM ontology + CollecTRI), not hardcoded biology."""

import pytest

from hallsim.reporter_wiring import (
    ObservableKind,
    classify_ontology,
    classify_reporter,
    recommend_reporters,
    resolve_ontology,
    store_ontology_map,
    _human_symbol,
)


def test_classify_ontology_by_namespace():
    assert (
        classify_ontology({"chebi": "CHEBI:26523"})
        is ObservableKind.METABOLITE
    )
    assert classify_ontology({"uniprot": "P04637"}) is ObservableKind.PROTEIN
    assert classify_ontology({"go": "GO:0005739"}) is ObservableKind.PHYSICAL
    assert classify_ontology({"go": "GO:0006974"}) is ObservableKind.PROCESS
    assert classify_ontology({}) is ObservableKind.UNKNOWN


def test_uniprot_symbol_crosswalk_and_orthologs():
    assert _human_symbol("P04637") == ("TP53", None)  # human
    assert _human_symbol("O43524") == ("FOXO3", None)  # human
    assert _human_symbol("Q9Z1E3") == ("NFKBIA", None)  # mouse → ortholog
    assert _human_symbol("BOGUS123") == (
        None,
        "uniprot-missing",
    )  # loud on miss


@pytest.fixture(scope="module")
def composite():
    from demos.models.multi_hallmark import build_multi_hallmark_composite

    return build_multi_hallmark_composite(validate=False)


def test_observer_hop_resolves_to_annotated_source():
    """An unannotated derived path written by an observer with a single INPUT
    resolves to that source's annotation (one hop)."""
    from hallsim.composite import Composite
    from hallsim.process import Port, PortRole, Process

    class Src(Process):
        def ports_schema(self):
            return {
                "x": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    ontology={"uniprot": "P04637"},
                )
            }

        def derivative(self, t, state):
            return {"x": 0.0}

    class Observer(Process):
        def ports_schema(self):
            return {
                "x": Port(role=PortRole.INPUT, default=0.0),
                "derived": Port(role=PortRole.EVOLVED, default=0.0),
            }

        def derivative(self, t, state):
            return {"derived": state["x"]}

    comp = Composite(
        {"src": Src(), "obs": Observer()},
        topology={
            "src": {"x": "src/x"},
            "obs": {"x": "src/x", "derived": "obs/derived"},
        },
    )
    ont, resolved = resolve_ontology("obs/derived", comp)
    assert resolved == "src/x"
    assert ont.get("uniprot") == "P04637"


@pytest.mark.demo
@pytest.mark.slow
def test_multi_hallmark_reporter_verdicts(composite):
    from demos.models.multi_hallmark import MULTI_HALLMARK_REPORTERS

    valid = {
        "ok",
        "category-error",
        "self-map",
        "sign-conflict",
        "proxy",
        "tf-target-absent",
        "unannotated",
    }
    ontmap = store_ontology_map(composite)
    status = {
        r.key: classify_reporter(r, composite, ontmap).status
        for r in MULTI_HALLMARK_REPORTERS
    }
    # Central: every demo reporter classifies to a known status.
    for gene, s in status.items():
        assert s in valid, (gene, s)
    # Anchored cases with an unambiguous wiring interpretation:
    assert status["DDB2"] == "ok"  # gz06/x=TP53 → DDB2 CollecTRI target
    assert status["CDKN1A"] == "proxy"  # protein read as own transcript


@pytest.mark.demo
@pytest.mark.slow
def test_recommender_finds_foxo3_targets(composite):
    recs = recommend_reporters(composite, ["SOD2", "BNIP3", "IL6"])
    foxo = {(r["gene"], r["sign"]) for r in recs if r["tf"] == "FOXO3"}
    assert ("SOD2", 1) in foxo
    assert ("IL6", -1) in foxo  # FOXO3 represses IL6 in CollecTRI


def test_an_ambiguous_ortholog_is_unresolved_not_collapsed():
    """MGI lists a mouse gene once per human ortholog, so the MHC class I
    locus has several rows. Keeping the last resolved every one of them to
    whichever sorted last."""
    from hallsim.reporter_wiring import _orthologs

    table = _orthologs()
    assert table.get("H2-K1") is None
    assert table.get("H2-D1") is None
    assert table.get("Trp53") == "TP53"  # unambiguous rows still resolve


class TestChebiSkeletonMatching:
    """A deposit annotates the compound and an assay annotates the species it
    measured, so identity has to survive stereochemistry and protonation."""

    MET, L_MET, MET_ZW = "CHEBI:16811", "chebi:16643", "chebi:57844"
    L_LEU, L_ILE = "CHEBI:15603", "chebi:17191"
    AMINO_ACID, GLUCOSE_CLASS = "CHEBI:33709", "CHEBI:17234"

    def test_stereoisomer_and_zwitterion_share_the_parent_skeleton(self):
        from hallsim.reporter_wiring import _skeleton

        skel = {
            _skeleton("chebi", i) for i in (self.MET, self.L_MET, self.MET_ZW)
        }
        assert len(skel) == 1 and None not in skel

    def test_constitutional_isomers_do_not_collapse(self):
        from hallsim.reporter_wiring import _skeleton

        assert _skeleton("chebi", self.L_LEU) != _skeleton("chebi", self.L_ILE)

    def test_a_class_has_no_structure_and_so_no_skeleton(self):
        from hallsim.reporter_wiring import _skeleton

        assert _skeleton("chebi", self.AMINO_ACID) is None

    def test_another_namespace_is_untouched(self):
        from hallsim.reporter_wiring import _skeleton

        assert _skeleton("uniprot", "P04637") is None
        assert _skeleton("hgnc.symbol", "TP53") is None

    def test_the_parent_joins_its_measured_stereoisomer(self):
        from hallsim.reporter_wiring import paths_measuring

        got = paths_measuring(
            {"cell/met": {"chebi": self.MET}}, "chebi", [self.L_MET]
        )
        assert got == {"cell/met": self.L_MET}

    def test_an_exact_match_wins_and_keeps_the_callers_spelling(self):
        from hallsim.reporter_wiring import paths_measuring

        got = paths_measuring(
            {"cell/met": {"chebi": self.MET}},
            "chebi",
            ["chebi:16811", self.L_MET],
        )
        assert got == {"cell/met": "chebi:16811"}

    def test_an_isomer_the_panel_measured_is_not_a_match(self):
        from hallsim.reporter_wiring import paths_measuring

        assert (
            paths_measuring(
                {"cell/leu": {"chebi": self.L_LEU}}, "chebi", [self.L_ILE]
            )
            == {}
        )

    def test_a_structureless_parent_still_misses(self):
        """ChEBI gives some parents a structure and not others; `glucose` has
        none, so it cannot reach D-glucose this way."""
        from hallsim.reporter_wiring import paths_measuring

        assert (
            paths_measuring(
                {"cell/glc": {"chebi": self.GLUCOSE_CLASS}},
                "chebi",
                ["chebi:4167"],
            )
            == {}
        )

    def test_exact_matching_survives_a_missing_table(self, monkeypatch):
        import hallsim.reporter_wiring as rw

        monkeypatch.setattr(rw, "_chebi_skeletons", lambda: {})
        ontmap = {"cell/met": {"chebi": self.MET}}
        assert rw.paths_measuring(ontmap, "chebi", ["chebi:16811"]) == {
            "cell/met": "chebi:16811"
        }
        assert rw.paths_measuring(ontmap, "chebi", [self.L_MET]) == {}

    def test_derive_readouts_joins_end_to_end(self):
        import pandas as pd

        from hallsim.composite import Composite
        from hallsim.gene_reporters import derive_readouts
        from hallsim.metabolites import MetaboliteDataset
        from hallsim.process import Port, PortRole, Process

        class Cell(Process):
            def ports_schema(self):
                return {
                    "met": Port(
                        role=PortRole.EVOLVED,
                        default=1.0,
                        ontology={"chebi": "CHEBI:16811"},
                    )
                }

            def derivative(self, t, state):
                return {"met": 0.0}

        comp = Composite(
            {"cell": Cell()},
            topology={"cell": {"met": "cell/met"}},
            semantic_validation=False,
        )
        data = MetaboliteDataset(
            quantities=pd.DataFrame({"s1": [1.0]}, index=["chebi:16643"]),
            sample_groups={"a": ["s1"]},
        )
        (got,) = derive_readouts(comp, data)
        assert (got.path, got.key, got.provenance) == (
            "cell/met",
            "chebi:16643",
            "derived",
        )


def test_the_shipped_chebi_table_is_what_the_join_assumes():
    """Properties of the vendored table itself, which the fixture subset
    cannot show: the collapse must be bounded, and a class must be absent."""
    import csv
    import gzip
    from collections import Counter
    from pathlib import Path

    import hallsim

    path = (
        Path(hallsim.__file__).parent
        / "reference"
        / "ontology"
        / "chebi_skeleton.tsv.gz"
    )
    assert path.exists(), "the vendored ChEBI skeleton table is missing"
    with gzip.open(path, "rt", newline="") as fh:
        table = {
            r["chebi_id"]: r["skeleton"]
            for r in csv.DictReader(fh, delimiter="\t")
        }

    assert len(table) > 150_000
    assert all(len(s) == 14 for s in table.values())
    # the case that motivated it
    assert table["16811"] == table["16643"] == table["57844"]
    # constitutional isomers stay apart
    assert table["15603"] != table["17191"]
    # a structureless class has no row at all, so it cannot match anything
    for klass in ("33709", "17234", "33575"):
        assert klass not in table
    # the collapse is bounded: no skeleton stands for a large set of compounds
    biggest = Counter(table.values()).most_common(1)[0]
    assert biggest[1] <= 80, f"{biggest[0]} covers {biggest[1]} compounds"


@pytest.mark.network
def test_a_real_deposit_joins_a_real_panel_by_skeleton():
    """The case that found this: a hepatic methionine deposit against the
    Metabolomics Workbench panel whose ids are its stereoisomers."""
    from hallsim.composite import single_process_composite
    from hallsim.gene_reporters import derive_readouts
    from hallsim.metabolites import MetaboliteDataset
    from hallsim.sbml_import import process_from_sbml

    comp = single_process_composite(
        process_from_sbml("BIOMD0000000698", name="met"), "met"
    )
    data = MetaboliteDataset.from_workbench("ST000058")
    got = derive_readouts(comp, data)
    assert got, "no readout derived from a panel that measures methionine"
    assert any(r.key in set(data.measured) for r in got)
