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
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

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
def test_flagship_reporter_verdicts(composite):
    from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS

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
        r.gene_symbol: classify_reporter(r, composite, ontmap).status
        for r in MULTI_HALLMARK_REPORTERS
    }
    # Central: every flagship reporter classifies to a known status.
    for gene, s in status.items():
        assert s in valid, (gene, s)
    # Anchored cases with an unambiguous wiring interpretation:
    assert status["DDB2"] == "ok"  # gz06/x=TP53 → DDB2 CollecTRI target
    assert status["CDKN1A"] == "proxy"  # protein read as own transcript
    assert status["NFKBIA"] == "unannotated"  # no MIRIAM on IkBat


@pytest.mark.demo
@pytest.mark.slow
def test_recommender_finds_foxo3_targets(composite):
    recs = recommend_reporters(composite, ["SOD2", "BNIP3", "IL6"])
    foxo = {(r["gene"], r["sign"]) for r in recs if r["tf"] == "FOXO3"}
    assert ("SOD2", 1) in foxo
    assert ("IL6", -1) in foxo  # FOXO3 represses IL6 in CollecTRI
