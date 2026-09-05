"""Discovery source adapters — the parts that are logic, not network.

The four registered sources differ in what they can do: BioModels answers a
query server-side; ModelDB, BioSimulations and Physiome expose a listing only
and are matched client-side against a cached index. These cover the matching,
the kwarg routing and the record normalisation, none of which need a network.
"""

import inspect

import pytest

from hallsim import discovery
from hallsim.discovery import ModelCandidate, _accepted, _score


def test_all_sources_registered():
    assert set(discovery.SOURCES) == {
        "biomodels",
        "modeldb",
        "biosimulations",
        "physiome",
    }


def test_every_source_takes_query_and_limit():
    for name, search in discovery.SOURCES.items():
        params = inspect.signature(search).parameters
        assert "query" in params, name
        assert "limit" in params, name


def test_score_requires_every_term():
    assert _score("dna damage", "DNA damage response") > 0
    # one term present, the other absent -> no match, so a two-word query
    # cannot return everything matching either word
    assert _score("dna telomere", "DNA damage response") == 0
    assert _score("", "anything") == 0


def test_score_matches_a_stem_at_a_word_boundary():
    assert _score("senesc", "cellular senescence") > 0
    assert _score("senesc", "senescent fibroblast") > 0


def test_score_does_not_match_mid_word():
    """Plain substring matching made 'ros' hit 'interossei', 'Rosenbaum' and
    'cross-bridge', so a Physiome search for reactive oxygen species returned
    hand muscles and cardiac cross-bridge models."""
    assert _score("ros", "dorsal interossei I") == 0
    assert _score("ros", "cross-bridge model of shortening heat") == 0
    assert _score("ros", "Zeng, Laurita, Rosenbaum, Rudy, 1995") == 0
    assert _score("ros", "reactive oxygen species ROS") > 0


def test_score_ranks_by_term_count():
    many = _score("p53", "p53 p53 p53 oscillator")
    few = _score("p53", "p53 oscillator")
    assert many > few


def test_accepted_drops_kwargs_a_source_does_not_take():
    def biomodels_like(query, limit=25, curated_only=True):
        pass

    def listing_like(query, limit=25, refresh=False):
        pass

    kwargs = {"curated_only": False}
    assert _accepted(biomodels_like, kwargs) == {"curated_only": False}
    # curated_only means nothing to a listing source; passing it would be a
    # TypeError, and one source's option must not break the others
    assert _accepted(listing_like, kwargs) == {}


def test_accepted_passes_everything_to_a_var_keyword_source():
    def anything(query, limit=25, **kw):
        pass

    assert _accepted(anything, {"curated_only": False}) == {
        "curated_only": False
    }


def test_search_for_model_rejects_an_unknown_source():
    with pytest.raises(KeyError):
        discovery.search_for_model("x", sources=["nosuchrepo"])


def test_candidate_kind_defaults_to_unknown():
    c = ModelCandidate(
        source="biomodels",
        id="BIOMD1",
        name="n",
        format="SBML",
        url="u",
        curated=True,
    )
    assert c.kind == "unknown"
    assert c.description == ""


def test_fetch_names_the_manual_route_for_unsupported_sources():
    c = ModelCandidate(
        source="modeldb",
        id="3343",
        name="n",
        format="XPP",
        url="https://modeldb.science/3343",
        curated=True,
    )
    with pytest.raises(NotImplementedError, match="modeldb.science/3343"):
        c.fetch()


# --- BioModels record + multi-file fetch -----------------------------------
# A search hit says what a model is called; the record says whether anyone
# curated it and what else the deposit ships. Both change decisions, and
# neither was reachable before.


def test_accession_pads_an_integer_id():
    from hallsim.discovery import _accession

    assert _accession(10) == "BIOMD0000000010"
    assert _accession(632) == "BIOMD0000000632"
    # a string accession passes through, including the uncurated MODEL branch
    assert _accession("MODEL2307050001") == "MODEL2307050001"


def test_record_filenames_covers_main_and_additional():
    from hallsim.discovery import _record_filenames

    record = {
        "files": {
            "main": [{"name": "il6_model.xml"}],
            "additional": [{"name": "ReadMe.txt"}, {"name": ""}],
        }
    }
    # the empty name is dropped, and additional files are not lost
    assert _record_filenames(record) == ("il6_model.xml", "ReadMe.txt")
    assert _record_filenames({}) == ()


def test_from_record_reads_curation_rather_than_guessing_it():
    from hallsim.discovery import _from_record

    # An uncurated deposit in the MODEL branch: the accession prefix and the
    # record agree here, but the record is the one that is authoritative.
    c = _from_record(
        "MODEL2307050001",
        {
            "name": "Sobotta2017 - IL-6-induced JAK1-STAT3-signaling",
            "curationStatus": "NON_CURATED",
            "publication": {"title": "Model Based Targeting"},
            "files": {"main": [{"name": "il6_model.xml"}]},
        },
    )
    assert c.curation == "NON_CURATED"
    assert c.curated is False
    assert c.publication == "Model Based Targeting"
    assert c.files == ("il6_model.xml",)


def test_from_record_curation_overrides_the_accession_guess():
    from hallsim.discovery import _from_record

    # A BIOMD accession would be guessed curated; the record says otherwise
    # and must win, because that is the field that decides whether ontology
    # and unit annotations exist.
    c = _from_record("BIOMD0000000632", {"curationStatus": "NON_CURATED"})
    assert c.curated is False
    c = _from_record("BIOMD0000000632", {"curationStatus": "CURATED"})
    assert c.curated is True


def test_from_record_falls_back_to_the_prefix_when_unstated():
    from hallsim.discovery import _from_record

    assert _from_record("BIOMD0000000632", {}).curated is True
    assert _from_record("MODEL2307050001", {}).curated is False


def test_candidate_record_and_fetch_all_refuse_non_biomodels_sources():
    c = ModelCandidate(
        source="modeldb",
        id="3343",
        name="n",
        format="XPP",
        url="https://modeldb.science/3343",
        curated=True,
    )
    for call in (c.record, c.fetch_all):
        with pytest.raises(NotImplementedError):
            call()


# --- SBML event delays -----------------------------------------------------
# Here only because it shares the "what a repository actually hands you" theme:
# COPASI writes <delay>0</delay> on every event it exports, so the presence of
# a delay element is not evidence of a delay.
#
# These drive _delay_seconds through stub objects rather than libsbml Events.
# libsbml segfaults the interpreter when an Event is assembled through its own
# API outside a fully-populated document, which is a fault in the library, not
# a reason to leave the branching untested. The real-deposit path is exercised
# by importing BIOMD0000000632, which is a network test.


class _Math:
    def __init__(self, number):
        self._number = number

    def isNumber(self):
        return self._number is not None


class _Delay:
    def __init__(self, math):
        self._math = math

    def isSetMath(self):
        return self._math is not None

    def getMath(self):
        return self._math


class _Event:
    def __init__(self, delay):
        self._delay = delay

    def getDelay(self):
        return self._delay


def test_absent_delay_reads_as_zero(monkeypatch):
    from hallsim import sbml_events

    assert sbml_events._delay_seconds(_Event(None)) == 0.0
    assert sbml_events._delay_seconds(_Event(_Delay(None))) == 0.0


def test_constant_delay_reads_its_value(monkeypatch):
    import libsbml

    from hallsim import sbml_events

    monkeypatch.setattr(libsbml, "formulaToL3String", lambda m: str(m._number))
    # the COPASI-emitted form: a delay element whose math is a literal zero
    assert sbml_events._delay_seconds(_Event(_Delay(_Math(0.0)))) == 0.0
    assert sbml_events._delay_seconds(_Event(_Delay(_Math(5.0)))) == 5.0


def test_nonconstant_delay_is_nan_so_it_is_rejected():
    import math

    from hallsim import sbml_events

    # a state- or time-dependent delay is not a number; NaN compares unequal
    # to zero, so the caller rejects it rather than silently dropping it
    assert math.isnan(sbml_events._delay_seconds(_Event(_Delay(_Math(None)))))


# --- Event trigger pathologies ---------------------------------------------
# Both are properties of the trigger expressions alone, so triage decides them
# without integrating anything. Both were originally found by a referee running
# tolerance sweeps in COPASI on Stucki 2005 (BIOMD0000001059).


class _Ev:
    def __init__(self, name, trigger_ir):
        self._name = name
        self._trigger_ir = trigger_ir


def test_complementary_triggers_sharing_a_boundary_are_caught():
    from hallsim.sbml_events import trigger_pathologies

    # cascade <= 20 (and c3 >= 4.5)   vs   cascade > 20
    a = _Ev(
        "latch_on",
        (
            "and",
            [
                ("leq", ("var", "cascade"), ("const", 20.0)),
                ("geq", ("var", "c3"), ("const", 4.5)),
            ],
        ),
    )
    b = _Ev("latch_off", ("gt", ("var", "cascade"), ("const", 20.0)))
    found = trigger_pathologies([a, b])
    assert any("round-off" in f and "hysteresis" in f for f in found), found


def test_a_hysteresis_band_is_not_flagged():
    from hallsim.sbml_events import trigger_pathologies

    # arm at 20, disarm at 18 — no value satisfies both, so no chatter
    a = _Ev("arm", ("gt", ("var", "cascade"), ("const", 20.0)))
    b = _Ev("disarm", ("lt", ("var", "cascade"), ("const", 18.0)))
    assert trigger_pathologies([a, b]) == []


def test_equality_against_time_is_caught():
    from hallsim.sbml_events import trigger_pathologies

    ev = _Ev("release", ("eq", ("time",), ("const", 2000.0)))
    found = trigger_pathologies([ev])
    assert any("equality against time" in f for f in found), found


def test_a_time_threshold_crossing_is_not_flagged():
    from hallsim.sbml_events import trigger_pathologies

    ev = _Ev("release", ("geq", ("time",), ("const", 2000.0)))
    assert trigger_pathologies([ev]) == []


# --- gene search: symbol and accession reach different models --------------


def test_search_by_gene_unions_symbol_and_accession_hits(monkeypatch):
    """BioModels indexes MIRIAM annotations as well as free text, so a model
    whose species are annotated but whose title never writes the symbol is
    invisible to a symbol search. Measured: TP53 returns 4 hits by symbol and
    32 by P04637, and the miss runs both ways."""
    from hallsim import discovery

    def fake_accessions(symbol, taxon=9606, timeout=30.0):
        return ("P04637",)

    calls = []

    def fake_search(term, limit=25, **kw):
        calls.append(term)
        by_term = {
            "TP53": ["BIOMD1", "BIOMD2"],
            "P04637": ["BIOMD2", "BIOMD3"],
        }
        return [
            ModelCandidate(
                source="biomodels",
                id=i,
                name=i,
                format="SBML",
                url="",
                curated=True,
            )
            for i in by_term.get(term, [])
        ]

    monkeypatch.setattr(discovery, "uniprot_accessions", fake_accessions)
    monkeypatch.setattr(discovery, "search_biomodels", fake_search)

    got = discovery.search_by_gene("TP53")
    assert calls == ["TP53", "P04637"]
    # union, de-duplicated on the shared hit
    assert sorted(c.id for c in got) == ["BIOMD1", "BIOMD2", "BIOMD3"]


def test_uniprot_accessions_prefers_the_local_table(monkeypatch):
    """The repo ships a small symbol table for its reporters; it is offline
    and instant, so it answers before any network call."""
    from hallsim import discovery

    def boom(*a, **k):
        raise AssertionError("should not hit the network")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", boom)
    assert discovery.uniprot_accessions("TP53") == ("P04637",)
