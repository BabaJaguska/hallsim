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


def test_score_is_substring_so_stems_match():
    assert _score("senesc", "cellular senescence") > 0
    assert _score("senesc", "senescent fibroblast") > 0


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
