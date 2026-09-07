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
        "jws",
        "europepmc",
        "modeldb",
        "biosimulations",
        "physiome",
    }


def test_every_source_takes_query_and_limit():
    """A source is called uniformly by `search_for_model`, so a thin
    late-import wrapper is unwrapped before its signature is read."""
    for name, search in discovery.SOURCES.items():
        target = search
        if search.__name__.startswith("_search_"):
            target = getattr(
                __import__("hallsim.literature", fromlist=["x"]),
                search.__name__.removeprefix("_"),
            )
        params = inspect.signature(target).parameters
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


class TestJWSOnline:
    """JWS Online is the second SBML-serving source. Its index is assembled
    from three endpoints, so the parts that can be checked without the network
    are the record shaping and the filters."""

    INDEX = [
        {
            "slug": "achcar2",
            "name": "achcar2",
            "cbm": False,
            "status": "CURATED",
            "title": "Topological and parameter uncertainty in glycolysis",
            "authors": "Achcar Barrett",
            "species": "Glucose (Glucose) ATP (ATP)",
            "reactions": "hexokinase (hexokinase)",
        },
        {
            "slug": "fba1",
            "name": "fba1",
            "cbm": True,  # constraint-based: no rate laws to integrate
            "status": "CURATED",
            "title": "A genome-scale reconstruction",
            "authors": "",
            "species": "Glucose (Glucose)",
            "reactions": "",
        },
        {
            "slug": "draft1",
            "name": "draft1",
            "cbm": False,
            "status": "SUBMITTED",
            "title": "An uncurated glycolysis draft",
            "authors": "",
            "species": "",
            "reactions": "",
        },
    ]

    def _patched(self, monkeypatch):
        monkeypatch.setattr(
            discovery, "cached_index", lambda *a, **k: self.INDEX
        )

    def test_registered_as_a_source(self):
        assert "jws" in discovery.SOURCES
        assert discovery.SOURCES["jws"] is discovery.search_jws

    def test_matches_on_title_and_shapes_a_candidate(self, monkeypatch):
        self._patched(monkeypatch)
        (hit,) = [
            c for c in discovery.search_jws("glycolysis") if c.id == "achcar2"
        ]
        assert hit.source == "jws"
        assert hit.format == "SBML"
        assert hit.curated
        assert "jjj.bio.vu.nl" in hit.url

    def test_matches_on_species_a_title_never_writes(self, monkeypatch):
        """The miss `search_by_gene` exists to close, on the other source: a
        model whose title omits the molecule is still reachable through the
        species it contains."""
        self._patched(monkeypatch)
        assert [c.id for c in discovery.search_jws("hexokinase")] == [
            "achcar2"
        ]

    def test_drops_constraint_based_models(self, monkeypatch):
        """A `cbm` model is stoichiometry with no rate laws — solved by linear
        programming, not integrated, so importing it yields no dynamics."""
        self._patched(monkeypatch)
        assert "fba1" not in [c.id for c in discovery.search_jws("genome")]

    def test_uncurated_returned_by_default_and_flagged(self, monkeypatch):
        """Curation status is the repository's editorial queue, not a property
        of the model, so it is reported rather than filtered on — `triage_sbml`
        is the admission test."""
        self._patched(monkeypatch)
        (hit,) = [c for c in discovery.search_jws("draft") if c.id == "draft1"]
        assert hit.curated is False
        assert hit.curation.upper() != "CURATED"

    def test_curated_only_restricts_on_request(self, monkeypatch):
        self._patched(monkeypatch)
        assert "draft1" not in [
            c.id for c in discovery.search_jws("draft", curated_only=True)
        ]


def test_jws_source_scheme_resolves_without_touching_biomodels(monkeypatch):
    """`jws:<slug>` must route to JWS, not fall through to a BioModels fetch."""
    from hallsim import sbml_import

    monkeypatch.setattr(
        sbml_import, "_download_jws_to_cache", lambda slug: f"/tmp/{slug}.xml"
    )
    monkeypatch.setattr(
        sbml_import,
        "_download_biomodel_to_cache",
        lambda i: pytest.fail("routed to BioModels"),
    )
    path, name = sbml_import._resolve_source("jws:achcar2", None)
    assert path.endswith("achcar2.xml")
    assert name == "jws_achcar2"


class TestBioModelsSearchFilters:
    """A filter applied to a server-truncated page silently shrinks the result
    set, and the shrunk set reads as "the repository has nothing"."""

    def _payload(self, n_uncurated=6, n_curated=6, n_nonsbml=6):
        models = (
            [
                {"id": f"MODEL230700000{i}", "name": f"u{i}", "format": "SBML"}
                for i in range(n_uncurated)
            ]
            + [
                {"id": f"BIOMD000000000{i}", "name": f"c{i}", "format": "SBML"}
                for i in range(n_curated)
            ]
            + [
                {
                    "id": f"MODEL230800000{i}",
                    "name": f"m{i}",
                    "format": "MATLAB",
                }
                for i in range(n_nonsbml)
            ]
        )
        return {"matches": 999, "models": models}

    def _spy(self, monkeypatch, payload=None):
        seen = {}

        def fake(url, params, timeout):
            seen.update(params)
            return payload if payload is not None else self._payload()

        monkeypatch.setattr(discovery, "_get_json", fake)
        return seen

    def test_uncurated_returned_by_default_and_flagged(self, monkeypatch):
        """Curation status is EBI's editorial queue, not a property of the
        model; `triage_sbml` is the admission test."""
        self._spy(monkeypatch)
        got = discovery.search_biomodels("q", limit=25)
        uncurated = [c for c in got if not c.curated]
        assert uncurated, "MODEL accessions must survive the default search"
        assert all(c.id.startswith("MODEL") for c in uncurated)
        assert any(c.curated for c in got)

    def test_curated_only_restricts_on_request(self, monkeypatch):
        self._spy(monkeypatch)
        got = discovery.search_biomodels("q", limit=25, curated_only=True)
        assert got and all(c.curated for c in got)

    def test_overfetches_so_a_filter_cannot_eat_the_page(self, monkeypatch):
        seen = self._spy(monkeypatch)
        discovery.search_biomodels("q", limit=4)
        assert seen["numResults"] == 4 * discovery.OVERFETCH

    def test_trims_after_filtering_not_before(self, monkeypatch):
        """12 SBML hits survive `sbml_only` out of 18 fetched; asking for 6
        must return 6, not 6-minus-whatever the format filter removed."""
        self._spy(monkeypatch)
        assert len(discovery.search_biomodels("q", limit=6)) == 6

    def test_a_format_heavy_page_still_yields_its_sbml(self, monkeypatch):
        """The failure this guards: every early hit is unimportable, so a
        limit-sized fetch would have returned nothing at all."""
        self._spy(
            monkeypatch,
            self._payload(n_uncurated=0, n_curated=2, n_nonsbml=20),
        )
        got = discovery.search_biomodels("q", limit=2)
        assert [c.id for c in got] == ["BIOMD0000000000", "BIOMD0000000001"]


class TestProducedSpeciesScreen:
    class _Model:
        def __init__(self, n_species=0, reactions=(), names=None):
            self._n, self._rx = n_species, reactions
            if names is None:
                # No display names declared: id is the label, as in a
                # hand-written or COPASI-exported deposit.
                names = {p: p for r in reactions for p in getattr(r, "_p", ())}
            self._names = names

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
        monkeypatch.setattr(
            discovery, "_sbml_paths_for", lambda *a: ["/tmp/x.xml"]
        )
        monkeypatch.setattr(
            discovery, "_first_readable_sbml", lambda paths: (model, "")
        )

    def test_a_reactionless_deposit_is_not_a_no_match(self, monkeypatch):
        """SBML-qual parses to an empty core model. Calling that `no-match` says
        the reactions were read and produced nothing — the opposite conclusion,
        and how three Boolean deposits read as screened negatives."""
        self._patch(monkeypatch, self._Model(n_species=0))
        (row,) = discovery.screen_produced_species(["MODEL1"], "IL6")
        assert row.status == "no-reactions"
        assert "no reactions" in row.note

    def test_no_match_still_means_read_and_absent(self, monkeypatch):
        self._patch(
            monkeypatch,
            self._Model(n_species=2, reactions=(self._Reaction(["TNFR"]),)),
        )
        (row,) = discovery.screen_produced_species(["MODEL1"], r"\bIL6\b")
        assert row.status == "no-match"

    def test_produces_reports_the_matching_products(self, monkeypatch):
        self._patch(
            monkeypatch,
            self._Model(
                n_species=3,
                reactions=(self._Reaction(["IL6", "junk"]),),
            ),
        )
        (row,) = discovery.screen_produced_species(["MODEL1"], r"IL6")
        assert row.status == "produces" and row.produced == ("IL6",)

    def test_matches_a_display_name_when_the_id_is_a_uuid(self, monkeypatch):
        """CellDesigner exports — a large part of BioModels — put the gene
        symbol in the name and a UUID in the id. Dwivedi2014 produces IL6 in
        three compartments and scored `no-match` on an id-only screen."""
        uuid = "mwf626e95e_543f_41e4_aad4_c6bf60ab345b"
        model = TestProducedSpeciesScreen._Model(
            reactions=(TestProducedSpeciesScreen._Reaction([uuid]),),
            names={uuid: "IL6"},
        )
        monkeypatch.setattr(
            discovery, "_sbml_paths_for", lambda *a: ["/tmp/x.xml"]
        )
        monkeypatch.setattr(
            discovery, "_first_readable_sbml", lambda paths: (model, "")
        )
        (row,) = discovery.screen_produced_species(["MODEL1"], r"\bIL6\b")
        assert row.status == "produces"
        assert row.produced == ("IL6",), "report the name, not the UUID"

    def test_a_drawn_map_does_not_produce_anything(self, monkeypatch):
        """A CellDesigner disease map draws reactions with no rate law. Wu2010
        has 254 of them and 0 parameters; counting an arrow as production
        promotes a diagram to a model."""
        model = TestProducedSpeciesScreen._Model(
            reactions=(
                TestProducedSpeciesScreen._Reaction(["IL6"], kinetic=False),
            ),
        )
        monkeypatch.setattr(
            discovery, "_sbml_paths_for", lambda *a: ["/tmp/x.xml"]
        )
        monkeypatch.setattr(
            discovery, "_first_readable_sbml", lambda paths: (model, "")
        )
        (row,) = discovery.screen_produced_species(["MODEL1"], r"\bIL6\b")
        assert row.status == "no-rate-laws"
        assert row.n_reactions == 1
