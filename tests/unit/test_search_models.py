"""Model search adapters — the parts that are logic, not network.

The four registered sources differ in what they can do: BioModels answers a
query server-side; ModelDB, BioSimulations and Physiome expose a listing only
and are matched client-side against a cached index. These cover the matching,
the kwarg routing and the record normalisation, none of which need a network.
"""

import inspect

import pytest

from hallsim.search import fetch, models
from hallsim.search.models import ModelCandidate, _accepted, term_score


def test_all_sources_registered():
    assert set(models.SOURCES) == {
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
    for name, search in models.SOURCES.items():
        target = search
        if search.__name__.startswith("_search_"):
            target = getattr(
                __import__("hallsim.search.literature", fromlist=["x"]),
                search.__name__.removeprefix("_"),
            )
        params = inspect.signature(target).parameters
        assert "query" in params, name
        assert "limit" in params, name


def test_term_score_requires_every_term():
    assert term_score("dna damage", "DNA damage response") > 0
    # one term present, the other absent -> no match, so a two-word query
    # cannot return everything matching either word
    assert term_score("dna telomere", "DNA damage response") == 0
    assert term_score("", "anything") == 0


def test_term_score_matches_a_stem_at_a_word_boundary():
    assert term_score("senesc", "cellular senescence") > 0
    assert term_score("senesc", "senescent fibroblast") > 0


def test_term_score_does_not_match_mid_word():
    """Plain substring matching made 'ros' hit 'interossei', 'Rosenbaum' and
    'cross-bridge', so a Physiome search for reactive oxygen species returned
    hand muscles and cardiac cross-bridge models."""
    assert term_score("ros", "dorsal interossei I") == 0
    assert term_score("ros", "cross-bridge model of shortening heat") == 0
    assert term_score("ros", "Zeng, Laurita, Rosenbaum, Rudy, 1995") == 0
    assert term_score("ros", "reactive oxygen species ROS") > 0


def test_term_score_ranks_by_term_count():
    many = term_score("p53", "p53 p53 p53 oscillator")
    few = term_score("p53", "p53 oscillator")
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
        models.search_for_model("x", sources=["nosuchrepo"])


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
    from hallsim.search.models import _accession

    assert _accession(10) == "BIOMD0000000010"
    assert _accession(632) == "BIOMD0000000632"
    # a string accession passes through, including the uncurated MODEL branch
    assert _accession("MODEL2307050001") == "MODEL2307050001"


def test_record_filenames_covers_main_and_additional():
    from hallsim.search.models import _record_filenames

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
    from hallsim.search.models import _from_record

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
    from hallsim.search.models import _from_record

    # A BIOMD accession would be guessed curated; the record says otherwise
    # and must win, because that is the field that decides whether ontology
    # and unit annotations exist.
    c = _from_record("BIOMD0000000632", {"curationStatus": "NON_CURATED"})
    assert c.curated is False
    c = _from_record("BIOMD0000000632", {"curationStatus": "CURATED"})
    assert c.curated is True


def test_from_record_falls_back_to_the_prefix_when_unstated():
    from hallsim.search.models import _from_record

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


# --- gene search: symbol and accession reach different models --------------


def test_search_by_gene_unions_symbol_and_accession_hits(monkeypatch):
    """BioModels indexes MIRIAM annotations as well as free text, so a model
    whose species are annotated but whose title never writes the symbol is
    invisible to a symbol search. Measured: TP53 returns 4 hits by symbol and
    32 by P04637, and the miss runs both ways."""
    from hallsim.search import models

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

    monkeypatch.setattr(models, "uniprot_accessions", fake_accessions)
    monkeypatch.setattr(models, "search_biomodels", fake_search)

    got = models.search_by_gene("TP53")
    assert calls == ["TP53", "P04637"]
    # union, de-duplicated on the shared hit
    assert sorted(c.id for c in got) == ["BIOMD1", "BIOMD2", "BIOMD3"]


def test_uniprot_accessions_reads_the_rest_answer_once(monkeypatch, tmp_path):
    """UniProt answers a TSV with a header row; the accessions are kept on
    disk, so the second lookup of a symbol makes no request."""
    import io
    import urllib.request

    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _Resp(b"Entry\nP04637\nQ9H3D4\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert models.uniprot_accessions("TP53") == ("P04637", "Q9H3D4")
    assert models.uniprot_accessions("TP53") == ("P04637", "Q9H3D4")
    assert len(calls) == 1 and "gene_exact%3ATP53" in calls[0]


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
        monkeypatch.setattr(models, "cached_index", lambda *a, **k: self.INDEX)

    def test_registered_as_a_source(self):
        assert "jws" in models.SOURCES
        assert models.SOURCES["jws"] is models.search_jws

    def test_matches_on_title_and_shapes_a_candidate(self, monkeypatch):
        self._patched(monkeypatch)
        (hit,) = [
            c for c in models.search_jws("glycolysis") if c.id == "achcar2"
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
        assert [c.id for c in models.search_jws("hexokinase")] == ["achcar2"]

    def test_drops_constraint_based_models(self, monkeypatch):
        """A `cbm` model is stoichiometry with no rate laws — solved by linear
        programming, not integrated, so importing it yields no dynamics."""
        self._patched(monkeypatch)
        assert "fba1" not in [c.id for c in models.search_jws("genome")]

    def test_uncurated_returned_by_default_and_flagged(self, monkeypatch):
        """Curation status is the repository's editorial queue, not a property
        of the model, so it is reported rather than filtered on — `triage_sbml`
        is the admission test."""
        self._patched(monkeypatch)
        (hit,) = [c for c in models.search_jws("draft") if c.id == "draft1"]
        assert hit.curated is False
        assert hit.curation.upper() != "CURATED"

    def test_curated_only_restricts_on_request(self, monkeypatch):
        self._patched(monkeypatch)
        assert "draft1" not in [
            c.id for c in models.search_jws("draft", curated_only=True)
        ]


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

        monkeypatch.setattr(models, "get_json", fake)
        return seen

    def test_uncurated_returned_by_default_and_flagged(self, monkeypatch):
        """Curation status is EBI's editorial queue, not a property of the
        model; `triage_sbml` is the admission test."""
        self._spy(monkeypatch)
        got = models.search_biomodels("q", limit=25)
        uncurated = [c for c in got if not c.curated]
        assert uncurated, "MODEL accessions must survive the default search"
        assert all(c.id.startswith("MODEL") for c in uncurated)
        assert any(c.curated for c in got)

    def test_curated_only_restricts_on_request(self, monkeypatch):
        self._spy(monkeypatch)
        got = models.search_biomodels("q", limit=25, curated_only=True)
        assert got and all(c.curated for c in got)

    def test_overfetches_so_a_filter_cannot_eat_the_page(self, monkeypatch):
        seen = self._spy(monkeypatch)
        models.search_biomodels("q", limit=4)
        assert seen["numResults"] == 4 * models.OVERFETCH

    def test_trims_after_filtering_not_before(self, monkeypatch):
        """12 SBML hits survive `sbml_only` out of 18 fetched; asking for 6
        must return 6, not 6-minus-whatever the format filter removed."""
        self._spy(monkeypatch)
        assert len(models.search_biomodels("q", limit=6)) == 6

    def test_a_format_heavy_page_still_yields_its_sbml(self, monkeypatch):
        """The failure this guards: every early hit is unimportable, so a
        limit-sized fetch would have returned nothing at all."""
        self._spy(
            monkeypatch,
            self._payload(n_uncurated=0, n_curated=2, n_nonsbml=20),
        )
        got = models.search_biomodels("q", limit=2)
        assert [c.id for c in got] == ["BIOMD0000000000", "BIOMD0000000001"]


def test_biomodels_search_drops_phrase_quotes(monkeypatch):
    """The search endpoint answers a quoted phrase with HTTP 400."""
    from hallsim.search import models

    sent = {}

    def fake(url, params, timeout):
        sent.update(params)
        return {"models": []}

    monkeypatch.setattr(models, "get_json", fake)
    models.search_biomodels('"Down syndrome" AND glutathione')
    assert sent["query"] == "Down syndrome AND glutathione"


def test_sources_are_asked_in_parallel_and_kept_in_order(monkeypatch):
    import time

    from hallsim.search import models
    from hallsim.search.models import ModelCandidate

    def slow(tag):
        def search(query, limit=25, **_):
            time.sleep(0.4)
            return [ModelCandidate(tag, f"{tag}-1", tag, "sbml", "", False)]

        return search

    monkeypatch.setattr(
        models, "SOURCES", {"a": slow("a"), "b": slow("b"), "c": slow("c")}
    )
    started = time.perf_counter()
    hits = models.search_for_model("x")
    assert time.perf_counter() - started < 1.0  # not 3 x 0.4 s
    assert [h.source for h in hits] == ["a", "b", "c"]


def test_fetch_routes_a_paper_and_a_jws_hit_to_their_files(monkeypatch):
    seen = []
    monkeypatch.setattr(
        models,
        "model_files",
        lambda mid, src, timeout: seen.append((mid, src)) or ["/tmp/m.xml"],
    )
    for source, mid in (("europepmc", "PMC1"), ("jws", "glycolysis1")):
        c = ModelCandidate(source, mid, "n", "SBML", "u", False)
        assert str(c.fetch()) == "/tmp/m.xml"
    assert seen == [("PMC1", "europepmc"), ("glycolysis1", "jws")]


def test_biomodel_download_falls_back_to_the_records_main_file(monkeypatch):
    """Newer deposits keep the author's filename; the conventional
    ``<accession>_url.xml`` answers 400 and the record names the file."""
    import io
    import urllib.error
    import urllib.request

    served = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        served.setdefault("urls", []).append(url)
        if "_url.xml" in url:
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)
        return _Resp(b"<sbml/>")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        models,
        "biomodels_record",
        lambda acc, timeout=30.0: {
            "files": {"main": [{"name": "Csikasz-Nagy2006.xml"}]}
        },
    )
    text = models._biomodel_main_text("BIOMD0000001044")
    assert text == "<sbml/>"
    assert len(served["urls"]) == 2
    assert "Csikasz-Nagy2006.xml" in served["urls"][1]
