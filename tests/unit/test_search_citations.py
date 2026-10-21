"""One hop through the citation graph, screened before it is returned."""

import pytest

from hallsim.search import citations, literature
from hallsim.search.citations import Neighbour, expand, identifier
from hallsim.search.literature import Paper
from hallsim.search.models import ModelCandidate


def _rec(pmid, title, cites, pmcid=None, doi=None, oa=False):
    return {
        "source": "MED",
        "id": pmid,
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "pubYear": "2020",
        "authorString": "A B",
        "citedByCount": cites,
        "isOpenAccess": "Y" if oa else "N",
    }


#: Europe PMC's view: the seed, three citers (one also in OpenAlex), and
#: the records a lookup returns for each identifier.
RECORDS = {
    "1": _rec("1", "Seed model", 500, doi="10.1/seed"),
    "11": _rec("11", "Extension with code", 40, pmcid="PMC11", oa=True),
    "12": _rec("12", "A review", 300, doi="10.1/review"),
    "13": _rec("13", "Closed follow-up", 2, doi="10.1/closed"),
}


class FakeGraphs:
    """Europe PMC lists 11 and 12 as citing the seed; OpenAlex lists 11,
    13 (by DOI only, no PMID) and the seed itself."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, params, timeout):
        self.calls.append(url)
        if url.endswith("/search"):
            query = params["query"]
            hits = [
                rec
                for pmid, rec in RECORDS.items()
                if f"EXT_ID:{pmid} " in query
                or (rec["doi"] and f'"{rec["doi"]}"' in query)
            ]
            return {"resultList": {"result": hits}}
        if url.endswith("/MED/1/citations"):
            rows = [
                {"source": "MED", "id": p, "title": RECORDS[p]["title"]}
                for p in ("11", "12")
            ]
            return {"hitCount": 2, "citationList": {"citation": rows}}
        if url.endswith("/MED/1/references"):
            # Europe PMC's reference rows carry no citation count.
            rows = [
                {"source": "MED", "id": p, "title": RECORDS[p]["title"]}
                for p in ("13", "12")
            ]
            return {"hitCount": 2, "referenceList": {"reference": rows}}
        if url.endswith("/works/pmid:1"):
            return {"id": "https://openalex.org/W1", "referenced_works": []}
        if url.endswith("/works") and params["filter"] == "cites:W1":
            if params["cursor"] != "*":
                return {"results": [], "meta": {"next_cursor": None}}
            return {
                "results": [
                    {
                        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/11"},
                        "title": "Extension with code",
                        "cited_by_count": 41,
                    },
                    {
                        "ids": {},
                        "doi": "https://doi.org/10.1/CLOSED",
                        "title": "Closed follow-up",
                        "cited_by_count": 3,
                    },
                ],
                "meta": {"next_cursor": "next"},
            }
        raise AssertionError(url)


@pytest.fixture
def graphs(monkeypatch):
    fake = FakeGraphs()
    monkeypatch.setattr(citations, "get_json", fake)
    monkeypatch.setattr(citations, "cached_json", lambda key, fetch: fetch())
    return fake


@pytest.mark.parametrize(
    "text, expected",
    [
        ("17111212", "MED:17111212"),
        ("PMID: 17111212", "MED:17111212"),
        ("pmc5919631", "PMC:PMC5919631"),
        (
            "https://doi.org/10.1007/S11538-023-01242-4",
            "DOI:10.1007/s11538-023-01242-4",
        ),
        ("PPR:PPR846460", "PPR:PPR846460"),
    ],
)
def test_identifiers_take_every_usual_spelling(text, expected):
    assert identifier(text) == expected


def test_identifier_refuses_a_title():
    with pytest.raises(ValueError):
        identifier("Kim 2007")


def test_expansion_pools_both_graphs_and_drops_the_seed(graphs):
    found = expand(["PMID:1"], directions=("cited-by",))
    assert found.reached == {
        ("PMID:1", "cited-by", "europepmc"): 2,
        ("PMID:1", "cited-by", "openalex"): 2,
    }
    by = {n.paper.pmid: n for n in found.neighbours}
    assert set(by) == {"11", "12", "13"}
    assert by["11"].graphs == ("europepmc", "openalex")
    assert by["11"].paper.pmcid == "PMC11"
    assert by["13"].graphs == ("openalex",)
    assert by["12"].via == (("PMID:1", "cited-by"),)


def test_the_cap_keeps_the_most_cited(graphs):
    found = expand(["PMID:1"], directions=("cited-by",), per_seed=2)
    assert {n.paper.pmid for n in found.neighbours} == {"12", "11"}


def test_references_without_counts_are_ranked_on_their_records(graphs):
    found = expand(
        ["PMID:1"], directions=("cites",), graphs=("europepmc",), per_seed=1
    )
    assert [n.paper.pmid for n in found.neighbours] == ["12"]


def test_a_lookup_keeps_the_readable_record_of_a_paper(monkeypatch):
    preprint = {
        "source": "PPR",
        "id": "PPR9",
        "doi": "10.1/shared",
        "title": "Preprint",
        "isOpenAccess": "N",
    }
    published = _rec("9", "Published", 5, pmcid="PMC9", oa=True)
    published["doi"] = "10.1/shared"
    monkeypatch.setattr(
        citations,
        "get_json",
        lambda url, params, timeout: {
            "resultList": {"result": [preprint, published]}
        },
    )
    monkeypatch.setattr(citations, "cached_json", lambda key, fetch: fetch())
    found = citations.lookup_papers(["10.1/shared"])
    assert found["DOI:10.1/shared"].pmcid == "PMC9"


def test_a_client_error_is_not_retried():
    import urllib.error

    from hallsim.search.fetch import retrying

    calls = []

    def missing():
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

    with pytest.raises(urllib.error.HTTPError):
        retrying(missing)
    assert len(calls) == 1


def test_unknown_direction_is_refused(graphs):
    with pytest.raises(ValueError):
        expand(["PMID:1"], directions=("sideways",))


def _neighbour(pmid, pmcid=None, doi=None, oa=False):
    paper = Paper(pmid, pmcid, 2020, "J", f"paper {pmid}", "A", 0, doi, oa)
    return Neighbour(paper=paper, via=(("PMID:1", "cited-by"),), graphs=())


def _classified(cited, *, limit, timeout):
    return [
        (
            ModelCandidate(
                source=forge,
                id=pointer,
                name=pointer,
                format="matlab",
                url="",
                curated=False,
                kind="source:matlab",
            ),
            papers,
        )
        for (forge, pointer), papers in list(cited.items())[:limit]
    ]


def test_screen_returns_only_what_it_found_something_in(monkeypatch):
    readable = _neighbour("21", pmcid="PMC21", oa=True)
    with_file = _neighbour("22", pmcid="PMC22", oa=True)
    closed = _neighbour("23", doi="10.1/Closed")
    nothing = _neighbour("24", doi="10.1/none")
    pointers = {
        "PMC21": {"github": ["lab/model", "lab/second"], "zenodo": ["99"]},
        "PMC22": {},
    }
    monkeypatch.setattr(
        citations, "model_pointers", lambda pmcid, timeout: pointers[pmcid]
    )
    monkeypatch.setattr(
        citations,
        "supplementary_model_files",
        lambda pmcid, timeout: (
            ["/c/PMC22__model.cps"] if pmcid == "PMC22" else []
        ),
    )
    asked = []

    def naming(dois, timeout):
        asked.extend(dois)
        return {"10.1/closed": ["layton/exercise"], "10.1/none": []}

    monkeypatch.setattr(citations, "repositories_naming", naming)
    monkeypatch.setattr(citations, "classify_pointers", _classified)

    hits = citations.screen_neighbours(
        [readable, with_file, closed, nothing], repo_limit=2
    )
    by = {h.paper.pmid: h for h in hits}
    assert set(by) == {"21", "22", "23"}
    assert sorted(asked) == ["10.1/closed", "10.1/none"]
    assert [r.kind for r in by["21"].repositories] == [
        "source:matlab",
        "source:matlab",
    ]
    assert by["21"].pointers == {"zenodo": ["99"]}
    assert by["22"].model_files == ("/c/PMC22__model.cps",)
    assert by["23"].repositories[0].kind == "unclassified"
    assert by["23"].via == (("PMID:1", "cited-by"),)


def test_readme_search_attributes_each_hit_by_its_readme(monkeypatch):
    queries = []

    def search(query, timeout):
        queries.append(query)
        return ["lab/a", "lab/b"] if "10.1/a" in query else []

    readmes = {"lab/a": "Code for doi 10.1/A.", "lab/b": "see 10.1/x"}
    monkeypatch.setattr(literature, "_search_github", search)
    monkeypatch.setattr(literature, "_readme", lambda r, t: readmes[r])
    monkeypatch.setattr(literature, "cached_json", lambda key, fetch: fetch())

    dois = ["10.1/A"] + [f"10.1/{i}" for i in range(7)]
    found = literature.repositories_naming(dois)
    assert len(queries) == 2
    assert found["10.1/a"] == ["lab/a"]
    assert all(v == [] for k, v in found.items() if k != "10.1/a")


@pytest.mark.parametrize(
    "text, cites",
    [
        ("doi: 10.1371/pcbi.100123.", True),
        ("https://doi.org/10.1371/pcbi.100123)", True),
        ("doi: 10.1371/pcbi.1001234", False),
        ("doi: 10.1371/pcbi.100123.g001", False),
        ("doi: 10.1371/pcbi.100123-supp", False),
    ],
)
def test_a_readme_names_a_doi_not_a_longer_one(text, cites):
    assert literature._names_doi(text, "10.1371/pcbi.100123") is cites


def test_a_failed_readme_batch_leaves_out_only_its_own_dois(monkeypatch):
    def search(query, timeout):
        return ["lab/a"] if "10.1/0" in query else ["lab/broken"]

    def readme(repo, timeout):
        if repo == "lab/broken":
            raise OSError("rate limited")
        return "10.1/0"

    monkeypatch.setattr(literature, "_search_github", search)
    monkeypatch.setattr(literature, "_readme", readme)
    monkeypatch.setattr(literature, "cached_json", lambda key, fetch: fetch())

    found = literature.repositories_naming([f"10.1/{i}" for i in range(8)])
    assert found["10.1/0"] == ["lab/a"]
    assert "10.1/6" not in found and "10.1/7" not in found


def test_a_repository_without_a_readme_reads_as_empty(monkeypatch):
    import urllib.error

    def missing(url, params, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(literature, "get_json", missing)
    assert literature._readme("lab/none", 1.0) == ""


def test_a_supplement_timeout_is_not_kept_as_a_miss(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(literature, "cache_dir", lambda name: tmp_path)
    monkeypatch.setattr(literature.urllib.request, "urlopen", timeout)
    assert literature.supplementary_model_files("PMC1") == []
    assert not (tmp_path / "PMC1" / ".harvested").exists()


def test_harvested_files_exclude_the_cached_full_text(monkeypatch, tmp_path):
    monkeypatch.setattr(literature, "cache_dir", lambda name: tmp_path)
    paper = tmp_path / "PMC1"
    paper.mkdir()
    for name in (".harvested", "fulltext.xml", "PMC1__model.cps"):
        (paper / name).write_text("")
    found = literature.supplementary_model_files("PMC1")
    assert [p.name for p in found] == ["PMC1__model.cps"]


@pytest.mark.network
@pytest.mark.parametrize(
    "seed, direction, wanted",
    [
        # Kim, Saidel & Cabrera 2007: two Palumbo extensions and the
        # sex-specific model of Abo, Casella & Layton 2024.
        ("PMID:17111212", "cited-by", {"29698395", "37390762", "38228814"}),
        # Contrepois 2020: Li 2022, the exercise-inducible metabolite.
        ("PMID:32470399", "cited-by", {"35705806"}),
        # Palumbo 2018: the Hirsch 1991 clamp data it was fitted to.
        ("PMC5919631", "cites", {"2035626"}),
    ],
)
def test_expansion_reaches_the_known_descendants_and_sources(
    seed, direction, wanted
):
    found = expand([seed], directions=(direction,))
    assert wanted <= {n.paper.pmid for n in found.neighbours}


@pytest.mark.network
def test_a_closed_paper_is_screened_by_the_repositories_that_cite_it():
    found = expand(["PMID:17111212"], directions=("cited-by",))
    abo = [n for n in found.neighbours if n.paper.pmid == "38228814"]
    hits = citations.screen_neighbours(abo, supplements=False)
    kinds = {r.id: r.kind for r in hits[0].repositories}
    assert kinds["Layton-Lab/Sex-Specific-Exercise-Model"] == "source:matlab"
