"""Search-to-paper-to-source discovery, including incomplete retrievals."""

import io
import json
from email.message import Message

import pytest
from click.testing import CliRunner

from hallsim import web_discovery as web
from hallsim.cli import simulate
from hallsim.discovery import ModelCandidate
from hallsim.literature import pointers_in


def test_forge_names_are_not_truncated_at_punctuation():
    found = pointers_in(
        "https://github.com/cosbi-research/FocmDownSyndrome "
        "https://github.com/lab/my-model.v2.git. "
        "https://zenodo.org/records/123456 "
        "https://bitbucket.org/lab/my-model/src/master/"
    )
    assert found["github"] == [
        "cosbi-research/FocmDownSyndrome",
        "lab/my-model.v2",
    ]
    assert found["bitbucket"] == ["lab/my-model"]
    assert found["zenodo"] == ["123456"]


def test_concatenated_pdf_links_do_not_become_repository_names():
    url = "https://github.com/cosbi-research/FocmDownSyndrome"
    assert pointers_in(url + url)["github"] == [
        "cosbi-research/FocmDownSyndrome"
    ]


def test_expansion_is_explicit_and_deduplicated():
    queries = web.query_plan(
        "Down syndrome",
        aliases=["trisomy 21", "Down syndrome"],
        mechanisms=["DYRK1A NFAT"],
    )
    assert len(queries) == 5
    assert any('"trisomy 21"' in q for q in queries)
    assert queries[-1] == 'DYRK1A NFAT "mathematical model"'
    assert not any("CBS" in q for q in queries)


def test_publisher_pdf_repository_chain_with_shared_provenance(monkeypatch):
    class Provider:
        def search(self, query, **kwargs):
            return [
                web.WebHit("Paper", "https://journal.test/article"),
                web.WebHit("Other", "https://journal.test/other"),
            ]

    requested = []

    def read(url, **kwargs):
        requested.append(url)
        if url.endswith("/article"):
            return web.Document(
                url,
                "Model in accepted manuscript",
                ["https://journal.test/model.pdf"],
            )
        if url.endswith(".pdf"):
            return web.Document(
                url, "Code https://github.com/cosbi-research/FocmDownSyndrome"
            )
        raise AssertionError("PDF must be prioritized within the budget")

    monkeypatch.setattr(web, "read_document", read)
    monkeypatch.setattr(
        web,
        "classify_repository",
        lambda pointer, forge, **kw: ModelCandidate(
            forge,
            pointer,
            pointer,
            "matlab",
            "https://github.com/" + pointer,
            False,
            kind="source:matlab",
            files=("FOCM_model.m",),
        ),
    )
    report = web.discover_models(
        "Down syndrome", provider=Provider(), europepmc=False, max_documents=2
    )
    repo = report.leads[0]
    assert repo.inspected and repo.candidate.kind == "source:matlab"
    assert len(repo.evidence) == 2  # both queries survive deduplication
    assert all(
        e.url.endswith("/model.pdf") and e.via.endswith("/article")
        for e in repo.evidence
    )
    assert requested == [
        "https://journal.test/article",
        "https://journal.test/model.pdf",
    ]
    assert report.deferred_documents == 1
    assert report.documents_read == 2
    assert not report.errors
    assert json.loads(json.dumps(report.to_dict()))["leads"][0]["candidate"][
        "files"
    ] == ["FOCM_model.m"]


def test_failures_keep_uninspected_link_and_explain_missing_results(
    monkeypatch,
):
    def fail(*args, **kwargs):
        raise OSError("service unavailable")

    monkeypatch.setattr(web, "read_document", fail)
    monkeypatch.setattr(web, "classify_repository", fail)
    report = web.discover_models(
        urls=["https://journal.test/article", "https://github.com/lab/model"],
        europepmc=False,
    )
    assert report.leads[0].candidate.kind == "linked-unverified"
    assert not report.leads[0].inspected
    assert [e["stage"] for e in report.errors] == ["document", "repository"]


def test_zero_budgets_never_fetch(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("budget is zero")

    monkeypatch.setattr(web, "read_document", fail)
    monkeypatch.setattr(web, "classify_repository", fail)
    report = web.discover_models(
        urls=["https://journal.test/article", "https://github.com/lab/model"],
        max_documents=0,
        max_repositories=0,
    )
    assert report.deferred_documents == report.deferred_repositories == 1
    assert not report.leads[0].inspected


def test_forge_pages_are_classified_not_read(monkeypatch):
    class Provider:
        def search(self, query, **kwargs):
            return [web.WebHit("Repo", "https://github.com/lab/from-search")]

    def fail(*args, **kwargs):
        pytest.fail("a forge page must not be read as a document")

    monkeypatch.setattr(web, "read_document", fail)
    monkeypatch.setattr(
        web,
        "classify_repository",
        lambda pointer, forge, **kw: ModelCandidate(
            forge,
            pointer,
            pointer,
            "python",
            "https://github.com/" + pointer,
            False,
            kind="source:python",
        ),
    )
    report = web.discover_models(
        "aging",
        urls=["https://gitlab.com/lab/seeded"],
        provider=Provider(),
        europepmc=False,
    )
    assert report.documents_read == report.deferred_documents == 0
    assert {lead.candidate.id for lead in report.leads if lead.inspected} == {
        "lab/from-search",
        "lab/seeded",
    }


class Response(io.BytesIO):
    def __init__(
        self,
        blob,
        content_type="text/html",
        url="https://journal.test/article",
    ):
        super().__init__(blob)
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.url = url

    def geturl(self):
        return self.url


def test_html_relative_pdf_and_hidden_repository_href(monkeypatch):
    html = b"""<meta name="citation_pdf_url" content="/accepted.pdf">
        <a href="https://github.com/lab/model">Code</a>
        <a href="/accepted.pdf">PDF</a><script>ignore me</script>"""
    monkeypatch.setattr(
        web.urllib.request, "urlopen", lambda *a, **kw: Response(html)
    )
    doc = web.read_document("https://journal.test/article")
    assert doc.pdf_links == ["https://journal.test/accepted.pdf"]
    assert pointers_in(doc.text)["github"] == ["lab/model"]
    assert "ignore me" not in doc.text


def test_pdf_reader_extracts_real_pdf(monkeypatch):
    pytest.importorskip("pypdf")
    # Minimal PDF with a content stream; no external PDF generator needed.
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 20 700 Td (https://github.com/lab/my-) Tj "
        b"0 -15 Td (model) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    monkeypatch.setattr(
        web.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(buf.getvalue(), "application/pdf"),
    )
    assert pointers_in(
        web.read_document("https://journal.test/model.pdf").text
    )["github"] == ["lab/my-model"]


def test_document_limits_and_scheme(monkeypatch):
    with pytest.raises(ValueError, match="HTTP"):
        web.read_document("file:///tmp/private")
    monkeypatch.setattr(web, "MAX_DOCUMENT_BYTES", 3)
    monkeypatch.setattr(
        web.urllib.request, "urlopen", lambda *a, **kw: Response(b"1234")
    )
    with pytest.raises(ValueError, match="retrieval limit"):
        web.read_document("https://journal.test/article")


def test_brave_api_contract(monkeypatch):
    def open_request(req, timeout):
        assert req.get_header("X-subscription-token") == "test-key"
        assert "count=3" in req.full_url
        return Response(
            json.dumps(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Model",
                                "url": "https://paper.test",
                                "description": "ODE",
                            }
                        ]
                    }
                }
            ).encode()
        )

    monkeypatch.setattr(web.urllib.request, "urlopen", open_request)
    assert web.BraveSearch("test-key").search("model", limit=3) == [
        web.WebHit("Model", "https://paper.test", "ODE")
    ]


def test_cli_no_key_and_url_only_report(monkeypatch, tmp_path):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(simulate, ["discover", "aging", "--web"])
    assert result.exit_code != 0 and "BRAVE_SEARCH_API_KEY" in result.output
    target = tmp_path / "report.json"
    result = runner.invoke(
        simulate,
        [
            "discover",
            "--url",
            "https://github.com/lab/model",
            "--no-papers",
            "--max-documents",
            "0",
            "--max-repositories",
            "0",
            "--json-output",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(target.read_text())["deferred_repositories"] == 1
    assert "linked-unverified" in result.output


def test_search_failure_is_reported_not_an_empty_success(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("search unavailable")

    monkeypatch.setattr(web, "search_europepmc", fail)
    result = CliRunner().invoke(simulate, ["discover", "aging"])
    assert result.exit_code != 0
    assert "search unavailable" in result.output


def test_pmc_xml_link_targets_are_read(monkeypatch):
    xml = b'<article><ext-link xlink:href="https://github.com/lab/model">code</ext-link></article>'
    monkeypatch.setattr(
        web.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(xml, "application/xml"),
    )
    assert pointers_in(web.read_document("https://journal.test/article").text)[
        "github"
    ] == ["lab/model"]


def test_classifier_label_is_reported_unchanged(monkeypatch):
    monkeypatch.setattr(
        web,
        "classify_repository",
        lambda pointer, forge, **kw: ModelCandidate(
            forge,
            pointer,
            pointer,
            "sbml",
            "https://github.com/" + pointer,
            False,
            kind="importable:sbml",
            files=("metadata.xml",),
        ),
    )
    report = web.discover_models(
        urls=["https://github.com/lab/model"], max_documents=0
    )
    assert report.leads[0].candidate.kind == "importable:sbml"
    assert report.leads[0].inspected


def test_pmc_fulltext_does_not_follow_relative_publisher_pdf(monkeypatch):
    paper = ModelCandidate(
        "europepmc",
        "PMC1",
        "Model paper",
        "paper",
        "https://europepmc.org/article/PMC/PMC1",
        False,
    )
    monkeypatch.setattr(web, "search_europepmc", lambda *a, **kw: [paper])
    visited = []

    def read(url, **kwargs):
        visited.append(url)
        return web.Document(
            url,
            "https://github.com/lab/model",
            ["https://bad-relative.test/model.pdf"],
        )

    monkeypatch.setattr(web, "read_document", read)
    report = web.discover_models("aging", max_repositories=0)
    assert len(visited) == 1 and visited[0].endswith("/fullTextXML")
    assert any(lead.candidate.id == "lab/model" for lead in report.leads)
    assert report.deferred_documents == 0
