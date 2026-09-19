"""Web and publisher discovery with explicit provenance and retrieval budgets."""

from __future__ import annotations

import io
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Protocol

from hallsim.discovery import ModelCandidate, USER_AGENT
from hallsim.literature import (
    FORGE_URL,
    classify_repository,
    pointers_in,
    search_europepmc,
)

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    def search(
        self, query: str, *, limit: int, timeout: float
    ) -> list[WebHit]:
        """Return public web results in the provider's relevance order."""
        ...


class BraveSearch:
    """Brave Web Search; reads BRAVE_SEARCH_API_KEY unless a key is supplied."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("Web search requires BRAVE_SEARCH_API_KEY.")

    def search(self, query, *, limit=10, timeout=30.0):
        if not 1 <= limit <= 20:
            raise ValueError("Brave search limit must be between 1 and 20")
        params = urllib.parse.urlencode({"q": query, "count": limit})
        req = urllib.request.Request(
            "https://api.search.brave.com/res/v1/web/search?" + params,
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
        return [
            WebHit(r.get("title", ""), r["url"], r.get("description", ""))
            for r in payload.get("web", {}).get("results", [])[:limit]
        ]


def query_plan(topic: str, *, aliases=(), mechanisms=()) -> list[str]:
    """Bounded keyword expansion; biological aliases are supplied by callers."""
    terms = list(
        dict.fromkeys(t.strip() for t in (topic, *aliases) if t.strip())
    )
    queries = []
    for term in terms:
        quoted = '"' + term.replace('"', "") + '"'
        queries.extend(
            [
                f'{quoted} ("mathematical model" OR "kinetic model" OR "computational model")',
                f"{quoted} (SBML OR code OR GitHub OR Bitbucket)",
            ]
        )
    for mechanism in dict.fromkeys(mechanisms):
        if mechanism.strip():
            queries.append(f'{mechanism.strip()} "mathematical model"')
    return queries


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.links = []
        self.pdfs = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self.hidden += 1
        href = attrs.get("href") or attrs.get("xlink:href")
        if href:
            self.links.append(href)
        if (
            tag == "meta"
            and attrs.get("name", "").lower() == "citation_pdf_url"
        ):
            if attrs.get("content"):
                self.pdfs.append(attrs["content"])

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


@dataclass
class Document:
    url: str
    text: str
    pdf_links: list[str] = field(default_factory=list)


def read_document(url: str, *, timeout=30.0) -> Document:
    """Read public HTML/XML/text or PDF, without executing page scripts."""
    if urllib.parse.urlsplit(url).scheme not in ("https", "http"):
        raise ValueError("Document URLs must use HTTP or HTTPS")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        blob = response.read(MAX_DOCUMENT_BYTES + 1)
        resolved = response.geturl()
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
    if len(blob) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 20 MB retrieval limit")
    if blob.startswith(b"%PDF-") or content_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError(
                'PDF reading requires pip install "hallsim[search]"'
            ) from exc
        reader = PdfReader(io.BytesIO(blob))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # Preserve hyphens inside URLs split across typeset lines.
        text = re.sub(
            r"(https?://[^\s]+[-/])[ \t]*\n[ \t]*(?=[\w])", r"\1", text
        )
        links = []
        for page in reader.pages:
            for ref in page.get("/Annots", []):
                action = ref.get_object().get("/A", {})
                if hasattr(action, "get_object"):
                    action = action.get_object()
                if action.get("/URI"):
                    links.append(str(action["/URI"]))
        text += "\n" + "\n".join(links)
        if not text.strip():
            raise ValueError(
                "PDF has no extractable text; OCR is not available"
            )
        return Document(resolved, text)
    if not (content_type.startswith("text/") or "xml" in content_type):
        raise ValueError(f"Unsupported document type: {content_type}")
    raw = blob.decode(charset, "replace")
    page = _Page()
    page.feed(raw)
    links = [urllib.parse.urljoin(resolved, link) for link in page.links]
    pdfs = [urllib.parse.urljoin(resolved, link) for link in page.pdfs]
    pdfs += [
        link
        for link in links
        if urllib.parse.urlsplit(link).path.lower().endswith(".pdf")
    ]
    # Link targets often carry repository names absent from the visible text.
    text = "\n".join(page.text + links)
    return Document(resolved, text, list(dict.fromkeys(pdfs)))


@dataclass
class Evidence:
    query: str
    url: str
    stage: str
    via: str = ""


@dataclass
class Lead:
    candidate: ModelCandidate
    evidence: list[Evidence] = field(default_factory=list)
    inspected: bool = False


@dataclass
class DiscoveryReport:
    queries: list[str]
    leads: list[Lead] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    documents_read: int = 0
    deferred_documents: int = 0
    deferred_repositories: int = 0

    def to_dict(self):
        return asdict(self)


_POINTER_URLS = {
    **FORGE_URL,
    "biomodels": "https://www.ebi.ac.uk/biomodels/{}",
    "zenodo": "https://zenodo.org/records/{}",
    "modeldb": "https://modeldb.science/{}",
    "figshare": "https://{}",
}
_FORGE_HOSTS = frozenset(
    urllib.parse.urlsplit(url.format("")).hostname
    for url in FORGE_URL.values()
)


def discover_models(
    topic: str = "",
    *,
    aliases=(),
    mechanisms=(),
    urls=(),
    provider: SearchProvider | None = None,
    europepmc=True,
    limit=10,
    max_documents=20,
    max_repositories=10,
    pdfs_per_page=2,
    timeout=30.0,
) -> DiscoveryReport:
    """Find papers and follow source links; nothing is executed or validated.

    Providers are optional. Seed URLs work without a search key. Limits bound
    search results per query, documents fetched, and repositories inspected.
    Failures and uninspected leads remain in the returned report.
    """
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    if min(max_documents, max_repositories, pdfs_per_page) < 0 or timeout <= 0:
        raise ValueError("Budgets must be nonnegative and timeout positive")
    if not topic.strip() and not urls:
        raise ValueError("Supply a topic or at least one URL")
    queries = query_plan(topic, aliases=aliases, mechanisms=mechanisms)
    report = DiscoveryReport(queries)
    leads = {}
    pending = {}

    def add(candidate, evidence):
        key = (candidate.source, candidate.id)
        lead = leads.setdefault(key, Lead(candidate))
        if evidence not in lead.evidence:
            lead.evidence.append(evidence)
        return lead

    def pointers(text, evidence):
        for source, identifiers in pointers_in(text).items():
            for identifier in identifiers:
                if source not in _POINTER_URLS:
                    continue
                add(
                    ModelCandidate(
                        source,
                        identifier,
                        identifier,
                        "unknown",
                        _POINTER_URLS[source].format(identifier),
                        False,
                        kind="linked-unverified",
                    ),
                    evidence,
                )

    def enqueue(url, evidence, follow_pdfs=True):
        url = urllib.parse.urldefrag(url)[0]
        # A forge page is a repository to classify through its API, not a
        # document to read: its HTML links the whole site.
        if urllib.parse.urlsplit(url).hostname in _FORGE_HOSTS:
            return
        entry = pending.setdefault(url, ([], follow_pdfs))
        if evidence not in entry[0]:
            entry[0].append(evidence)

    for url in urls:
        evidence = Evidence("", url, "seed")
        pointers(url, evidence)
        enqueue(url, evidence)

    for query in queries:
        if europepmc:
            try:
                papers = search_europepmc(query, limit=limit, timeout=timeout)
                for paper in papers:
                    evidence = Evidence(query, paper.url, "europepmc")
                    add(paper, evidence)
                    enqueue(
                        f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper.id}/fullTextXML",
                        evidence,
                        follow_pdfs=False,
                    )
            except Exception as exc:
                report.errors.append(
                    dict(stage="europepmc", query=query, error=str(exc))
                )
        if provider is not None:
            try:
                for hit in provider.search(
                    query, limit=limit, timeout=timeout
                ):
                    evidence = Evidence(query, hit.url, "search-snippet")
                    pointers(hit.url + "\n" + hit.snippet, evidence)
                    add(
                        ModelCandidate(
                            "web",
                            hit.url,
                            hit.title,
                            "page",
                            hit.url,
                            False,
                            kind="unverified",
                            description=hit.snippet,
                        ),
                        evidence,
                    )
                    enqueue(hit.url, evidence)
            except Exception as exc:
                report.errors.append(
                    dict(stage="web-search", query=query, error=str(exc))
                )

    visited = set()
    while len(visited) < max_documents:
        next_url = next((url for url in pending if url not in visited), None)
        if next_url is None:
            break
        visited.add(next_url)
        evidence_list, follow_pdfs = pending[next_url]
        try:
            doc = read_document(next_url, timeout=timeout)
            report.documents_read += 1
            for evidence in evidence_list:
                pointers(
                    doc.text,
                    Evidence(
                        evidence.query, doc.url, "document", evidence.url
                    ),
                )
            if follow_pdfs:
                for pdf in doc.pdf_links[:pdfs_per_page]:
                    for evidence in evidence_list:
                        enqueue(
                            pdf,
                            Evidence(evidence.query, doc.url, "linked-pdf"),
                            False,
                        )
                # Read a paper's PDFs before spending the budget on more hits.
                priority = {
                    url: pending[url]
                    for url in doc.pdf_links[:pdfs_per_page]
                    if url in pending
                }
                pending = {**priority, **pending}
        except Exception as exc:
            report.errors.append(
                dict(stage="document", url=next_url, error=str(exc))
            )
    report.deferred_documents = len(pending.keys() - visited)

    repositories = [
        lead for lead in leads.values() if lead.candidate.source in FORGE_URL
    ]
    for lead in repositories[:max_repositories]:
        candidate = lead.candidate
        try:
            lead.candidate = classify_repository(
                candidate.id, candidate.source, timeout=timeout
            )
            lead.inspected = True
        except Exception as exc:
            report.errors.append(
                dict(stage="repository", url=candidate.url, error=str(exc))
            )
    report.deferred_repositories = max(0, len(repositories) - max_repositories)
    # Artifact evidence is useful first; it does not establish biological relevance.
    report.leads = sorted(
        leads.values(),
        key=lambda lead: (
            not lead.inspected,
            lead.candidate.source in ("web", "europepmc"),
        ),
    )
    return report
