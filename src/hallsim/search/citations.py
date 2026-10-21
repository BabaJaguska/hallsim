"""One hop through the citation graph from a set of seed papers.

A model's descendants — the extension, the code release, the reuse of its
data — cite it; the data it was fitted to is in what it cites. A text search
reaches neither unless the words happen to match. This expands seeds both
ways, through Europe PMC's citation lists and OpenAlex's, which overlap
without either containing the other, and screens what it reaches with the
same tests a first round gets: where the paper says its code lives, what that
repository holds, and whether its supplement carries a model file. Only a
paper the screen finds something in comes back, each saying which seed
reached it and in which direction.

    from hallsim.search.citations import expand, screen_neighbours
    found = expand(["PMID:17111212"], directions=("cited-by",))
    for hit in screen_neighbours(found.neighbours):
        print(hit.via, hit.paper.title, [r.kind for r in hit.repositories])

The fan-out is capped per seed and direction, most-cited first: a classic is
cited hundreds of times, and every reached paper costs a full-text read.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from hallsim.search.fetch import cached_json, get_json, retrying
from hallsim.search.literature import (
    EUROPEPMC,
    FORGE_URL,
    Paper,
    _year,
    classify_pointers,
    model_pointers,
    paper_from_record,
    repositories_naming,
    supplementary_model_files,
)
from hallsim.search.models import ModelCandidate

log = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org"
#: ``cited-by`` reaches the papers citing a seed, ``cites`` its references.
DIRECTIONS = ("cited-by", "cites")
GRAPHS = ("europepmc", "openalex")
#: Neighbours kept per seed and direction, most-cited first.
PER_SEED = 100
#: Pointer kinds that name a deposit rather than a forge.
DEPOSIT_KINDS = ("biomodels", "modeldb", "zenodo", "figshare")
_LOOKUP_BATCH = 50
_OPENALEX_BATCH = 50
_WORKERS = 8
_OPENALEX_FIELDS = "id,ids,doi,title,publication_year,cited_by_count"


@dataclass(frozen=True)
class Neighbour:
    """A paper one hop from a seed, and what the screen found in it.

    ``via`` lists every ``(seed, direction)`` that reached it, ``graphs``
    the citation graphs that hold the edge. The screen fills
    ``repositories`` (classified forge links), ``model_files`` (supplement
    files an importer or a translator reads) and ``pointers`` (deposit
    accessions such as BioModels or Zenodo).
    """

    paper: Paper
    via: tuple[tuple[str, str], ...]
    graphs: tuple[str, ...]
    repositories: tuple[ModelCandidate, ...] = ()
    model_files: tuple[str, ...] = ()
    pointers: dict = field(default_factory=dict, compare=False)

    @property
    def found(self) -> bool:
        return bool(self.repositories or self.model_files or self.pointers)


@dataclass(frozen=True)
class Expansion:
    """The seeds, how many papers each graph returned for each seed and
    direction before deduplication and the cap, and the distinct
    neighbours kept."""

    seeds: tuple[Paper, ...]
    reached: dict
    neighbours: tuple[Neighbour, ...]


# ── Identifiers ──────────────────────────────────────────────────────────


def identifier(text: str) -> str:
    """``MED:<pmid>``, ``PMC:<pmcid>``, ``PPR:<id>`` or ``DOI:<doi>`` for a
    PMID, PMC id, Europe PMC preprint id or DOI, written any usual way."""
    raw = text.strip()
    bare = re.sub(r"^(pmid|pmcid|pmc|ppr|doi|med)\s*:\s*", "", raw, flags=re.I)
    bare = re.sub(r"^https?://(dx\.)?doi\.org/", "", bare, flags=re.I)
    if re.fullmatch(r"\d+", bare):
        return f"MED:{bare}"
    if re.fullmatch(r"PMC\d+", bare, flags=re.I):
        return f"PMC:{bare.upper()}"
    if re.fullmatch(r"PPR\d+", bare, flags=re.I):
        return f"PPR:{bare.upper()}"
    if bare.startswith("10."):
        return f"DOI:{bare.lower()}"
    raise ValueError(f"not a PMID, PMC id, preprint id or DOI: {text!r}")


def _clause(ident: str) -> str:
    kind, value = ident.split(":", 1)
    if kind == "MED":
        return f"(EXT_ID:{value} AND SRC:MED)"
    if kind == "PPR":
        return f"(EXT_ID:{value} AND SRC:PPR)"
    if kind == "PMC":
        return f"PMCID:{value}"
    return f'DOI:"{value}"'


def _keys(rec: dict) -> list[str]:
    keys = [f"{rec.get('source')}:{rec.get('id')}"]
    if rec.get("pmid"):
        keys.append(f"MED:{rec['pmid']}")
    if rec.get("pmcid"):
        keys.append(f"PMC:{rec['pmcid']}")
    if rec.get("doi"):
        keys.append(f"DOI:{rec['doi'].lower()}")
    return keys


def _month() -> str:
    return time.strftime("%Y-%m")


def lookup_papers(
    identifiers: Iterable[str], *, timeout: float = 45.0
) -> dict[str, Paper]:
    """Europe PMC's record for each identifier, keyed by its
    :func:`identifier` form; one missing from Europe PMC is absent. Batched,
    and cached for a month — open access and citation counts change."""
    wanted = list(dict.fromkeys(identifier(i) for i in identifiers))
    out: dict[str, Paper] = {}
    for start in range(0, len(wanted), _LOOKUP_BATCH):
        batch = wanted[start : start + _LOOKUP_BATCH]
        query = " OR ".join(_clause(i) for i in batch)
        records = cached_json(
            f"europepmc-lookup|{query}|{_month()}",
            lambda q=query: retrying(
                lambda: get_json(
                    f"{EUROPEPMC}/search",
                    {
                        "query": q,
                        "format": "json",
                        "resultType": "lite",
                        "pageSize": 1000,
                    },
                    timeout,
                )
            )
            .get("resultList", {})
            .get("result", []),
        )
        for rec in records:
            paper = paper_from_record(rec)
            for key in _keys(rec):
                held = out.get(key)
                if held is None or _readability(paper) > _readability(held):
                    out[key] = paper
    return {i: out[i] for i in wanted if i in out}


def _readability(paper: Paper) -> tuple[bool, bool, bool]:
    """Ranks two records of one paper, such as a preprint and its published
    version: the one whose full text can be read, then the one in PMC, then
    the one in PubMed."""
    return (_readable(paper), bool(paper.pmcid), bool(paper.pmid))


def _identity(paper: Paper) -> str:
    if paper.pmid:
        return f"MED:{paper.pmid}"
    if paper.doi:
        return f"DOI:{paper.doi.lower()}"
    if paper.pmcid:
        return f"PMC:{paper.pmcid}"
    return "TITLE:" + re.sub(r"\W+", " ", paper.title.lower()).strip()


# ── The two graphs ───────────────────────────────────────────────────────


def _stub(title, year, authors="", journal="", cited_by=0, doi=None):
    return Paper(
        pmid=None,
        pmcid=None,
        year=year,
        journal=journal or "",
        title=(title or "").strip().rstrip("."),
        authors=(authors or "")[:120],
        cited_by=int(cited_by or 0),
        doi=doi,
        open_access=False,
    )


def europepmc_neighbours(
    seed: Paper, direction: str, *, timeout: float = 45.0
) -> list[tuple[str | None, Paper]]:
    """``(identifier, stub)`` for each paper Europe PMC lists as citing
    ``seed`` (``cited-by``) or cited by it (``cites``). A reference Europe
    PMC could not match to a record has no identifier."""
    if seed.pmid:
        src, ext = "MED", seed.pmid
    elif seed.pmcid:
        src, ext = "PMC", seed.pmcid
    else:
        return []
    endpoint = "citations" if direction == "cited-by" else "references"
    holder = "citationList" if direction == "cited-by" else "referenceList"
    item = holder.removesuffix("List")
    rows: list[dict] = []
    page = 1
    while True:
        payload = cached_json(
            f"europepmc-{endpoint}|{src}:{ext}|{page}|{_month()}",
            lambda p=page: retrying(
                lambda: get_json(
                    f"{EUROPEPMC}/{src}/{ext}/{endpoint}",
                    {"format": "json", "pageSize": 1000, "page": p},
                    timeout,
                )
            ),
        )
        batch = (payload.get(holder) or {}).get(item) or []
        rows += batch
        if len(batch) < 1000 or len(rows) >= int(payload.get("hitCount", 0)):
            break
        page += 1
    out = []
    for rec in rows:
        ident = None
        if rec.get("id") and rec.get("source") in ("MED", "PPR"):
            ident = f"{rec['source']}:{rec['id']}"
        stub = _stub(
            rec.get("title"),
            _year(rec.get("pubYear")),
            rec.get("authorString"),
            rec.get("journalAbbreviation"),
            rec.get("citedByCount"),
        )
        out.append((ident, stub))
    return out


def _openalex_get(path: str, params: dict, timeout: float) -> dict:
    return retrying(lambda: get_json(f"{OPENALEX}{path}", params, timeout))


def _openalex_row(work: dict) -> tuple[str | None, Paper]:
    ids = work.get("ids") or {}
    pmid = (ids.get("pmid") or "").rsplit("/", 1)[-1] or None
    doi = (work.get("doi") or ids.get("doi") or "").removeprefix(
        "https://doi.org/"
    ) or None
    ident = f"MED:{pmid}" if pmid else (f"DOI:{doi.lower()}" if doi else None)
    stub = _stub(
        work.get("title"),
        work.get("publication_year"),
        cited_by=work.get("cited_by_count"),
        doi=doi,
    )
    return ident, replace(stub, pmid=pmid)


def openalex_neighbours(
    seed: Paper, direction: str, *, timeout: float = 45.0
) -> list[tuple[str | None, Paper]]:
    """``(identifier, stub)`` for each paper OpenAlex lists as citing
    ``seed`` or cited by it. OpenAlex resolves a seed by PMID or DOI."""
    if seed.pmid:
        handle = f"pmid:{seed.pmid}"
    elif seed.doi:
        handle = f"doi:{seed.doi}"
    else:
        return []
    month = _month()
    work = cached_json(
        f"openalex-work|{handle}|{month}",
        lambda: _openalex_get(
            f"/works/{handle}", {"select": "id,referenced_works"}, timeout
        ),
    )
    wid = work["id"].rsplit("/", 1)[-1]
    works: list[dict] = []
    if direction == "cited-by":
        cursor = "*"
        while cursor:
            page = cached_json(
                f"openalex-cites|{wid}|{cursor}|{month}",
                lambda c=cursor: _openalex_get(
                    "/works",
                    {
                        "filter": f"cites:{wid}",
                        "per-page": 200,
                        "cursor": c,
                        "select": _OPENALEX_FIELDS,
                    },
                    timeout,
                ),
            )
            works += page.get("results", [])
            cursor = (page.get("meta") or {}).get("next_cursor")
            if not page.get("results"):
                break
    else:
        refs = [w.rsplit("/", 1)[-1] for w in work.get("referenced_works", [])]
        for start in range(0, len(refs), _OPENALEX_BATCH):
            batch = "|".join(refs[start : start + _OPENALEX_BATCH])
            page = cached_json(
                f"openalex-refs|{batch}",
                lambda b=batch: _openalex_get(
                    "/works",
                    {
                        "filter": f"openalex:{b}",
                        "per-page": _OPENALEX_BATCH,
                        "select": _OPENALEX_FIELDS,
                    },
                    timeout,
                ),
            )
            works += page.get("results", [])
    return [_openalex_row(w) for w in works]


_GRAPH = {
    "europepmc": europepmc_neighbours,
    "openalex": openalex_neighbours,
}


# ── Expansion ────────────────────────────────────────────────────────────


def _most_cited(rows, per_seed, timeout):
    """The ``per_seed`` most-cited of ``(identifier, stub)`` rows, each as
    Europe PMC's record where it has one, so every graph's row carries the
    same PMC id and DOI. A row its graph gave no citation count, as Europe
    PMC's reference lists give none, is looked up before it is ranked."""
    counted = lookup_papers(
        (i for i, stub in rows if i and not stub.cited_by), timeout=timeout
    )
    rows = sorted(
        ((i, counted.get(i, stub)) for i, stub in rows),
        key=lambda row: -row[1].cited_by,
    )[:per_seed]
    found = lookup_papers(
        (i for i, _ in rows if i and i not in counted), timeout=timeout
    )
    return [found.get(i, paper) if i else paper for i, paper in rows]


def _walk(seed, direction, graph, per_seed, timeout):
    """``(how many papers the graph lists, the most-cited of them)``; a
    graph that fails lists none."""
    try:
        rows = _GRAPH[graph](seed, direction, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - one graph, not all
        log.warning(
            "%s %s of %s failed: %s", graph, direction, seed.label, exc
        )
        return 0, []
    return len(rows), _most_cited(rows, per_seed, timeout)


def expand(
    seeds: Sequence[str | Paper],
    *,
    directions: Sequence[str] = DIRECTIONS,
    graphs: Sequence[str] = GRAPHS,
    per_seed: int = PER_SEED,
    timeout: float = 45.0,
    workers: int = _WORKERS,
) -> Expansion:
    """Every paper one hop from ``seeds``, deduplicated across graphs and
    seeds, at most ``per_seed`` per seed and direction (most-cited first).
    A seed is a :class:`Paper` or a PMID, PMC id or DOI. Each seed,
    direction and graph is walked concurrently. Unscreened — pass the
    result to :func:`screen_neighbours`."""
    bad = set(directions) - set(DIRECTIONS) | set(graphs) - set(GRAPHS)
    if bad:
        raise ValueError(
            f"unknown {sorted(bad)}; directions {DIRECTIONS}, graphs {GRAPHS}"
        )
    named = [s for s in seeds if isinstance(s, str)]
    resolved = lookup_papers(named, timeout=timeout) if named else {}
    papers = []
    for s in seeds:
        if isinstance(s, Paper):
            papers.append(s)
        elif identifier(s) in resolved:
            papers.append(resolved[identifier(s)])
        else:
            log.warning("seed %s is not in Europe PMC; skipped", s)
    seed_ids = {_identity(p) for p in papers}

    walks = [
        (seed, direction, graph)
        for seed in papers
        for direction in directions
        for graph in graphs
    ]
    with ThreadPoolExecutor(workers) as pool:
        walked = dict(
            zip(
                walks,
                pool.map(
                    lambda w: _walk(*w, per_seed=per_seed, timeout=timeout),
                    walks,
                ),
            )
        )

    reached: dict[tuple[str, str, str], int] = {
        (seed.label, direction, graph): n
        for (seed, direction, graph), (n, _) in walked.items()
    }
    merged: dict[str, dict] = {}
    for seed in papers:
        for direction in directions:
            pooled: dict[str, tuple[Paper, set]] = {}
            for graph in graphs:
                rows = walked[(seed, direction, graph)][1]
                for paper in rows:
                    key = _identity(paper)
                    if key in seed_ids or key == "TITLE:":
                        continue
                    held, where = pooled.setdefault(key, (paper, set()))
                    if paper.pmcid and not held.pmcid:
                        pooled[key] = (paper, where)
                    where.add(graph)
            ranked = sorted(
                pooled.items(),
                key=lambda kv: (-kv[1][0].cited_by, -(kv[1][0].year or 0)),
            )[:per_seed]
            for key, (paper, where) in ranked:
                entry = merged.setdefault(
                    key, {"paper": paper, "via": [], "graphs": set()}
                )
                entry["via"].append((seed.label, direction))
                entry["graphs"] |= where
    neighbours = tuple(
        Neighbour(
            paper=e["paper"],
            via=tuple(e["via"]),
            graphs=tuple(g for g in GRAPHS if g in e["graphs"]),
        )
        for e in merged.values()
    )
    log.info(
        "expanded %d seed(s): %d distinct neighbours",
        len(papers),
        len(neighbours),
    )
    return Expansion(tuple(papers), reached, neighbours)


# ── Screening ────────────────────────────────────────────────────────────


def _readable(paper: Paper) -> bool:
    return bool(paper.pmcid and paper.open_access)


def _pointers(paper: Paper, timeout: float) -> dict:
    """Where the paper's full text says its code lives."""
    if not _readable(paper):
        return {}
    try:
        return model_pointers(paper.pmcid, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - one paper, not the run
        log.info("full text of %s unavailable: %s", paper.pmcid, exc)
        return {}


def _supplement(paper: Paper, timeout: float) -> tuple[str, ...]:
    if not (paper.pmcid and paper.open_access):
        return ()
    try:
        files = supplementary_model_files(paper.pmcid, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - one paper, not the run
        log.info("supplement of %s unavailable: %s", paper.pmcid, exc)
        return ()
    return tuple(str(f) for f in files)


def screen_neighbours(
    neighbours: Iterable[Neighbour],
    *,
    repo_limit: int = 20,
    supplements: bool = True,
    readme: bool = True,
    timeout: float = 60.0,
    workers: int = _WORKERS,
) -> list[Neighbour]:
    """The neighbours the screen finds something in, each filled with what:
    the forge repositories its text links, what each holds, its
    supplement's model files, and deposit accessions.

    A paper whose full text cannot be read is searched the other way round
    when ``readme`` is set: the GitHub repositories whose README names its
    DOI. Keyless, that is ten papers a minute, cached after the first run.
    ``repo_limit`` bounds the repositories classified, since the forges
    rate-limit keyless calls; one past it is still listed, as
    ``unclassified``."""
    items = list(neighbours)
    closed = (
        {
            n.paper.doi.lower(): index
            for index, n in enumerate(items)
            if n.paper.doi and not _readable(n.paper)
        }
        if readme
        else {}
    )
    with ThreadPoolExecutor(workers) as pool:
        naming = pool.submit(
            repositories_naming, list(closed), timeout=timeout
        )
        reading = pool.map(lambda n: _pointers(n.paper, timeout), items)
        harvesting = (
            pool.map(lambda n: _supplement(n.paper, timeout), items)
            if supplements
            else [()] * len(items)
        )
        pointers, files = list(reading), list(harvesting)
        try:
            named = naming.result()
        except Exception as exc:  # noqa: BLE001 - degrade, not fail
            log.warning("README search failed: %s", exc)
            named = {}
    for doi, repos in named.items():
        if repos:
            pointers[closed[doi]] = {"github": repos}
    cited: dict[tuple[str, str], list[int]] = {}
    for index, found in enumerate(pointers):
        for forge in FORGE_URL:
            for pointer in found.get(forge, []):
                cited.setdefault((forge, pointer), []).append(index)
    classified = {
        (cand.source, cand.id): cand
        for cand, _ in classify_pointers(
            cited, limit=repo_limit, timeout=timeout
        )
    }
    out = []
    for index, item in enumerate(items):
        repos = tuple(
            classified.get(
                (forge, pointer),
                ModelCandidate(
                    source=forge,
                    id=pointer,
                    name=pointer,
                    format="unknown",
                    url=FORGE_URL[forge].format(pointer),
                    curated=False,
                    kind="unclassified",
                ),
            )
            for forge in FORGE_URL
            for pointer in pointers[index].get(forge, [])
        )
        deposits = {
            k: v for k, v in pointers[index].items() if k in DEPOSIT_KINDS
        }
        hit = replace(
            item,
            repositories=repos,
            model_files=files[index],
            pointers=deposits,
        )
        if hit.found:
            out.append(hit)
    log.info("screened %d neighbours: %d with a finding", len(items), len(out))
    return out
