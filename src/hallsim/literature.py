"""Europe PMC as a model source — the papers whose models were never deposited.

Deduplicated across repositories, the curated kinetic-ODE corpus is a few
thousand models. Far more exist only as a file attached to a paper: a COPASI
``.cps``, an SBML export, a MATLAB script. Those are complete, parameterised
models that no repository indexes, and this reaches them.

Two stages, and only the first is implemented here because only the first is
mechanical:

1. **Harvest** — find open-access papers, pull their supplementary archives,
   and recover any model file inside. No inference: a ``.cps`` or an SBML file
   in a supplement *is* the model, and :mod:`hallsim.cps_import` already reads
   the former.
2. **Extraction** — recovering a model from the equations when no file was
   deposited. Not done here. It produces plausible-looking SBML, and plausible
   is the failure mode that matters, so it needs the triage gate in front of it
   before it is worth anything.

    from hallsim.literature import search_europepmc
    from hallsim.discovery import screen_produced_species
    hits = search_europepmc("senescence SASP kinetic model")
    screen_produced_species(hits, r"IL6|CXCL8|MMP1")
"""

from __future__ import annotations

import io
import json
import logging
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from hallsim.discovery import ModelCandidate

log = logging.getLogger(__name__)

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CACHE = Path.home() / ".cache" / "hallsim" / "europepmc"

#: Extensions worth opening. SBML is also detected by content, because a
#: supplement names it anything at all.
MODEL_SUFFIXES = (".cps", ".xml", ".sbml", ".m", ".ode", ".cellml")
#: Archives to descend into. A publisher wraps the real supplement in one.
ARCHIVE_SUFFIXES = (".zip",)
#: Depth limit on that descent — an archive containing itself is not a model.
MAX_DEPTH = 3


def _get_json(url: str, params: dict, timeout: float):
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{q}", timeout=timeout) as fh:
        return json.load(fh)


def search_europepmc(
    query: str,
    limit: int = 25,
    *,
    open_access_only: bool = True,
    timeout: float = 45.0,
    **_,
) -> list[ModelCandidate]:
    """Europe PMC full-text search, restricted to articles we can actually open.

    ``OPEN_ACCESS:y AND HAS_FT:y`` is what makes this useful: it keeps only
    papers whose full text and supplementary files are retrievable, which is
    the difference between a citation and a model.

    Precision is poor by construction — a text search cannot tell a review of
    IL-6 models from an IL-6 model. That is what
    :func:`hallsim.discovery.screen_produced_species` and
    :func:`hallsim.intake.triage_sbml` are for, and why the harvest is worth
    running wide.
    """
    q = query
    if open_access_only:
        q = f"({query}) AND OPEN_ACCESS:y AND HAS_FT:y"
    payload = _get_json(
        f"{EUROPEPMC}/search",
        {
            "query": q,
            "format": "json",
            "pageSize": min(limit, 100),
            "resultType": "core",
        },
        timeout,
    )
    out = []
    for rec in payload.get("resultList", {}).get("result", []):
        pmcid = rec.get("pmcid")
        if not pmcid:
            continue  # no PMC id means no retrievable supplement
        out.append(
            ModelCandidate(
                source="europepmc",
                id=pmcid,
                name=rec.get("title", "").strip().rstrip("."),
                format="paper",
                url=f"https://europepmc.org/article/PMC/{pmcid}",
                curated=False,
                submitter=rec.get("authorString", "")[:120] or None,
                description=(rec.get("abstractText") or "")[:2000],
                publication=rec.get("journalTitle", ""),
            )
        )
    log.info(
        "europepmc '%s': %d hits, %d with a retrievable PMC id",
        query,
        payload.get("hitCount", 0),
        len(out),
    )
    return out[:limit]


from hallsim.discovery import _is_junk_archive_entry  # noqa: E402


def _looks_like_sbml(blob: bytes) -> bool:
    return b"<sbml" in blob[:8000]


def _harvest(blob: bytes, dest: Path, stem: str, depth: int) -> list[Path]:
    """Recover model files from one archive, descending into nested ones."""
    found: list[Path] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return found
    for info in archive.infolist():
        if info.is_dir() or info.file_size > 64 * 1024 * 1024:
            continue
        if _is_junk_archive_entry(info.filename):
            continue
        name = Path(info.filename).name
        low = name.lower()
        if low.endswith(ARCHIVE_SUFFIXES) and depth < MAX_DEPTH:
            found += _harvest(
                archive.read(info), dest, f"{stem}_{Path(low).stem}", depth + 1
            )
            continue
        if not low.endswith(MODEL_SUFFIXES):
            continue
        data = archive.read(info)
        # An .xml in a supplement is usually not SBML; check before keeping it.
        if low.endswith((".xml", ".sbml")) and not _looks_like_sbml(data):
            continue
        out = dest / f"{stem}__{name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        found.append(out)
    return found


#: Where a paper's model actually lives when it is not an attachment. Measured
#: on a 12-paper sample: 0 shipped a model file, 4 linked GitHub, 2 Zenodo.
POINTER_PATTERNS = {
    # An owner alone is kept: a code-availability statement often names the
    # lab's organisation and leaves the repository to the reader.
    "github": re.compile(
        r"github\.com/([\w.\-]+(?:/[\w.\-]+?)?)(?:\.git)?\b", re.I
    ),
    "bitbucket": re.compile(
        r"bitbucket\.org/([\w.\-]+(?:/[\w.\-]+?)?)(?:\.git)?\b", re.I
    ),
    "gitlab": re.compile(
        r"gitlab\.com/([\w.\-]+(?:/[\w.\-]+?)?)(?:\.git)?\b", re.I
    ),
    "zenodo": re.compile(
        r"(?:zenodo\.org/record/|10\.5281/zenodo\.)(\d+)", re.I
    ),
    "biomodels": re.compile(r"\b((?:BIOMD|MODEL)\d{10})\b"),
    "figshare": re.compile(r"figshare\.com/[\w/.\-]+", re.I),
    "modeldb": re.compile(
        r"modeldb\.(?:science|yale\.edu)[\w/.\-]*?(\d{4,7})", re.I
    ),
}


def full_text(pmcid: str, *, timeout: float = 60.0) -> str:
    """The article's full text as XML, cached on disk. Empty when absent."""
    if not re.fullmatch(r"PMC\d+", pmcid):
        raise ValueError(f"not a PMC id: {pmcid!r}")
    path = CACHE / pmcid / "fulltext.xml"
    if path.exists():
        return path.read_text(errors="replace")
    try:
        with urllib.request.urlopen(
            f"{EUROPEPMC}/{pmcid}/fullTextXML", timeout=timeout
        ) as fh:
            text = fh.read().decode("utf-8", "replace")
    except Exception as exc:
        log.info("europepmc %s: no full text (%s)", pmcid, exc)
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def model_pointers(pmcid: str, *, timeout: float = 60.0) -> dict:
    """``{kind: [identifier, ...]}`` — where the paper says its model lives.

    A modern paper rarely attaches its model; it cites a repository in the
    data- or code-availability statement. Measured on the first 12 papers of a
    senescence/autophagy sweep: **none** shipped a model file, a third linked
    GitHub and a sixth linked Zenodo. Reading only the attachment therefore
    misses most of what is actually available, which is the gap between "the
    paper deposited nothing" and "the model cannot be obtained".
    """
    text = full_text(pmcid, timeout=timeout)
    return pointers_in(text) if text else {}


def pointers_in(text: str) -> dict:
    """``{kind: [identifier, ...]}`` for every repository, archive or
    accession ``text`` names, in order of first mention."""
    out = {}
    for kind, pattern in POINTER_PATTERNS.items():
        seen = []
        for m in pattern.finditer(text):
            value = m.group(1) if m.groups() else m.group(0)
            if value not in seen:
                seen.append(value)
        if seen:
            out[kind] = seen
    return out


def supplementary_model_files(
    pmcid: str, *, timeout: float = 120.0, refresh: bool = False
) -> list[Path]:
    """Model files inside a paper's supplementary archive, cached on disk.

    Returns SBML and COPASI files ready to import, plus MATLAB/XPP sources,
    which are kept because they are the model even though nothing here reads
    them yet. An empty list means the paper deposited no model file — the
    common case, and the reason stage 2 exists.
    """
    if not re.fullmatch(r"PMC\d+", pmcid):
        raise ValueError(f"not a PMC id: {pmcid!r}")
    dest = CACHE / pmcid
    marker = dest / ".harvested"
    if marker.exists() and not refresh:
        return sorted(p for p in dest.iterdir() if p.name != ".harvested")

    url = f"{EUROPEPMC}/{pmcid}/supplementaryFiles"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            blob = fh.read()
    except Exception as exc:
        log.info("europepmc %s: no supplement (%s)", pmcid, exc)
        dest.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return []
    dest.mkdir(parents=True, exist_ok=True)
    found = _harvest(blob, dest, pmcid, depth=0)
    marker.touch()
    log.info(
        "europepmc %s: %.1f MB supplement -> %d model file(s)",
        pmcid,
        len(blob) / 1e6,
        len(found),
    )
    return sorted(found)


# ── What a cited repository holds ───────────────────────────────────────

#: Files an importer reads, by suffix.
IMPORTABLE_SUFFIXES = {
    ".xml": "sbml",
    ".sbml": "sbml",
    ".cps": "copasi",
    ".ode": "xpp",
}
#: Source a model would have to be translated from, by suffix.
SOURCE_SUFFIXES = {
    ".m": "matlab",
    ".py": "python",
    ".ipynb": "python",
    ".c": "c",
    ".cpp": "c++",
    ".h": "c",
    ".r": "r",
    ".jl": "julia",
}
FORGE_URL = {
    "github": "https://github.com/{}",
    "gitlab": "https://gitlab.com/{}",
    "bitbucket": "https://bitbucket.org/{}",
}
_TREE_PAGES = 5


def repository_files(
    pointer: str, forge: str = "github", *, timeout: float = 30.0
) -> list[str]:
    """Every path in a repository's default branch, read keyless from
    GitHub, GitLab or Bitbucket. An owner alone (an organisation) returns
    its repositories as ``owner/repo`` names instead."""
    import urllib.parse

    owner, _, repo = pointer.partition("/")
    if forge == "github":
        if not repo:
            rows = _get_json(
                f"https://api.github.com/users/{owner}/repos",
                {"per_page": 100},
                timeout,
            )
            return [r["full_name"] for r in rows]
        meta = _get_json(
            f"https://api.github.com/repos/{pointer}", {}, timeout
        )
        tree = _get_json(
            f"https://api.github.com/repos/{pointer}/git/trees/"
            f"{meta['default_branch']}",
            {"recursive": 1},
            timeout,
        )
        return [t["path"] for t in tree.get("tree", []) if t["type"] == "blob"]
    if forge == "gitlab":
        if not repo:
            rows = _get_json(
                f"https://gitlab.com/api/v4/groups/{owner}/projects",
                {"per_page": 100},
                timeout,
            )
            return [r["path_with_namespace"] for r in rows]
        project = urllib.parse.quote(pointer, safe="")
        out = []
        for page in range(1, _TREE_PAGES + 1):
            rows = _get_json(
                f"https://gitlab.com/api/v4/projects/{project}/repository/tree",
                {"recursive": "true", "per_page": 100, "page": page},
                timeout,
            )
            out += [r["path"] for r in rows if r["type"] == "blob"]
            if len(rows) < 100:
                break
        return out
    if forge == "bitbucket":
        base = "https://api.bitbucket.org/2.0/repositories"
        if not repo:
            rows = _get_json(f"{base}/{owner}", {"pagelen": 100}, timeout)
            return [r["full_name"] for r in rows.get("values", [])]
        meta = _get_json(f"{base}/{pointer}", {}, timeout)
        branch = meta["mainbranch"]["name"]
        out, params = [], {"max_depth": 8, "pagelen": 100}
        for _ in range(_TREE_PAGES):
            page = _get_json(
                f"{base}/{pointer}/src/{branch}/", params, timeout
            )
            out += [
                v["path"]
                for v in page.get("values", [])
                if v.get("type") == "commit_file"
            ]
            if not page.get("next"):
                break
            params = {"page": page["next"].rsplit("page=", 1)[-1], **params}
        return out
    raise ValueError(f"unknown forge {forge!r}; have {sorted(FORGE_URL)}")


def classify_repository(
    pointer: str, forge: str = "github", *, timeout: float = 30.0
) -> ModelCandidate:
    """What a cited repository holds, as a candidate ``simulate find`` can
    list beside the deposits: ``importable:<format>`` when a file an
    importer reads is there (judged by suffix), ``source:<language>`` when
    the model exists only as code to translate, ``organisation`` for an
    owner-level link, ``unknown`` otherwise."""
    from collections import Counter
    from pathlib import Path

    files = repository_files(pointer, forge, timeout=timeout)
    url = FORGE_URL[forge].format(pointer)
    if "/" not in pointer:
        return ModelCandidate(
            source=forge,
            id=pointer,
            name=pointer,
            format="organisation",
            url=url,
            curated=False,
            kind="organisation",
            description=f"{len(files)} repositories: " + ", ".join(files[:20]),
        )
    suffix = {f: Path(f).suffix.lower() for f in files}
    importable = [f for f in files if suffix[f] in IMPORTABLE_SUFFIXES]
    languages = Counter(
        SOURCE_SUFFIXES[suffix[f]]
        for f in files
        if suffix[f] in SOURCE_SUFFIXES
    )
    if importable:
        fmt = IMPORTABLE_SUFFIXES[suffix[importable[0]]]
        kind, listed = f"importable:{fmt}", importable
    elif languages:
        fmt = languages.most_common(1)[0][0]
        kind = f"source:{fmt}"
        listed = [f for f in files if SOURCE_SUFFIXES.get(suffix[f]) == fmt]
    else:
        fmt, kind, listed = "unknown", "unknown", []
    return ModelCandidate(
        source=forge,
        id=pointer,
        name=pointer,
        format=fmt,
        url=url,
        curated=False,
        kind=kind,
        description=f"{len(files)} files; " + ", ".join(listed[:12]),
    )


def repositories_cited(
    papers, *, limit: int = 10, timeout: float = 30.0
) -> list[tuple[ModelCandidate, list[str]]]:
    """The repositories the given Europe PMC candidates cite, classified,
    each with the PMC ids that cite it. Stops at ``limit`` repositories
    and records a forge it could not reach (rate limit, private, gone) as
    kind ``unreachable`` rather than dropping it."""
    import urllib.error

    cited: dict[tuple[str, str], list[str]] = {}
    for paper in papers:
        try:
            pointers = model_pointers(paper.id, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - one paper, not the run
            log.warning("full text of %s unavailable: %s", paper.id, exc)
            continue
        for forge in FORGE_URL:
            for pointer in pointers.get(forge, []):
                cited.setdefault((forge, pointer), []).append(paper.id)
    out = []
    for (forge, pointer), by_papers in list(cited.items())[:limit]:
        try:
            cand = classify_repository(pointer, forge, timeout=timeout)
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            cand = ModelCandidate(
                source=forge,
                id=pointer,
                name=pointer,
                format="unknown",
                url=FORGE_URL[forge].format(pointer),
                curated=False,
                kind="unreachable",
                description=str(exc)[:120],
            )
        out.append((cand, by_papers))
    return out
