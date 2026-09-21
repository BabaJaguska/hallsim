"""A census of a model repository: what survives each intake gate, and why.

Stage 0 of the composition benchmark. ``simulate find`` answers "is there a
deposit for X"; this answers the prior question — of everything the
repository holds, how much can HallSim use at all, and what stops the rest.
It runs the same mechanical gate as ``simulate screen``
(:func:`hallsim.intake.triage_sbml`) over every SBML deposit in BioModels,
curated (``BIOMD…``) and uncurated (``MODEL…``), keeps the deposit's own
account of itself (paper, authors, what it models) beside the verdict, and
classifies each failure by the work that would lift it. The curated branch
has had a curator reproduce a figure; the uncurated one has not been run by
anyone, which is where a mechanical screen is worth most.

    simulate census run                    # both branches, in parallel
    simulate census run --branch uncurated
    simulate census report                 # tables, figures, write-up

The gates, in the order a model has to clear them:

=============  ==============================================================
gate           what it asks
=============  ==============================================================
``listed``     the deposit is an SBML file in the branch
``kinetic``    it is a rate-law model — not SBML-qual, SBML-fbc or a map
``imports``    the importer compiles it (no unsupported construct)
``solves``     one solve over ``t_end`` native time units gives a finite,
               bounded trajectory
``clock``      it declares a time unit, so its clock is not a guess
``annotated``  at least half its species carry an ontology id
``at_rest``    the published initial condition is near a rest state
``clean``      nothing else was flagged (tolerance, domain, gradient)
=============  ==============================================================

These are the numerical screen and nothing more. A model that clears all of
them can still be about the wrong cell type, cite a paper that does not say
what it is cited for, or fail to reproduce its own figure; the review panel
and :func:`hallsim.intake.published_fit_chi2` are for that, and both cost a
reviewer. What this measures is the supply: how many deposits are worth a
reviewer's time at all.

Every row carries a **salvage** verdict from a closed vocabulary
(:data:`SALVAGE`), so the census says not only that a deposit fails but
what class of work lifts it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

#: The search endpoint pages the whole repository; the curation facet
#: filter answers HTTP 400, so branches are told apart by accession prefix.
PAGE = 100

#: Screen horizon, in the model's native time unit — the same default as
#: ``simulate screen``. A limitation, recorded in the write-up: ten seconds
#: for a second-scale model and ten days for a day-scale one.
DEFAULT_T_END = 10.0
DEFAULT_TIMEOUT = 300.0
#: A worker is recycled after this many deposits. Each deposit's compile
#: leaves memory behind in the worker (2–4 GB after twenty deposits), and six
#: such workers push a laptop into swap, where every worker stalls at once and
#: a whole batch times out.
TASKS_PER_WORKER = 6

#: Local kinetic-law parameters a COBRA export writes on every reaction.
COBRA_BOUNDS = frozenset(
    {"LOWER_BOUND", "UPPER_BOUND", "OBJECTIVE_COEFFICIENT", "FLUX_VALUE"}
)
#: Above this many reactions a model is a genome-scale network and is not
#: sent to the compiler, whatever its kinetic laws say.
GENOME_SCALE_REACTIONS = 2000

GATES = (
    "listed",
    "kinetic",
    "imports",
    "solves",
    "clock",
    "annotated",
    "at_rest",
    "clean",
)

#: Salvage classes, closed so they can be counted. ``how`` on each row
#: names the concrete action.
SALVAGE = (
    "as-is",
    "cheap-fix",
    "needs-review",
    "importer-work",
    "wrong-formalism",
    "deposit-defect",
    "framework-defect",
    "timeout",
)

SALVAGE_MEANING = {
    "as-is": "clears every gate; nothing to do before a reviewer sees it",
    "cheap-fix": "runs; one documented action at the call site lifts the "
    "flag (declare the clock from the paper, integrate at the tolerance the "
    "screen names, equilibrate before the experiment, annotate the species)",
    "needs-review": "runs; a reader has to decide (a state went negative, "
    "the model only moves when driven, the gradient is non-finite, the "
    "trajectory is still growing at the horizon, the rate laws are "
    "stochastic propensities)",
    "importer-work": "a legitimate SBML construct HallSim does not compile "
    "yet — algebraic rules, a varying compartment, a delayed event, symbolic "
    "stoichiometry",
    "wrong-formalism": "a logical, constraint-based or reaction-only "
    "deposit: there is no ODE in it to integrate",
    "deposit-defect": "the file is incomplete or inconsistent as deposited: "
    "a reaction with no kinetic law, an unresolved initial value, an "
    "undefined symbol, or a file libsbml cannot read",
    "framework-defect": "HallSim's own error: an unexpected exception, or a "
    "divergence an independent stepper does not reproduce",
    "timeout": "not screened inside the per-deposit budget",
}


# ---------------------------------------------------------------------------
# Enumeration and description — network, cached
# ---------------------------------------------------------------------------


BRANCH_PREFIX = {"curated": "BIOMD", "uncurated": "MODEL"}


def biomodels_index(refresh: bool = False) -> list[dict]:
    """Every deposit the search lists — id, format, name — cached."""
    from hallsim.discovery import BIOMODELS_SEARCH, _get_json, cached_index

    def build():
        rows, offset, matches = [], 0, None
        while matches is None or offset < matches:
            page = _get_json(
                BIOMODELS_SEARCH,
                {
                    "query": "*:*",
                    "domain": "biomodels",
                    "numResults": PAGE,
                    "offset": offset,
                    "format": "json",
                },
                30.0,
            )
            matches = int(page.get("matches", 0))
            rows.extend(
                {
                    "id": m.get("id", ""),
                    "format": m.get("format", ""),
                    "name": m.get("name", ""),
                }
                for m in page.get("models", [])
            )
            offset += PAGE
        ids = {r["id"] for r in rows}
        if len(ids) != matches:
            log.warning(
                "biomodels index: %d distinct ids against %d matches",
                len(ids),
                matches,
            )
        return rows

    return cached_index("biomodels_all", build, refresh=refresh)


def list_accessions(branch: str = "all", refresh: bool = False) -> list[str]:
    """SBML accessions in ``branch`` — ``curated`` (BIOMD…), ``uncurated``
    (MODEL…) or ``all`` — sorted. Other formats (COMBINE archives, Python,
    MATLAB) are listed by the repository but have no importer."""
    if branch not in ("curated", "uncurated", "all"):
        raise ValueError(
            f"branch must be curated, uncurated or all: {branch!r}"
        )
    prefixes = (
        tuple(BRANCH_PREFIX.values())
        if branch == "all"
        else (BRANCH_PREFIX[branch],)
    )
    return sorted(
        {
            r["id"]
            for r in biomodels_index(refresh)
            if r["id"].startswith(prefixes) and r["format"].upper() == "SBML"
        }
    )


def branch_of(accession: str) -> str:
    return "curated" if accession.startswith("BIOMD") else "uncurated"


def _records_dir() -> Path:
    d = Path.home() / ".cache" / "hallsim" / "biomodels_records"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_record(accession: str, tries: int = 3) -> dict:
    """The BioModels record, cached on disk per accession."""
    from hallsim.discovery import biomodels_record

    path = _records_dir() / f"{accession}.json"
    if path.exists() and path.stat().st_size:
        return json.loads(path.read_text())
    last = None
    for attempt in range(tries):
        try:
            record = biomodels_record(accession)
            path.write_text(json.dumps(record))
            return record
        except Exception as exc:  # pragma: no cover - network
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{accession}: record not fetched: {last}")


_TAG = re.compile(r"<[^>]+>")
_BOILERPLATE = (
    "To the extent possible under law",
    "This model is hosted on",
    "To cite BioModels",
    "This model originates from BioModels",
)


def _notes_text(html: str) -> str:
    text = _TAG.sub(" ", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    cut = min(
        [text.find(b) for b in _BOILERPLATE if text.find(b) >= 0] + [len(text)]
    )
    return text[:cut].strip()


def _first_sentence(text: str, limit: int = 240) -> str:
    text = (text or "").strip()
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    out = m.group(1) if m else text
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _surname(name: str) -> str:
    """``"Naama Geva-Zatorsky"`` → Geva-Zatorsky; ``"Novak B"`` → Novak;
    ``"Le Novère N"`` → Le Novère. Initials-last records put the surname
    first."""
    tokens = name.replace(",", " ").split()
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[-1].isupper() and len(tokens[-1]) <= 3:
        return " ".join(tokens[:-1])
    return tokens[-1]


def describe(record: dict, accession: str) -> dict:
    """What the deposit says about itself, flattened for a table."""
    pub = record.get("publication") or {}
    authors = pub.get("authors") or []
    first = (authors[0].get("name") or "") if authors else ""
    if first:
        cite = _surname(first)
        if len(authors) == 2:
            cite += " & " + _surname(authors[1].get("name") or "")
        elif len(authors) > 2:
            cite += " et al."
        if pub.get("year"):
            cite += f" {pub['year']}"
    else:
        cite = ""
    ann = record.get("modelLevelAnnotations") or []
    taxon = next(
        (
            a.get("name")
            for a in ann
            if a.get("qualifier") == "bqbiol:hasTaxon"
        ),
        "",
    )
    go = [
        a.get("name")
        for a in ann
        if a.get("resource") == "Gene Ontology" and a.get("name")
    ]
    files = [
        f.get("name", "")
        for g in ("main", "additional")
        for f in (
            (record.get("files") or {}).get(
                "main" if g == "main" else "additional"
            )
            or []
        )
    ]
    data_exts = (".csv", ".xls", ".xlsx", ".tsv", ".dat", ".txt")
    skip = ("curation_notes", "readme", "manifest")
    notes = _notes_text(record.get("description") or "")
    name = record.get("name") or ""
    if name and notes.startswith(name):
        notes = notes[len(name) :].lstrip(" -:")
    synopsis = re.sub(
        r"\s+", " ", _TAG.sub(" ", pub.get("synopsis") or "")
    ).strip()
    return {
        "accession": accession,
        "name": record.get("name") or "",
        "paper": pub.get("title") or "",
        "cite": cite,
        "first_author": first,
        "n_authors": len(authors),
        "journal": pub.get("journal") or "",
        "year": pub.get("year"),
        "pubmed": (
            pub.get("accession") if pub.get("type") == "PubMed ID" else ""
        ),
        "about": _first_sentence(synopsis) or _first_sentence(notes),
        "curator_note": _first_sentence(notes),
        "synopsis": synopsis,
        "approach": (record.get("modellingApproach") or {}).get("name") or "",
        "format": (record.get("format") or {}).get("name") or "",
        "taxon": taxon or "",
        "go_terms": "; ".join(go[:5]),
        "curation": record.get("curationStatus") or "",
        "has_sedml": any(f.lower().endswith(".sedml") for f in files),
        "data_files": "; ".join(
            f
            for f in files
            if f.lower().endswith(data_exts) and not f.lower().startswith(skip)
        ),
        "n_files": len(files),
    }


def fetch_deposits(accessions, workers: int = 12) -> dict[str, dict]:
    """``{accession: describe(record)}`` for every accession, concurrently."""

    def one(acc):
        try:
            return acc, describe(fetch_record(acc), acc)
        except Exception as exc:
            log.warning("%s: %s", acc, exc)
            return acc, {"accession": acc, "name": "", "cite": "", "about": ""}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, accessions))


# ---------------------------------------------------------------------------
# One deposit through the gates — runs in a worker process
# ---------------------------------------------------------------------------

_SCREEN_FIELDS = (
    "exploding",
    "vanishing",
    "tolerance_sensitive",
    "negative",
    "undriven",
    "not_at_rest",
    "rest_tau",
    "framework_suspect",
    "tunes",
    "did_not_construct",
    "tol_rel_diff",
    "max_abs",
    "detail",
)


def _blank_row(accession: str) -> dict:
    row = {
        "accession": accession,
        "kind": "unknown",
        "n_species": 0,
        "n_reactions": 0,
        "n_parameters": 0,
        "n_events": 0,
        "n_rate_rules": 0,
        "status": "",
        "blockers": [],
        "flags": [],
        "import_error": "",
        "screen_raised": "",
        "trigger_pathology": "",
        "time_unit_declared": False,
        "annotation_coverage": float("nan"),
        "rest_residual": float("nan"),
        "stochastic_propensities": False,
        "error": "",
        "seconds": 0.0,
    }
    for f in _SCREEN_FIELDS:
        row[f] = None
    return row


def _formalism(model) -> tuple[str, str]:
    """``(kind, note)`` from the packages and rate laws the file carries."""
    if model.getPlugin("qual") is not None:
        return "qualitative", (
            "SBML-qual: a logical model over discrete levels, with update "
            "rules rather than rate laws"
        )
    n_rx = model.getNumReactions()
    n_rate = sum(
        1 for i in range(model.getNumRules()) if model.getRule(i).isRate()
    )
    with_law = sum(
        1
        for i in range(n_rx)
        if model.getReaction(i).isSetKineticLaw()
        and model.getReaction(i).getKineticLaw().isSetMath()
    )
    if model.getPlugin("fbc") is not None and with_law == 0:
        return "constraint-based", (
            "SBML-fbc: flux bounds and an objective, no rate laws — solved "
            "by linear programming, not integrated"
        )
    # A COBRA-style export carries no fbc package: every reaction has a
    # "kinetic law" whose local parameters are the flux bounds and the
    # objective coefficient, and the math is a placeholder.
    bounds = 0
    for i in range(n_rx):
        kl = model.getReaction(i).getKineticLaw()
        if kl is None:
            continue
        ids = {
            kl.getParameter(j).getId() for j in range(kl.getNumParameters())
        }
        ids |= {
            kl.getLocalParameter(j).getId()
            for j in range(kl.getNumLocalParameters())
        }
        if ids & COBRA_BOUNDS:
            bounds += 1
    if n_rx and bounds >= 0.5 * n_rx:
        return "constraint-based", (
            f"COBRA-style SBML: {bounds} of {n_rx} reactions carry flux "
            "bounds as kinetic-law parameters and no rate law — solved by "
            "linear programming, not integrated"
        )
    if n_rx > GENOME_SCALE_REACTIONS:
        return "genome-scale", (
            f"{n_rx} reactions over {model.getNumSpecies()} species: a "
            "genome-scale network, outside the ODE path"
        )
    if n_rx == 0 and n_rate == 0:
        return "no-rate-laws", (
            "declares no reactions and no rate rules: a drawn map or an "
            "algebraic model, nothing to integrate"
        )
    if n_rx and with_law == 0 and n_rate == 0:
        return "no-rate-laws", (
            f"{n_rx} reactions and none carries a kinetic law: a pathway "
            "map, not a kinetic model"
        )
    return "ode", ""


def _classify_import_error(text: str) -> tuple[str, str]:
    """``(salvage, how)`` for the message an import raised."""
    m = re.match(r"import failed: (\w+): (.*)", text, re.S)
    exc_type, msg = (m.group(1), m.group(2)) if m else ("", text)
    low = msg.lower()
    # Every importer refusal is an ``Unsupported*Error``; any other type is
    # the importer failing rather than declining.
    if not exc_type.startswith("Unsupported") and not text.startswith(
        "unsupported constructs"
    ):
        return (
            "framework-defect",
            f"unexpected {exc_type or 'error'}: {msg[:160]}",
        )
    if "sbml qual" in low:
        return "wrong-formalism", "SBML-qual logical model"
    if "algebraic rules" in low:
        return "importer-work", "algebraic rules (a DAE, not an ODE)"
    if (
        "varying volume" in low
        or "is not constant" in low
        or "set by a rule" in low
    ):
        return "importer-work", "compartment volume varies in time"
    if "stoichiometrymath" in low:
        return "importer-work", "symbolic stoichiometry"
    if "has a delay" in low:
        return "importer-work", "delayed event"
    if "has a priority" in low:
        return "importer-work", "event priority"
    if "has no translation" in low:
        node = re.search(r"math node (\S+)", msg)
        return (
            "importer-work",
            f"MathML node {node.group(1) if node else '?'} has no translation",
        )
    if "nest too deeply" in low:
        return "importer-work", "function definitions nest too deeply"
    if "could not parse" in low:
        return "deposit-defect", "libsbml cannot read the file"
    if "no kinetic law" in low:
        return "deposit-defect", "a reaction has no kinetic law"
    if "no initial value" in low:
        return "importer-work", (
            "a species has no initial value; other simulators default it to "
            "zero, and adopting that convention (with a warning) lifts this"
        )
    if "cannot be resolved" in low:
        return "deposit-defect", "an initial value is unresolved"
    if "is not a species, parameter or compartment" in low:
        return "deposit-defect", "an expression refers to an undefined symbol"
    if "form a cycle" in low:
        return "deposit-defect", "assignment rules form a cycle"
    if "undefined function" in low or "argument(s)" in low:
        return "deposit-defect", "a function is called wrongly"
    if "missing math" in low:
        return "deposit-defect", "an element has no math"
    if "not an integrated quantity" in low:
        return "importer-work", "rate rule on a non-integrated quantity"
    return "importer-work", msg[:160]


def stage_of(row: dict) -> str:
    """The first gate the deposit fails, or ``"pass"``."""
    if row.get("error", "").startswith("timeout"):
        return "solves"
    if row.get("error"):
        return "imports"
    if row.get("kind") != "ode":
        return "kinetic"
    if row.get("import_error"):
        return "imports"
    if (
        row.get("exploding")
        or row.get("did_not_construct")
        or row.get("screen_raised")
        or row.get("trigger_pathology")
    ):
        return "solves"
    if not row.get("time_unit_declared"):
        return "clock"
    from hallsim.intake import ANNOTATION_FLAG, REST_RESIDUAL_FLAG

    cov = row.get("annotation_coverage")
    if cov is None or cov != cov or cov < ANNOTATION_FLAG:
        return "annotated"
    res = row.get("rest_residual")
    if row.get("not_at_rest") or (
        res == res and res is not None and res > REST_RESIDUAL_FLAG
    ):
        return "at_rest"
    if salvage_verdict(row)[0] != "as-is":
        return "clean"
    return "pass"


def salvage_verdict(row: dict) -> tuple[str, str]:
    """``(salvage, how)`` — the class of work that lifts this deposit."""
    err = row.get("error", "")
    if err.startswith("timeout"):
        return "timeout", err
    if err:
        return "framework-defect", err[:160]
    kind = row.get("kind")
    if kind == "unreadable":
        return "deposit-defect", "libsbml cannot read the file"
    if kind != "ode":
        return "wrong-formalism", row.get("kind_note") or kind
    if row.get("import_error"):
        return _classify_import_error(row["import_error"])
    if row.get("screen_raised"):
        return "framework-defect", row["screen_raised"][:160]
    if row.get("did_not_construct"):
        return "framework-defect", "the composite could not be built"
    if row.get("trigger_pathology"):
        return "needs-review", row["trigger_pathology"][:160]
    if row.get("exploding"):
        if row.get("framework_suspect"):
            return "framework-defect", (
                "diverges here and is bounded under an independent stepper"
            )
        return "needs-review", (
            "still growing past 1000x at the horizon; check the paper's own "
            "time span before calling it unstable"
        )
    review, cheap = [], []
    if row.get("tunes") is False:
        review.append("non-finite gradient: not calibratable as configured")
    if row.get("negative"):
        review.append("a non-negative state went negative")
    if row.get("undriven"):
        review.append(
            "moves only when an INPUT is driven; a component, not a model"
        )
    if row.get("vanishing"):
        review.append("every state decays to zero")
    if row.get("stochastic_propensities"):
        review.append(
            "rate laws are stochastic propensities; the ODE is a mean field"
        )
    if row.get("tolerance_sensitive"):
        cheap.append("integrate at the tolerance the screen names")
    if not row.get("time_unit_declared"):
        cheap.append("declare the time unit from the paper")
    if row.get("not_at_rest") or (
        row.get("rest_residual") is not None
        and row["rest_residual"] == row["rest_residual"]
        and row["rest_residual"] > 1.0
    ):
        cheap.append("equilibrate before applying the experiment")
    cov = row.get("annotation_coverage")
    if cov is not None and cov == cov and cov < 0.5:
        cheap.append("annotate the species before composing")
    if review:
        return "needs-review", "; ".join(review + cheap)
    if cheap:
        return "cheap-fix", "; ".join(cheap)
    return "as-is", ""


def reason_of(row: dict) -> str:
    """One line on why the deposit stopped where it did."""
    stage = row.get("stage") or stage_of(row)
    if stage == "pass":
        return ""
    if stage == "kinetic":
        return row.get("kind_note") or row.get("kind", "")
    if stage == "imports":
        return row.get("error") or row.get("import_error", "")
    if stage == "solves":
        if row.get("error"):
            return row["error"]
        for key in ("screen_raised", "trigger_pathology"):
            if row.get(key):
                return row[key]
        return row.get("detail") or "exploding"
    if stage == "clock":
        return "no declared time unit"
    if stage == "annotated":
        cov = row.get("annotation_coverage")
        return (
            f"{cov:.0%} of species carry an ontology id"
            if cov == cov
            else "no annotations"
        )
    if stage == "at_rest":
        res = row.get("rest_residual")
        tau = row.get("rest_tau")
        parts = []
        if res is not None and res == res:
            parts.append(f"‖f(y0)‖/‖y0‖ = {res:.3g}")
        if tau not in (None, float("inf")) and tau == tau:
            parts.append(f"rest tau {tau:.3g}")
        return "IC is not a rest state (" + ", ".join(parts) + ")"
    return "; ".join(row.get("flags") or [])


def census_one(accession: str, t_end: float = DEFAULT_T_END) -> dict:
    """Run one deposit through every gate. Never raises."""
    t0 = time.time()
    row = _blank_row(accession)
    try:
        import libsbml

        from hallsim.sbml_import import _download_biomodel_to_cache

        fmt = ""
        try:
            fmt = (fetch_record(accession).get("format") or {}).get("name", "")
        except Exception:
            pass
        if fmt and fmt.upper() != "SBML":
            row["kind"] = "not-sbml"
            row["kind_note"] = f"deposited as {fmt}, not SBML"
            return _finish(row, t0)
        path = _download_biomodel_to_cache(accession)
        doc = libsbml.readSBMLFromFile(str(path))
        model = doc.getModel()
        if model is None:
            row["kind"] = "unreadable"
            row["kind_note"] = "libsbml cannot read the file"
            return _finish(row, t0)
        row["n_species"] = model.getNumSpecies()
        row["n_reactions"] = model.getNumReactions()
        row["n_parameters"] = model.getNumParameters()
        row["n_events"] = model.getNumEvents()
        row["n_rate_rules"] = sum(
            1 for i in range(model.getNumRules()) if model.getRule(i).isRate()
        )
        kind, note = _formalism(model)
        row["kind"], row["kind_note"] = kind, note
        if kind != "ode":
            return _finish(row, t0)

        from hallsim.intake import triage_sbml

        verdict = triage_sbml(accession, t_end=t_end, name="m")
        row["status"] = verdict.status
        row["blockers"] = list(verdict.blockers)
        row["flags"] = list(verdict.flags)
        row["n_parameters"] = verdict.n_parameters or row["n_parameters"]
        row["time_unit_declared"] = bool(verdict.time_unit_declared)
        row["annotation_coverage"] = float(verdict.annotation_coverage)
        row["rest_residual"] = float(verdict.rest_residual)
        for b in verdict.blockers:
            if b.startswith("import failed:") or b.startswith(
                "unsupported constructs"
            ):
                row["import_error"] = b
            elif b.startswith("screen raised"):
                row["screen_raised"] = b
            elif b.startswith("numerical screen"):
                pass
            else:
                row["trigger_pathology"] = b
        row["stochastic_propensities"] = any(
            "stochastic propensity" in f for f in verdict.flags
        )
        screen = verdict.screen
        if screen is not None:
            for f in _SCREEN_FIELDS:
                v = getattr(screen, f, None)
                if isinstance(v, float) and v != v:
                    v = None
                row[f] = v
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return _finish(row, t0)


def _finish(row: dict, t0: float) -> dict:
    row["seconds"] = round(time.time() - t0, 2)
    row["stage"] = stage_of(row)
    row["salvage"], row["how"] = salvage_verdict(row)
    row["reason"] = reason_of(row)
    return row


def timeout_row(accession: str, timeout: float) -> dict:
    row = _blank_row(accession)
    row["error"] = f"timeout: not screened within {timeout:.0f} s"
    row["seconds"] = timeout
    return _finish(row, time.time() - timeout)


# ---------------------------------------------------------------------------
# The run — a process pool with a per-deposit timeout, streamed to disk
# ---------------------------------------------------------------------------


def run_census(
    accessions=None,
    *,
    run_dir: Path | str | None = None,
    workers: int = 4,
    t_end: float = DEFAULT_T_END,
    timeout: float = DEFAULT_TIMEOUT,
    limit: int | None = None,
    retry_timeouts: bool = False,
    tasks_per_worker: int = TASKS_PER_WORKER,
) -> Path:
    """Screen every accession, appending one JSON row per deposit to
    ``rows.jsonl`` as it completes and a line per deposit to
    ``progress.log``. Resumable: an existing ``rows.jsonl`` in ``run_dir``
    is read first and its accessions skipped; ``retry_timeouts`` re-screens
    the ones that ran out of budget, and :func:`load_rows` keeps the last
    row per accession."""
    import multiprocessing
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures import as_completed

    from pebble import ProcessExpired, ProcessPool

    from hallsim.io import make_run_dir

    run = Path(run_dir) if run_dir else make_run_dir("census")
    run.mkdir(parents=True, exist_ok=True)
    accessions = list(accessions or list_accessions())
    if limit:
        accessions = accessions[:limit]

    deposits_path = run / "deposits.jsonl"
    known = {}
    if deposits_path.exists():
        for line in deposits_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                known[d["accession"]] = d
    missing = [a for a in accessions if a not in known]
    if missing:
        log.info("fetching %d deposit records", len(missing))
        fetched = fetch_deposits(missing)
        with deposits_path.open("a") as fh:
            for acc in missing:
                fh.write(json.dumps(fetched[acc]) + "\n")
        known.update(fetched)

    rows_path = run / "rows.jsonl"
    wanted = set(accessions)
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if retry_timeouts and prior.get("error", "").startswith("timeout"):
                continue
            if prior["accession"] in wanted:
                done.add(prior["accession"])
    todo = [a for a in accessions if a not in done]
    total = len(accessions)
    (run / "config.json").write_text(
        json.dumps(
            {
                "n_accessions": total,
                "workers": workers,
                "t_end": t_end,
                "timeout": timeout,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=1,
        )
    )
    log.info(
        "census: %d deposits, %d already done, %d to screen, %d workers",
        total,
        len(done),
        len(todo),
        workers,
    )
    if not todo:
        return run

    progress = (run / "progress.log").open("a")
    t_start = time.time()
    n = len(done)
    ctx = multiprocessing.get_context("spawn")
    with (
        ProcessPool(
            max_workers=workers, max_tasks=tasks_per_worker, context=ctx
        ) as pool,
        rows_path.open("a") as out,
    ):
        futures = {}
        for acc in todo:
            fut = pool.schedule(census_one, args=(acc, t_end), timeout=timeout)
            futures[fut] = acc
        for fut in as_completed(futures):
            acc = futures[fut]
            try:
                row = fut.result()
            except FutureTimeout:
                row = timeout_row(acc, timeout)
            except ProcessExpired as exc:
                row = _blank_row(acc)
                row["error"] = f"worker died: {exc}"
                row = _finish(row, time.time())
            except Exception as exc:  # pragma: no cover - defensive
                row = _blank_row(acc)
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                row = _finish(row, time.time())
            out.write(json.dumps(row) + "\n")
            out.flush()
            n += 1
            elapsed = time.time() - t_start
            rate = (n - len(done)) / elapsed if elapsed > 0 else 0.0
            eta = (total - n) / rate / 60.0 if rate > 0 else float("nan")
            name = (known.get(acc) or {}).get("name", "")[:48]
            line = (
                f"{time.strftime('%H:%M:%S')} [{n}/{total}] {acc} "
                f"{row['stage']:9s} "
                f"{row['salvage']:16s} {row['seconds']:6.1f}s "
                f"eta {eta:5.0f} min  {name}"
            )
            progress.write(line + "\n")
            progress.flush()
            log.info(line)
    progress.close()
    return run


# ---------------------------------------------------------------------------
# The report — tables, figures, and the prose
# ---------------------------------------------------------------------------


def refresh_deposits(run_dir: Path | str) -> int:
    """Rewrite ``deposits.jsonl`` from the cached records, so the report
    carries the current :func:`describe` rather than the one the run
    started with. Returns the number of rows rewritten; a record not in the
    cache keeps its stored description."""
    run = Path(run_dir)
    dpath = run / "deposits.jsonl"
    if not dpath.exists():
        return 0
    stored = []
    for line in dpath.read_text().splitlines():
        if line.strip():
            stored.append(json.loads(line))
    out, refreshed = [], 0
    for d in stored:
        acc = d["accession"]
        cached = _records_dir() / f"{acc}.json"
        if cached.exists() and cached.stat().st_size:
            out.append(describe(json.loads(cached.read_text()), acc))
            refreshed += 1
        else:
            out.append(d)
    tmp = dpath.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(d) + "\n" for d in out))
    tmp.replace(dpath)
    return refreshed


def load_rows(run_dir: Path | str):
    """One frame, deposit description joined onto the verdict."""
    import pandas as pd

    run = Path(run_dir)
    by_accession = {}
    for line in (run / "rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_accession[r["accession"]] = r
    rows = list(by_accession.values())
    deps = {}
    dpath = run / "deposits.jsonl"
    if dpath.exists():
        for line in dpath.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                deps[d["accession"]] = d
    for r in rows:
        d = deps.get(r["accession"], {})
        for k, v in d.items():
            if k not in r:
                r[k] = v
        # The verdict is a function of the stored evidence; deriving it here
        # keeps every row on the current rubric, whichever version screened it.
        r["stage"] = stage_of(r)
        r["salvage"], r["how"] = salvage_verdict(r)
        r["reason"] = reason_of(r)
        fmt = r.get("format") or ""
        if fmt.upper() not in ("", "SBML") and r.get("kind") in (
            "unreadable",
            "unknown",
        ):
            # The record says the deposit is not SBML; a file the reader
            # cannot open is then the wrong formalism, not a broken one.
            note = f"deposited as {fmt}, not SBML"
            r.update(
                kind="not-sbml",
                kind_note=note,
                stage="kinetic",
                salvage="wrong-formalism",
                how=note,
                reason=note,
                error="",
            )
        r["blockers"] = "; ".join(r.get("blockers") or [])
        r["flags"] = "; ".join(r.get("flags") or [])
    df = pd.DataFrame(rows)
    if "stage" in df:
        order = {g: i for i, g in enumerate(GATES)}
        order["pass"] = len(GATES)
        df["stage_index"] = df["stage"].map(order).fillna(-1).astype(int)
    df["branch"] = df["accession"].map(branch_of)
    return df.sort_values("accession").reset_index(drop=True)


def funnel(df) -> list[tuple[str, int]]:
    """Survivors at each gate, cumulative."""
    out = [("listed", int(len(df)))]
    for i, gate in enumerate(GATES[1:], start=1):
        out.append((gate, int((df["stage_index"] > i).sum())))
    return out


def _summarize_frame(df) -> dict:
    fun = funnel(df)
    solved = df[df["stage_index"] > GATES.index("solves")]
    prevalence = {}
    if len(solved):
        prevalence = {
            "no_time_unit": int(
                (~solved["time_unit_declared"].astype(bool)).sum()
            ),
            "thin_annotation": int(
                (solved["annotation_coverage"].fillna(0) < 0.5).sum()
            ),
            "not_at_rest": int(
                (
                    solved["not_at_rest"].fillna(False).astype(bool)
                    | (solved["rest_residual"].fillna(0) > 1.0)
                ).sum()
            ),
            "tolerance_sensitive": int(
                solved["tolerance_sensitive"].fillna(False).astype(bool).sum()
            ),
            "negative": int(
                solved["negative"].fillna(False).astype(bool).sum()
            ),
            "non_finite_gradient": int(solved["tunes"].eq(False).sum()),
            "undriven": int(
                solved["undriven"].fillna(False).astype(bool).sum()
            ),
            "stochastic_propensities": int(
                solved["stochastic_propensities"]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
        }
    fails = df[df["stage"] != "pass"]
    return {
        "n": int(len(df)),
        "funnel": fun,
        "pass": int((df["stage"] == "pass").sum()),
        "by_stage": {k: int(v) for k, v in df["stage"].value_counts().items()},
        "by_salvage": {
            k: int(v) for k, v in df["salvage"].value_counts().items()
        },
        "by_kind": {k: int(v) for k, v in df["kind"].value_counts().items()},
        "stage_x_salvage": {
            s: {k: int(v) for k, v in g["salvage"].value_counts().items()}
            for s, g in fails.groupby("stage")
        },
        "how_top": {
            k: int(v) for k, v in fails["how"].value_counts().head(25).items()
        },
        "prevalence_among_solved": prevalence,
        "with_sedml": (
            int(df["has_sedml"].fillna(False).astype(bool).sum())
            if "has_sedml" in df
            else 0
        ),
        "with_data_files": (
            int((df["data_files"].fillna("") != "").sum())
            if "data_files" in df
            else 0
        ),
        "seconds_total": float(df["seconds"].sum()),
        "seconds_median": float(df["seconds"].median()),
        "timeouts": int(
            df["error"].fillna("").str.startswith("timeout").sum()
        ),
    }


def summarize(df) -> dict:
    """Counts over the whole frame, and per branch when both are present."""
    out = _summarize_frame(df)
    if "branch" in df and df["branch"].nunique() > 1:
        out["branches"] = {
            b: _summarize_frame(g) for b, g in df.groupby("branch")
        }
    return out


def _bar_figure(labels, series, title, subtitle, path, xlabel="deposits"):
    """Horizontal bars per label; ``series`` maps a legend entry to its
    values. One series draws no legend; two draw grouped bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(series)
    k = max(len(names), 1)
    n = len(labels)
    height = 0.72 / k
    fig, ax = plt.subplots(figsize=(7.2, 0.30 * n * k + 2.2))
    base = [float(i) for i in range(n)][::-1]
    vmax = max((max(v) for v in series.values() if v), default=1) or 1
    for j, name in enumerate(names):
        vals = series[name]
        ys = [b + ((k - 1) / 2 - j) * height for b in base]
        ax.barh(ys, vals, height=height * 0.92, color=f"C{j}", label=name)
        for y, v in zip(ys, vals):
            ax.text(v + vmax * 0.01, y, f"{v:,}", va="center", fontsize=8)
    ax.set_yticks(base)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(0, vmax * 1.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    if len(names) > 1:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", y=0.995, wrap=True)
    fig.text(
        0.01, 0.935, subtitle, fontsize=8, color="0.35", ha="left", wrap=True
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _md_table(headers, rows) -> str:
    def cell(x):
        s = "" if x is None else str(x)
        return s.replace("|", "／").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


GATE_QUESTION = {
    "listed": "in the branch",
    "kinetic": "a rate-law model, not logical / constraint-based / a map",
    "imports": "the importer compiles it",
    "solves": "one solve gives a finite, bounded trajectory",
    "clock": "declares a time unit",
    "annotated": "≥ 50% of species carry an ontology id",
    "at_rest": "published initial condition is a rest state",
    "clean": "nothing else flagged",
}

BRANCH_LABEL = {
    "curated": "manually curated (BIOMD) deposits",
    "uncurated": "uncurated (MODEL) deposits",
}


def notable_cases(df, per_class: int = 4):
    """Rows worth showing by name: every framework defect and timeout,
    then the largest deposits in each other failing class."""
    import pandas as pd

    fails = df[df["stage"] != "pass"].copy()
    fails["size"] = fails["n_species"].fillna(0) + fails["n_reactions"].fillna(
        0
    )
    picked = []
    for cls in ("framework-defect", "timeout"):
        picked.append(
            fails[fails["salvage"] == cls].sort_values("size", ascending=False)
        )
    for cls in (
        "deposit-defect",
        "importer-work",
        "needs-review",
        "wrong-formalism",
        "cheap-fix",
    ):
        picked.append(
            fails[fails["salvage"] == cls]
            .sort_values("size", ascending=False)
            .head(per_class)
        )
    out = pd.concat(picked) if picked else fails.head(0)
    return out.drop_duplicates("accession")


def _text(x) -> str:
    """A frame cell as text: a missing string column reads back as NaN,
    which is truthy and prints as ``nan``."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return str(x)


def _pct(k, n) -> str:
    return f"{100.0 * k / n:.0f}%" if n else "–"


def _paragraph(s: dict, label: str, t_end: float, stamp: str) -> str:
    """The preprint paragraph for one summary."""
    fun = dict(s["funnel"])
    n, n_pass = s["n"], s["pass"]
    prev = s["prevalence_among_solved"]
    by_stage = s["by_stage"]
    return (
        f"We ran the numerical intake gate over the {label} in BioModels "
        f"({n:,} SBML accessions, {stamp}). {fun['kinetic']:,} "
        f"({_pct(fun['kinetic'], n)}) are rate-law models; the rest "
        f"are logical (SBML-qual), constraint-based (SBML-fbc) or reaction "
        f"maps with no kinetic law. Of those, the importer compiles "
        f"{fun['imports']:,} ({_pct(fun['imports'], n)}); "
        f"{by_stage.get('imports', 0)} fail on a construct it does not yet "
        f"support or on a defect in the file as deposited. "
        f"{fun['solves']:,} ({_pct(fun['solves'], n)}) produce a finite, "
        f"bounded trajectory over {t_end:g} native time units. Among those, "
        f"{prev.get('no_time_unit', 0)} declare no time unit, so their clock "
        f"is a guess before any composition; {prev.get('thin_annotation', 0)} "
        f"annotate fewer than half their species, so the semantic checks that "
        f"decide merge-or-couple are blind on them; "
        f"{prev.get('not_at_rest', 0)} start far from a rest state, so a run "
        f"is mostly relaxation; and {prev.get('non_finite_gradient', 0)} have "
        f"a non-finite gradient as deposited. {n_pass} ({_pct(n_pass, n)}) "
        f"clear every gate with nothing flagged."
    )


def write_report(run_dir: Path | str) -> Path:
    """``census.csv``, ``summary.json``, ``failures.md``, two figures and
    ``writeup.md`` (a preprint paragraph and a blog section) in ``run_dir``.
    With both branches present the figures and tables compare them."""
    run = Path(run_dir)
    refresh_deposits(run)
    df = load_rows(run)
    summary = summarize(df)
    cfg = {}
    if (run / "config.json").exists():
        cfg = json.loads((run / "config.json").read_text())
    t_end = cfg.get("t_end", DEFAULT_T_END)
    stamp = time.strftime("%Y-%m-%d")
    branches = summary.get("branches") or {}
    order = [b for b in ("curated", "uncurated") if b in branches]
    n, n_pass = summary["n"], summary["pass"]

    cols = [
        "accession", "branch", "name", "cite", "journal", "year", "pubmed",
        "paper", "about", "curator_note", "approach", "taxon", "go_terms",
        "kind", "n_species", "n_reactions", "n_parameters", "n_events",
        "stage", "reason", "salvage", "how", "status", "time_unit_declared",
        "annotation_coverage", "rest_residual", "rest_tau", "exploding",
        "tolerance_sensitive", "negative", "undriven", "tunes",
        "framework_suspect", "stochastic_propensities", "has_sedml",
        "data_files", "blockers", "flags", "error", "seconds",
    ]  # fmt: skip
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(run / "census.csv", index=False)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))

    # --- figures ------------------------------------------------------------
    labels = [f"{g}: {GATE_QUESTION[g]}" for g, _ in summary["funnel"]]
    if order:
        series = {b: [v for _, v in branches[b]["funnel"]] for b in order}
        title = (
            " and ".join(
                f"{branches[b]['pass']} of {branches[b]['n']:,} {b}"
                for b in order
            )
            + " BioModels deposits clear every numerical gate"
        )
    else:
        series = {"deposits": [v for _, v in summary["funnel"]]}
        title = (
            f"{n_pass} of {n:,} BioModels deposits clear every numerical gate"
        )
    _bar_figure(
        labels,
        series,
        title,
        f"survivors per gate, cumulative · screen horizon {t_end:g} native "
        f"time units · {stamp}",
        run / "funnel.png",
    )

    def _fail_classes(s):
        return sorted(
            ((k, v) for k, v in s["by_salvage"].items() if k != "as-is"),
            key=lambda kv: -kv[1],
        )

    overall = _fail_classes(summary)
    if overall:
        classes = [k for k, _ in overall]
        if order:
            series = {
                b: [branches[b]["by_salvage"].get(k, 0) for k in classes]
                for b in order
            }
        else:
            series = {"deposits": [v for _, v in overall]}
        top, top_n = overall[0]
        _bar_figure(
            classes,
            series,
            f"{top} accounts for {top_n} of the {n - n_pass} deposits that do "
            f"not pass",
            f"salvage class per non-passing deposit · {stamp}",
            run / "salvage.png",
        )

    # --- failures.md: every deposit that did not pass, by name -------------
    def _authors(r):
        cite = _text(r.get("cite"))
        journal = _text(r.get("journal"))
        return f"{cite}, {journal}" if cite and journal else cite or journal

    def _fail_rows(frame):
        return [
            (
                r.accession,
                _text(r.get("name")),
                _text(r.get("paper")),
                _authors(r),
                _text(r.get("about")),
                r.stage,
                _text(r.get("reason")),
                r.salvage,
                _text(r.get("how")),
            )
            for _, r in frame.iterrows()
        ]

    fail_headers = [
        "accession", "model", "paper", "authors", "about", "gate", "reason",
        "salvage", "how",
    ]  # fmt: skip
    fails = df[df["stage"] != "pass"].sort_values(["stage_index", "accession"])
    parts = [
        f"# Census failures — {stamp}",
        "",
        f"{len(fails)} of {n:,} deposits stop at a gate. One row per deposit, "
        "in gate order. `about` is the first sentence of the paper's "
        "abstract, or the curator's note when the record has no abstract; "
        "`reason` is what the gate saw; `salvage` is the class of work that "
        "lifts it and `how` the concrete action.",
        "",
    ]
    if order:
        for b in order:
            sub = fails[fails["branch"] == b]
            parts += [
                f"## {BRANCH_LABEL[b]} — {len(sub)} of {branches[b]['n']:,}",
                "",
                _md_table(fail_headers, _fail_rows(sub)),
                "",
            ]
    else:
        parts += [_md_table(fail_headers, _fail_rows(fails))]
    (run / "failures.md").write_text("\n".join(parts))

    # --- writeup.md ----------------------------------------------------------
    fun = dict(summary["funnel"])
    if order:
        paragraphs = [
            _paragraph(branches[b], BRANCH_LABEL[b], t_end, stamp)
            for b in order
        ]
        c, u = branches.get("curated"), branches.get("uncurated")
        if c and u:
            top_c = max(c["by_stage"].items(), key=lambda kv: kv[1])
            top_u = max(u["by_stage"].items(), key=lambda kv: kv[1])
            paragraphs.append(
                f"Curation is visible in the numbers: {_pct(c['pass'], c['n'])} "
                f"of curated deposits clear every gate against "
                f"{_pct(u['pass'], u['n'])} of uncurated ones, and the gate "
                f"most curated deposits stop at is `{top_c[0]}` "
                f"({top_c[1]}) while uncurated ones most often stop at "
                f"`{top_u[0]}` ({top_u[1]}). A pass on an uncurated deposit "
                f"says the file runs, not that it reproduces its paper — no "
                f"curator has checked that — so the uncurated survivors are "
                f"the set worth a reproduction check first."
            )
        preprint = "\n\n".join(paragraphs)
    else:
        preprint = _paragraph(summary, "SBML deposits", t_end, stamp)
    preprint += (
        f"\n\n{summary['with_sedml']} deposits ship a SED-ML file, nearly all "
        f"of them the repository's autogenerated template rather than the "
        f"curator's reproduction run, and {summary['with_data_files']} ship a "
        f"data file, which bounds how many can be checked against their own "
        f"paper without a reviewer. The gate is mechanical — it cannot see a "
        f"wrong cell type or a citation that does not say what it is cited "
        f"for — so these are upper bounds on the usable supply, not "
        f"estimates of it."
    )

    def _col(s, key):
        return s[key]

    if order:
        funnel_headers = ["gate", "asks"] + [
            f"{b} ({branches[b]['n']:,})" for b in order
        ]
        funnel_rows = [
            (g, GATE_QUESTION[g])
            + tuple(
                f"{dict(branches[b]['funnel'])[g]:,} "
                f"({_pct(dict(branches[b]['funnel'])[g], branches[b]['n'])})"
                for b in order
            )
            for g, _ in summary["funnel"]
        ]
        stage_headers = ["stopped at"] + order
        stage_rows = [
            (g,) + tuple(branches[b]["by_stage"].get(g, 0) for b in order)
            for g in list(GATES[1:]) + ["pass"]
        ]
        salvage_headers = ["salvage"] + order + ["meaning"]
        salvage_rows = [
            (k,)
            + tuple(branches[b]["by_salvage"].get(k, 0) for b in order)
            + (SALVAGE_MEANING.get(k, ""),)
            for k, _ in sorted(
                summary["by_salvage"].items(), key=lambda kv: -kv[1]
            )
        ]
        flag_headers = ["flag among the models that solve"] + [
            f"{b} (of {dict(branches[b]['funnel'])['solves']})" for b in order
        ]
        flag_keys = list(summary["prevalence_among_solved"])
        flag_rows = [
            (k,)
            + tuple(
                branches[b]["prevalence_among_solved"].get(k, 0) for b in order
            )
            for k in flag_keys
        ]
        headline = (
            "**"
            + " and ".join(
                f"{branches[b]['pass']} of {branches[b]['n']:,} {b}"
                for b in order
            )
            + " deposits clear the numerical intake gate with nothing "
            "flagged.**"
        )
    else:
        funnel_headers = ["gate", "asks", "survivors", "of listed"]
        funnel_rows = [
            (g, GATE_QUESTION[g], v, _pct(v, n)) for g, v in summary["funnel"]
        ]
        stage_headers = ["stopped at", "deposits"]
        stage_rows = [
            (g, summary["by_stage"].get(g, 0))
            for g in list(GATES[1:]) + ["pass"]
        ]
        salvage_headers = ["salvage", "deposits", "of listed", "meaning"]
        salvage_rows = [
            (k, v, _pct(v, n), SALVAGE_MEANING.get(k, ""))
            for k, v in sorted(
                summary["by_salvage"].items(), key=lambda kv: -kv[1]
            )
        ]
        flag_headers = ["flag among the models that solve", "deposits"]
        flag_rows = list(summary["prevalence_among_solved"].items())
        headline = (
            f"**{n_pass} of {n:,} deposits clear the numerical intake gate "
            "with nothing flagged.**"
        )

    how_rows = list(summary["how_top"].items())
    notable = notable_cases(df)
    notable_rows = [
        (
            r.accession,
            _text(r.get("name"))[:70],
            _text(r.get("paper"))[:90],
            _authors(r)[:60],
            _text(r.get("about"))[:160],
            r.stage,
            _text(r.get("reason"))[:140],
            r.salvage,
            _text(r.get("how"))[:120],
        )
        for _, r in notable.iterrows()
    ]
    kinds = ", ".join(
        f"{k} {v}"
        for k, v in sorted(summary["by_kind"].items(), key=lambda kv: -kv[1])
    )

    blog = "\n".join(
        [
            f"# How much of BioModels can a composition framework actually "
            f"use? — {stamp}",
            "",
            headline
            + f" {fun['kinetic']:,} of {n:,} are kinetic models at all, "
            f"{fun['imports']:,} compile, {fun['solves']:,} solve. The rest "
            "is not noise: every failure has a class, and most classes have "
            "a fix.",
            "",
            "## What was run",
            "",
            f"Every SBML accession the BioModels search lists ({n:,} on "
            f"{stamp}) went through the same gate `simulate screen` applies "
            "to one model: read the file, classify its formalism from the "
            "SBML packages it carries, compile it with HallSim's native SBML "
            "importer, solve it once on its own clock over "
            f"{t_end:g} native time units at two tolerances, take one "
            "gradient, and read its metadata (declared time unit, ontology "
            "coverage, distance of the published initial condition from a "
            "rest state). Each deposit had "
            f"{cfg.get('timeout', DEFAULT_TIMEOUT):.0f} s; "
            f"{summary['timeouts']} ran out. Median time per deposit "
            f"{summary['seconds_median']:.1f} s.",
            "",
            "## The funnel",
            "",
            _md_table(funnel_headers, funnel_rows),
            "",
            "![funnel](funnel.png)",
            "",
            "## Where they stop",
            "",
            _md_table(stage_headers, stage_rows),
            "",
            f"Formalism of everything listed: {kinds}.",
            "",
            "## What would lift them",
            "",
            _md_table(salvage_headers, salvage_rows),
            "",
            "![salvage](salvage.png)",
            "",
            "The most common concrete actions:",
            "",
            _md_table(["action", "deposits"], how_rows),
            "",
            "## Flags among the models that solve",
            "",
            _md_table(flag_headers, flag_rows),
            "",
            "## Cases worth naming",
            "",
            "Every framework defect and timeout, then the largest deposits "
            "in each other failing class. The full per-deposit table is in "
            "`failures.md`; every column, including the screen's numbers, is "
            "in `census.csv`.",
            "",
            _md_table(fail_headers, notable_rows),
            "",
            "## What this does not measure",
            "",
            "- The horizon is the same number of native time units for every "
            "model, so a second-scale model is screened over seconds and a "
            "day-scale one over days. A model that is still moving at the "
            "horizon is flagged, not rejected.",
            "- Whether a deposit reproduces its own paper is not checked "
            "here: `published_fit_chi2` needs deposited fitting data, which "
            f"{summary['with_data_files']} deposits ship, and the SED-ML "
            "runner is not built. An uncurated deposit that passes has been "
            "shown to run, not to match its figure.",
            "- Nothing here reads the paper. A model can pass every gate and "
            "be about the wrong cell type, the wrong stimulus, or the "
            "opposite fate from the one a slot needs; that is the review "
            "panel's job.",
            "- `annotated` is a threshold on species with any ontology id; "
            "it does not check that the ids are the right ones.",
        ]
    )

    writeup = "\n".join(
        [
            f"# BioModels census — {stamp}",
            "",
            f"Run directory: `{run}`. Files: `census.csv` (every column), "
            "`failures.md` (every failing deposit by name), `summary.json` "
            "(the counts), `funnel.png`, `salvage.png`.",
            "",
            "## Preprint paragraph",
            "",
            preprint,
            "",
            "## Blog section",
            "",
            blog,
            "",
        ]
    )
    (run / "writeup.md").write_text(writeup)
    return run / "writeup.md"
