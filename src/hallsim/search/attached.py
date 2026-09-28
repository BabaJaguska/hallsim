"""The data a model ships beside itself.

The time courses that built the kinetic models in the repositories rarely
reach a data repository: they sit next to the model. Three places hold
them in a form a machine can read. The PEtab benchmark collection pairs
each model with its measurement table, condition by condition and time by
time, read here through the ``petab`` library (:func:`petab_problems`,
:func:`search_petab`). A BioModels deposit's additional files sometimes
include the fitting data as a table, and a COPASI file's parameter
estimation task names the experiment files it was fitted to and says
whether each is a time course, read through COPASI's own bindings
(:func:`copasi_experiments`, :func:`biomodels_data`). Each becomes a
:class:`~hallsim.search.datasets.DatasetCandidate` whose design is stated
by the table rather than read from sample titles.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from hallsim.search.datasets import (
    CONTROL_WORDS,
    TABLE_SUFFIXES,
    DatasetCandidate,
    Design,
    Measured,
    curie,
    table_design,
)
from hallsim.search.fetch import cached_index, get_json
from hallsim.search.models import download_biomodel_files, term_score

log = logging.getLogger(__name__)


# ── SBML annotations ────────────────────────────────────────────────

_IDENTIFIERS = re.compile(
    r"(?:identifiers\.org/|urn:miriam:)(?P<ns>[A-Za-z0-9.]+)[/:]"
    r"(?P<id>[^\s\"'/]+)"
)


def sbml_species_curies(model) -> tuple[str, ...]:
    """Every ``namespace:id`` a libsbml model's species annotations name,
    from the identifiers.org and MIRIAM URIs in their CV terms; the join
    key a dataset's measured quantities are matched on."""
    out: list[str] = []
    for i in range(model.getNumSpecies()):
        species = model.getSpecies(i)
        for k in range(species.getNumCVTerms()):
            term = species.getCVTerm(k)
            for j in range(term.getNumResources()):
                m = _IDENTIFIERS.search(term.getResourceURI(j))
                if m:
                    out.append(curie(m.group("ns"), unquote(m.group("id"))))
    return tuple(dict.fromkeys(out))


# ── PEtab: the benchmark collection of models with their data ───────

PETAB_REPO = "Benchmarking-Initiative/Benchmark-Models-PEtab"
PETAB_TREE = f"https://api.github.com/repos/{PETAB_REPO}/git/trees/master"
PETAB_RAW = f"https://raw.githubusercontent.com/{PETAB_REPO}/master"
PETAB_URL = (
    f"https://github.com/{PETAB_REPO}/tree/master/Benchmark-Models/{{}}"
)


def petab_design(measurements, conditions=None) -> Design:
    """The design a PEtab measurement table states: one arm per simulation
    condition (named through the condition table where it names them),
    the ``time`` column's distinct values per arm. PEtab declares no time
    unit; the model's clock is the unit."""
    names = {}
    if conditions is not None and "conditionId" in conditions:
        label = (
            conditions["conditionName"]
            if "conditionName" in conditions
            else conditions["conditionId"]
        )
        names = dict(zip(conditions["conditionId"], label.fillna("")))
    per_arm: dict[str, set] = {}
    cond = measurements.get("simulationConditionId")
    times = pd.to_numeric(measurements.get("time"), errors="coerce")
    for c, t in zip(cond if cond is not None else [""] * len(times), times):
        arm = str(names.get(c) or c or "")
        per_arm.setdefault(arm, set())
        if pd.notna(t):
            per_arm[arm].add(round(float(t), 6))
    arms = tuple(sorted(per_arm))
    control = next(
        (
            a
            for a in arms
            if any(tok.lower() in CONTROL_WORDS for tok in a.split())
        ),
        None,
    )
    return Design(
        arms=arms,
        control=control,
        per_arm=tuple((a, tuple(sorted(per_arm[a]))) for a in arms),
        n_titles=int(len(measurements)),
    )


def load_problem(url: str):
    """A PEtab problem from its YAML, local or remote; ``petab`` reads the
    tables and the model the YAML names."""
    from petab.v1 import Problem

    return Problem.from_yaml(url)


def _build_petab_index() -> list[dict]:
    """One record per benchmark problem: its files, the design its
    measurement table states, the species its model annotates."""
    tree = get_json(PETAB_TREE, {"recursive": 1}, 60.0).get("tree", [])
    by_problem: dict[str, list[str]] = {}
    for entry in tree:
        parts = entry["path"].split("/")
        if parts[0] == "Benchmark-Models" and len(parts) == 3:
            by_problem.setdefault(parts[1], []).append(parts[2])
    log.info("petab: %d problems (one-time, cached)", len(by_problem))
    out = []
    for name, files in sorted(by_problem.items()):
        yaml = next((f for f in files if f.endswith(".yaml")), "")
        if not yaml:
            continue
        try:
            problem = load_problem(
                f"{PETAB_RAW}/Benchmark-Models/{name}/{yaml}"
            )
        except Exception as exc:  # noqa: BLE001 - one problem
            log.info("petab %s: not read (%s)", name, exc)
            continue
        conditions = problem.condition_df
        design = petab_design(
            problem.measurement_df,
            conditions.reset_index() if conditions is not None else None,
        )
        observables = problem.observable_df
        sbml = getattr(problem.model, "sbml_model", None)
        out.append(
            {
                "name": name,
                "files": files,
                "design": design.to_dict(),
                "observables": (
                    sorted(str(o) for o in observables.index)
                    if observables is not None
                    else []
                ),
                "ids": list(sbml_species_curies(sbml)) if sbml else [],
                "n_measurements": int(len(problem.measurement_df)),
            }
        )
    return out


def _petab_candidate(rec: dict) -> DatasetCandidate:
    return DatasetCandidate(
        source="petab",
        accession=rec["name"],
        title=rec["name"].replace("_", " "),
        kind="model with measurements",
        organism="",
        n_samples=int(rec.get("n_measurements") or 0),
        platform="",
        url=PETAB_URL.format(rec["name"]),
        summary="observables: " + ", ".join(rec.get("observables", [])),
        files=tuple(rec.get("files", ())),
        measured=Measured("targeted", False, tuple(rec.get("ids", ()))),
        stated=Design.from_dict(rec.get("design", {})),
    )


def petab_problems(refresh: bool = False) -> list[DatasetCandidate]:
    """Every problem in the PEtab benchmark collection as a candidate."""
    return [
        _petab_candidate(r)
        for r in cached_index("petab", _build_petab_index, refresh=refresh)
    ]


def search_petab(
    query: str,
    limit: int = 25,
    *,
    organism: str | None = None,
    timeout: float = 30.0,
    refresh: bool = False,
) -> list[DatasetCandidate]:
    """PEtab benchmark problems matching ``query`` by name, observable or
    condition. Thirty-odd problems, each a model with the time courses it
    was fitted to."""
    scored = []
    for rec in cached_index("petab", _build_petab_index, refresh=refresh):
        score = term_score(
            query,
            rec["name"].replace("_", " "),
            " ".join(rec.get("observables", [])),
            " ".join(a for a in rec.get("design", {}).get("arms", [])),
        )
        if score:
            scored.append((score, rec))
    scored.sort(key=lambda sr: -sr[0])
    log.info("petab '%s': %d candidates", query, len(scored))
    return [_petab_candidate(r) for _, r in scored[:limit]]


# ── COPASI: what a model was fitted to ──────────────────────────────


@dataclass(frozen=True)
class CopasiExperiment:
    """One experiment in a COPASI parameter-estimation task."""

    name: str
    #: The data file as the task names it, often a path on the author's
    #: machine; only its base name can be looked for in the deposit.
    file: str
    #: ``time course`` or ``steady state``.
    kind: str
    n_rows: int = 0

    @property
    def basename(self) -> str:
        return Path(self.file.replace("\\", "/")).name


def copasi_experiments(path) -> tuple[CopasiExperiment, ...]:
    """The experiments a COPASI file's fitting task declares, read by
    COPASI itself through ``basico``. An empty tuple means the model
    carries no fitting experiments, or the file is not COPASI's."""
    try:
        import basico
        from COPASI import CTaskEnum
    except ImportError as exc:
        raise ImportError(
            'reading a COPASI file needs pip install "hallsim[copasi]"'
        ) from exc
    model = basico.load_model(str(path))
    if model is None:
        return ()
    try:
        out = []
        for i, name in enumerate(basico.get_experiment_names()):
            exp = basico.get_experiment(i)
            first, last = exp.getFirstRow(), exp.getLastRow()
            out.append(
                CopasiExperiment(
                    name=name,
                    file=exp.getFileName(),
                    kind=(
                        "time course"
                        if exp.getExperimentType() == CTaskEnum.Task_timeCourse
                        else "steady state"
                    ),
                    n_rows=max(int(last) - int(first), 0),
                )
            )
        return tuple(out)
    finally:
        basico.remove_datamodel(model)


# ── A BioModels deposit's own data ──────────────────────────────────


def biomodels_data(
    accession: str,
    files,
    *,
    name: str = "",
    organism: str = "",
    pubmed: str = "",
    ids=(),
    timeout: float = 60.0,
) -> DatasetCandidate | None:
    """The data a BioModels deposit ships beside its model, as a candidate,
    or ``None`` when it ships none. ``files`` is the deposit's file list.
    Table-like files are downloaded and read for a design; a COPASI file's
    fitting experiments are read for what the model was fitted to, and
    their data files read too where the deposit ships them."""
    files = [f for f in files if f]
    tables = [
        f
        for f in files
        if f.lower().endswith(TABLE_SUFFIXES)
        and not f.lower().startswith(("curation_notes", "readme", "manifest"))
    ]
    copasi = [f for f in files if f.lower().endswith(".cps")]
    if not tables and not copasi:
        return None
    local = {
        p.name: p
        for p in download_biomodel_files(
            accession, timeout=timeout, names=tables + copasi
        )
    }
    factors: list[str] = []
    for f in copasi:
        path = local.get(f)
        if path is None:
            continue
        try:
            experiments = copasi_experiments(path)
        except ImportError as exc:
            log.info("%s: %s", accession, exc)
            break
        except Exception as exc:  # noqa: BLE001 - one file
            log.info("%s: %s not read (%s)", accession, f, exc)
            continue
        for exp in experiments:
            factors.append(f"COPASI {exp.kind}: {exp.basename}")
            if exp.basename in files and exp.basename not in tables:
                tables.append(exp.basename)
    if not tables and not factors:
        return None
    missing = [f for f in tables if f not in local]
    if missing:
        local.update(
            {
                p.name: p
                for p in download_biomodel_files(
                    accession, timeout=timeout, names=missing
                )
            }
        )
    stated = table_design([local[f] for f in tables if f in local])
    return DatasetCandidate(
        source="biomodels",
        accession=accession,
        title=name or accession,
        kind="model deposit",
        organism=organism,
        n_samples=stated.n_titles if stated else 0,
        platform="",
        url=f"https://www.ebi.ac.uk/biomodels/{accession}#Files",
        summary="data shipped with the model: " + ", ".join(tables),
        files=tuple(tables),
        measured=Measured("targeted", False, tuple(ids)),
        pubmed=pubmed,
        factors=tuple(dict.fromkeys(factors)),
        stated=stated,
    )


__all__ = [
    "CopasiExperiment",
    "biomodels_data",
    "copasi_experiments",
    "load_problem",
    "petab_design",
    "petab_problems",
    "sbml_species_curies",
    "search_petab",
]
