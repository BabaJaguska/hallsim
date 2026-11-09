"""The search, and what opens a hit, as tools an agent can call: an MCP
server over :mod:`hallsim.search` and the intake gate.

:func:`build_server` returns a FastMCP server with ``find_models``,
``find_data``, ``dataset_design``, ``paper_datasets`` and ``sources``,
each a thin call into the search package that returns plain
dictionaries. :func:`framework_tools` adds the tools that open a deposit:
``screen_models``, ``triage_model`` and the shipped census snapshots.
Composing and simulating need a live Python object and stay in code.
``supply mcp`` serves both over stdio for Claude Code or Claude Desktop,
and anything that speaks MCP can register it. Needs the ``mcp`` extra.
"""

from __future__ import annotations

from dataclasses import asdict

from hallsim.search.datasets import (
    DEFAULT_SOURCES,
    SOURCES,
    DatasetCandidate,
    gds_design,
    gds_subsets,
    geo_by_accession,
    paper_data,
    search_for_dataset,
)
from hallsim.search.models import SOURCES as MODEL_SOURCES
from hallsim.search.models import search_for_model


def dataset_record(cand: DatasetCandidate) -> dict:
    """A candidate as a dictionary, its design spelled out."""
    record = asdict(cand)
    record["design"] = cand.design.to_dict()
    record["design_summary"] = cand.design.summary()
    return record


def build_server(name: str = "hallsim-search"):
    """The MCP server, tools registered; more can be added with
    ``server.tool`` before it runs."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            'serving MCP needs pip install "hallsim[mcp]"'
        ) from exc

    mcp = FastMCP(
        name,
        instructions=(
            "Search for deposited models and datasets across the public "
            "repositories. find_data reads each hit's experimental design "
            "(arms, control, timepoints); find_models returns deposits an "
            "importer can take by id."
        ),
    )

    @mcp.tool
    def find_models(
        query: str, limit: int = 10, sources: list[str] | None = None
    ) -> list[dict]:
        """Deposited models matching a query, across BioModels, JWS Online,
        ModelDB, BioSimulations, Physiome and Europe PMC supplements.
        ``sources`` narrows to names from ``sources()``."""
        return [
            asdict(c)
            for c in search_for_model(query, limit=limit, sources=sources)
        ]

    @mcp.tool
    def find_data(
        query: str,
        limit: int = 10,
        organism: str | None = None,
        sources: list[str] | None = None,
    ) -> list[dict]:
        """Datasets matching a query, each with its design: arms, control
        arm, timepoints, and whether the source states the design or it was
        read from sample titles. Default sources: OmicsDI (one index over
        29 repositories), Zenodo, GEO DataSets, the PEtab benchmark
        collection and ENCODE; ``sources=["all"]`` asks every repository
        directly. A deposit several sources list comes back once."""
        names = list(SOURCES) if sources == ["all"] else sources
        return [
            dataset_record(c)
            for c in search_for_dataset(
                query, limit=limit, sources=names, organism=organism
            )
        ]

    @mcp.tool
    def dataset_design(accession: str) -> dict:
        """The design of a GEO series (``GSE…``) read from its sample titles,
        or of a GEO DataSet (``GDS…``) read from its curated subsets."""
        acc = accession.strip().upper()
        if acc.startswith("GDS"):
            design = gds_design(gds_subsets(acc))
            return {
                "accession": acc,
                **design.to_dict(),
                "summary": design.summary(),
            }
        cand = geo_by_accession(acc)
        if cand is None:
            return {"accession": acc, "error": "not a GEO series"}
        return dataset_record(cand)

    @mcp.tool
    def paper_datasets(pubmed: str, with_files: bool = False) -> dict:
        """What Europe PMC links to a paper: the datasets it deposits or
        cites, its supplement, the chemicals mined from its text and the
        BioModels deposits built from it."""
        paper = paper_data(pubmed, with_files=with_files)
        record = asdict(paper)
        record["datasets"] = [dataset_record(d) for d in paper.datasets]
        return record

    @mcp.tool
    def sources() -> dict:
        """The registered model and data sources, and the data sources
        searched by default."""
        return {
            "models": list(MODEL_SOURCES),
            "data": list(SOURCES),
            "data_default": list(DEFAULT_SOURCES),
        }

    return mcp


def _number(x) -> float | None:
    """A float for JSON: NaN and the infinities become ``None``."""
    x = float(x)
    return x if x == x and abs(x) != float("inf") else None


def verdict_record(verdict) -> dict:
    """A triage verdict as a dictionary, the numerical screen as its text."""
    return {
        "name": verdict.name,
        "status": verdict.status,
        "escalate": verdict.escalate,
        "blockers": list(verdict.blockers),
        "flags": list(verdict.flags),
        "n_species": verdict.n_species,
        "n_parameters": verdict.n_parameters,
        "time_unit_declared": verdict.time_unit_declared,
        "annotation_coverage": _number(verdict.annotation_coverage),
        "rest_residual": _number(verdict.rest_residual),
        "screen": None if verdict.screen is None else str(verdict.screen),
    }


def framework_tools(mcp):
    """Register, on a server from :func:`build_server`, the tools that open
    a deposit: the role a species plays in it, the intake triage, and the
    shipped census tables. Kept apart from the search tools because these
    import the framework."""
    from hallsim.census_snapshots import (
        find_shipped_datasets,
        find_shipped_models,
    )

    @mcp.tool
    def screen_models(
        model_ids: list[str], pattern: str, roles: list[str] | None = None
    ) -> list[dict]:
        """What role each species matching ``pattern`` (a regex against
        species ids and names) plays in each deposit (BioModels accessions,
        ``jws:`` slugs or ``PMC`` ids from find_models): produces, consumes,
        reads (a modifier or rate-law symbol) or input (boundary or
        constant). ``status`` is ``match`` where one plays any of ``roles``
        (default all). A consumer or reader is the downstream half of a
        composite; the producer is an edge to attach."""
        from dataclasses import asdict

        from hallsim.screens import ROLES, screen_species_roles

        return [
            asdict(r)
            for r in screen_species_roles(
                model_ids, pattern, roles=tuple(roles or ROLES)
            )
        ]

    @mcp.tool
    def triage_model(
        model_id: str,
        source: str = "biomodels",
        t_end: float | None = None,
        produces: str | None = None,
    ) -> dict:
        """Import one deposit and run the intake gate on it, from any
        repository find_models lists: ``source`` is the hit's source
        (biomodels, jws, modeldb, physiome, biosimulations, europepmc) and
        ``model_id`` its id. ``status`` is ``pass``; ``flag`` (it runs, and
        each flag says what to do); ``reject`` (it will not import, or
        produced no usable trajectory); or ``unscreenable`` (it could not be
        fetched, or holds no readable model file, so nothing is known about
        the model). ``t_end`` is the solo solve window in the model's own
        time unit, by default the deposit's authored SED-ML horizon or 10;
        ``produces`` is a regex for a quantity the deposit must emit.
        Imports and solves the model, so a call takes seconds to minutes."""
        from hallsim import intake

        return verdict_record(
            intake.triage_model(
                model_id, source, t_end=t_end, produces=produces
            )
        )

    mcp.tool(find_shipped_models, name="census_models")
    mcp.tool(find_shipped_datasets, name="census_datasets")
    return mcp
