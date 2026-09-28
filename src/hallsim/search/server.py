"""The search as tools an agent can call: an MCP server over
:mod:`hallsim.search`.

:func:`build_server` returns a FastMCP server with ``find_models``,
``find_data``, ``dataset_design``, ``paper_datasets`` and ``sources``,
each a thin call into the search package that returns plain
dictionaries. ``simulate mcp`` serves it over stdio for Claude Code or
Claude Desktop, adding the framework's screens, and anything that speaks
MCP can register it. Needs the ``mcp`` extra.
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
        29 repositories), Zenodo, GEO DataSets and the PEtab benchmark
        collection; ``sources=["all"]`` asks every repository directly."""
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
